# -*- coding: utf-8 -*-
# utils_flow_demo.py : file containing flow base class (simplified version)
# Authors: Zhidi Lin, Juan Maroñas

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softplus
# custom
from EnVI.utils import settings as cg


class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask):
        super(RealNVP, self).__init__()

        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(mask))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(mask))])

    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self, x):
        z, logp = self.f(x)
        return self.prior.log_prob(z) + logp

class Four_Layer_NN(nn.Module):
  def __init__(self, input_dim, output_dim, residual=True, activation="relu", hidden_dim_list=None):
    super().__init__()
    if hidden_dim_list is None:
        hidden_dim_list = [16, 64, 32]
    if activation == "relu":
      self.activation = torch.relu
    elif activation == "tanh":
      self.activation = torch.tanh
    self.residual = residual
    self.output_dim = output_dim
    self.layer1 = nn.Linear(input_dim, hidden_dim_list[0])
    self.layer2 = nn.Linear(hidden_dim_list[0], hidden_dim_list[1])
    self.layer3 = nn.Linear(hidden_dim_list[1], hidden_dim_list[2])
    self.layer4 = nn.Linear(hidden_dim_list[2], output_dim)

  def forward(self, x):
    res = self.activation(self.layer1(x))
    res = self.activation(self.layer2(res))
    res = self.activation(self.layer3(res))
    res = self.layer4(res)
    if self.residual:
      out = res + x
    else:
      out = res
    return out

class Flow(nn.Module):
    """ General Flow Class.
        All flows should inherit and overwrite this method
    """

    def __init__(self) -> None:
        super(Flow, self).__init__()

    def forward(self, f0: torch.tensor, X: torch.tensor = None) -> torch.tensor:
        raise NotImplementedError("Not Implemented")

    def forward_initializer(self, X):
        # just return 0 if it is not needed
        raise NotImplementedError("Not Implemented")

class Linear_Flow(Flow):
    def __init__(self, output_dim, NN_parameterized=False, input_dim=None) -> None:
        """
        control_input_dim: int,  control input dimension

        """
        super(Linear_Flow, self).__init__()
        self.NN_parameterized = NN_parameterized
        self.output_dim = output_dim

        if NN_parameterized:
            if input_dim is None:
                self.input_dim = 1
            else:
                self.input_dim = input_dim
            self.NN_par = Four_Layer_NN(input_dim=self.input_dim, output_dim=2*output_dim, residual=False)

        else:
            init_c = np.random.randn(output_dim)
            init_d = np.zeros(output_dim)
            # linear part
            self.c = nn.Parameter(torch.tensor(init_c, dtype=cg.dtype))
            self.d = nn.Parameter(torch.tensor(init_d, dtype=cg.dtype))

    def forward(self, f0: torch.tensor, input_data: torch.tensor = None) -> torch.tensor:
        """
        f0:          shape: ***** x self.output_dim
        input_data:  shape: batch_size x (control_input_dim + 1)
                        includes control input, c_t, with shape: batch_size x control_input_dim
                        and the time step t.
        """
        if f0.shape[-1] == 1:
            f0 = torch.ones(self.output_dim, device=cg.device, dtype=cg.dtype) * f0   # f0: batch_size x output_dim
        assert f0.shape[-1] == self.output_dim

        if self.NN_parameterized:
            assert input_data is not None, "should have input for neural networks"
            assert input_data.shape[-1] == self.input_dim, "input_data dimension is expected to be [***, ctrl_input_dim + 1]"
            params = self.NN_par(input_data)   # shape: batch_size x (2 * output_dim)
            c = params[:, :self.output_dim]    # shape: batch_size x output_dim
            d = params[:, self.output_dim:]    # shape: batch_size x output_dim

            # Linear part
            fk = c * f0 + d  # f0: batch_size x output_dim     fk: batch_size x output_dim
        else:
            # Linear part
            fk = self.c * f0 + self.d    # f0: batch_size x output_dim     fk: batch_size x output_dim
        return fk

class SAL_Flow(Flow):
    def __init__(self, output_dim, NN_parameterized=False) -> None:
        super(SAL_Flow, self).__init__()
        if NN_parameterized:
            # TODO
            # self.NN = Four_Layer_NN(input_dim=)
            pass
        else:
            init_a = np.zeros(output_dim)
            # init_b = np.random.randn(output_dim)
            init_b = np.ones(output_dim)

            # init_c = np.random.randn(output_dim)
            init_c = np.ones(output_dim)
            init_d = np.zeros(output_dim)

            # Sinh-Asinh part
            self.a = nn.Parameter(torch.tensor(init_a, dtype=cg.dtype))
            self.b = nn.Parameter(torch.tensor(init_b, dtype=cg.dtype))

            # linear part
            self.c = nn.Parameter(torch.tensor(init_c, dtype=cg.dtype))
            self.d = nn.Parameter(torch.tensor(init_d, dtype=cg.dtype))

        self.set_restrictions = False

    def asinh(self, f: torch.tensor) -> torch.tensor:
        return torch.log(f + (f ** 2 + 1) ** (0.5))

    def forward(self, f0: torch.tensor, X: torch.tensor = None) -> torch.tensor:
        a = self.a
        b = self.b
        if self.set_restrictions:
            b = softplus(b)

        # SA part
        fk_ = torch.sinh(b * self.asinh(f0) - a)    # f0: [batch_size x 1]     fk_: batch_size x output_dim

        # Linear part
        fk = self.c * fk_ + self.d    # fk_: batch_size x output_dim     fk: batch_size x output_dim
        return fk

class SAL_Flows(Flow):
    def __init__(self, num_blocks, output_dim) -> None:
        super(SAL_Flows, self).__init__()

         # list of blocks of SAL flow
        self.SAL_blocks_list = [SAL_Flow(output_dim=output_dim) for _ in range(num_blocks)]
        self.SAL_blocks = nn.Sequential(*self.SAL_blocks_list)

    def forward(self, f0: torch.tensor, X: torch.tensor = None) -> torch.tensor:
        fk = self.SAL_blocks(f0)
        return fk

class Tanh_Flow(Flow):
    def __init__(self, output_dim, NN_parameterized=False) -> None:
        super(Tanh_Flow, self).__init__()
        if NN_parameterized:
            # TODO
            # self.NN = Four_Layer_NN(input_dim=)
            pass
        else:
            init_a = np.ones(output_dim)
            # init_b = np.random.randn(output_dim)
            init_b = np.ones(output_dim)

            # init_c = np.random.randn(output_dim)
            init_c = np.zeros(output_dim)
            init_d = np.zeros(output_dim)

            # linear part
            self.a = nn.Parameter(torch.tensor(init_a, dtype=cg.dtype))
            self.d = nn.Parameter(torch.tensor(init_d, dtype=cg.dtype))

            # Tanh part
            self.b = nn.Parameter(torch.tensor(init_b, dtype=cg.dtype))
            self.c = nn.Parameter(torch.tensor(init_c, dtype=cg.dtype))

        self.set_restrictions = False

    def forward(self, f0: torch.tensor, X: torch.tensor = None) -> torch.tensor:
        a = self.a
        b = self.b
        if self.set_restrictions:
            b = softplus(b)

        # tanh part
        fk_ = torch.tanh(b * (f0 + self.c))    # f0: [batch_size x 1]     fk_: batch_size x output_dim

        # Linear part
        fk = a * fk_ + self.d                 # fk_: batch_size x output_dim     fk: batch_size x output_dim
        return fk

class Tanh_Flows(Flow):
    def __init__(self, num_blocks, output_dim) -> None:
        super(Tanh_Flows, self).__init__()

         # list of blocks of Tanh flow
        self.Tanh_blocks_list = [Tanh_Flow(output_dim=output_dim) for _ in range(num_blocks)]
        self.Tanh_blocks = nn.Sequential(*self.Tanh_blocks_list)

    def forward(self, f0: torch.tensor, X: torch.tensor = None) -> torch.tensor:
        fk = self.Tanh_blocks(f0)
        return fk


def initialize_flows(flow_specs):
    """
    Initializes the flows applied on the prior. Flow_specs is a list with an instance of the flow used per output GP.
    """
    G_matrix = []
    for idx, fl in enumerate(flow_specs):
        G_matrix.append(fl)
    G_matrix = nn.ModuleList(G_matrix)
    return G_matrix

def instance_flow(flow_list, is_composite=True):
    """
     From these flows only Box-Cox, sinh-arcsinh and affine return to the identity
    """
    FL = []
    for flow_name in flow_list:

        flow_name, init_values = flow_name

        if flow_name == 'affine':
            fl = AffineFlow(**init_values)

        elif flow_name == 'sinh_arcsinh':
            fl = Sinh_ArcsinhFlow(**init_values)

        elif flow_name == 'identity':
            fl = IdentityFlow()

        else:
            raise ValueError("Unkown flow identifier {}".format(flow_name))

        FL.append(fl)

    if is_composite:
        return CompositeFlow(FL)
    return FL

class CompositeFlow(Flow):
    def __init__(self, flow_arr: list) -> None:
        """
            Args:
                flow_arr: is an array of flows. The first element is the first flow applied.
        """
        super(CompositeFlow, self).__init__()
        self.flow_arr = nn.ModuleList(flow_arr)

    def forward(self, f: torch.tensor, X: torch.tensor = None) -> torch.tensor:
        for flow in self.flow_arr:
            f = flow.forward(f, X)
        return f

    def forward_initializer(self, X: torch.tensor):
        loss = 0.0
        for flow in self.flow_arr:
            loss += flow.forward_initializer(X)
        return loss

class IdentityFlow(Flow):
    """ Identity Flow
           fk = f0
    """

    def __init__(self) -> None:
        super(IdentityFlow, self).__init__()
        self.input_dependent = False

    def forward(self, f0: torch.tensor, X: torch.tensor = None) -> torch.tensor:
        return f0

class AffineFlow(Flow):
    def __init__(self, init_a: float, init_b: float, set_restrictions: bool) -> None:
        ''' Affine Flow
            fk = a*f0+b
            * recovers the identity for a = 1 b = 0
            * a has to be strictly possitive to ensure invertibility if this flow is used in a linear
            combination, i.e with the step flow
            Args:
                a                (float) :->: Initial value for the slope
                b                (float) :->: Initial value for the bias
                set_restrictions (bool)  :->: If true then a >= 0 using  a = softplus(a)
        '''
        super(AffineFlow, self).__init__()
        self.a = nn.Parameter(torch.tensor(init_a, dtype=cg.dtype))
        self.b = nn.Parameter(torch.tensor(init_b, dtype=cg.dtype))

        self.set_restrictions = set_restrictions

    def forward(self, f0: torch.tensor, X: torch.tensor = None) -> torch.tensor:
        a = self.a
        if self.set_restrictions:
            a = softplus(a)
        b = self.b
        return a * f0 + b

class Sinh_ArcsinhFlow(Flow):
    def __init__(self, init_a: float, init_b: float, add_init_f0: bool, set_restrictions: bool) -> None:
        ''' SinhArcsinh Flow
          fk = sinh( b*arcsinh(f) - a)
          * b has to be strictkly possitive when used in a linear combination so that function is invertible.
          * Recovers the identity function

          Args:
                 init_a           (float) :->: initial value for a. Only used if input_dependent = False. Also used by the initializer if input_dependent = True so
                                               that NNets parameters are matched to take this value.
                 init_b           (float) :->: initial value for b. Only used if input_dependent = False. Also used by the initializer if input_dependent = True so
                                               that NNets parameters are matched to take this value.
                 set_restrictions (bool)  :->: if true then b > 0 with b = softplus(b)
                 add_init_f0      (bool)  :->: if true then fk = f0 + sinh( b*arcsinh(f) - a)
                                               If true then set_restrictions = True
        '''
        super(Sinh_ArcsinhFlow, self).__init__()

        self.a = nn.Parameter(torch.tensor(init_a, dtype=cg.dtype))
        self.b = nn.Parameter(torch.tensor(init_b, dtype=cg.dtype))

        if add_init_f0:
            set_restrictions = True

        self.set_restrictions = set_restrictions
        self.add_init_f0 = add_init_f0

    def asinh(self, f: torch.tensor) -> torch.tensor:
        return torch.log(f + (f ** 2 + 1) ** (0.5))

    def forward(self, f0: torch.tensor, X: torch.tensor = None) -> torch.tensor:
        # assert self.is_initialized, "This flow hasnt been initialized. Either set self.is_initialized = False or use an initializer"

        a = self.a
        b = self.b
        if self.set_restrictions:
            b = softplus(b)
        fk = torch.sinh(b * self.asinh(f0) - a)

        if self.add_init_f0:
            return fk + f0
        return fk

def SAL(num_blocks):
    block_array = []
    for nb in range(num_blocks):
        a_aff, b_aff = np.random.randn(1), 0.0
        a_sal, b_sal = 0.0, np.random.randn(1)

        init_affine = {'init_a': a_aff, 'init_b': b_aff, 'set_restrictions': False}
        init_sinh_arcsinh = {'init_a': a_sal, 'init_b': b_sal, 'add_init_f0': False, 'set_restrictions': False}
        block = [('sinh_arcsinh', init_sinh_arcsinh), ('affine', init_affine)]
        block_array.extend(block)
    return block_array

def linearFlow():
    block_array = []
    a_aff, b_aff = np.random.randn(1), 0.0
    init_affine = {'init_a': a_aff, 'init_b': b_aff, 'set_restrictions': False}
    block = [('affine', init_affine)]
    block_array.extend(block)
    return block_array

