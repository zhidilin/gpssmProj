# recognition network is using to parameterize the initial latent state
# i.e.,  q(x0 | y[1:T] ) = N(m0, Sigma0), where Sigma0 is a diagonal covariance matrix
# T is the observations length

import torch
import torch.nn as nn
from gpytorch.distributions import MultivariateNormal

def safe_softplus(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Safe softplus to return a softplus larger than epsilon.

    Parameters
    ----------
    x: torch.Tensor.
        Input tensor to transform.
    eps: float.
        Safety jitter.

    Returns
    -------
    output: torch.Tensor.
        Transformed tensor.
    """
    return nn.functional.softplus(x) + eps

def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    """Inverse function to torch.functional.softplus.

    Parameters
    ----------
    x: torch.Tensor.
        Input tensor to transform.

    Returns
    -------
    output: torch.Tensor.
        Transformed tensor.
    """
    return torch.log(torch.exp(x) - 1.)


class One_Layer_NN(nn.Module):
  def __init__(self, input_dim, output_dim, H=None, residual=False, bias=False):
    super().__init__()
    self.residual = residual
    self.layer1 = nn.Linear(input_dim, output_dim, bias=bias)
    if H is not None:
      self.layer1.weight.data = H
      self.H = H

  def forward(self, x):
    res = self.layer1(x)
    if self.residual:
      out = res + x
    else:
      out = res
    return out

class Two_Layer_NN(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_dim, residual=True, activation="relu", batchnorm=False):
    super().__init__()
    if activation == "relu":
      self.activation = torch.relu
    elif activation == "tanh":
      self.activation = torch.tanh
    self.residual = residual
    self.batchnorm = batchnorm
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.layer1 = nn.Linear(input_dim, hidden_dim)
    self.layer2 = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    # x = x[:,1].unsqueeze(1)
    res = x
    if self.batchnorm:
      res = nn.BatchNorm1d(self.input_dim)(res)
    res = self.activation(self.layer1(res))
    res = self.layer2(res)
    if self.residual:
      out = res + x
    else:
      out = res
    # out = res + 0.5*x[:,1].unsqueeze(1)
    return out

class Three_Layer_NN(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_dim, residual=True, activation="relu"):
    super().__init__()
    if activation == "relu":
      self.activation = torch.relu
    elif activation == "tanh":
      self.activation = torch.tanh
    self.residual = residual
    self.output_dim = output_dim
    self.layer1 = nn.Linear(input_dim, hidden_dim[0])
    self.layer2 = nn.Linear(hidden_dim[0], hidden_dim[1])
    self.layer3 = nn.Linear(hidden_dim[1], output_dim)

  def forward(self, x):
    res = self.activation(self.layer1(x))
    res = self.activation(self.layer2(res))
    res = self.layer3(res)
    if self.residual:
      out = res + x
    else:
      out = res
    return out

class Four_Layer_NN(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_dim, residual=True, activation="relu"):
    super().__init__()
    if activation == "relu":
      self.activation = torch.relu
    elif activation == "tanh":
      self.activation = torch.tanh
    self.residual = residual
    self.output_dim = output_dim
    self.layer1 = nn.Linear(input_dim, hidden_dim[0])
    self.layer2 = nn.Linear(hidden_dim[0], hidden_dim[1])
    self.layer3 = nn.Linear(hidden_dim[1], hidden_dim[2])
    self.layer4 = nn.Linear(hidden_dim[2], output_dim)

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

class Recognition(nn.Module):
    """Base Class for recognition Module.

    Parameters
    ----------
    dim_outputs: int.
        Dimension of the outputs.
    dim_inputs: int.
        Dimension of the inputs.
    dim_states: int.
        Dimension of the state.
    length: int.
        Recognition length/ sequence length
    """

    def __init__(self, dim_outputs: int, dim_inputs: int, dim_states: int, length: int ) -> None:
        super().__init__()
        self.dim_outputs = dim_outputs
        self.dim_inputs = dim_inputs
        self.dim_states = dim_states
        self.length = length

class LSTMRecognition(Recognition):
    """LSTM Based Recognition."""

    def __init__(self, dim_outputs, dim_inputs, dim_states, length,
                 hidden_size=32, num_layers=2, variance = 1.0, batch_first=True, bd=False):

        super().__init__(dim_outputs, dim_inputs, dim_states, length)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bd = bd

        self.lstm = nn.LSTM(input_size=self.dim_inputs + self.dim_outputs,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=batch_first,
                            bidirectional=bd
                            )
        in_features = self.hidden_size * (1+bd)
        self.mean = nn.Linear(in_features=in_features, out_features=dim_states)
        self.var = nn.Linear(in_features=in_features, out_features=dim_states)
        self.var.bias = nn.Parameter(torch.ones(self.dim_states) * torch.tensor(variance), requires_grad=True)

    def forward(self, output_sequence, input_sequence=None):
        """Forward execution of the recognition model.
        N_MC: number of particles
        """

        # Reshape input/output sequence:
        # output_sequence: [batch_sz x seq_len x dim_output]
        # input_sequence: [batch_sz x seq_len x dim_input]

        # 将输入序列倒序一下
        device = output_sequence.device
        output_sequence = output_sequence.flip(dims=[1])

        if input_sequence is None:
            input_sequence = torch.tensor([]).to(device)
        else:
            input_sequence = input_sequence.flip(dims=[1])

        batch_size = output_sequence.shape[0]

        io_sequence = torch.cat((output_sequence, input_sequence), dim=-1)

        num_layers = self.lstm.num_layers * (1 + self.lstm.bidirectional)

        hidden = (torch.zeros(num_layers, batch_size, self.lstm.hidden_size, device=device),
                  torch.zeros(num_layers, batch_size, self.lstm.hidden_size, device=device))

        out, _ = self.lstm(io_sequence, hidden)
        x = out[:, -1]

        # loc = self.mean(x).expand(num_particles, batch_size, self.dim_states).permute(1, 0, 2)
        # cov = safe_softplus(self.var(x)).expand(num_particles, batch_size, self.dim_states).permute(1, 0, 2)

        loc = self.mean(x)                   # shape: batch_size x dim_state
        cov = safe_softplus(self.var(x))     # shape: batch_size x dim_state

        # # expand along with N_MC
        # loc = loc.expand(N_MC, batch_size, self.dim_states).permute(1, 2, 0)   # shape: batch_size x dim_state x N_MC
        # cov = cov.expand(N_MC, batch_size, self.dim_states).permute(1, 2, 0)   # shape: batch_size x dim_state x N_MC

        return MultivariateNormal(loc, covariance_matrix=torch.diag_embed(cov))

class MFInference(Recognition):
    """LSTM Based Inference Network. Joint Gaussian variation distribution for the latent states.

    With Markov Gaussian Structure. Based on the paper:
        S. Eleftheriadis, et al. "Identification of Gaussian process state space models."  NeurIPS'2017.

    """

    def __init__(self, dim_outputs, dim_inputs, dim_states, length,
                 hidden_size=32, num_layers=2, batch_first=True, bd=True):

        super().__init__(dim_outputs, dim_inputs, dim_states, length)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=self.dim_inputs + self.dim_outputs,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=batch_first,
                            bidirectional=bd
                            )

        in_features = self.hidden_size * (1+bd)
        # self.At = nn.Linear(in_features=in_features, out_features=self.dim_states * self.dim_states)
        # self.raw_covar = nn.Linear(in_features=in_features, out_features=self.dim_states * self.dim_states)
        self.At = nn.Linear(in_features=in_features, out_features=self.dim_states)
        self.raw_covar = nn.Linear(in_features=in_features, out_features=self.dim_states)

    def forward(self, output_sequence, input_sequence=None):
        """Forward execution of the recognition model."""

        device = self.raw_covar.bias.device

        if input_sequence is None:
            input_sequence = torch.tensor([], device=device)
        else:
            input_sequence = input_sequence.flip(dims=[1])

        # Reshape input/output sequence:
        # output_sequence: [batch_sz x seq_len x dim_output]
        # input_sequence: [batch_sz x seq_len x dim_input]

        batch_size = output_sequence.shape[0]

        io_sequence = torch.cat((output_sequence, input_sequence), dim=-1)

        num_layers = self.lstm.num_layers * (1 + self.lstm.bidirectional)

        # hidden = (torch.zeros(num_layers, batch_size, self.lstm.hidden_size, device=device),
        #           torch.zeros(num_layers, batch_size, self.lstm.hidden_size, device=device))

        # out shpae: [batch_size, seq_len, (1+bd)*hidden_size], where we set 'batch_first = True' in LSTM
        # out, _ = self.lstm(io_sequence, hidden)
        out, _ = self.lstm(io_sequence)

        x = out

        At_all = self.At(x)  # At_all shape: batch_size x seq_len x dim_state
        At_all = torch.diag_embed(At_all) # At_all shape: batch_size x seq_len x dim_state x dim_state

        raw_cov = safe_softplus(self.raw_covar(x)) # raw_cov shape: batch_size x seq_len x dim_state
        Lt_all = torch.diag_embed(raw_cov) # raw_cov shape: batch_size x seq_len x dim_state x dim_state

        ''' the following part can be very unstable, by constructing a PSD covariance matrix '''
        # At_all = self.At(x)  # At_all shape: batch_size x seq_len x (dim_state x dim_state)
        # At_all = At_all.reshape(batch_size, self.length, self.dim_states, self.dim_states)
        # At_all = torch.clamp(At_all, min=-2, max=2)
        # # print(f"min: {At_all.min()}, max: {At_all.max()}")
        #
        # raw_cov = self.raw_covar(x) # raw_cov shape: batch_size x seq_len x (dim_state*dim_state)
        # raw_cov = raw_cov.reshape(batch_size, self.length, self.dim_states, self.dim_states)
        # raw_cov = safe_softplus(raw_cov)
        #
        # # First make the cholesky factor is lower triangular
        # lower_mask = torch.ones(raw_cov.shape[-2:], device=device).tril(0).expand(batch_size, self.length, self.dim_states, self.dim_states)
        # Lt_all = raw_cov.mul(lower_mask)

        return At_all, Lt_all


class LSTMencoder(Recognition):
    """LSTM Based Inference Network. Need to use with PostNet

    With Markov Gaussian Structure. Based on the paper:
        Rahul G. Krishnan, et al.
            "Structured Inference Networks for Nonlinear State Space Models." AAAI 2017

    """

    def __init__(self, dim_outputs, dim_inputs, dim_states, length,
                 hidden_size=32, num_layers=2, batch_first=True, bd=True):
        super().__init__(dim_outputs, dim_inputs, dim_states, length)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bd = bd

        self.lstm = nn.LSTM(input_size=self.dim_inputs + self.dim_outputs,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=batch_first,
                            bidirectional=bd
                            )

    def forward(self, output_sequence, input_sequence=None):
        """Forward execution of the recognition model."""

        device = output_sequence.device

        if input_sequence is None:
            input_sequence = torch.tensor([]).to(device)
        else:
            input_sequence = input_sequence.flip(dims=[1])

        # Reshape input/output sequence:
        # output_sequence: [batch_sz x seq_len x dim_output]
        # input_sequence: [batch_sz x seq_len x dim_input]

        batch_size = output_sequence.shape[0]

        io_sequence = torch.cat((output_sequence, input_sequence.to(device)), dim=-1)

        num_layers = self.lstm.num_layers * (1 + self.lstm.bidirectional)

        hidden = (torch.zeros(num_layers, batch_size, self.lstm.hidden_size).to(device),
                  torch.zeros(num_layers, batch_size, self.lstm.hidden_size).to(device))

        # out shpae: [batch_size, seq_len, (1+bd)*hidden_size], where we set 'batch_first = True' in LSTM
        out, _ = self.lstm(io_sequence, hidden)

        return out

class PostNet(nn.Module):
    """
    Parameterizes `q(x_t|x_{t-1}, y_{1:T})`, which is the basic building block of the inference (i.e. the variational distribution).
    The dependence on `y_{1:T}` is through the hidden state of the RNN

    With Markov Gaussian Structure. Based on the Structured Inference Networks mentioned in Section IV-B
    """
    def __init__(self, x_dim, h_dim, bd=True, mf_flag=True):
        super(PostNet, self).__init__()
        self.bd = bd
        self.flag_mf = mf_flag

        if not mf_flag:
            self.x_to_h = nn.Sequential(nn.Linear(x_dim, h_dim), nn.Tanh() )

        ''' 
        future work: 
        build the correlations between each dimensions of the latent state, i.e., out_feature = dim_states x dim_states
        '''
        self.h_to_inv_softplus_var = nn.Linear(in_features=h_dim, out_features=x_dim)
        self.h_to_mu = nn.Linear(in_features=h_dim, out_features=x_dim)

    def forward(self, x_t_1, hidden):
        """
        Given the latent x_t_1 at a particular time step t-1 as well as the hidden
        state of the RNN `h(y_{1:T})` we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution `q(x_t|x_{t-1}, y_{1:T})`

        # hidden shape: [batch_size x  (1+bd) * hidden_size]
        # x_t_1 shape:  [batch_size x state_dim]
        """
        h_tmp = hidden.view(hidden.shape[0], 1+self.bd, -1)    # hidden shape: [batch_size x (1+bd) x hidden_size]
        h_combined = h_tmp.sum(1).div(2)                       # h_combined shape: [batch_size x  hidden_size]

        if not self.flag_mf:
            h_combined = 1/3 * (self.x_to_h(x_t_1) + 2 * h_combined) # combine the LSTM hidden state with a transformed version of x_t_1

        mu = self.h_to_mu(h_combined)
        inv_soft_plus_var = self.h_to_inv_softplus_var(h_combined)
        var = safe_softplus(inv_soft_plus_var)

        epsilon = torch.randn(x_t_1.size(), device=x_t_1.device) # sampling x by re-parameterization
        x_t = epsilon * torch.sqrt(var) + mu                     # [batch_size x dim_states]
        return x_t, MultivariateNormal(mu, torch.diag_embed(var)).add_jitter()