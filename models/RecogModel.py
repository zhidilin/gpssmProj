# recognition network is using to parameterize the initial latent state
# i.e.,  q(x0 | y[1:T] ) = N(m0, Sigma0), where Sigma0 is a diagonal covariance matrix
# T is the observations length


from torch import Tensor
import torch
import torch.nn as nn
from gpytorch.distributions import MultivariateNormal
from .utils import inverse_softplus, safe_softplus

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
        Recognition length.
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

        self.lstm = nn.LSTM(input_size=self.dim_inputs + self.dim_outputs,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=batch_first,
                            bidirectional=bd
                            )
        in_features = self.hidden_size * (1+bd)
        self.mean = nn.Linear(in_features=in_features, out_features=dim_states)
        self.var = nn.Linear(in_features=in_features, out_features=dim_states)
        self.var.bias = nn.Parameter(torch.ones(self.dim_states) * variance, requires_grad=True)

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

        hidden = (torch.zeros(num_layers, batch_size, self.lstm.hidden_size).to(device),
                  torch.zeros(num_layers, batch_size, self.lstm.hidden_size).to(device))

        out, _ = self.lstm(io_sequence, hidden)
        x = out[:, -1]

        # loc = self.mean(x).expand(num_particles, batch_size, self.dim_states).permute(1, 0, 2)
        # cov = safe_softplus(self.var(x)).expand(num_particles, batch_size, self.dim_states).permute(1, 0, 2)

        loc = self.mean(x)                   # shape: batch_size x dim_state
        cov = safe_softplus(self.var(x))     # shape: batch_size x dim_state

        # # expand along with N_MC
        # loc = loc.expand(N_MC, batch_size, self.dim_states).permute(1, 2, 0)   # shape: batch_size x dim_state x N_MC
        # cov = cov.expand(N_MC, batch_size, self.dim_states).permute(1, 2, 0)   # shape: batch_size x dim_state x N_MC

        return loc, torch.diag_embed(cov)