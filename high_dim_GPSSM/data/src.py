"""Python Script Template."""
import torch
import numpy as np
from typing import Callable, Tuple

__all__ = ['safe_softplus', 'inverse_softplus', 'KL_divergence',
           'get_data_split', 'generate_trajectory', 'generate_batches', 'Normalizer']

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
    return torch.nn.functional.softplus(x) + eps


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

def KL_divergence(P, Q):

    """
    P: Multivariate
    Q: Multivariate

    return:
        KL( P||Q )
    """
    res = torch.distributions.kl.kl_divergence(P, Q)
    return res


def get_data_split(array: np.ndarray,
                   split_idx: int = None,
                   train: bool = True
                   ) -> np.ndarray:
    """Get data split by index.

    Parameters
    ----------
    array: np.ndarray
        Array to split with shape [n_experiment, time, dim].

    split_idx: int, optional.
        Splitting index (Default half).
        Indexes lower than `split_idx' are assigned to the train set and higher to test.

    train: bool, optional.
        Flag that indicates if the set is the train or the test set. (Default True)

    Returns
    -------
    array: np.ndarray
        Split array of shape [n_experiment, time, dim].

    """
    assert array.ndim == 3, "array must have shape [n_experiment, time, dim]"

    if split_idx is None:
        split_idx = array.shape[1] // 2    # // 取整除赋值运算符

    if train:
        if array.shape[0] > split_idx:  # Experiment split.
            array = array[:split_idx, :, :]
        else:  # Trajectory split.
            array = array[:, :split_idx, :]
    else: # test
        if array.shape[0] > split_idx:
            array = array[split_idx:, :, :]
        else:
            array = array[:, split_idx:, :]

    return array


def generate_batches(array: np.ndarray, sequence_length: int, stride_length: int
                     ) -> np.ndarray:
    """Generate batches from an array.

    An array has size [n_experiment, time, dim] and it returns an array of size
    [n_sub_sequences, sequence_length, dim]

    Parameters
    ----------
    array: np.ndarray
        array to reshape of size [n_experiment, time, dim].
    sequence_length: int
        length of batch sequence.
    stride_length: int
        down-sampling rate of array.Generate batches from an array.

    An array has size [n_experiment, time, dim] and it returns an array of size
    [n_sub_sequences, sequence_length, dim]

    Parameters
    ----------
    array: np.ndarray.
        array to reshape of shape [n_experiment, time, dim].

    sequence_length: int.
        length of batch sequence.

    stride_length: int.
        down-sampling rate of array.

    Returns
    -------
    array: np.ndarray.
        reshaped array  of size [n_sub_sequences, sequence_length, dim].

    """
    assert array.ndim == 3, "array must have shape [n_experiment, time, dim]"
    trajectory_length = array.shape[1]
    if sequence_length is None:
        sequence_length = trajectory_length
    assert trajectory_length >= sequence_length, """
    sequence length can't be larger than data."""

    sequences = []
    for exp_array in array:  # Each exp_array is [time, dim].
        for time in range(0, trajectory_length - sequence_length + 1, stride_length):
            sequences.append(exp_array[time:time + sequence_length, :])

        if (trajectory_length - sequence_length) % stride_length > 0:
            sequences.append(exp_array[-sequence_length:, :])

    return np.stack(sequences, axis=0)


def generate_trajectory(transition_function: Callable,
                        observation_function: Callable,
                        inputs: np.ndarray = None,
                        trajectory_length: int = 120,
                        x0: np.ndarray = np.array(0.5),
                        process_noise_sd: np.ndarray = np.array(0.05),
                        observation_noise_sd: np.ndarray = np.array(np.sqrt(0.8))
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a trajectory.

    The trajectory is generated as:
            x_{k+1} = f(x_k, u_k) + process_noise
            y_k = g(x_k, u_k) + observation_noise

    where f() and g() are the transition and observation functions, respectively.

    Parameters
    ----------
    transition_function: callable.
        Function that accepts (state, input) and returns (next_state).

    observation_function: callable.
        Function that accepts (state, input) and returns (observation).

    inputs: np.ndarray, optional (default: None).
        Sequence of inputs to apply.
        If not None, it has to be longer than the trajectory length.

    trajectory_length: optional (default: 120).
        Length of trajectory to simulate.

    x0: np.ndarray, optional (default: np.array(0.5)).
        Initial state. The state dimension is inferred from here.

    process_noise_sd: np.ndarray, optional (default: np.array(0.05)).
        Standard deviation of the process noise [state_dim x state_dim].

    observation_noise_sd: np.ndarray, optional (default: np.array(0.8)).
        Standard deviation of the observation noise [output_dim x output_dim].

    Returns
    -------
    states: np.ndarray.
        Array of states [trajectory_length x state_dim].

    outputs: np.ndarray.
        Array of outputs [trajectory_length x output_dim].

    """
    states = []
    outputs = []
    state = x0
    state_dim = x0.shape[0]
    out_dim = observation_noise_sd.shape[0]

    if inputs is None:
        inputs = np.zeros((trajectory_length, 0))
    assert inputs.shape[0] >= trajectory_length, """
        Input sequence must be longer to trajectory length"""

    for idx in range(trajectory_length):
        output = observation_function(state, inputs[idx])
        output += (observation_noise_sd @ np.random.randn(out_dim)).squeeze()
        next_state = transition_function(state, inputs[idx])
        next_state += (process_noise_sd @ np.random.randn(state_dim)).squeeze()

        states.append(state)
        outputs.append(output)

        state = next_state

    return np.array(states), np.array(outputs)


class Normalizer(object):
    """Normalizer Transformation for data sets.

    Parameters
    ----------
    array: np.ndarray.
        Array with dimensions [n_sequences x sequence_length x dimension].
    """

    def __init__(self, array: np.ndarray, normalize: bool = True) -> None:
        assert array.ndim == 3, """Array must have 3 dimensions"""
        dim = array.shape[2]

        if normalize:
            self.mean = np.mean(array, axis=(0, 1))
            self.sd = np.std(array, axis=(0, 1))
        else:
            self.mean = np.zeros((dim,))
            self.sd = np.ones((dim,))

        if np.all(self.sd == 0.):  # This is for constant sequences.
            self.sd = np.ones((dim,))

        assert self.mean.shape == (dim,)
        assert self.sd.shape == (dim,)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Transform data."""
        return (data - self.mean) / self.sd

    def inverse(self, data: np.ndarray) -> np.ndarray:
        """Inverse transformation."""
        return self.mean + data * self.sd