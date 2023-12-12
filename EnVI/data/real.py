"""Classes to access data."""
import torch
from torch.utils import data
import numpy as np
import scipy.io as sio
import os
from .src import get_data_split, generate_batches, generate_trajectory, Normalizer

__all__ = ['Actuator', 'BallBeam', 'Drive', 'Dryer',   'GasFurnace',  'KinkFunction', 'Dataset', 'get_dataset']
DATA_DIR = 'data/datasets'


class Dataset(data.TensorDataset):
    """Dataset handler for time-series data.

    Parameters
    ----------
    outputs: np.ndarray.
        Array of shape [n_experiment, time, dim] with outputs of the time series.

    inputs: np.ndarray, optional (default: data/).
        Array of shape [n_experiment, time, dim] with inputs of the time series.

    states: bool, optional (default: True).
        Array of shape [n_experiment, time, dim] with hidden states of the time series.

    sequence_length: int, optional (default: full sequence).
        Integer that indicates how long the sequences should be.

    sequence_stride: int, optional (default: 1).
        Integer that indicates every how many time-steps the sequences start.

    """

    dim_outputs = None  # type: int
    dim_inputs = None  # type: int

    def __init__(self,
                 outputs: np.ndarray = np.empty((1,)),
                 inputs: np.ndarray = None,
                 sequence_length: int = None,
                 sequence_stride: int = 1,
                 split_idx: int = None,
                 normalize: bool = True,
                 data_dir: str = DATA_DIR,
                 train: bool = True) -> None:

        assert outputs.ndim == 3, 'Outputs shape is [n_experiment, time, dim]'
        assert self.dim_outputs == outputs.shape[2]

        self.train = train
        num_experiments, experiment_length, _ = outputs.shape
        if sequence_length is None:
            sequence_length = min(20, experiment_length)
        self._sequence_length = sequence_length
        self._sequence_stride = sequence_stride

        if inputs is not None:
            assert inputs.ndim == 3, 'Inputs shape is [n_experiment, time, dim]'
            assert inputs.shape[0] == outputs.shape[0], """
                Inputs and outputs must have the same number of experiments"""
            assert inputs.shape[1] == outputs.shape[1], """
                Inputs and outputs experiments should be equally long"""
        else:
            inputs = np.zeros((num_experiments, experiment_length, 0))

        assert self.dim_inputs == inputs.shape[2]

        train_inputs = get_data_split(inputs, split_idx, train=True)
        train_outputs = get_data_split(outputs, split_idx, train=True)

        test_inputs = get_data_split(inputs, split_idx, train=False)
        test_outputs = get_data_split(outputs, split_idx, train=False)

        # Store normalized inputs, outputs, states.
        self.input_normalizer = Normalizer(train_inputs, normalize=normalize)
        self.output_normalizer = Normalizer(train_outputs, normalize=normalize)

        if self.train:
            self.inputs = self.input_normalizer(train_inputs)
            self.outputs = self.output_normalizer(train_outputs)
        else:
            self.inputs = self.input_normalizer(test_inputs)
            self.outputs = self.output_normalizer(test_outputs)

        self.num_experiments, self.experiment_length, _ = self.outputs.shape

        super().__init__(*[torch.tensor(
            generate_batches(x, self._sequence_length, self._sequence_stride)).float()
                           for x in [self.inputs, self.outputs]])

    def __str__(self):
        """Return string with dataset statistics."""
        string = 'input dim: {} \noutput dim: {} \n'.format(self.dim_inputs, self.dim_outputs)

        string += 'sequence length: {} \n'.format(self.tensors[0].shape[1] )
        key = 'train' if self.train else 'test'
        string += '{}_samples: {} \n{}_sequences: {} \n'.format(key, self.experiment_length, key, self.tensors[0].shape[0])
        return string

    @property
    def sequence_length(self) -> int:
        """Get sequence length."""
        return self._sequence_length

    @sequence_length.setter
    def sequence_length(self, new_seq_length):
        """Set sequence length and reshape the tensors.

        Parameters
        ----------
        new_seq_length: int.
        """
        self._sequence_length = new_seq_length
        self.tensors = [torch.tensor(
            generate_batches(x, self._sequence_length, self._sequence_stride)).float()
                        for x in [self.inputs, self.outputs]]

    @staticmethod
    def f(x: np.ndarray, u: np.ndarray = None) -> np.ndarray:
        """Transition function."""
        return x

    @staticmethod
    def g(x: np.ndarray, u: np.ndarray = None) -> np.ndarray:
        """Observation function."""
        return x


class Actuator(Dataset):
    """Actuator dataset implementation.

    Train Split: first half of the data.
    Test Split: second half of the data.

    Parameters
    ----------
    data_dir: str, optional (default: data/).
        Directory where `actuator.mat' is located.

    train: bool, optional (default: True).
        Flag that indicates dataset split.

    sequence_length: int, optional (default: full sequence).
        Integer that indicates how long the sequences should be.

    sequence_stride: int, optional (default: 1).
        Integer that indicates every how many time-steps the sequences start.

    References
    ----------
    https://github.com/zhenwendai/RGP/tree/master/datasets/system_identification

    """

    dim_outputs = 1
    dim_inputs = 1

    def __init__(self,
                 data_dir: str = DATA_DIR,
                 train: bool = True,
                 normalize: bool = True,
                 sequence_length: int = None,
                 sequence_stride: int = 1) -> None:
        raw_data = sio.loadmat(os.path.join(data_dir, 'actuator.mat'))

        inputs = raw_data['u'][np.newaxis]
        outputs = raw_data['p'][np.newaxis]

        super().__init__(inputs=inputs,
                         outputs=outputs,
                         normalize=normalize,
                         sequence_length=sequence_length,
                         sequence_stride=sequence_stride,
                         train=train)


class BallBeam(Dataset):
    """BallBeam dataset implementation.

    Train Split: first half of the data.
    Test Split: second half of the data.

    Parameters
    ----------
    data_dir: str, optional (default: data/).
        Directory where `actuator.mat' is located.

    train: bool, optional (default: True).
        Flag that indicates dataset split.

    sequence_length: int, optional (default: full sequence).
        Integer that indicates how long the sequences should be.

    sequence_stride: int, optional (default: 1).
        Integer that indicates every how many time-steps the sequences start.

    References
    ----------
    http://homes.esat.kuleuven.be/~smc/daisy/daisydata.html
    [96-004] Data of the ball-and-beam setup in STADIUS

    """

    dim_outputs = 1
    dim_inputs = 1

    def __init__(self, data_dir: str = DATA_DIR,
                 train: bool = True,
                 normalize: bool = True,
                 sequence_length: int = None,
                 sequence_stride: int = 1) -> None:
        raw_data = np.loadtxt(os.path.join(data_dir, 'ballbeam.dat'))

        inputs = raw_data[np.newaxis, :, 0, np.newaxis]
        outputs = raw_data[np.newaxis, :, 1, np.newaxis]

        super().__init__(inputs=inputs, outputs=outputs, normalize=normalize,
                         sequence_length=sequence_length,
                         sequence_stride=sequence_stride, train=train)


class Drive(Dataset):
    """Drive dataset implementation.

    Train Split: first half of the data.
    Test Split: second half of the data.

    Parameters
    ----------
    data_dir: str, optional (default: data/).
        Directory where `actuator.mat' is located.

    train: bool, optional (default: True).
        Flag that indicates dataset split.

    sequence_length: int, optional (default: full sequence).
        Integer that indicates how long the sequences should be.

    sequence_stride: int, optional (default: 1).
        Integer that indicates every how many time-steps the sequences start.

    References
    ----------
    http://homes.esat.kuleuven.be/~smc/daisy/daisydata.html

    """

    dim_outputs = 1
    dim_inputs = 1

    def __init__(self, data_dir: str = DATA_DIR, train: bool = True,
                 normalize: bool = True,
                 sequence_length: int = None, sequence_stride: int = 1) -> None:
        raw_data = sio.loadmat(os.path.join(data_dir, 'drive.mat'))

        inputs = raw_data['u1'][np.newaxis]
        outputs = raw_data['z1'][np.newaxis]

        super().__init__(inputs=inputs, outputs=outputs, normalize=normalize,
                         sequence_length=sequence_length,
                         sequence_stride=sequence_stride, train=train)


class Dryer(Dataset):
    """Hair Dryer dataset implementation.

    Train Split: first half of the data.
    Test Split: second half of the data.

    Parameters
    ----------
    data_dir: str, optional (default: data/).
        Directory where `actuator.mat' is located.

    train: bool, optional (default: True).
        Flag that indicates dataset split.

    sequence_length: int, optional (default: full sequence).
        Integer that indicates how long the sequences should be.

    sequence_stride: int, optional (default: 1).
        Integer that indicates every how many time-steps the sequences start.

    References
    ----------
    http://homes.esat.kuleuven.be/~smc/daisy/daisydata.html
    [96-006] Data of a laboratory setup acting like a hair dryer

    """

    dim_outputs = 1
    dim_inputs = 1

    def __init__(self, data_dir: str = DATA_DIR, train: bool = True,
                 normalize: bool = True,
                 sequence_length: int = None, sequence_stride: int = 1) -> None:
        raw_data = np.loadtxt(os.path.join(data_dir, 'dryer.dat'))

        inputs = raw_data[np.newaxis, :, 0, np.newaxis]
        outputs = raw_data[np.newaxis, :, 1, np.newaxis]

        super().__init__(inputs=inputs, outputs=outputs, normalize=normalize,
                         sequence_length=sequence_length,
                         sequence_stride=sequence_stride, train=train)


class GasFurnace(Dataset):
    """Gas Furnace dataset implementation.

    Train Split: first half of the data.
    Test Split: second half of the data.

    Parameters
    ----------
    data_dir: str, optional (default: data/).
        Directory where `actuator.mat' is located.

    train: bool, optional (default: True).
        Flag that indicates dataset split.

    sequence_length: int, optional (default: full sequence).
        Integer that indicates how long the sequences should be.

    sequence_stride: int, optional (default: 1).
        Integer that indicates every how many time-steps the sequences start.

    References
    ----------
    https://openmv.net/info/gas-furnace

    """

    dim_outputs = 1
    dim_inputs = 1

    def __init__(self, data_dir: str = DATA_DIR, train: bool = True,
                 normalize: bool = True,
                 sequence_length: int = None, sequence_stride: int = 1) -> None:
        raw_data = np.loadtxt(os.path.join(data_dir, 'gas_furnace.csv'),
                              skiprows=1, delimiter=',')

        inputs = raw_data[np.newaxis, :, 0, np.newaxis]
        outputs = raw_data[np.newaxis, :, 1, np.newaxis]

        super().__init__(inputs=inputs, outputs=outputs, normalize=normalize,
                         sequence_length=sequence_length,
                         sequence_stride=sequence_stride, train=train)


class KinkFunction(Dataset):
    """Kink Function dataset implementation.

    The Kink function is:
        f(x) = 0.8 + (x + 0.2) * (1 - 5 / (1 + exp(-2 * x)))

    The trajectory is generated as:
        x_{k+1} = f(x_k) + process_noise
        y_k = x_k + observation_noise

    Parameters
    ----------
    data_dir: str, optional (default: data/).
        Directory where `actuator.mat' is located.

    train: bool, optional (default: True).
        Flag that indicates dataset split.

    sequence_length: int, optional (default: full sequence).
        Integer that indicates how long the sequences should be.

    sequence_stride: int, optional (default: 1).
        Integer that indicates every how many time-steps the sequences start.

    trajectory_length: int, optional (default = 120).
            Length of trajectory.

    x0: float, optional (default = 0.5).
        Initial state.

    process_noise_sd: float, optional (default = 0.05).
        Standard deviation of the process noise.

    observation_noise_sd: float, optional (default = 0.2).
        Standard deviation of observation noise.

    References
    ----------
    Ialongo, A. D., Van Der Wilk, M., Hensman, J., & Rasmussen, C. E. (2019, May).
    Overcoming Mean-Field Approximations in Recurrent Gaussian Process Models.
    In International Conference on Machine Learning (pp. 2931-2940).

    """

    dim_outputs = 1
    dim_inputs = 0

    def __init__(self,
                 data_dir: str = DATA_DIR,
                 train: bool = True,
                 normalize: bool = False,
                 sequence_length: int = None,
                 sequence_stride: int = 1,
                 trajectory_length: int = 600,
                 x0: float = 0.5,
                 process_noise_sd: float = 0.05,
                 observation_noise_sd: float = 0.2) -> None:

        file_name = os.path.join(data_dir, 'kink_function.mat')
        if not os.path.exists(file_name):
            states, outputs = generate_trajectory(
                self.f, self.g, trajectory_length=trajectory_length, x0=np.array([x0]),
                process_noise_sd=np.array([process_noise_sd]),
                observation_noise_sd=np.array([observation_noise_sd]))

            sio.savemat(file_name, {
                'ds_x': states,
                'ds_y': outputs,
                'title': 'Kink Function'
            })
            # states = states[np.newaxis]
            outputs = outputs[np.newaxis]

        else:
            raw_data = sio.loadmat(file_name)
            # states = raw_data['ds_x'][np.newaxis]
            outputs = raw_data['ds_y'][np.newaxis]

        super().__init__(outputs=outputs, normalize=normalize,
                         sequence_length=sequence_length, split_idx=500,
                         sequence_stride=sequence_stride, train=train)

    @staticmethod
    def f(x: np.ndarray, _: np.ndarray = None) -> np.ndarray:
        """Kink transition function."""
        return 0.8 + (x + 0.2) * (1 - 5 / (1 + np.exp(- 2 * x)))


def get_dataset(dataset_: str):
    """Get Dataset."""
    if dataset_.lower() == 'actuator':
        return Actuator
    elif dataset_.lower() == 'ballbeam':
        return BallBeam
    elif dataset_.lower() == 'drive':
        return Drive
    elif dataset_.lower() == 'dryer':
        return Dryer
    elif dataset_.lower() == 'gasfurnace':
        return GasFurnace
    elif dataset_.lower() == 'kinkfunction':
        return KinkFunction
    else:
        raise NotImplementedError("{}".format(dataset_))