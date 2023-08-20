import numpy as np

def kink(x, noise_std):
    """
    Compute the kink function value for a given input.

    :param x: Input to the kink function
    :param noise_std: Standard deviation of the noise
    :return: Output of the kink function
    """
    f = 0.8 + (x + 0.2) * (1 - 5 / (1 + np.exp(-2 * x)))
    y = f + np.random.normal(0.0, noise_std)
    return y


def Kink_SSMs(trajectory_length=2000, state_int=0.5, process_noise_sd=np.sqrt(0.01),
              observation_noise_sd=np.sqrt(0.1)):
    """
    Simulate a state space model (SSM) with a kink function.

    :param trajectory_length: Length of the trajectory
    :param state_int: Initial state value
    :param process_noise_sd: Standard deviation of process noise
    :param observation_noise_sd: Standard deviation of observation noise
    :return: States and observations of the SSM (Numpy arrays)
    """
    states = np.zeros(trajectory_length)
    observations = np.zeros(trajectory_length)
    states[0] = state_int
    observations[0] = state_int + np.random.normal(0.0, observation_noise_sd)

    for i in range(trajectory_length - 1):
        states[i + 1] = kink(states[i], process_noise_sd)
        observations[i + 1] = states[i + 1] + np.random.normal(0.0, observation_noise_sd)

    return states, observations


def Kink_SSMs_2D(trajectory_length=2000, state_int=np.array([-1, 0]), process_noise_sd=np.sqrt(0.001),
                 observation_noise_sd=np.sqrt(0.01)):
    """
    Simulate a state space model (SSM) with a 2D kink function.

    :param trajectory_length: Length of the trajectory
    :param state_int: Initial state value
    :param process_noise_sd: Standard deviation of process noise
    :param observation_noise_sd: Standard deviation of observation noise
    :return: States and observations of the SSM (Numpy arrays)
    """
    states = np.zeros([trajectory_length, 2])
    observations = np.zeros([trajectory_length, 2])
    states[0] = state_int
    observations[0] = state_int + np.random.normal(0.0, observation_noise_sd, 2)

    for i in range(trajectory_length - 1):
        tmp =  kink(states[i, 0], process_noise_sd) + states[i, 1]
        states[i + 1, 0] = tmp + states[i, 0]
        states[i + 1, 1] = -.5 * tmp + states[i, 1]
        observations[i + 1] = states[i + 1] + np.random.normal(0.0, observation_noise_sd, 2)

    return states, observations


