import numpy as np
import torch
import matplotlib.pyplot as plt
from numpy.linalg import cholesky

'''  # define a linear state space model  '''
def LGSSM(s, A, Q, H, m0, T):
    X = np.zeros((T, A.shape[0], 1))
    X_hat = np.zeros((T, A.shape[0], 1))
    Y = np.zeros((T, H.shape[0], 1))
    Y_hat = np.zeros((T, A.shape[0], 1))
    x = m0
    for k in range(T):
        # keep input
        X_hat[k] = x
        # noise
        q = cholesky(Q) @ np.random.normal(size=(A.shape[0], 1))
        x = A @ x + q
        y = H @ x + s * np.random.normal(size=(4, 1))
        # keep output
        X[k] = x
        Y[k] = y
        Y_hat[k] = y
    return X_hat, X, Y


def Kink_function(trajectory_length = 2000, state_int = 0.5, process_noise_sd = np.sqrt(0.01),
                  observation_noise_sd = np.sqrt(0.1)):
    """
    ----------       Kink function data generation

    Parameters
    ----------
    trajectory_length: int
    state_int: initialized state, i.e. x[0]
    process_noise_sd:  transition process noise variance standard deviation
    observation_noise_sd: observation noise variance standard deviation

    Returns
    -------
    """

    states, observations = np.zeros(trajectory_length), np.zeros(trajectory_length)
    states[0], observations[0] = state_int, state_int + np.random.normal(0.0, observation_noise_sd)
    for i in range(trajectory_length-1):
        f = 0.8 + (states[i] + 0.2) * (1 - 5 / (1 + np.exp(-2 * states[i])))
        states[i+1] = f + np.random.normal(0.0, process_noise_sd)
        observations[i+1] = states[i + 1] + np.random.normal(0.0, observation_noise_sd)
    return states, observations


def KS_function(trajectory_length = 2000, state_int = 0.5, process_noise_sd = np.sqrt(0.01),
                  observation_noise_sd = np.sqrt(0.1)):
    """
    ----------       Kink-step function data generation

    Parameters
    ----------
    trajectory_length: int
    state_int: initialized state, i.e. x[0]
    process_noise_sd:  transition process noise variance standard deviation
    observation_noise_sd: observation noise variance standard deviation

    Returns
    -------
    """
    states, observations = np.zeros(trajectory_length), np.zeros(trajectory_length)
    states[0], observations[0] = state_int, state_int + np.random.normal(0.0, observation_noise_sd)
    for i in range(trajectory_length-1):
        if (5 > states[i] >= 4) or (states[i] < 3):
            f = states[i] + 1
        elif 4 > states[i] >= 3:
            f = 0.0
        else:
            f = 16 - 2 * states[i]
        states[i+1] = f + np.random.normal(0.0, process_noise_sd)
        observations[i+1] = states[i + 1] + np.random.normal(0.0, observation_noise_sd)
    return states, observations


def move_one(A):

    """
    for data visualization purpose

    Parameters
    ----------
    A: input data of size n x m, where n is the number of data, m is the dimensionality

    Returns
    -------
    b: output data of size (n-1) x m, where (n-1) is the number of data, m is the dimensionality

    """
    b = np.zeros(len(A))
    for i in range(len(A) - 1):
        b[i] = (A[i + 1])
    return b


''' ---  some useful functions --- '''

def KS_func(x):
    if (5 > x >= 4) or (x < 3):
        f = x + 1
    elif 4 > x >= 3:
        f = 0
    else:
        f = 16 - 2 * x
    return f

def kink_func(x):
    f = 0.8 + (x + 0.2) * (1 - 5 / (1 + torch.exp(-2 * x)))
    return f


def syn_data_generation(func, traj_len, process_noise_sd, observation_noise_sd, number_ips, state_int=0.5, if_plot=False):
    """
    data generation for training the GPSSMs

    Parameters:
    ----------
        func: "kinkfunc" or "ksfunc"
        process_noise_sd
        observation_noise_sd
        number_ips
        state_int
        if_plot

    Returns:
    -------
        inducing_points
        true_state_np
        obsers_np
    """

    if func == 'kinkfunc':
        lo = -3.15
        up = 1.15

        true_state_np, obsers_np = Kink_function(trajectory_length=traj_len,
                                                 state_int=state_int,
                                                 process_noise_sd=process_noise_sd,
                                                 observation_noise_sd=observation_noise_sd)

        # initialize inducing points: shape: state_dim x number_ips x state_dim
        inducing_points = torch.linspace(lo, up, number_ips)
        inducing_points = inducing_points.repeat(1, 1, 1).permute(0, 2, 1)

        # 画出数据
        fig, (ax1, ax) = plt.subplots(1, 2, figsize=(10, 5))
        ax.plot(obsers_np, move_one(obsers_np), 'r*', label='Data', markersize=10)
        ax.set(xlabel="y[t]", ylabel="y[t+1]")
        ax1.plot(true_state_np, move_one(true_state_np), 'b*', label='Data', markersize=10)
        ax1.set(xlabel="x[t]", ylabel="x[t+1]")
        if if_plot:
            plt.show()
        plt.close('all')

    elif func == 'ksfunc':
        lo = -0.5
        up = 6.5

        true_state_np, obsers_np = KS_function(trajectory_length=traj_len,
                                               state_int=state_int,
                                               process_noise_sd=process_noise_sd,
                                               observation_noise_sd=observation_noise_sd)

        # initialize inducing points: shape: state_dim x number_ips x state_dim
        inducing_points = torch.linspace(lo, up, number_ips)
        inducing_points = inducing_points.repeat(1, 1, 1).permute(0, 2, 1)

        # plot the dataset
        fig, (ax1, ax) = plt.subplots(1, 2, figsize=(10, 5))
        ax.plot(obsers_np, move_one(obsers_np), 'r*', label='Data', markersize=10)
        ax.set(xlabel="y[t]", ylabel="y[t+1]")
        ax1.plot(true_state_np, move_one(true_state_np), 'b*', label='Data', markersize=10)
        ax1.set(xlabel="x[t]", ylabel="x[t+1]")
        if if_plot:
            plt.show()
        plt.close()
    else:
        raise NotImplementedError("Function format not implemented.")


    return inducing_points, true_state_np, obsers_np


def plot_kink_data(x_filter, x, y, save_fig=False, DIR=None, func=None, epoch=None):
    # plot the dataset
    plt.figure(figsize=(6, 6))
    plt.plot(y, move_one(y), 'k*', label='observation', markersize=10)
    plt.plot(x_filter, move_one(x_filter), 'r*', label='filtering', markersize=10)
    plt.plot(x, move_one(x), 'b*', label='latent state', markersize=10)
    plt.legend(loc=0, fontsize=10)
    if save_fig:
        plt.savefig(DIR + f"func_{func}_epoch_{epoch}_data.pdf")
    else:
        plt.show()
    plt.close('all')