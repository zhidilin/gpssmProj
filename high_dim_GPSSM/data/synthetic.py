import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from numpy.linalg import cholesky


''' # Lorenz96 data generation'''
def lorenz96_drift(x):
    F = 8
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F


def lorenz96_data_gen(T=6, dt=0.01, D=40, SDE_sigma=0.1, obs_sigma=1.0):
    state_all = []
    obs_all = []
    state_target = 10 * np.random.rand(D)
    for i in range(int(T / dt)):
        # ground truth
        state_target += dt * lorenz96_drift(state_target) + np.sqrt(dt) * SDE_sigma * np.random.randn(*state_target.shape)
        obs = state_target + obs_sigma * np.random.randn(*state_target.shape)
        state_all.append(state_target.copy())
        obs_all.append(obs.copy())

    return np.array(state_all), np.array(obs_all)


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

# Define the basic kink function using NumPy
def kink_func(x):
    return 0.8 + (x + 0.2) * (1 - 5 / (1 + np.exp(-2 * x)))

# Add local jitter and variations for a non-stationary kink function using NumPy
def nonstationary_kink_func_with_jitter(x):
    # Base kink function
    f_base = kink_func(x)

    # Adjust the slope for x > 0 using NumPy
    slope_adjustment = np.where(x > 0, 1 - 0.5 * np.exp(-0.5 * x), 1.0)
    f_slowed = f_base * slope_adjustment

    # Add more pronounced jitter to the x > 0 area
    local_jitter = np.where(x > 0, 0.5 * np.sin(8 * x), 0.5 * np.sin(2 * x))

    # Overlay jitter and noise on the adjusted kink function
    f_nonstationary = f_slowed - local_jitter
    return f_nonstationary


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


def NS_Kink_function(trajectory_length = 2000, state_int = 0.5, process_noise_sd = np.sqrt(0.01),
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
        states[i+1] = nonstationary_kink_func_with_jitter(states[i]) + np.random.normal(0.0, process_noise_sd)
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

    elif func == 'ns-kink':
        lo = -6
        up = 2

        true_state_np, obsers_np = NS_Kink_function(trajectory_length=traj_len,
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


def plot_1D_all(model, epoch, func='kinkfunc', save=False, path='./fig_MF1D/2layer_learned_kink_Epoch'):
    device = model.H.device
    fontsize = 28
    N_test = 100

    if func == 'ns-kink':
        label = "kink function"
        X_test = np.linspace(-6, 2, N_test)
        y_test = nonstationary_kink_func_with_jitter(X_test)


    elif func == 'kinkfunc':
        label = "kink function"
        X_test = np.linspace(-6, 2, N_test)
        y_test = 0.8 + (X_test + 0.2) * (1 - 5 / (1 + np.exp(-2 * X_test)))

    else:
        raise NotImplementedError("The 'func' input only supports kinkfunc and ns-kink")

    if model.BNN is not None:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.tensor(X_test, dtype=torch.float).to(device)
            test_xx = test_x.reshape(-1, 1)  # shape: batch_size x (input_dim + state_dim)

            func_pred = model.transition(test_xx)  # shape: (batch_size, )
            func_pred = func_pred.sample(torch.Size([500])) # shape: 500 x batch_size

            # get the 500 random sample weights for transforming the GP
            weights = torch.stack([model.BNN(test_xx) for _ in range(500)])  # weights: 500 x batch_size x (2*state_dim)

            # _x_t shape: 500 x batch_size x state_dim
            if model.pureNN:
                _x_t = weights
            else:
                _x_t = func_pred.unsqueeze(-1) * weights[..., :model.state_dim] + weights[..., model.state_dim:]

            if model.residual_trans:
               # compute the mean and variance of the 100 samples
                pred_val_mean = _x_t.mean(dim=0) + test_xx # pred_val_mean shape: batch_size x state_dim
            else:
                # compute the mean and variance of the 100 samples
                pred_val_mean = _x_t.mean(dim=0)    # pred_val_mean shape: batch_size x state_dim

            pred_val_var = _x_t.std(dim=0) ** 2 + model.likelihood.noise  # pred_val_var shape: batch_size x state_dim

            # Get upper and lower confidence bounds
            lower, upper = pred_val_mean - 2 * pred_val_var.sqrt(), \
                           pred_val_mean + 2 * pred_val_var.sqrt()

            # compute prediction MSE
            MSE_preGP = 1 / (N_test) * np.linalg.norm(pred_val_mean.cpu().numpy().reshape(-1) - y_test) ** 2
            print(f"\nMSE_preGP: {MSE_preGP.item()}")

    else:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.tensor(X_test, dtype=torch.float).to(device)
            test_xx = test_x.reshape(-1, 1)  # shape: batch_size x (input_dim + state_dim)

            # x expected shape: state_dim x batch_size x (input_dim + state_dim)
            test_xx = test_xx.repeat(model.state_dim, 1, 1)

            func_pred = model.transition(test_xx)             # shape: state_dim x batch_size
            observed_pred = model.likelihood(func_pred)       # shape: state_dim x batch_size

            if model.residual_trans:
                pred_val_mean = observed_pred.mean + test_xx[0, :, :].transpose(-1, -2)
                # Get upper and lower confidence bounds
                lower, upper = observed_pred.confidence_region()
                lower, upper = lower.reshape(-1, ) + test_xx[0, :, :].transpose(-1, -2).mean(dim=[0]), \
                               upper.reshape(-1, ) + test_xx[0, :, :].transpose(-1, -2).mean(dim=[0])

            else:
                pred_val_mean = observed_pred.mean
                # Get upper and lower confidence bounds
                lower, upper = observed_pred.confidence_region()

            # compute prediction MSE
            MSE_preGP = 1 / (N_test) * np.linalg.norm(pred_val_mean.cpu().numpy().reshape(-1) - y_test) ** 2
            print(f"\nMSE_preGP: {MSE_preGP.item()}")

    with torch.no_grad():

        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(15, 15))

        # Plot test data as read stars
        ax.plot(X_test, y_test, 'r', label=label)
        # Plot predictive means as blue line
        ax.plot(X_test, pred_val_mean.cpu().numpy().reshape(-1, ), 'b', label='learned function')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.cpu().numpy(),
                        lower.reshape(-1) .cpu().numpy(),
                        upper.reshape(-1) .cpu().numpy(),
                        label='$\pm 2 \sigma$', alpha=0.2)

        ax.legend(loc=0, fontsize=fontsize)
        # plt.title(f"Epoch: {epoch}", fontsize=15)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.grid()

        ax.set_xlim([-6, 2])
        ax.set_ylim([-6, 2])

        if save:
            plt.savefig(path + f"{func}_epoch_{epoch}.pdf")
        else:
            plt.show()

    return MSE_preGP
