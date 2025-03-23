"""
This is a script for running experiments on the Lorenz 96 datasets using EnVI-GPSSM and EnVI_EGPSSM.

- For the EGPSSM, the EnVI model using a 'Shared GP + Normalizing Flows' for the transition function modeling
"""
import os
import time
import torch
import gpytorch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import utils_h as cg
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from data.synthetic import lorenz96_data_gen, lorenz96_drift
from modules.Filter import ensemble_kalman_filter
from modules.GPSSM_GPyTorch import EnVI
from modules.EGPSSM import EnVI as EGPSSM

cg.reset_seed(0)  # setting random seed
device = cg.device  # setting device
dtype = cg.dtype  # setting dtype
plt.rcParams["figure.figsize"] = (20, 10)

dt = 0.01
Lorenz_noise_std = 0.1
obs_noise_std = 2.0
D = 100
number_particles = 200  # number of sample using in EnKF
""" ----------------------------------------
##########    Data preparation
----------------------------------------"""

x_true, y_true = lorenz96_data_gen(T=6, dt=dt, D=D, SDE_sigma=Lorenz_noise_std, obs_sigma=obs_noise_std)
T, _ = x_true.shape
saving_dims = [0, 3, 6,  9]
seq_len = 60                   # sequence length for learning GPSSMs/ETGPSSMs
batch_size = int(T / seq_len)  # batch size

# 获取观测数据
observe_T = torch.tensor(y_true, dtype=dtype).to(device)  # shape: T x obs_dim
observe_pretrain_label = observe_T[1:, 0]                 # shape: (T-1),
observe_pretrain_input = observe_T[:-1, :]                # shape:(T-1) x obs_dim
observe = observe_T.reshape([batch_size, seq_len, D])     # shape: batch_size x seq_len x obs_dim
# 获取真实状态
state_T = torch.tensor(x_true, dtype=dtype).to(device)    # shape: T x state_dim
state = state_T.reshape([batch_size, seq_len, D])         # shape: batch_size x seq_len x state_dim
## 获取初始状态
# state_init = state[:, -1, :]  # shape: batch_size x state_dim



""" ---------------------------------------- EnKF  ---------------------------------------- """
enkf_pred = 10 * np.random.randn(number_particles, D)
x_enkf_t_all = []
x_enkf = []
x_enkf_std = []
for i in range(T):
    # make prediction
    enkf_pred += dt * lorenz96_drift(enkf_pred) + np.sqrt(dt) * Lorenz_noise_std * np.random.randn(*enkf_pred.shape)
    # update
    x_enkf_t = ensemble_kalman_filter(y_obs=y_true[i],
                                      particles=enkf_pred,
                                      A=np.eye(D),
                                      noise_cov=obs_noise_std * np.eye(D),
                                      n_samples=number_particles, bias=0.0)

    # save results and proceed to the next step
    x_enkf_t_all.append(x_enkf_t)
    enkf_pred = x_enkf_t
    x_enkf.append(x_enkf_t.mean(axis=0))
    x_enkf_std.append(x_enkf_t.std(axis=0))

rmse_EnKF, coverage_EnKF, spread_EnKF, crps_EnKF = cg.measure_calculate(state_T, torch.tensor(x_enkf_t_all, dtype=dtype, device=device))

# convert to numpy array
x_enkf = np.array(x_enkf)
x_enkf_var = np.array(x_enkf_std)

# plot in a single figure
plt.figure(figsize=(15, 4 * len(saving_dims)))  # 根据 saving_dims 的长度调整图形高度
for d in range(len(saving_dims)):
    # 第一个子图：状态和 EnKF 估计
    plt.subplot(len(saving_dims), 2, 2 * d + 1)  # 行数为 len(saving_dims)，列数为 2
    plt.plot(x_true[:, saving_dims[d]], label='state')
    plt.plot(x_enkf[:, saving_dims[d]], label='EnKF')
    plt.fill_between(np.arange(T),
                     x_enkf[:, saving_dims[d]] - 2 * x_enkf_var[:, saving_dims[d]],
                     x_enkf[:, saving_dims[d]] + 2 * x_enkf_var[:, saving_dims[d]], alpha=0.3)
    plt.legend()
    plt.title(f'state/est (dim={saving_dims[d]})')

    # 第二个子图：观测数据
    plt.subplot(len(saving_dims), 2, 2 * d + 2)
    plt.plot(y_true[:, saving_dims[d]])
    plt.title(f'obs (dim={saving_dims[d]})')

plt.tight_layout()
plt.show()

""" ---------------------------------------- GPSSMs  ---------------------------------------- """
model_name = 'EGPSSM'
number_ips = 200
residual_trans = True       # if True, the transition model is a residual model
if_BNN = False              # if True, the transition model is a Bayesian neural network
if_pureNN_list = [False, True]           # if True, the transition model is a pure neural network
fixEmission = False          # fix the emission model likelihood
fixTransition = False       # fix the transition model likelihood
lr = 1e-3                   # learning rate
num_epochs = 800           # number of epochs
x_egpssm_all = []
for if_pureNN in if_pureNN_list:
    print(f'\nRunning {model_name} model, with settings: \n'
          f'number_ips: {number_ips} \n'
          f'if_BNN: {if_BNN}\n'
          f'if_pureNN: {if_pureNN}\n'
          f'fixEmission: {fixEmission}\n'
          f'fixTransition: {fixTransition}\n')

    """ ----------------------------------------
    ##########  Model preparation
    ----------------------------------------"""
    if model_name == 'GPSSM':
        # uniformly generate inducing points from [-2, 2],  shape: state_dim x number_ips x (state_dim + input_dim)
        ips = state.reshape(-1, D)[:number_ips, :].unsqueeze(0).repeat(D, 1, 1)
        model = EnVI(dim_x=D,
                     dim_y=D,
                     seq_len=seq_len,
                     ips=ips,
                     dim_c=0,
                     N_MC=number_particles,
                     process_noise_sd=Lorenz_noise_std,
                     emission_noise_sd=obs_noise_std,
                     consistentSampling=False,
                     learn_emission=False,
                     residual_trans=residual_trans).to(device)
        model.BNN = None
    else:
        # shape: number_ips x (state_dim + input_dim)
        ips = state.reshape(-1, D)[:number_ips, :]
        model = EGPSSM(dim_x=D,
                       dim_y=D,
                       seq_len=seq_len,
                       ips=ips.squeeze(0),
                       dim_c=0,
                       N_MC=number_particles,
                       process_noise_sd=Lorenz_noise_std,
                       emission_noise_sd=obs_noise_std,
                       BayesianNN=if_BNN,
                       if_pureNN=if_pureNN,
                       consistentSampling=False,
                       learn_emission=False,
                       residual_trans=residual_trans).to(device)
        if if_pureNN:
            model.transition.requires_grad_(False)

    if fixEmission:
        model.emission_likelihood.requires_grad_(False)
    if fixTransition:
        model.likelihood.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # """ ------------------------------------------------------------------------------------------------------
    # ##########    Pre-training for EGPSSM using observations (initialize the transition model)
    # ----------------------------------------------------------------------------------------------------------"""
    # # training the transition model using observations
    # optimizer_gp_pretrain = torch.optim.Adam([{'params': model.transition.parameters(), 'lr': 5e-3},
    #                                           {'params': model.likelihood.parameters(), 'lr': 1e-3},
    #                                           ])
    # model.transition.train()
    # model.likelihood.train()
    # epochs_iter = tqdm(range(2000), desc="Epoch")
    # for i in epochs_iter:
    #     optimizer_gp_pretrain.zero_grad()
    #     output = model.transition(observe_pretrain_input).rsample()
    #     loss = F.mse_loss(output, observe_pretrain_label) + model.transition.kl_divergence()
    #     epochs_iter.set_postfix({'loss': '{0:1.5f}'.format(loss.item()),
    #                              'KL': '{0:1.5f}'.format(model.transition.kl_divergence().item())})
    #     loss.backward()
    #     optimizer_gp_pretrain.step()


    """ ----------------------------------------------------------------------------------------------------------
    ##########    Training
    ---------------------------------------------------------------------------------------------------------------"""
    MSE = []
    losses = []
    epochiter = tqdm(range(0, num_epochs), desc='Epoch:')
    t0 = time.time()
    for epoch in epochiter:
        model.train()
        optimizer.zero_grad()
        # state_init[0] = 10 * torch.randn_like(state_init[0])  # same distribution as the true initial state
        ELBO, X_filter, X_filter_var, X_filter_ensemble = model(observations=observe,
                                                                input_sequence=None,
                                                                state_init=None)
        loss = -ELBO
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        epochiter.set_postfix({'loss': '{0:1.5f}'.format(loss.item())})

        if epoch % 100 == 0:
            rmse_EGPSSM, coverage_EGPSSM, spread_EGPSSM, crps_EGPSSM = cg.measure_calculate(state_T, X_filter_ensemble.permute(1, 0, 2))
            Time = time.time() - t0
            x_egpssm = X_filter.cpu().detach().numpy().reshape(-1, D)
            x_egpssm_var = X_filter_var.cpu().detach().numpy().reshape(-1, D)
            """ ----------------------------------------"""
            # plot in a single figure
            plt.figure(figsize=(15, 4 * len(saving_dims)))  # 根据 saving_dims 的长度调整图形高度
            for d in range(len(saving_dims)):
                # 第一个子图：状态和 EnKF 估计
                plt.subplot(len(saving_dims), 2, 2 * d + 1)  # 行数为 len(saving_dims)，列数为 2
                plt.plot(x_true[:, saving_dims[d]], label='state')
                plt.plot(x_egpssm[:, saving_dims[d]], label='EGPSSM')
                plt.fill_between(np.arange(T),
                                 x_egpssm[:, saving_dims[d]] - 2 * x_egpssm_var[:, saving_dims[d]] ** 0.5,
                                 x_egpssm[:, saving_dims[d]] + 2 * x_egpssm_var[:, saving_dims[d]] ** 0.5, alpha=0.3)
                plt.legend()
                plt.title(f'state/est (dim={saving_dims[d]})')

                # 第二个子图：观测数据
                plt.subplot(len(saving_dims), 2, 2 * d + 2)
                plt.plot(y_true[:, saving_dims[d]])
                plt.title(f'obs (dim={saving_dims[d]})')

            plt.tight_layout()
            plt.show()

    Time = time.time() - t0
    ELBO, X_filter, X_filter_var, X_filter_ensemble = model(observations=observe,
                                                            input_sequence=None,
                                                            state_init=None)
    rmse_EGPSSM, coverage_EGPSSM, spread_EGPSSM, crps_EGPSSM = cg.measure_calculate(state_T, X_filter_ensemble.permute(1, 0, 2))
    x_egpssm = X_filter.cpu().detach().numpy().reshape(-1, D)
    x_egpssm_var = X_filter_var.cpu().detach().numpy().reshape(-1, D)
    x_egpssm_all.append(X_filter_ensemble.transpose(0, 1).detach())

""" ----------------------------------------"""
x_egpssm_all = torch.cat(x_egpssm_all, dim=0)   # shape: 2T x ensemble_size x D
# post-processing: plot the trajectory
fig = cg.plot_lorenz_trajectory_all(state_T, x_egpssm_all[:T], x_egpssm_all[T:],
                                    torch.tensor(x_enkf_t_all), observe_T,
                                    T, f"results/_Lorenz96")

