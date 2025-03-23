"""
This script compares EnVI-GPSSM and EnVI-ETGPSSM in terms of computational efficiency and the number of parameters.

- For the EGPSSM, the EnVI model using a 'Shared GP + Normalizing Flows' for the transition function modeling
"""

import time
import torch
import numpy as np
from tqdm import tqdm
import utils_h as cg
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from data.synthetic import lorenz96_data_gen
from modules.GPSSM_GPyTorch import EnVI
from modules.EGPSSM import EnVI as EGPSSM

cg.reset_seed(0)  # setting random seed
device = torch.device("cpu") #
# device = cg.device  # setting device
dtype = cg.dtype  # setting dtype
plt.rcParams["figure.figsize"] = (20, 10)

dt = 0.01
Lorenz_noise_std = 0.1
obs_noise_std = 2.0
number_particles = 150  # number of sample using in EnKF
""" ---------------------------------------- GPSSMs  ---------------------------------------- """
model_name_list = ['GPSSM', 'ETGPSSM']
number_ips = 100
residual_trans = True  # if True, the transition model is a residual model
if_BNN = False  # if True, the transition model is a Bayesian neural network
if_pureNN = False  # if True, the transition model is a pure neural network
fixEmission = False  # fix the emission model likelihood
fixTransition = False  # fix the transition model likelihood
lr = 1e-3  # learning rate
parameters_count = []
running_time = []
for model_name in model_name_list:
    D_list = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # state dimension
    for D in D_list:
        print(f'Running {model_name} model in Device {device} with Dimension={D}')
        x_true, y_true = lorenz96_data_gen(T=6, dt=dt, D=D, SDE_sigma=Lorenz_noise_std, obs_sigma=obs_noise_std)
        T, _ = x_true.shape
        seq_len = 30  # sequence length
        batch_size = int(T / seq_len)  # batch size

        """ ----------------------------------------
        ##########    Data preparation
        ----------------------------------------"""
        # 获取观测数据
        observe = torch.tensor(y_true, dtype=dtype).to(device)  # shape: T x obs_dim
        observe_pretrain_label = observe[1:, 0]  # shape: (T-1),
        observe_pretrain_input = observe[:-1, :]  # shape:(T-1) x obs_dim
        observe = observe.reshape([batch_size, seq_len, D])  # shape: batch_size x seq_len x obs_dim
        # 获取真实状态
        state = torch.tensor(x_true, dtype=dtype).to(device)  # shape: T x state_dim
        state = state.reshape([batch_size, seq_len, D])  # shape: batch_size x seq_len x state_dim

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

        t0 = time.time()
        ELBO, X_filter, X_filter_var = model(observations=observe,
                                             input_sequence=None,
                                             state_init=None)
        t1 = time.time()
        print(f"Time cost: {t1 - t0}")
        running_time.append(t1 - t0)


        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())

        number_parameters = count_parameters(model)
        print("Number of parameters:", number_parameters)
        parameters_count.append(number_parameters)


parameters_count = np.array(parameters_count).reshape(2, -1)
running_time = np.array(running_time).reshape(2, -1)

# 创建两个子图并排
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# === 第一个子图：参数数量对比 ===
ax1 = axes[0]
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax1.plot(D_list, parameters_count[0], label=model_name_list[0], marker='o', linestyle='-', linewidth=3, markersize=8)
ax1.plot(D_list, parameters_count[1], label=model_name_list[1], marker='s', linestyle='--', linewidth=3, markersize=8)

ax1.set_xlabel(r"$d_x$", fontsize=18)
ax1.set_ylabel('# Parameters (log-scaled)', fontsize=18)
ax1.set_yscale('log')
ax1.tick_params(axis='both', labelsize=16)

# 在特定数据点上标注数值
for i in [2, 6, -1]:
    ax1.text(D_list[i], parameters_count[0][i], f"{parameters_count[0][i]:,}",
             fontsize=16, ha='center', va='bottom', color='b')
    ax1.text(D_list[i], parameters_count[1][i], f"{parameters_count[1][i]:,}",
             fontsize=16, ha='center', va='bottom', color='r')

ax1.legend(fontsize=16)
ax1.grid(True)

# === 第二个子图：运行时间对比 ===
ax2 = axes[1]
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax2.plot(D_list, running_time[0], label=model_name_list[0], marker='o', linestyle='-', linewidth=3, markersize=8)
ax2.plot(D_list, running_time[1], label=model_name_list[1], marker='s', linestyle='--', linewidth=3, markersize=8)

ax2.set_xlabel(r"$d_x$", fontsize=18)
ax2.set_ylabel('Runtime (s)', fontsize=18)
ax2.tick_params(axis='both', labelsize=16)

# 在特定数据点上标注数值
for i in [2, 6, -1]:
    ax2.text(D_list[i], running_time[0][i], f"{running_time[0][i]:.2f}",
             fontsize=16, ha='center', va='bottom', color='b')
    ax2.text(D_list[i], running_time[1][i], f"{running_time[1][i]:.2f}",
             fontsize=16, ha='center', va='bottom', color='r')

ax2.legend(fontsize=16)
ax2.grid(True)
# 自动调整布局
plt.tight_layout()
# 保存图片
plt.savefig('./results/efficiency_comparison/parameters_vs_runtime.pdf', dpi=300, bbox_inches='tight')
# 显示图像
plt.show()

