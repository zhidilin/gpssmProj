"""
 This file is to test the performance of GPSSM and ETGPSSM, on a modified input-varying kink function.
    - modified Kink function
    - original GPSSM with EnVI algorithm
"""

import os
import time
import torch
import math
import numpy as np
from tqdm import tqdm
import utils_h as cg
from matplotlib import pyplot as plt
from modules.GPSSM_GPyTorch import EnVI
from modules.EGPSSM import EnVI as EGPSSM
from data.synthetic import syn_data_generation, plot_kink_data, plot_1D_all

device = cg.device  # setting device
dtype = cg.dtype  # setting dtype
plt.rcParams["figure.figsize"] = (20, 10)

# setting parameters
state_dim = 1  # latent state dimension
output_dim = 1  # observation dimension
input_dim = 0  # control input dimension
seq_len = 20  # sub-trajectories length
episode = 30
number_ips = 15  # number of inducing points
number_particles = 150  # number of sample using in EnKF
process_noise_sd = 0.05  # initial process noise std
observation_noise_sd_list = [math.sqrt(0.0008), math.sqrt(0.008), math.sqrt(0.08), math.sqrt(0.8)]
# observation_noise_sd_list = [math.sqrt(0.08)]
# observation_noise_sd_list  = [math.sqrt(0.008)]
num_repeat = 1  # repeat experiments
lr = 0.005
save_fig = True
save_model = True
fixEmission = True
fixTransition = False
num_epoch = 2000  # number of epoch

sampleU = True
if_BNN = False
if_pureNN = True
func = 'ns-kink'
model_name = 'EGPSSM'
if if_pureNN:
    residual_trans = False
else:
    residual_trans = True

for iii in observation_noise_sd_list:
    cg.reset_seed(0)  # setting random seed
    observation_noise_sd = iii  # initial observation noise std
    if model_name == 'GPSSM':
        DIR = f'results/{func}/{model_name}/fixEmission_{fixEmission}_fixTrans_{fixTransition}_eNoise_{round(observation_noise_sd ** 2, 3)}_tNoise_{round(process_noise_sd, 3)}_consistentELBO_{sampleU}/'
    else:
        if if_pureNN:
            DIR = f'results/{func}/{model_name}_BNN_{if_BNN}_pure{if_pureNN}/fixEmission_{fixEmission}_fixTrans_{fixTransition}_eNoise_{round(observation_noise_sd ** 2, 3)}_tNoise_{round(process_noise_sd, 3)}_consistentELBO_{sampleU}/'
        else:
            DIR = f'results/{func}/{model_name}_BNN_{if_BNN}/fixEmission_{fixEmission}_fixTrans_{fixTransition}_eNoise_{round(observation_noise_sd ** 2, 3)}_tNoise_{round(process_noise_sd, 3)}_consistentELBO_{sampleU}/'
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    ############    Data preparation
    ips, state_np, observe_np = syn_data_generation(func=func,
                                                    traj_len=episode * seq_len,
                                                    process_noise_sd=process_noise_sd,
                                                    observation_noise_sd=observation_noise_sd,
                                                    number_ips=number_ips,
                                                    state_int=1.7,
                                                    if_plot=False)

    observe = torch.tensor(observe_np.reshape([episode, seq_len, output_dim]), dtype=torch.float).to(device)
    state = torch.tensor(state_np.reshape([episode, seq_len, state_dim]), dtype=torch.float).to(device)

    ##########    Model preparation
    if model_name == 'GPSSM':
        model = EnVI(dim_x=state_dim, dim_y=output_dim, seq_len=seq_len, ips=ips, dim_c=input_dim,
                     N_MC=number_particles,
                     process_noise_sd=process_noise_sd,
                     emission_noise_sd=observation_noise_sd,
                     consistentSampling=sampleU,
                     learn_emission=False,
                     residual_trans=residual_trans).to(device)
        model.BNN = None
    else:
        model = EGPSSM(dim_x=state_dim, dim_y=output_dim, seq_len=seq_len, ips=ips.squeeze(0), dim_c=input_dim,
                       N_MC=number_particles,
                       process_noise_sd=process_noise_sd,
                       emission_noise_sd=observation_noise_sd,
                       BayesianNN=if_BNN,
                       if_pureNN=if_pureNN,
                       consistentSampling=sampleU,
                       learn_emission=False,
                       residual_trans=residual_trans).to(device)
        if if_pureNN:
            model.transition.requires_grad_(False)

    if fixEmission:
        model.emission_likelihood.requires_grad_(False)
    if fixTransition:
        model.likelihood.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ############    Training
    Time = []
    MSE = []
    losses = []
    epochiter = tqdm(range(0, num_epoch), desc='Epoch:')
    t0 = time.time()
    for epoch in epochiter:
        model.train()
        optimizer.zero_grad()
        ELBO, X_filter, _ = model(observe)
        loss = -ELBO
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        epochiter.set_postfix({'loss': '{0:1.5f}'.format(loss.item())})

        if epoch % 100 == 0:  # plot the results
            MSE_tmp = plot_1D_all(model=model, epoch=epoch, func=func, save=save_fig, path=DIR)
            MSE.append(MSE_tmp)
            Time.append(time.time() - t0)
            X_filter = X_filter.squeeze().view(-1)
            plot_kink_data(x_filter=X_filter.detach().cpu().numpy(),
                           x=state_np,
                           y=observe_np,
                           epoch=epoch,
                           func=func,
                           save_fig=save_fig,
                           DIR=DIR)

            ''' save model '''
            state = {'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'epoch': epoch,
                     }

            if save_model:
                log_dir = DIR + f"{func}_epoch{epoch}_MSE_{round(MSE_tmp, 4)}_eNoise_{round(observation_noise_sd ** 2, 3)}_tNoise_{round(process_noise_sd, 3)}.pt"
                torch.save(state, log_dir)

    # Plot and save the results
    model.eval()

    plt.figure(figsize=(6, 6))
    plt.plot(np.arange(len(MSE)), np.array(MSE), c='r', label='MSE (train)')
    plt.xscale('log')
    plt.title(r'training MSE, {} data'.format(func), fontsize=15)
    plt.ylabel(r'$\mathcal{MSE}$', fontsize=15)
    plt.legend(fontsize=12)
    if save_fig:
        plt.savefig(DIR + f"func_{func}_mse.pdf")
    else:
        plt.show()

    plt.figure(figsize=(6, 6))
    plt.plot(np.arange(len(losses)), -np.array(losses), c='r', label='ELBO (train)')
    plt.xscale('log')
    plt.title(r'training loss, {} data'.format(func), fontsize=15)
    plt.ylabel(r'$\mathcal{L}$', fontsize=15)
    plt.legend(fontsize=12)
    if save_fig:
        plt.savefig(DIR + f"func_{func}_loss.pdf")
    else:
        plt.show()

    MSE_preTGP = plot_1D_all(model=model,
                             epoch=epoch,
                             func=func,
                             save=save_fig,
                             path=DIR)
    X_filter = X_filter.squeeze().view(-1)
    plot_kink_data(x_filter=X_filter.detach().cpu().numpy(),
                   x=state_np,
                   y=observe_np,
                   epoch=epoch,
                   func=func,
                   save_fig=save_fig, DIR=DIR)

    ''' save model '''
    state = {'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch,
             'losses': losses,
             'time': np.array(Time),
             'MSE': np.array(MSE),
             }

    if save_model:
        log_dir = DIR + f"{func}_epoch{epoch}_MSE_{round(MSE_preTGP, 4)}_eNoise_{round(observation_noise_sd ** 2, 3)}_tNoise_{round(process_noise_sd, 3)}.pt"
        torch.save(state, log_dir)
