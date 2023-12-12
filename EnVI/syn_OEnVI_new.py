"""
Main function for synthetic datasets (kink function datasets)

OEnVI - GPSSM
"""
import math
import os
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
from data.synthetic import syn_data_generation, plot_kink_data
from tqdm import tqdm
from models.EnVI import OnlineEnVI
from utils import settings as cg
from utils.plotResult import plot_1D_all
cg.reset_seed(33)
device = cg.device
dtype = cg.dtype

'''######################  parameters to generate the dataset from a SSM with kink dynamics  #####################'''
number_ips = 15  # number of inducing points
dim_states = 1
dim_output = 1
dim_input = 0
batch_size = 1
data_length = 1500  # number of data in the online setting
func ='kinkfunc'

process_noise_sd =  0.05
observation_noise_sd_list  = [math.sqrt(0.008), math.sqrt(0.08), math.sqrt(0.8)]
MSE_all = np.zeros([len(observation_noise_sd_list), int(data_length/50)+1])
data_index = 0
for observation_noise_sd in observation_noise_sd_list:

    '''########################   Simulated data   ########################################'''
    ips, state_np, observe_np = syn_data_generation(func=func, traj_len=data_length, process_noise_sd=process_noise_sd,
                                                    observation_noise_sd=observation_noise_sd, number_ips=number_ips,
                                                    if_plot=False)

    # data_length x dim_output
    observe = torch.tensor(observe_np.reshape([data_length, dim_output]), dtype=dtype).to(device)
    # data_length x dim_states
    state = torch.tensor(state_np.reshape([data_length, dim_states]), dtype=dtype).to(device)

    """ # ##-------------------   OEnVI GPSSM & experiment settings -------------------## #"""
    save_fig = True     # True:  save figure
    save_model = True   # True:  save model
    fixEmission = True
    fixTransition = False
    sample_u = True  # consistent sampling (see Ialongo'19); (False corresponds to Doerr'18)
    num_epoch = 1  # number of epochs for the inner optimization loop (at each time step t)
    lr = 0.01  # learning rate
    DIR = f'results/kinkFunc/OnlineEnVI/fixEmission_{fixEmission}_fixTrans_{fixTransition}_eNoise_{round(observation_noise_sd ** 2, 3)}_tNoise_{round(process_noise_sd, 3)}_consistentELBO_{sample_u}/'
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    # define model
    model = OnlineEnVI(dim_x=dim_states, dim_y=dim_output, ips=ips, dim_c=0, N_MC=50,
                       process_noise_sd=process_noise_sd, emission_noise_sd=observation_noise_sd, seq_len=1,
                       consistentSampling=True, learn_emission=False, residual_trans=True).to(cg.device)
    # conditions
    if fixEmission:
        model.emission_likelihood.requires_grad_(False)
    if fixTransition:
        model.likelihood.requires_grad_(False)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    """ # ##-------------------  Start main loop (data sequentially comes in), OEnVI    -------------------## #"""
    true_state = []
    true_obser = []
    filter_X = []
    filter_Var = []
    RMSE_RAW = []
    RMSE_OEnVI = []
    MSE_fitting = []

    X = state[0, :]
    x_t_1 = X.expand(model.N_MC, dim_states, batch_size, dim_states)  # shape: N_MC x state_dim x batch_size x state_dim

    DataIter = tqdm(range(0, data_length), desc='Data index:')
    for ii in DataIter:

        true_state.append(state_np[ii])
        true_obser.append(observe_np[ii])

        """ -------------------   OEnVI  -------------------"""
        # Y: steps x 4,    X: steps x 4
        Y = observe[ii, :]
        Y = Y.reshape(batch_size, -1)  # shape: batch_size x output_dim

        # start optimization: inner optimization loop
        losses = []
        for epoch in range(num_epoch):
            model.train()
            optimizer.zero_grad()
            # filtered_x shape: batch_size x seq_len x state_dim
            # x_t shape: N_MC x state_dim x batch_size x state_dim
            ELBO, x_t, filtered_x, filtered_var = model(x_t_1=x_t_1, y_t=Y)
            loss = -ELBO
            loss.backward()
            optimizer.step()

            ##### save results
            losses.append(loss.item())
            filter_X.append(filtered_x)
            filter_Var.append(filtered_var)

        x_t_1 = x_t.detach()

        DataIter.set_postfix({'loss': '{0:1.5f}'.format(np.stack(losses).mean())})

        if ii % 50 == 0 or ii==data_length-1:
            X_EnVI = torch.cat(filter_X, dim=1)                 # shape: batch_size x seq_len x state_dim
            var_EnVI = torch.stack(filter_Var).transpose(0, 1)  # shape: batch_size x seq_len x state_dim
            # reshape
            X_EnVI = X_EnVI.reshape(-1)                         # shape: (batch_size x seq_len)
            rmse_raw = np.sqrt(np.mean((np.stack(true_obser) - np.stack(true_state)) ** 2))
            rmse_EnVI = np.sqrt(np.mean((X_EnVI.detach().cpu().numpy() - np.stack(true_state)) ** 2))
            print("\n")
            print('Baseline:', rmse_raw)
            print('EnVI:', rmse_EnVI)
            MSE_preTGP, LL_preGP = plot_1D_all(model=model, epoch=ii, func=func, condition_u=sample_u,
                                               save=save_fig,  path=DIR)
            print("_" * 80)
            RMSE_RAW.append(rmse_raw)
            RMSE_OEnVI.append(rmse_EnVI)
            MSE_fitting.append(MSE_preTGP)

    MSE_fitting = np.stack(MSE_fitting)
    MSE_all[data_index, :] = MSE_fitting
    data_index += 1

plt.figure(figsize=(6, 4.5))
plt.figure()
fontsize = 16
aa = np.linspace(0, data_length, MSE_all.shape[1])
for t in range(data_index):
    plt.plot(aa, MSE_all[t, :], '-*', lw=2)

plt.legend(['$\sigma_{\mathrm{R}}^2=0.008$', '$\sigma_{\mathrm{R}}^2=0.08$', '$\sigma_{\mathrm{R}}^2=0.8$'],
           loc='best', fontsize=fontsize-2)

plt.plot(aa, MSE_all[0, :].min()*np.ones_like(aa), 'k--', lw=0.5, alpha=0.5)
plt.plot(aa, MSE_all[1, :].min()*np.ones_like(aa), 'k--', lw=0.5, alpha=0.5)
plt.plot(aa, MSE_all[2, :].min()*np.ones_like(aa), 'k--', lw=0.5, alpha=0.5)
plt.text(20, 10, f"{MSE_all[0, :].min()}")
plt.xlim([0, data_length])
plt.xlabel('time step $t$', fontsize=fontsize-2)
plt.ylabel('MSE', fontsize=fontsize-2)
plt.tick_params(labelsize=fontsize-2)
plt.tick_params(labelsize=fontsize-2)
DIR = f'results/kinkFunc/OnlineEnVI/'
plt.savefig(DIR + f"OEnVI_{func}_MSE.pdf")

