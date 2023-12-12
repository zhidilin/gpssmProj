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
from models.EnVI import OEnVI
from utils import settings as cg
from utils.plotResult import plot_1D_all
cg.reset_seed(33)
device = cg.device
dtype = cg.dtype


'''######################  parameters to generate the dataset from a SSM with kink dynamics  #####################'''
func ='kinkfunc'
process_noise_sd =  0.05
# observation_noise_sd_list  = [math.sqrt(0.008), math.sqrt(0.08), math.sqrt(0.8)]
observation_noise_sd_list  = [math.sqrt(0.008)]

for observation_noise_sd in observation_noise_sd_list:
    number_ips = 15  # number of inducing points
    dim_states = 1
    dim_output = 1
    dim_input = 0
    batch_size = 1
    data_length = 1000  # number of data in the online setting

    '''########################   Simulated data   ########################################'''
    ips, state_np, observe_np = syn_data_generation(func=func, traj_len=data_length, process_noise_sd=process_noise_sd,
                                                    observation_noise_sd=observation_noise_sd, number_ips=number_ips,
                                                    if_plot=False)

    # data_length x dim_output
    observe = torch.tensor(observe_np.reshape([data_length, dim_output]), dtype=dtype).to(device)
    # data_length x dim_states
    state = torch.tensor(state_np.reshape([data_length, dim_states]), dtype=dtype).to(device)

    """ # ##-------------------   OEnVI GPSSM & experiment settings -------------------## #"""
    save_fig = True  # True:  save figure
    save_model = True  # True:  save model
    fixEmission = True
    fixTransition = False
    sample_u = True  # consistent sampling (see Ialongo'19); (False corresponds to Doerr'18)
    num_epoch = 1  # number of epochs for the inner optimization loop (at each time step t)
    lr = 0.01  # learning rate
    DIR = f'results/kinkFunc/OEnVI/fixEmission_{fixEmission}_fixTrans_{fixTransition}_eNoise_{round(observation_noise_sd ** 2, 3)}_tNoise_{round(process_noise_sd, 3)}_consistentELBO_{sample_u}/'
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    # initial observation matrix
    indices = [i for i in range(dim_output)]
    H_true = torch.eye(dim_states, device=cg.device, dtype=cg.dtype)[indices]

    # define model
    model = OEnVI(dim_x=dim_states, dim_y=dim_output, seq_len=1, ips=ips, dim_c=0, N_MC=100,
                  process_noise_sd=process_noise_sd, emission_noise_sd=observation_noise_sd,
                  consistentSampling=sample_u).to(cg.device)
    model.variance_output = True

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
        Y = Y.reshape(batch_size, 1, -1)  # shape: batch_size x seq_len x output_dim

        # start optimization: inner optimization loop
        losses = []
        for epoch in range(num_epoch):
            model.train()
            optimizer.zero_grad()
            ELBO, x_t, filtered_x, filtered_var = model(Y, x_t_1, H_true)
            loss = -ELBO
            loss.backward()
            optimizer.step()

            ##### save results
            losses.append(loss.item())
            filter_X.append(filtered_x)
            filter_Var.append(filtered_var)

        x_t_1 = x_t.detach()

        DataIter.set_postfix({'loss': '{0:1.5f}'.format(np.stack(losses).mean())})

        if ii % 50 == 0:
            X_EnVI = torch.cat(filter_X, dim=1)  # shape: batch_size x seq_len x state_dim
            var_EnVI = torch.stack(filter_Var).transpose(0, 1)  # shape: batch_size x seq_len x state_dim x state_dim
            # reshape
            X_EnVI = X_EnVI.reshape(-1)  # shape: (batch_size x seq_len)
            rmse_raw = np.sqrt(np.mean(np.sum((np.stack(true_obser) - np.stack(true_state)) ** 2)))
            rmse_EnVI = np.sqrt(np.mean(np.sum((X_EnVI.detach().cpu().numpy() - np.stack(true_state)) ** 2)))
            print("\n")
            print('Baseline:', rmse_raw)
            print('EnVI:', rmse_EnVI)
            MSE_preTGP, LL_preGP = plot_1D_all(model=model, epoch=ii, func=func, condition_u=sample_u,
                                               save=save_fig,  path=DIR)
            # print(f'fitting MSE: {MSE_preTGP},   Iteration: {ii}')
            # print(f'fitting Log-likelihood: {LL_preGP},   Iteration: {ii}')
            print("_" * 80)
            RMSE_RAW.append(rmse_raw)
            RMSE_OEnVI.append(rmse_EnVI)
            MSE_fitting.append(MSE_preTGP)

    # """#  --------------------------------------  Plot and save the results  -------------------------------------- """
    # sss=[0, 120, 240, 360, 480, 600, 720, 840, 900, 0]
    # lll=[120, 120, 120, 120, 120, 120, 120, 120, 100, data_length]
    # for jjj in range(len(sss)):
    #     length_ = lll[jjj]
    #     start_ = sss[jjj]
    #     model.eval()
    #     with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #         rmse_raw = np.sqrt(np.mean(np.sum((np.stack(true_obser)[start_: start_ + length_, ] - np.stack(true_state)[start_: start_ + length_, ]) ** 2)))
    #
    #         X_EnVI = torch.cat(filter_X, dim=1)  # shape: batch_size x seq_len x state_dim
    #         var_EnVI = torch.stack(filter_Var).transpose(0, 1)  # shape: batch_size x seq_len x state_dim x state_dim
    #         # reshape
    #         X_EnVI = X_EnVI.reshape(-1)  # shape: (batch_size x seq_len)
    #         var_EnVI = var_EnVI.reshape(-1, dim_states, dim_states)  # shape: (batch_size x seq_len) x state_dim x state_dim
    #         rmse_EnVI = np.sqrt(np.mean(np.sum((X_EnVI.detach().cpu().numpy()[start_: start_ + length_, ] - np.stack(true_state)[start_: start_ + length_, ]) ** 2)))
    #         print('Baseline:', rmse_raw)
    #         print('EnVI:', rmse_EnVI)
    #         print(" ")
    #         sd_EnVI = torch.sqrt(var_EnVI.diagonal(dim1=-1, dim2=-2))  # shape: (batch_size x seq_len) x state_dim
    #         lower_EnVI = X_EnVI - 2 * sd_EnVI[:,0]  # shape: (batch_size x seq_len) x state_dim
    #         upper_EnVI = X_EnVI + 2 * sd_EnVI[:,0]  # shape: (batch_size x seq_len) x state_dim
    #         # Initialize plots
    #         fig = plt.figure(figsize=(5 * dim_output, 3))
    #         l1, = plt.plot(np.linspace(start_+1, start_+length_+1, length_), np.stack(true_obser)[start_ : start_+length_].squeeze(), 'k*')
    #         l2, = plt.plot(np.linspace(start_+1, start_+length_+1, length_), np.stack(true_state)[start_ : start_+length_].squeeze(), 'g*')
    #
    #         # plot EnVI results
    #         l5, = plt.plot(np.linspace(start_+1, start_+length_+1, length_), X_EnVI[start_ : start_+length_,].detach().cpu().numpy(), 'm-')
    #         l6 = plt.fill_between(np.linspace(start_+1, start_+length_+1, length_),
    #                              lower_EnVI[start_ : start_+length_, ].detach().cpu().numpy(),
    #                              upper_EnVI[start_ : start_+length_, ].detach().cpu().numpy(), alpha=0.5)
    #         # ax.set_ylim([-3, 3])
    #         plt.legend(handles=[l1, l2, l5, l6], labels=['obsers.', 'states', 'OEnVI', '$\pm 2 \sigma$'],fontsize=15, loc='best')
    #         plt.xlabel('$t$', fontsize=12)
    #         plt.tick_params(labelsize=15)
    #         # plt.tick_params(axis="x", labelsize=15, colors='r')
    #         fig.tight_layout()
    #         # plt.savefig(f"results/carTrack/carTrack_{start_}_to_{start_+length_}_EnVI.pdf")
    #         plt.show()
    #         plt.close()