"""
Main function for synthetic datasets

EnVI - GPSSM
"""

import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from utils import settings as cg
from utils.plotResult import plot_1D_all
from data.synthetic import syn_data_generation, plot_kink_data
from models.EnVI import GPSSMs

cg.reset_seed(0)            # setting random seed
device = cg.device          # setting device
dtype = cg.dtype            # setting dtype
number_ips = 15             # number of inducing points
num_epoch = 1000            # number of epochs

func ='kinkfunc'
num_experiments = 5

# process_noise_sd = 0.5
# observation_noise_sd_list = [1.]
# process_noise_sd = math.sqrt(0.01)
# observation_noise_sd_list = [math.sqrt(0.1)]
# process_noise_sd =  0.05
# observation_noise_sd_list = [math.sqrt(0.8)]
process_noise_sd = 0.05
observation_noise_sd_list  = [math.sqrt(0.008), math.sqrt(0.08), math.sqrt(0.8)]

save_fig = True
save_model = True
fixEmission = True
fixTransition = False
sampleU = True

# hidden_size = 32
state_dim = 1
output_dim = 1
input_dim = 0
episode = 30
seq_len = 20
batch_size = episode  # full batch training

for iii in observation_noise_sd_list:

    MSE_all = []
    LL_all = []

    observation_noise_sd = iii

    DIR = f'results/kinkFunc/EnVI/fixEmission_{fixEmission}_fixTrans_{fixTransition}_eNoise_{round(observation_noise_sd**2, 3)}_tNoise_{round(process_noise_sd, 3)}_consistentELBO_{sampleU}/'
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    for j in range(num_experiments):

        # emission matrix
        indices = [i for i in range(output_dim)]
        H_true = torch.eye(state_dim, device=device, dtype=dtype)[indices]

        ############    Data preparation
        ips, state_np, observe_np = syn_data_generation(func=func, traj_len=episode*seq_len, process_noise_sd=process_noise_sd,
                                                        observation_noise_sd=observation_noise_sd, number_ips=number_ips)

        observe = torch.tensor(observe_np.reshape([episode, seq_len, output_dim]), dtype=torch.float).to(device)
        state = torch.tensor(state_np.reshape([episode, seq_len, state_dim]), dtype=torch.float).to(device)

        ############    Model preparation
        model = GPSSMs(dim_x=state_dim, dim_y=output_dim, seq_len=seq_len, ips=ips, dim_c=0, N_MC=100,
                       process_noise_sd=process_noise_sd, emission_noise_sd=observation_noise_sd, consistentSampling=sampleU).to(device)
        if fixEmission:
            model.emission_likelihood.requires_grad_(False)
        if fixTransition:
            model.likelihood.requires_grad_(False)

        lr = 0.01
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        ''' -----------------------------------------------------------------  '''
        log_dir = DIR + f"kinkfunc_epoch{num_epoch}_MSE_0.0013_eNoise_{round(observation_noise_sd, 3)}_tNoise_{round(process_noise_sd, 3)}.pt"
        if os.path.exists(log_dir):
            checkpoint = torch.load(log_dir)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            losses = checkpoint['losses']
            print('Load epoch {} successfully!'.format(start_epoch))
        else:
            start_epoch = 0
            losses = []
            print('No existing models, training from beginning!')
        ''' -----------------------------------------------------------------  '''

        MSE = []
        LL = []
        epochiter = tqdm(range(start_epoch, start_epoch+num_epoch), desc='Epoch:')
        for epoch in epochiter:
            model.train()
            optimizer.zero_grad()
            ELBO, X_filter = model(observe, H_true)
            loss = -ELBO
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            epochiter.set_postfix({'loss': '{0:1.5f}'.format(loss.item())})

            if epoch % 100 == 0: # plot the results
                MSE_tmp, LL_tmp = plot_1D_all(model=model, epoch=epoch, func=func, condition_u=sampleU, save=save_fig, path=DIR)
                MSE.append(MSE_tmp)
                LL.append(LL_tmp.cpu().numpy())
                X_filter = torch.stack(X_filter, dim=1)
                X_filter = X_filter.squeeze().view(-1)
                plot_kink_data(x_filter=X_filter.detach().cpu().numpy(), x=state_np, y=observe_np, epoch=epoch, func=func,save_fig=save_fig, DIR=DIR)

        # Plot and save the results
        model.eval()

        plt.figure(figsize=(6, 6))
        plt.plot(np.arange(len(MSE)), np.array(MSE), c='r', label='MSE (train)')
        plt.plot(np.arange(len(LL)), np.array(LL), c='b', label='LL (train)')
        plt.xscale('log')
        plt.title(r'training MSE & Log-Likelihood, {} data'.format(func), fontsize=15)
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
        MSE_preTGP, LL_preGP = plot_1D_all(model=model, epoch=epoch, func=func, condition_u=sampleU, save=save_fig, path=DIR)
        X_filter = torch.stack(X_filter, dim=1)
        X_filter = X_filter.squeeze().view(-1)
        plot_kink_data(x_filter=X_filter.detach().cpu().numpy(), x=state_np, y=observe_np, epoch=epoch, func=func, save_fig=save_fig, DIR=DIR)

        ''' save model '''
        state = {'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'epoch': epoch,
                 'losses':losses,
                 'MSE': np.array(MSE),
                 'LL': np.array(LL),
                 }

        if save_model:
            log_dir = DIR + f"{func}_epoch{epoch}_MSE_{round(MSE_preTGP, 4)}_eNoise_{round(observation_noise_sd**2, 3)}_tNoise_{round(process_noise_sd, 3)}.pt"
            torch.save(state, log_dir)

        MSE_all.append(MSE_preTGP)
        LL_all.append(LL_preGP.cpu().numpy())

    ''' save results '''
    results = {'MSE_all': np.array(MSE_all),
               'MSE_all_mean': np.array(MSE_all).mean(),
               'MSE_all_std': np.array(MSE_all).std(),
               'LL_all': np.array(LL_all),
               'LL_all_mean': np.array(LL_all).mean(),
               'LL_all_std': np.array(LL_all).std(),
               }
    log_dir = DIR + f"MSE_LL_all_eNoise_{round(observation_noise_sd**2, 3)}_tNoise_{round(process_noise_sd, 3)}.pt"
    torch.save(results, log_dir)

