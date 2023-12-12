"""
Main function for System Identification datasets

EnVI - GPSSM
"""
import os
import time
import torch
import gpytorch
import numpy as np
from tqdm import tqdm
from utils import settings as cg
from matplotlib import pyplot as plt
from models.EnVI import GPSSMs, EnVI
from torch.utils.data import DataLoader
from data import real as dd
cg.reset_seed(0)                      # setting random seed
device = cg.device                    # setting device
dtype = cg.dtype                      # setting dtype
plt.rcParams["figure.figsize"] = (20,10)

state_dim = 4                         # latent state dimension
output_dim = 1                        # observation dimension
input_dim = 1                         # control input dimension
seq_len = 50                          # sub-trajectories length
wd_slides = 50                        # steps interval for sampling the sub-trajectories (down-sampling rate)
number_ips = 20                       # number of inducing points
number_particles = 50                 # number of sample using in EnKF
# process_noise_sd = 0.05               # initial process noise std
# observation_noise_sd=0.1              # initial observation noise std
process_noise_sd = 1                    # initial process noise std
observation_noise_sd=1                  # initial observation noise std
lr = 0.005                            # learning rate
num_epoch = 600                       # number of epoch
num_repeat = 10                       # repeat experiments
save_fig = True
save_model = True
fixEmission = False
fixTransition = False
sampleU = False
data_name_all = [ 'actuator', 'ballbeam',  'dryer',  'gasfurnace',  'drive' ]
# data_name_all = [ 'actuator', 'ballbeam',  'dryer' ]
# data_name_all = ['gasfurnace',  'drive' ]

for _, data_name in enumerate(data_name_all):

    print('\n')
    print("*" * 50)
    print(f"dataset: {data_name}")
    print("*" * 50)

    DIR = f'results/SysID/EnVI_{data_name}/fixEmission_{fixEmission}_fixTrans_{fixTransition}_eNoise_{round(observation_noise_sd, 3)}_tNoise_{round(process_noise_sd, 3)}_consistentELBO_{sampleU}/'
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    '''  --------------------- get dataset ----------------------'''
    dataset = dd.get_dataset(data_name)

    batch_size = 128  # batch_size    # basically is full-batch
    train_set = dataset(train=True, sequence_length=seq_len, sequence_stride=wd_slides)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    test_set = dataset(train=False, sequence_length=seq_len, sequence_stride=wd_slides)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    '''  --------------------- repeat experiments ----------------------'''
    RMSE_ALL = []
    LOG_LL_ALL = []
    for jj in range(num_repeat):
        print('\n')
        print("!" * 30)
        print(f"Experiment repeated: {jj}")
        print("!" * 30)

        ''' --------############    Data preparation and Inducing Points initialization --------------------- '''
        # uniformly generate inducing points from [-2, 2],  shape: state_dim x number_ips x (state_dim + input_dim)
        ips = 4 * torch.rand((state_dim, number_ips, state_dim+input_dim)) - 2

        ############    Model preparation
        model = EnVI(dim_x=state_dim, dim_y=output_dim, seq_len=seq_len, ips=ips, dim_c=input_dim, N_MC=number_particles,
                     process_noise_sd=process_noise_sd, emission_noise_sd=observation_noise_sd,
                     consistentSampling=sampleU, learn_emission=False, residual_trans=False).to(device)

        if fixEmission:
            model.emission_likelihood.requires_grad_(False)
        if fixTransition:
            model.likelihood.requires_grad_(False)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        ''' -----------------------------------------------------------------  '''
        log_dir = DIR + f"{data_name}_epoch799_Repeat{jj}.pt"
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
        # Record the training time
        start_time = time.time()
        epochs_iter = tqdm(range(start_epoch, start_epoch+num_epoch), desc="Epoch")
        for epoch in epochs_iter:
            model.train()  # indicator of starting training
            loss_temp = []
            for i_iter, (U_train, Y_train) in enumerate(train_loader):
                optimizer.zero_grad()
                ELBO, filtered_x, filtered_x_var = model(observations=Y_train.to(cg.device),
                                                         input_sequence=U_train.to(cg.device))
                loss = -ELBO
                loss.backward()
                optimizer.step()
                print(f'Epoch/iter: {epoch}/{i_iter}   ELBO: {ELBO.item()}')
                loss_temp.append(loss.item())

            losses.append(np.mean(loss_temp))
            print("-" * 80)
            if epoch%100==99:
                ''' save model '''
                cg.save_models(model, optimizer, epoch, losses, DIR, data_name, jj, save_model=save_model)

                """make prediction """
                model.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    # y_pred_1,              shape: batch_size x seq_len x output_dim
                    # y_pred_sigma_1,        shape: batch_size x seq_len x output_dim x output_dim
                    y_pred_1, y_pred_sigma_1 = model.emission(filtered_x.unsqueeze(-1),
                                                              torch.diag_embed(filtered_x_var))

                    # shape: 1 x T x output_dim
                    y_pred_sigma_1 = torch.diagonal(y_pred_sigma_1, dim1=-1, dim2=-2)
                    y_pred_1, y_pred_sigma_1 = y_pred_1.view(1, -1, output_dim), y_pred_sigma_1.view(1, -1, output_dim)
                    train_len = y_pred_1.shape[1]

                    for i_iter, (U_test, Y_test) in enumerate(test_loader):


                        # Y_test: shape: batch_size x seq_len x output_dim
                        U_test = U_test.view(1, -1, input_dim).to(device) # shape: 1 x T x input_dim
                        Y_test = Y_test.view(1, -1, output_dim).to(device)# shape: 1 x T x output_dim

                        x = filtered_x[-1, -1, :]
                        # shape: N_MC x state_dim x batch_size x state_dim
                        x = x.repeat(model.N_MC, model.state_dim, 1, 1)

                        # prediction
                        # pred_x,          shape: 1 x T x state_dim
                        # pred_x_var,      shape: 1 x T x state_dim
                        NLL, pred_x, pred_x_var, pred_mu_2, pred_sigma_2 = model.Forcasting(T=100, x_0=x,
                                                                                            input_sequence=U_test,
                                                                                            observations=Y_test)

                        # shape: batch_size x seq_len x output_dim
                        pred_sigma_2 = torch.diagonal(pred_sigma_2, dim1=-1, dim2=-2)

                        pred_mu = torch.cat([y_pred_1, pred_mu_2], dim=1).detach().cpu().numpy()
                        pred_sigma = torch.cat([y_pred_sigma_1, pred_sigma_2], dim=1).detach().cpu().numpy()

                # post-processing of predictions
                data_sd = test_loader.dataset.output_normalizer.sd          # np array, shape (output_dim,)
                data_mean = test_loader.dataset.output_normalizer.mean

                y_pred_mean = pred_mu * data_sd + data_mean               # shape: 1 x T x output_dim
                Y_pred_var = pred_sigma * data_sd * data_sd               # shape: 1 x T x output_dim

                Y = torch.cat([Y_train.view(1, -1, output_dim), Y_test.detach().cpu()], dim=1).numpy()
                Y_test_original = Y * data_sd + data_mean                     # shape: 1 x T x output_dim

                MSE = np.mean(np.sum((Y_test_original[:, train_len:, :] - y_pred_mean[:, train_len:, :]) ** 2, axis=-1))
                RMSE = np.sqrt(MSE)
                print(f"\nRMSE: {RMSE}")
                print(f"log-likelihood: {-NLL}\n")

                # plot the results
                # shape: 1 x T x output_dim
                y_pred_std = np.sqrt(Y_pred_var)
                lower, upper = y_pred_mean-2*y_pred_std, y_pred_mean+2*y_pred_std

                T = Y.shape[1]

                f, ax = plt.subplots(1, 1)
                plt.plot(range(T), Y_test_original[0, :, 0], 'k-', label='true observations')
                plt.plot(range(T), y_pred_mean[0, :, 0], 'b-', label='predicted observations')
                ax.fill_between(range(T), lower[0, :, 0], upper[0, :, 0], color="b", alpha=0.2, label='95% CI')
                ax.legend(loc=0)  # , fontsize=28)
                plt.title(f'RMSE: {round(RMSE, 3)}, log-ll: {round(-NLL.item(), 3)}')
                if save_fig:
                    plt.savefig(DIR + f"prediction_performance_iter{jj}_epoch{epoch}.pdf")
                plt.show()
                plt.close()

        end_time = time.time()
        Time = start_time - end_time


        """make prediction """
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # y_pred_1,              shape: batch_size x seq_len x output_dim
            # y_pred_sigma_1,        shape: batch_size x seq_len x output_dim x output_dim
            y_pred_1, y_pred_sigma_1 = model.emission(filtered_x.unsqueeze(-1),
                                                      torch.diag_embed(filtered_x_var))

            # shape: 1 x T x output_dim
            y_pred_sigma_1 = torch.diagonal(y_pred_sigma_1, dim1=-1, dim2=-2)
            y_pred_1, y_pred_sigma_1 = y_pred_1.view(1, -1, output_dim), y_pred_sigma_1.view(1, -1, output_dim)
            train_len = y_pred_1.shape[1]

            for i_iter, (U_test, Y_test) in enumerate(test_loader):
                # Y_test: shape: batch_size x seq_len x output_dim
                U_test = U_test.view(1, -1, input_dim).to(device)    # shape: 1 x T x input_dim
                Y_test = Y_test.view(1, -1, output_dim).to(device)   # shape: 1 x T x output_dim

                x = filtered_x[-1, -1, :]
                # shape: N_MC x state_dim x batch_size x state_dim
                x = x.repeat(model.N_MC, model.state_dim, 1, 1)

                # prediction
                # pred_x,          shape: 1 x T x state_dim
                # pred_x_var,      shape: 1 x T x state_dim
                NLL, pred_x, pred_x_var, pred_mu_2, pred_sigma_2 = model.Forcasting(T=100, x_0=x,
                                                                                    input_sequence=U_test,
                                                                                    observations=Y_test)

                # shape: batch_size x seq_len x output_dim
                pred_sigma_2 = torch.diagonal(pred_sigma_2, dim1=-1, dim2=-2)

                pred_mu = torch.cat([y_pred_1, pred_mu_2], dim=1).detach().cpu().numpy()
                pred_sigma = torch.cat([y_pred_sigma_1, pred_sigma_2], dim=1).detach().cpu().numpy()

        # post-processing of predictions
        data_sd = test_loader.dataset.output_normalizer.sd  # np array, shape (output_dim,)
        data_mean = test_loader.dataset.output_normalizer.mean

        y_pred_mean = pred_mu * data_sd + data_mean  # shape: 1 x T x output_dim
        Y_pred_var = pred_sigma * data_sd * data_sd  # shape: 1 x T x output_dim

        Y = torch.cat([Y_train.view(1, -1, output_dim), Y_test.detach().cpu()], dim=1).numpy()
        Y_test_original = Y * data_sd + data_mean  # shape: 1 x T x output_dim

        MSE = np.mean(np.sum((Y_test_original[:, train_len:, :] - y_pred_mean[:, train_len:, :]) ** 2, axis=-1))
        RMSE = np.sqrt(MSE)
        print(f"\nRMSE: {RMSE}")
        print(f"log-likelihood: {-NLL}\n")

        # plot the results
        # shape: 1 x T x output_dim
        y_pred_std = np.sqrt(Y_pred_var)
        lower, upper = y_pred_mean - 2 * y_pred_std, y_pred_mean + 2 * y_pred_std

        T = Y.shape[1]

        f, ax = plt.subplots(1, 1)
        plt.plot(range(T), Y_test_original[0, :, 0], 'k-', label='true observations')
        plt.plot(range(T), y_pred_mean[0, :, 0], 'b-', label='predicted observations')
        ax.fill_between(range(T), lower[0, :, 0], upper[0, :, 0], color="b", alpha=0.2, label='95% CI')
        ax.legend(loc=0)  # , fontsize=28)
        plt.title(f'RMSE: {round(RMSE, 3)}, log-ll: {round(-NLL.item(), 3)}')
        if save_fig:
            plt.savefig(DIR + f"prediction_performance_iter{jj}.pdf")
        plt.show()
        plt.close()

        RMSE_ALL.append(RMSE)
        LOG_LL_ALL.append(NLL)
        cg.save_results(RMSE, NLL, DIR, jj)

    cg.save_results(RMSE_ALL, LOG_LL_ALL, DIR, '_all')