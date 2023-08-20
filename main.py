'''
# this file is to train the non-mean-field GPSSM on the 5 system identification data sets
# setup:
1. whiten the data
2. state_dim = 4, input_dim = 1, output_dim = 1, N_ips = 20, batch_size = full batch, seq_len = 50
3. inducing points initialization z ~ uniform[-2, 2]
'''
import torch
import gpytorch
import os
from tqdm import tqdm
import time
from models.GPSSMs import ODGPSSM, EGPSSM
# dataset
from torch.utils.data import DataLoader
import models.dataset as dd
from matplotlib import pyplot as plt
from models import utils as cg
cg.reset_seed(0)
device = cg.device
dtype = cg.dtype

"""-------------------- Common settings --------------------"""
state_dim = 4               # latent state dimension
num_latent_gp = state_dim   # number of latent GP for ODGPSSM
output_dim = 1              # observation dimension
input_dim = 1               # control input dimension
seq_len = 50                # sub-trajectories length
wd_slides = 50              # steps interval for sampling the sub-trajectories (down-sampling rate)
number_ips = 20             # number of inducing points
process_noise_sd = 0.05     # initial process noise std
observation_noise_sd=0.1    # initial observation noise std
number_particles = 50       # number of trajectories to be sampled
lr = 0.01                   # learning rate  %% default 0.01
num_epoch = 1000            # number of epoch
num_repeat = 5              # repeat experiments
save_fig = True
save_model = True

"""-------------------- GPSSM settings --------------------"""
LMC = False      # use LMC for GP or not
ARD = False

"""-------------------- EGPSSM settings --------------------"""
model_initialization = False
ETGP = True
SAL_flow = False
Affine_flow = (not SAL_flow)
nn_par = True            # Only in Linear Flow currently. If true, using NN to learn parameters in the linear flow
feedTime = nn_par        # Input-dependent non-stationary EGPSSM. TODO: Auto-Regularization needed
if nn_par and feedTime:
    input_dim = input_dim + 1
    lr = 0.05


"""-------------------- dataset settings --------------------"""
data_name_all = [ 'actuator', 'ballbeam',  'dryer',  'gasfurnace',  'drive' ]

for ii in range(len(data_name_all)):

    """------------------------ get dataset --------------------------"""
    data_name = data_name_all[ii]
    print('\n')
    print("*" * 50)
    print(f"dataset: {data_name}")
    print("*" * 50)
    dataset = dd.get_dataset(data_name)

    batch_size = 16  # batch_size
    train_set = dataset(train=True, sequence_length=seq_len, sequence_stride=wd_slides, if_time_step=feedTime)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_set = dataset(train=False, sequence_length=seq_len, sequence_stride=seq_len, if_time_step=feedTime)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # results dictionary
    if ETGP:
        if SAL_flow:
            result_dir = f'./results/{data_name}/EGPSSM_batchSize{batch_size}_seq{seq_len}_SALFlow/'
        else:
            if not nn_par:
                result_dir = f'./results/{data_name}/EGPSSM_batchSize{batch_size}_seq{seq_len}_Linear/'
            else:
                result_dir = f'./results/{data_name}/EGPSSM_batchSize{batch_size}_seq{seq_len}_nonStationary_Linear/'
        result_dir_ini = f'./results/{data_name}/LMC_{LMC}_batchSize{batch_size}_seq{seq_len}/'
    else:
        if LMC:
            result_dir = f'./results/{data_name}/LMC_{LMC}_batchSize{batch_size}_seq{seq_len}_nGPs_{num_latent_gp}/'
        else:
            result_dir = f'./results/{data_name}/LMC_{LMC}_batchSize{batch_size}_seq{seq_len}/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    '''  --------------------- repeat experiments ----------------------'''
    RMSE_ALL = []
    LOG_LL_ALL = []
    for jj in range(num_repeat):
        print('\n')
        print("!" * 30)
        print(f"Experiment repeated: {jj}")
        print("!" * 30)
        # uniformly generate inducing points from [-2, 2]

        # Model Initialization
        if ETGP:
            inducing_points = 4 * torch.rand(number_ips, state_dim + input_dim) - 2
            model = EGPSSM(state_dim=state_dim, output_dim=output_dim, seq_len=seq_len, inducing_points=inducing_points,
                           input_dim=input_dim, process_noise_sd=process_noise_sd,
                           N_MC=number_particles, ARD=ARD, SAL_flow=SAL_flow, linear_flow_nn_par=nn_par).to(device)

            if model_initialization:
                log_dir_inialization = result_dir_ini + f"{data_name}_epoch699_Repeat{jj}.pt"
                print(log_dir_inialization)
                PRSSM_ = model.initialization(log_dir_ini=log_dir_inialization)
                model.load_state_dict(PRSSM_, strict=False)
                model.GMatrix.requires_grad_(False)

        else:
            if LMC:
                inducing_points = 4 * torch.rand((num_latent_gp, number_ips, state_dim + input_dim)) - 2
            else:
                inducing_points = 4 * torch.rand((state_dim, number_ips, state_dim + input_dim)) - 2

            model = ODGPSSM(state_dim=state_dim, output_dim=output_dim, seq_len=seq_len, inducing_points=inducing_points,
                            input_dim=input_dim, process_noise_sd=process_noise_sd,
                            N_MC=number_particles, ARD=ARD, LMC=LMC).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)


        log_dir_ini = result_dir + f"{data_name}_epoch799_Repeat{jj}.pt"
        if os.path.exists(log_dir_ini):
            checkpoint = torch.load(log_dir_ini)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            losses = checkpoint['losses']
            print('加载 epoch {} 成功！'.format(start_epoch))
        else:
            start_epoch = 0
            losses = []
            print('无保存模型，将从头开始训练！')

        # Record the training time
        start_time = time.time()
        epochiter = tqdm(range(start_epoch, start_epoch+num_epoch), desc='Epoch:')
        best_loss = torch.tensor(0., device=cg.device)

        for epoch in epochiter:
            # indicator of starting training
            model.train()
            # if epoch == 5:
            #     print(epoch)
            for i_iter, (U_train, Y_train) in enumerate(train_loader):
                ELBO = model(observations=Y_train.to(device), input_sequence=U_train.to(device))
                loss = -ELBO
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                print(f'Epoch/iter: {epoch}/{i_iter}   ELBO: {ELBO.item()}')

                # save the best model
                if epoch==0 and i_iter == 0:
                    best_loss = loss.item()

                if loss < best_loss:
                    cg.save_best_model(model, optimizer, epoch, result_dir, data_name, jj)

            print("-" * 50)
            if epoch%200==0 or (epoch == start_epoch+num_epoch-1):
                cg.save_models(model, optimizer, epoch, losses, result_dir, data_name, jj, save_model=save_model)

                """make prediction """
                model.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    for i_iter, (U_test, Y_test) in enumerate(test_loader):
                        y_pred, log_ll = model.prediction(observations=Y_test.to(device),
                                                          input_sequence=U_test.to(device))
                        y_pred = y_pred.detach().cpu()  # shape: seq_len x N_MC x batch_size x output_dim

                # post-processing of predictions
                data_sd = torch.tensor(test_loader.dataset.output_normalizer.sd,
                                       dtype=dtype)  # np array, shape (output_dim, )
                data_mean = torch.tensor(test_loader.dataset.output_normalizer.mean, dtype=dtype)
                y_pred_original = y_pred * data_sd + data_mean  # shape: seq_len x N_MC x batch_size x output_dim
                y_pred_mean = y_pred_original.mean(dim=1)  # shape: seq_len x batch_size x output_dim
                y_pred_mean = y_pred_mean.transpose(0, 1)  # shape: batch_size x seq_len x output_dim

                Y_test_original = Y_test.detach().cpu() * data_sd + data_mean  # shape: batch_size x seq_len x output_dim
                MSE = torch.mean(torch.sum((Y_test_original - y_pred_mean) ** 2, dim=-1))
                RMSE = MSE.sqrt()
                print(f"\nRMSE: {RMSE}")
                print(f"log-likelihood: {log_ll}\n")


                # plot the results
                # shape: seq_len x batch_size x output_dim
                y_pred_std = data_sd * y_pred.std(dim=1) * data_sd \
                             + torch.sqrt(model.emission_likelihood.task_noises.detach().cpu()).view(-1, output_dim)

                test_len = y_pred_mean.shape[0] * y_pred_mean.shape[1]

                y_test_true = Y_test_original.reshape(test_len, )
                y_predict = y_pred_mean.reshape(test_len, )

                f, ax = plt.subplots(1, 1)
                plt.plot(range(test_len), y_test_true.cpu().numpy(), 'k-', label='true observations')
                plt.plot(range(test_len), y_predict.cpu().numpy(), 'b-', label='predicted observations')
                # ax.fill_between(range(test_len), lower.cpu().numpy(), upper.cpu().numpy(), color="b", alpha=0.2, label='95% CI')
                ax.legend(loc=0)  # , fontsize=28)
                plt.title(f'RMSE: {round(RMSE.item(), 3)}, log-ll: {round(log_ll.item(), 3)}')
                plt.savefig(result_dir + f"prediction_performance_iter{jj}_epoch{epoch}.pdf")
                plt.show()
                plt.close()

        end_time = time.time()
        Time = start_time - end_time

        """make prediction """
        model.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i_iter, (U_test, Y_test) in enumerate(test_loader):
                y_pred, log_ll = model.prediction(observations=Y_test.to(device), input_sequence=U_test.to(device))
                y_pred = y_pred.detach().cpu()                  # shape: seq_len x N_MC x batch_size x output_dim


        # post-processing of predictions
        data_sd = torch.tensor(test_loader.dataset.output_normalizer.sd, dtype=dtype)  # np array, shape (output_dim, )
        data_mean = torch.tensor(test_loader.dataset.output_normalizer.mean, dtype=dtype)
        y_pred_original = y_pred * data_sd + data_mean  # shape: seq_len x N_MC x batch_size x output_dim
        y_pred_mean = y_pred_original.mean(dim=1)       # shape: seq_len x batch_size x output_dim
        y_pred_mean = y_pred_mean.transpose(0, 1)       # shape: batch_size x seq_len x output_dim

        Y_test_original = Y_test.detach().cpu() * data_sd + data_mean  # shape: batch_size x seq_len x output_dim
        MSE = torch.mean(torch.sum((Y_test_original - y_pred_mean) ** 2, dim=-1))
        RMSE = MSE.sqrt()
        print(f"RMSE: {RMSE}")
        print(f"log-likelihood: {log_ll}")

        RMSE_ALL.append(RMSE)
        LOG_LL_ALL.append(log_ll)
        cg.save_results(RMSE, log_ll, result_dir, jj)


        # plot the results
        # shape: seq_len x batch_size x output_dim
        y_pred_std = data_sd * y_pred.std(dim=1) * data_sd \
                     + torch.sqrt(model.emission_likelihood.task_noises.detach().cpu()).view(-1, output_dim)

        test_len = y_pred_mean.shape[0] * y_pred_mean.shape[1]

        y_test_true = Y_test_original.reshape(test_len, )
        y_predict = y_pred_mean.reshape(test_len, )

        f, ax = plt.subplots(1, 1)
        plt.plot(range(test_len), y_test_true.cpu().numpy(), 'k-', label='true observations')
        plt.plot(range(test_len), y_predict.cpu().numpy(), 'b-', label='predicted observations')
        # ax.fill_between(range(test_len), lower.cpu().numpy(), upper.cpu().numpy(), color="b", alpha=0.2, label='95% CI')
        ax.legend(loc=0)  # , fontsize=28)
        plt.title(f'RMSE: {round(RMSE.item(), 3)}, log-ll: {round(log_ll.item(), 3)}')
        plt.savefig(result_dir + f"prediction_performance_iter{jj}.pdf")
        plt.show()
        plt.close()

    cg.save_results(RMSE_ALL, LOG_LL_ALL, result_dir, '_all')

