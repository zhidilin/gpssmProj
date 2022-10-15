import torch
import os
from tqdm import tqdm
import gpytorch
from models.ELBO import ELBO
# dataset
from torch.utils.data import DataLoader
import models.dataset as dd
import cg.src as cg
cg.reset_seed(0)
cg.check_torch()
device = cg.device
dtype = cg.dtype

hidden_size = 32            # hidden dimension size for bi-directional LSTM
state_dim = 4               # latent state dimension
output_dim = 1              # observation dimension
input_dim = 1               # control input dimension
seq_len = 50                # sub-trajectories length
test_seq_len = 100          # sub-trajectories length
wd_slides = 1               # steps interval for sampling the sub-trajectories (down-sampling rate)
number_ips = 20             # number of inducing points
process_noise_sd = 0.05     # initial process noise std
observation_noise_sd=0.1    # initial observation noise std
number_particles = 50       # number of trajectories to be sampled
LMC = False                  # use LMC for GP or not
lr = 0.01                   # learning rate
num_epoch = 70              # number of epoch
emi_idx = None
save_fig = True
save_model = True
# data_name_all = ['ballbeam']
data_name_all = ['drive']
# data_name_all = [ 'actuator', 'ballbeam',  'dryer',  'gasfurnace',  'drive' ]
for jjj in range(1):
    for ii in range(len(data_name_all)):

        '''  --------------------- get dataset ----------------------'''
        data_name = data_name_all[ii]
        dataset = dd.get_dataset(data_name)

        batch_size = 16  # batch_size
        train_set = dataset(train=True, sequence_length=seq_len, sequence_stride=wd_slides)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_set = dataset(train=False, sequence_length=test_seq_len, sequence_stride=test_seq_len)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

        # uniformly generate inducing points from [-2, 2]
        if LMC:
            num_latent_gp = state_dim
            inducing_points = 4 * torch.rand((num_latent_gp, number_ips, state_dim + input_dim)) - 2
        else:
            inducing_points = 4 * torch.rand((state_dim, number_ips, state_dim+input_dim)) - 2

        if data_name == 'ballbeam' or data_name == 'drive':
            ARD = True
        else:
            ARD = False

        # 准备模型
        model = ELBO(state_dim=state_dim,
                     output_dim=output_dim,
                     seq_len=seq_len,
                     inducing_points=inducing_points,
                     input_dim=input_dim,
                     process_noise_sd=process_noise_sd,
                     num_particles=number_particles,
                     ARD=ARD,
                     LMC=LMC
                     ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # results dictionary
        if LMC:
            result_dir = f'./figs_SID/{data_name}/LMC_{LMC}/batchSize{batch_size}_seq{seq_len}_nGPs_{num_latent_gp}/'
        else:
            result_dir = f'./figs_SID/{data_name}/LMC_{LMC}/batchSize{batch_size}_seq{seq_len}/'

        print(f"LMC: {LMC},      DIR: {result_dir}")

        if LMC:
            epoch_list = ["best_model", "epoch50", "epoch69"]
        else:
            epoch_list = ["best_model", "epoch50", "epoch69"]
        # epoch_list = ["epoch800"]
        for i in epoch_list:
            log_dir_ini = result_dir + f"{data_name}_{i}.pt"
            if os.path.exists(log_dir_ini):
                checkpoint = torch.load(log_dir_ini)
                model.load_state_dict(checkpoint['model'])
                # optimizer.load_state_dict(checkpoint['optimizer'])
                # start_epoch = checkpoint['epoch']
                # losses = checkpoint['losses']
                print(f'load model: {i} successfully!')
            else:
                start_epoch = 0
                losses = []
                print('No existing model, training from scratch！')

            """make prediction """
            model.eval()
            # Make predictions
            RMSE_list = []
            log_ll_list = []
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                for i_iter, (U_test, Y_test) in enumerate(test_loader):
                    if i_iter>5:
                        break
                    RMSE, log_ll = model.prediction(output_y=Y_test.to(device), input_c=U_test.to(device),
                                                    emi_idx=emi_idx,plt_result=True,
                                                    plt_save=True, model_str=i,
                                                    i_iter=i_iter, result_dir=result_dir)

                    print(f"RMSE: {RMSE},   log_ll: {log_ll}")
                    RMSE_list.append(RMSE)
                    log_ll_list.append(log_ll)

            RMSE_list = torch.tensor(RMSE_list).numpy()
            log_ll_list = torch.tensor(log_ll_list).numpy()
            print(f"Average RMSE: {RMSE_list.mean()}")
            print(f"Average Log_ll: {log_ll_list.mean()}")
            print("-"*50+"\n")

            f = open(result_dir + f"result_of_{i}.txt", "w")
            f.write("RMSE: ")
            f.write(str(RMSE_list) + "\n")
            f.write("Average RMSE: ")
            f.write(str(RMSE_list.mean()) + "\n")
            f.write("Log_ll: ")
            f.write(str(log_ll_list) + "\n")
            f.write("Average Log_ll: ")
            f.write(str(log_ll_list.mean()) )

            f.close()

