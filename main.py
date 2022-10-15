'''
# this file is to test the GPSSM on the 5 system identification data sets
# setup:
1. whiten the data
2. state_dim = 4, input_dim = 1, output_dim = 1, N_ips = 20, batch_size = 16, seq_len = 50
3. inducing points initialization z ~ uniform[-2, 2]
'''
import torch
import os
from tqdm import tqdm
import time
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
wd_slides = 1               # steps interval for sampling the sub-trajectories (down-sampling rate)
number_ips = 20             # number of inducing points
process_noise_sd = 0.05     # initial process noise std
observation_noise_sd=0.1    # initial observation noise std
number_particles = 50       # number of trajectories to be sampled
LMC = True                  # use LMC for GP or not
lr = 0.01                   # learning rate  %% default 0.01
num_epoch = 70              # number of epoch
emi_idx = None
save_fig = True
save_model = True
# data_name_all = ['ballbeam']
data_name_all = ['drive']
print(data_name_all)
# data_name_all = [ 'actuator', 'ballbeam',  'dryer',  'gasfurnace',  'drive' ]

for ii in range(len(data_name_all)):

    '''  --------------------- get dataset ----------------------'''
    data_name = data_name_all[ii]
    dataset = dd.get_dataset(data_name)

    batch_size = 16  # batch_size
    train_set = dataset(train=True, sequence_length=seq_len, sequence_stride=wd_slides)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
    test_set = dataset(train=False, sequence_length=seq_len, sequence_stride=seq_len)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)

    if data_name == 'ballbeam' or data_name == 'drive':
        ARD = True
    else:
        ARD = False

    # uniformly generate inducing points from [-2, 2]
    if LMC:
        num_latent_gp = state_dim
        inducing_points = 4 * torch.rand((num_latent_gp, number_ips, state_dim + input_dim)) - 2
    else:
        inducing_points = 4 * torch.rand((state_dim, number_ips, state_dim+input_dim)) - 2

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

    if data_name == 'ballbeam' or data_name == 'drive':
        # initialization for these two datasets (from doerr's paper); other datasets use default initializations of gpytorch
        lengthscale = 2 * torch.ones(inducing_points.shape[0], 1, inducing_points.shape[-1], dtype=dtype, device=device)
        outputscale = 0.5 * torch.ones(inducing_points.shape[0], dtype=dtype, device=device)
        model.transition.covar_module.base_kernel.lengthscale = lengthscale
        model.transition.covar_module.outputscale = outputscale

        # initialization for q(u)
        qu_mean = torch.normal(mean=0.,std=0.05, size=(inducing_points.shape[0], inducing_points.shape[1]),
                               dtype=dtype, device=device)
        model.transition.variational_strategy.variational_distribution.mean.data = qu_mean

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # results dictionary
    if LMC:
        result_dir = f'./figs_SID/{data_name}/LMC_{LMC}/batchSize{batch_size}_seq{seq_len}_nGPs_{num_latent_gp}/'
    else:
        result_dir = f'./figs_SID/{data_name}/LMC_{LMC}/batchSize{batch_size}_seq{seq_len}/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    log_dir_ini = result_dir + f"{data_name}_epoch1998.pt"
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
    losses = []
    epochiter = tqdm(range(start_epoch, start_epoch+num_epoch), desc='Epoch:')
    best_loss = torch.tensor(0.).to(device)
    for epoch in epochiter:
        model.train()  # indicator of starting training
        for i_iter, (U_train, Y_train) in enumerate(train_loader):
            ELBO1 = model(output_y=Y_train.to(device), input_c=U_train.to(device), emi_idx=emi_idx)
            loss = -ELBO1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            print(f'Epoch/iter: {epoch}/{i_iter}   ELBO: {ELBO1.item()}')


            # save the best model
            if epoch==0 and i_iter == 0:
                best_loss = loss.item()

            if loss < best_loss:
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                log_dir = result_dir + f"{data_name}_best_model.pt"
                torch.save(state, log_dir)


        print("-" * 50)
        if epoch%10==0 or (epoch == start_epoch+num_epoch-1):
            ''' save model '''
            state = {'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'epoch': epoch,
                     'losses': losses}
            if save_model:
                log_dir = result_dir + f"{data_name}_epoch{epoch}.pt"
                torch.save(state, log_dir)

    end_time = time.time()


    """make prediction """
    model.eval()








