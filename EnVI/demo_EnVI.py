"""
outdated
"""

import os
os.chdir('..')
# print(os.getcwd())

## don't display the figure in pycharm
# import matplotlib
# matplotlib.use('agg')
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
from models.EnVI import GPSSMs, EnVI
device='cpu'
# device='cuda:1'


plt.rcParams["figure.figsize"] = (20,10)
t = np.linspace(0, 10, 100)
y = np.stack([np.sin(t), np.cos(t)]).T + np.random.normal(0, 0.1, (100, 2))
mask = torch.ones_like(torch.Tensor(y))
plt.scatter(t, y[:,0])
plt.scatter(t, y[:,1])
plt.plot(t, np.sin(t))
plt.plot(t, np.cos(t))
plt.show()


ips = torch.randn((8, 20, 8))
model = EnVI(dim_x=8, dim_y=2, seq_len=100, ips=ips, dim_c=0, N_MC=50, process_noise_sd=1,
              emission_noise_sd=1, consistentSampling=False, learn_emission=True, residual_trans=False).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochIter = tqdm(range(0, 501), desc='Epoch:')
for i in epochIter:
    # with torch.autograd.profiler.profile() as prof:
    optimizer.zero_grad()
    elbo, _, _ = model(torch.tensor(y[None], device=device, dtype=torch.float))
    loss = -elbo
    loss.backward()
    optimizer.step()
    epochIter.set_postfix({'loss': '{0:1.5f}'.format(loss.item())})
    # print(prof.key_averages().table(sort_by="self_cpu_time_total"))

    if i % 100 == 0:
        # one-step prediction (with filtering at each step
        if model._get_name() == "EnVI":
            '''for EnVI model:'''
            x0 = torch.zeros(model.N_MC, model.state_dim, 1, model.state_dim, device=device, dtype=torch.float)
        elif model._get_name() == "EnVI_":
            '''## for EnVI_ model:'''
            x0 = torch.zeros(1, model.state_dim, model.N_MC, model.state_dim, device=device, dtype=torch.float)
        else:
            raise NotImplementedError("check model setting")

        # filtered_x,         shape: batch_size x seq_len x state_dim
        # filtered_x_var,     shape: batch_size x seq_len x state_dim
        _,  filtered_x, filtered_x_var = model.iterate_sequence(torch.tensor(y[None], device=device, dtype=torch.float),
                                                                x_0=x0)
        x = filtered_x[:, -1]
        if model._get_name() == "EnVI":
            '''## for EnVI model:'''
            # shape: N_MC x state_dim x batch_size x state_dim
            x = x.repeat(model.N_MC, model.state_dim, 1, 1)
        elif model._get_name() == "EnVI_":
            '''## for EnVI_ model:'''
            # shape: batch_size x state_dim x N_MC x state_dim
            x = x.repeat(model.N_MC, model.state_dim, 1, 1).permute(2,1,0,3)
        else:
            raise NotImplementedError("check model setting")


        # pred_mu_1,         shape: batch_size x seq_len x output_dim
        # pred_sigma_1,      shape: batch_size x seq_len x output_dim x output_dim
        pred_mu_1, pred_sigma_1 = model.emission(filtered_x.unsqueeze(-1), torch.diag_embed(filtered_x_var))
        pred_sigma_1 = torch.diagonal(pred_sigma_1, dim1=-1, dim2=-2)  # shape: batch_size x seq_len x output_dim

        # prediction
        # pred_x,          shape: batch_size x seq_len x state_dim
        # pred_x_var,      shape: batch_size x seq_len x state_dim
        pred_x, pred_x_var = model.Forcasting(T=100, x_0=x)

        # pred_mu_2,          shape: batch_size x seq_len x output_dim
        # pred_sigma_2,      shape: batch_size x seq_len x output_dim x output_dim
        pred_mu_2, pred_sigma_2 = model.emission(pred_x.unsqueeze(-1), torch.diag_embed(pred_x_var))
        pred_sigma_2 = torch.diagonal(pred_sigma_2, dim1=-1, dim2=-2)  # shape: batch_size x seq_len x output_dim

        pred_mu = torch.cat([pred_mu_1, pred_mu_2], dim=1).detach().cpu().numpy()
        pred_sigma = torch.cat([pred_sigma_1, pred_sigma_2], dim=1).detach().cpu().numpy()

        plt.scatter(t, y[:, 0], c='C0')
        plt.scatter(t, y[:, 1], c='C1')

        plt.plot(np.linspace(0, 20, 200), pred_mu[0, :, 0], c='C0')
        plt.plot(np.linspace(0, 20, 200), pred_mu[0, :, 1], c='C1')

        plt.fill_between(np.linspace(0, 20, 200),
                         pred_mu[0, :, 0] - np.sqrt(pred_sigma[0, :, 0]),
                         pred_mu[0, :, 0] + np.sqrt(pred_sigma[0, :, 0]), alpha=0.2)
        plt.fill_between(np.linspace(0, 20, 200),
                         pred_mu[0, :, 1] - np.sqrt(pred_sigma[0, :, 1]),
                         pred_mu[0, :, 1] + np.sqrt(pred_sigma[0, :, 1]), alpha=0.2)

        plt.plot()
        plt.show()

# one-step prediction (with filtering at each step
if model._get_name() == "EnVI":
    '''for EnVI model:'''
    x0 = torch.zeros(model.N_MC, model.state_dim, 1, model.state_dim, device=device, dtype=torch.float)
elif model._get_name() == "EnVI_":
    '''## for EnVI_ model:'''
    x0 = torch.zeros(1, model.state_dim, model.N_MC, model.state_dim, device=device, dtype=torch.float)
else:
    raise NotImplementedError("check model setting")

# filtered_x,         shape: batch_size x seq_len x state_dim
# filtered_x_var,     shape: batch_size x seq_len x state_dim
_, filtered_x, filtered_x_var = model.iterate_sequence(torch.tensor(y[None], device=device, dtype=torch.float),
                                                       x_0=x0)
x = filtered_x[:, -1]
if model._get_name() == "EnVI":
    '''## for EnVI model:'''
    # shape: N_MC x state_dim x batch_size x state_dim
    x = x.repeat(model.N_MC, model.state_dim, 1, 1)
elif model._get_name() == "EnVI_":
    '''## for EnVI_ model:'''
    # shape: batch_size x state_dim x N_MC x state_dim
    x = x.repeat(model.N_MC, model.state_dim, 1, 1).permute(2, 1, 0, 3)
else:
    raise NotImplementedError("check model setting")

# pred_mu_1,         shape: batch_size x seq_len x output_dim
# pred_sigma_1,      shape: batch_size x seq_len x output_dim x output_dim
pred_mu_1, pred_sigma_1 = model.emission(filtered_x.unsqueeze(-1), torch.diag_embed(filtered_x_var))
pred_sigma_1 = torch.diagonal(pred_sigma_1, dim1=-1, dim2=-2)  # shape: batch_size x seq_len x output_dim

# prediction
# pred_x,          shape: batch_size x seq_len x state_dim
# pred_x_var,      shape: batch_size x seq_len x state_dim
pred_x, pred_x_var = model.Forcasting(T=100, x_0=x)

# pred_mu_2,          shape: batch_size x seq_len x output_dim
# pred_sigma_2,      shape: batch_size x seq_len x output_dim x output_dim
pred_mu_2, pred_sigma_2 = model.emission(pred_x.unsqueeze(-1), torch.diag_embed(pred_x_var))
pred_sigma_2 = torch.diagonal(pred_sigma_2, dim1=-1, dim2=-2)  # shape: batch_size x seq_len x output_dim

pred_mu = torch.cat([pred_mu_1, pred_mu_2], dim=1).detach().cpu().numpy()
pred_sigma = torch.cat([pred_sigma_1, pred_sigma_2], dim=1).detach().cpu().numpy()

plt.scatter(t, y[:, 0], c='C0')
plt.scatter(t, y[:, 1], c='C1')

plt.plot(np.linspace(0, 20, 200), pred_mu[0, :, 0], c='C0')
plt.plot(np.linspace(0, 20, 200), pred_mu[0, :, 1], c='C1')

plt.fill_between(np.linspace(0, 20, 200),
                 pred_mu[0, :, 0] - np.sqrt(pred_sigma[0, :, 0]),
                 pred_mu[0, :, 0] + np.sqrt(pred_sigma[0, :, 0]), alpha=0.2)
plt.fill_between(np.linspace(0, 20, 200),
                 pred_mu[0, :, 1] - np.sqrt(pred_sigma[0, :, 1]),
                 pred_mu[0, :, 1] + np.sqrt(pred_sigma[0, :, 1]), alpha=0.2)

plt.plot()
plt.show()