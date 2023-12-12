import numpy as np
import numpy.random as npr
import torch
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
from models.EnVI import OnlineEnVI
from tqdm import tqdm
from utils import settings as cg
randseed = 523
cg.reset_seed(randseed)                      # setting random seed
device = cg.device                           # setting device
dtype = torch.float                          # setting dtype
plt.rcParams["figure.figsize"] = (20,15)


# The function below implements the RSLDS.
def rslds(x, A, B, R, r, K, dx=False):
    v = np.zeros(K - 1)
    p = np.zeros(K)

    for k in range(K - 1):
        v[k] = R[:, k] @ x + r[k]

    "Compute weighted LDS"
    fx = 0
    for k in range(K):
        w = 1
        # Compute weight
        for j in range(k):
            w *= sigmoid(-v[j])
        if k != K - 1:
            w *= sigmoid(v[k])

        fx += w * (A[:, :, k] @ x + B[:, k])

    if dx:
        fx = fx - x

    return fx


# Settings
d_latent = 2  # latent space dimensionality
d_obs = 10  # observation dimenionality
vary = 0.01  # variance of observation noise
varx = 1e-3  # variance of state noise

T1 = 2000  # length of training set
T2 = 500  # length of forecasting

# We generate NASCAR the trajectory using the recurrent switching linear dynamical system (RSLDS). Let's load the parameters and draw the velocity field.
rslds_data = np.load("svmc-main/notebooks/rslds_nascar.npy", allow_pickle=True)[()]
K = 4
dim = d_latent

Atrue = rslds_data['A']
btrue = rslds_data['b']
Rtrue = np.zeros((dim, K - 1))
Rtrue[0, 0] = 100
Rtrue[0, 1] = -100
Rtrue[1, 2] = 100
rtrue = -200 * np.ones(K - 1)
rtrue[-1] = 0

# Then we can generate the trajectory
np.random.seed(randseed)
T = T1 + T2
x = np.random.randn(T + 1, d_latent) * 10

for t in range(T):
    x[t + 1, :] = rslds(x[t, :], A=Atrue, B=btrue, R=Rtrue, r=rtrue, K=K) + np.sqrt(varx) * np.random.randn(d_latent)

x = x[1:, :].T
plt.figure()
plt.plot(*x)
plt.show()
plt.close()

# and the observation
C = npr.rand(d_obs, d_latent + 1) - 0.5  # parameters for emission (C = [C, D] for C*x + D
C[:, -1] = np.zeros(d_obs)
y = C[:, :-1] @ x + C[:, -1][:, None] + np.sqrt(vary) * npr.randn(d_obs, T)
Cobs = C[:, :-1]
dobs = C[:, -1]



""" ----------------   model preparation ----------------  """
fixEmission = True
fixTransition = False
m_inducing = 20  # number of GP inducing points
ips = 10 * torch.randn((d_latent, m_inducing, d_latent), dtype=dtype)
model = OnlineEnVI(dim_x=d_latent, dim_y=d_obs, ips=ips, dim_c=0, N_MC=50, process_noise_sd=1,
                   emission_noise_sd=np.sqrt(vary), consistentSampling=False, learn_emission=False, residual_trans=False,
                   H=torch.tensor(Cobs, dtype=dtype)).to(device)
if fixEmission:
    model.emission_likelihood.requires_grad_(False)
if fixTransition:
    model.likelihood.requires_grad_(False)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

initial_state = torch.tensor(x[:,0], dtype=dtype).expand(model.N_MC, model.state_dim, 1, model.state_dim)
maxEpoch=25
for epoch in range(maxEpoch):
    dataIter = tqdm(range(0, T1), desc='Data index')
    # one-step prediction (with filtering at each step
    x_t_1 = initial_state

    filter_x_all = []
    filter_x_var_all = []
    for i in dataIter:
        optimizer.zero_grad()
        elbo, x_t, filtered_mean, filtered_var = model(x_t_1=x_t_1,  y_t=torch.tensor(y[:, i][None, None],
                                                                                      device=device,
                                                                                      dtype=dtype),
                                                       )
                                # verbose=True, beta=1e3)
        filter_x_all.append(filtered_mean)
        filter_x_var_all.append(filtered_var)

        loss = -elbo
        loss.backward()
        optimizer.step()
        x_t_1 = x_t.detach()
        dataIter.set_postfix({'loss': '{0:1.5f}'.format(loss.item())})
        # print(prof.key_averages().table(sort_by="self_cpu_time_total"))

    # # Plot the inferred trajectory and velocity field.
    x_est = torch.cat(filter_x_all, dim=1).detach().cpu().numpy()
    x_est = x_est.squeeze().T             # shape: 2 x T1
    x_est_var = torch.cat(filter_x_var_all, dim=1).detach().cpu().numpy()
    x_est_var = x_est_var.squeeze().T    # shape: 2 x T1

    if epoch==(maxEpoch-1):
        import os
        DIR = f'results/NASCAR/OEnVI/'
        if not os.path.exists(DIR):
            os.makedirs(DIR)

        fig = plt.figure(figsize=(8,6))
        ax1 = fig.add_subplot(111)
        ax1.plot(*x[:, 1:T1], color='tab:blue', label='True')
        ax1.plot(*x_est, color='tab:red', label='Inferred')
        plt.legend()
        ax1.set_xlim(-10.1, 10.1)
        ax1.set_ylim(-8.1, 8.1)
        plt.savefig(DIR + f"filtered_performance_epoch{epoch}.pdf")
        fig.show()


        # make prediction
        # pred_x,          shape: batch_size x seq_len x state_dim
        # pred_x_var,      shape: batch_size x seq_len x state_dim
        x_test_ini = filtered_mean[:, -1].expand(model.N_MC, model.state_dim, -1, -1)
        pred_x, pred_x_var = model.Forcasting(T=T2, x_0=x_test_ini)
        x_pred = pred_x.squeeze().detach().cpu().numpy()

        # Plot over time
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(211)
        ax.plot(np.arange(T1 - 500, T), x[0, -500 - T2:], color='blue')

        ax.plot(np.arange(T1 - 500, T1), x_est[0, -500:], color='red', linestyle=':')
        # plt.fill_between(np.arange(T1 - 500, T1),
        #                  x_est[0, -500:] - np.sqrt(x_est_var[0, -500:]),
        #                  x_est[0, -500:] + np.sqrt(x_est_var[0, -500:]), alpha=0.2)

        ax.plot(np.arange(T1, T), x_pred[:, 0], color='red', linestyle='dashdot')

        ax.set_ylabel("x1")
        ax.axvline(x=T1, ymin=-25, ymax=25, color="grey")
        ax = fig.add_subplot(212)
        ax.plot(np.arange(T1 - 500, T), x[1, -500 - T2:], color='blue', label="true")
        ax.plot(np.arange(T1 - 500, T1), x_est[1, -500:], color='red', linestyle=':', label="filtered")
        # plt.fill_between(np.arange(T1 - 500, T1),
        #                  x_est[1, -500:] - np.sqrt(x_est_var[1, -500:]),
        #                  x_est[1, -500:] + np.sqrt(x_est_var[1, -500:]), alpha=0.2, label="uncertainty")

        ax.plot(np.arange(T1, T), x_pred[:, 1], color='red', linestyle='dashdot', label="predicted")
        ax.set_xlabel("t")
        ax.set_ylabel("x2")
        ax.axvline(x=T1, ymin=-25, ymax=25, color="grey")
        ax.legend(loc="upper left")
        plt.savefig(DIR + f"prediction_performance_epoch{epoch}.pdf")
        fig.show()

        RMSE = np.sqrt(np.mean(np.sum((x[:, -500:].T - x_pred)**2, axis=1)))
        print(f"Iteration: {epoch}, RMSE: {RMSE}")

        cg.save_models(model, optimizer, epoch, 'empty', DIR, 'NASCAR', '_final', save_model=True)


