import numpy as np
import torch
from matplotlib import pyplot as plt
import gpytorch

''' ---  some useful functions --- '''
def KS_func(x):
    if (5 > x >= 4) or (x < 3):
        f = x + 1
    elif 4 > x >= 3:
        f = 0
    else:
        f = 16 - 2 * x
    return f

def kink_func(x):
    f = 0.8 + (x + 0.2) * (1 - 5 / (1 + torch.exp(-2 * x)))
    return f

def plot_1D_all(model, epoch, func='kinkfunc',save=False, condition_u=True, path='./fig_MF1D/2layer_learned_kink_Epoch'):
    dtype = model.transition.variational_strategy.inducing_points.dtype
    device = model.transition.variational_strategy.inducing_points.device

    fontsize = 28
    N_test = 100

    if func == 'ksfunc':
        label = "kink-step function"
        X_test = np.linspace(-0.5, 6.5, N_test)
        y_test = np.zeros(N_test)
        for i in range(N_test):
            y_test[i] = KS_func(X_test[i])

    elif func == 'kinkfunc':
        label = "kink function"
        X_test = np.linspace(-3.15, 1.15, N_test)
        y_test = 0.8 + (X_test + 0.2) * (1 - 5 / (1 + np.exp(-2 * X_test)))

    else:
        raise NotImplementedError("The 'func' input only supports kinkfunc and KSfunc")

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_y = torch.tensor(y_test, dtype=torch.float).to(device)
        test_x = torch.tensor(X_test, dtype=torch.float).to(device)
        test_xx = test_x.reshape(-1, 1)  # shape: batch_size x (input_dim + state_dim)
        # test_xx = test_x.reshape(-1,)

        if condition_u:
            # shape [state_dim x num_ips x (state_dim + input_dim)]
            ips = model.transition.variational_strategy.inducing_points

            # shape: state_dim x num_ips
            U1 = model.transition(ips).mean + ips[0,:,:].transpose(-1,-2)
            U = U1.mean(dim=[0])

            # x expected shape: state_dim x batch_size x (input_dim + state_dim)
            test_xx = test_xx.repeat(model.state_dim, 1, 1)

            # MultivariateNormal, shape: state_dim x batch_size
            func_pred = model.transition(test_xx)

            # MultivariateNormal
            observed_pred = model.likelihood(func_pred)

            # shape: state_dim x batch_size (only for state_dim = 1 case)
            assert (model.state_dim == 1)
            pred_val_mean = observed_pred.mean.mean(dim=[0]) + test_xx[0,:,:].transpose(-1,-2).mean(dim=[0])

            # Get upper and lower confidence bounds
            # lower & upper,  shape: state_dim x batch_size
            lower, upper = observed_pred.confidence_region()
            lower, upper = lower.reshape(-1, ) + test_xx[0, :, :].transpose(-1, -2).mean(dim=[0]), \
                           upper.reshape(-1, ) + test_xx[0, :, :].transpose(-1, -2).mean(dim=[0])


        else:
            # x expected shape: state_dim x batch_size x (input_dim + state_dim)
            test_xx = test_xx.repeat(model.state_dim, 1, 1)

            func_pred = model.transition(test_xx)             # shape: state_dim x batch_size
            observed_pred = model.likelihood(func_pred)       # shape: state_dim x batch_size
            pred_val_mean = observed_pred.mean + test_xx[0, :, :].transpose(-1, -2)

            ips = model.transition.variational_strategy.inducing_points
            U = model.transition(ips).mean + ips[0, :, :].transpose(-1, -2)

            # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region()
            lower, upper = lower.reshape(-1, ) + test_xx[0, :, :].transpose(-1, -2).mean(dim=[0]), \
                           upper.reshape(-1, ) + test_xx[0, :, :].transpose(-1, -2).mean(dim=[0])


        # compute prediction MSE
        MSE_preGP = 1 / (N_test) * np.linalg.norm(pred_val_mean.cpu().numpy().reshape(-1) - y_test) ** 2
        print(f"\nMSE_preGP: {MSE_preGP.item()}")

        # compute log-likelihood
        variance = observed_pred.variance
        observed_pred_ = gpytorch.distributions.MultivariateNormal(pred_val_mean, torch.diag_embed(variance))
        LL_preGP =  1 / (N_test) * observed_pred_.log_prob(test_y.view(1, -1))
        print(f"LL_preGP: {LL_preGP.item()}")


    with torch.no_grad():

        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(15, 15))

        # Plot training data as black starss
        ax.plot(ips.cpu().numpy().reshape(-1, ),  U.cpu().numpy().reshape(-1, ), 'g*', label='inducing points', markersize=10)
        # Plot test data as read stars
        ax.plot(X_test, y_test, 'r', label=label)
        # Plot predictive means as blue line
        ax.plot(X_test, pred_val_mean.cpu().numpy().reshape(-1, ), 'b', label='learned function')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy(), label='$\pm 2 \sigma$', alpha=0.5)
        # Shade between the lower and upper confidence bounds
        # ax.fill_between(test_x.cpu().numpy(), lower1.cpu().numpy(), upper1.cpu().numpy(), facecolor = 'c', alpha=0.5, label='$\pm 2 \sigma$')
        ax.legend(loc=0, fontsize=fontsize)
        # plt.title(f"Epoch: {epoch}", fontsize=15)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.grid()
        if func == 'ksfunc':
            ax.set_xlim([-0.5, 6.5])
        elif func == 'kinkfunc':
            ax.set_xlim([-3.15, 1.15])
            ax.set_ylim([-3.15, 1.15])
        else:
            raise NotImplementedError("The 'func' input only supports kinkfunc and KSfunc")

        if save:
            plt.savefig(path + f"EnVI_{func}_epoch_{epoch}.pdf")
        else:
            plt.show()

    return MSE_preGP, LL_preGP


def plot_1D_NN(model, epoch, func='kinkfunc',save=False, path='./fig_MF1D/2layer_learned_kink_Epoch'):

    device = model.likelihood.noise.device

    fontsize = 28
    N_test = 100

    if func == 'ksfunc':
        label = "Kink-step function"
        X_test = np.linspace(-0.5, 6.5, N_test)
        y_test = np.zeros(N_test)
        for i in range(N_test):
            y_test[i] = KS_func(X_test[i])

    elif func == 'kinkfunc':
        label = "Kink function"
        X_test = np.linspace(-3.15, 1.15, N_test)
        y_test = 0.8 + (X_test + 0.2) * (1 - 5 / (1 + np.exp(-2 * X_test)))

    else:
        raise NotImplementedError("The 'func' input only supports kinkfunc and KSfunc")

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_y = torch.tensor(y_test, dtype=torch.float).to(device)
        test_x = torch.tensor(X_test, dtype=torch.float).to(device)
        test_xx = test_x.reshape(-1, 1)  # shape: batch_size x (input_dim + state_dim)
        # test_xx = test_x.reshape(-1,)

        pred_val_mean = model.transition(test_xx)   # shape: batch_size x state_dim
        lower = pred_val_mean - 2. * model.likelihood.noise.sqrt().view(1, -1)
        upper = pred_val_mean + 2. * model.likelihood.noise.sqrt().view(1, -1)

        MSE_preNN = 1 / (N_test) * np.linalg.norm(pred_val_mean.cpu().numpy().reshape(-1) - y_test) ** 2

        # compute log-likelihood
        variance = model.likelihood.noise
        observed_pred_ = gpytorch.distributions.MultivariateNormal(pred_val_mean, torch.diag_embed(variance))

        LL_preNN =  observed_pred_.log_prob(test_y.view(-1, 1)).mean()
        print(LL_preNN.item())


    with torch.no_grad():

        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(15, 10))

        # Plot test data as read stars
        ax.plot(X_test, y_test, 'r', label=label)
        # Plot predictive means as blue line
        ax.plot(X_test, pred_val_mean.cpu().numpy().reshape(-1, ), 'b', label='learned function')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.cpu().numpy(), lower.cpu().numpy().reshape(-1, ), upper.cpu().numpy().reshape(-1, ),
                        alpha=0.5, label='$\pm 2 \sigma$')
        ax.legend(loc=0, fontsize=fontsize)
        # plt.title(f"Epoch: {epoch}", fontsize=15)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.grid()
        if func == 'ksfunc':
            ax.set_xlim([-0.5, 6.5])
        elif func == 'kinkfunc':
            ax.set_xlim([-3.15, 1.15])
            ax.set_ylim([-3.15, 1.15])
        else:
            raise NotImplementedError("The 'func' input only supports kinkfunc and KSfunc")

        if save:
            plt.savefig(path + f"func_{func}_epoch_{epoch}.pdf")
        else:
            plt.show()

    return MSE_preNN, LL_preNN