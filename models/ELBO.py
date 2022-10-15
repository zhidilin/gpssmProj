"""
ELBO construction for variational GPSSM w/wo LMC.

The learning and inference algorithm is:
    from Doerr-ICML-2018/ Ialongo-ICML-2019

Features:
1. q(x_0) is directly parameterized by a LSTM-based recognition network.
2. stochastic gradient descent
3. sampling based methods for learning and inference

Author:
    Zhidi Lin   2022/08/16

"""
import os
import torch
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.constraints import GreaterThan, LessThan
from .GP import MultitaskGPModel
from .RecogModel import LSTMRecognition


class ELBO(nn.Module):
    """
    Doerr et al learning and inference scheme:
        Ref[1]: Doerr et al, Probabilistic Recurrent State-Space Models, ICML 2018

    """
    def __init__(self, state_dim, output_dim, seq_len, inducing_points, input_dim=0,
                 process_noise_sd=0.05,
                 num_particles=50,  # default value = 50
                 ARD=False,
                 LMC=True):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.N_MC = num_particles
        self.LMC = LMC
        self.ARD = ARD
        self.num_latent_gp = inducing_points.shape[0]

        # define GP transition
        self.transition = MultitaskGPModel(inducing_points=inducing_points, num_tasks=self.state_dim,
                                           num_latents=self.num_latent_gp, MoDep=self.LMC, ARD=self.ARD)

        self.proc_noise_model = MultitaskGaussianLikelihood( num_tasks=self.state_dim,  has_global_noise=True, noise_constraint=GreaterThan(1e-6) )
        self.obs_noise_model  = MultitaskGaussianLikelihood( num_tasks=self.output_dim, has_global_noise=True)

        # initialization
        self.proc_noise_model.noise = 2e-3 ** 2
        self.obs_noise_model.noise = 1


        # define recognition network
        self.recognet = LSTMRecognition(dim_outputs=self.output_dim, dim_inputs=self.input_dim,
                                        dim_states=self.state_dim, length=self.seq_len)

    def forward(self, output_y, input_c=None, emi_idx=None):
        """
        output_y: observation sequence with shape [ batch_size x seq_len x output_dim ]
        input_c:  control input with shape [ batch_size x seq_len x input_dim ]
        """
        batch_size = output_y.shape[0]
        dtype = output_y.dtype
        device = output_y.device

        # output_y.shape == [ batch_size x seq_len x output_dim ]
        assert (output_y.shape[-2] == self.seq_len)

        # emission index: indicates which dimensions of latent state are observable
        if emi_idx is None:
            emi_idx = [0]
        assert (len(emi_idx) == self.output_dim)



        ''' -----------------  1.  KL[ q(x0) || p(x0) ]  -----------------'''
        # construct variational distribution for x0:
        x0_mean, x0_cov = self.recognet(output_y, input_c)                  # shape: batch_size x dim_state
        # assumed x0 prior: p(x0) = N(0, I)
        # px0_mean = torch.zeros_like(qx0.mean).to(device)                  # shape: batch_size x dim_state
        # px0_cov = torch.diag_embed(torch.ones_like(qx0.mean)).to(device)  # shape: batch_size x dim_state x dim_state
        # px0 = MultivariateNormal(px0_mean, px0_cov)
        # KL_X01 = torch.distributions.kl.kl_divergence(qx0, px0).mean()     # take average over the batch_size

        KL_X0 = self.kl_mvn(m0=x0_mean, S0=x0_cov,
                            m1=torch.zeros_like(x0_mean).to(device),
                            S1=torch.diag_embed(torch.ones_like(x0_mean)).to(device))



        ''' -----------------  3.  data-fit: E_q(xt) [ log(yt | xt) ]   -----------------'''
        # sampling x_t_1 by re-parameterization
        epsilon = torch.randn(torch.Size([self.N_MC, batch_size, self.state_dim]), device=device)
        # shape: N_MC x batch_size x state_dim
        x_t_1 = x0_mean + epsilon * torch.sqrt(torch.diagonal(x0_cov, dim1=-1, dim2=-2))

        log_ll = torch.tensor(0.).to(device)          # initialization
        for t in range(self.seq_len):

            ''' -------- get x_[t+1] by using variational sparse GP, Eq.(12) in Ref[1]  --------  '''
            gp_input = x_t_1                                   # shape: N_MC x batch_size x state_dim
            if input_c is not None:
                c_t = input_c[:, t].repeat(self.N_MC, 1, 1)    # shape: N_MC x batch_size x input_dim
                gp_input = torch.cat((c_t, x_t_1), dim=-1)     # shape: N_MC x batch_size x (input_dim + state_dim)
            if self.LMC:
                tmp = gp_input.repeat(self.num_latent_gp,1,1,1)    # shape: num_latent_gp x N_MC x batch_size x state_dim
                tmp = tmp.transpose(0,1)                           # shape: N_MC x num_latent_gp x batch_size x state_dim
                qf_t = self.transition(tmp)                        # get function distribution: N_MC x batch_size x state_dim
                qx_t = self.proc_noise_model(qf_t)                 # get state distribution: N_MC x batch_size x state_dim
            else:
                tmp = gp_input.repeat(self.state_dim,1,1,1)        # shape: state_dim x N_MC x batch_size x state_dim
                tmp = tmp.transpose(0,1)                           # shape: N_MC x state_dim x batch_size x state_dim
                qf_t = self.transition(tmp)                        # get function distribution: N_MC x batch_size x state_dim
                qx_t = self.proc_noise_model(qf_t)                 # get state distribution: N_MC x batch_size x state_dim

            x_t = qx_t.rsample()                                   # shape: N_MC x batch_size x state_dim

            # emission model
            yt_mean = x_t[:, :, emi_idx]
            pyt = self.obs_noise_model(yt_mean)                    # shape:  N_MC x batch_size x output_dim

            y_tmp = output_y[:,t].expand(self.N_MC, batch_size, self.output_dim)
            log_ll = log_ll + pyt.log_prob(y_tmp).mean()#.div(self.seq_len)   # average over particles and batch

            # update x_t_1
            x_t_1 = x_t



        ''' -----------------  2.  KL[ q(U) || p(U) ]   -----------------'''
        beta = 1
        KL_GP = self.transition.kl_divergence().div(self.seq_len/beta)
        print(f"\nkl_x0:    {KL_X0}")
        print(f"kl_GP:     {KL_GP}")
        print(f"log_ll:    {log_ll}")

        return -KL_X0 - KL_GP + log_ll


    def prediction(self, output_y, input_c=None, emi_idx=None, plt_result=True, plt_save=True,
                   model_str=None, i_iter=None, result_dir=None):
        """
                output_y: observation sequence with shape [ batch_size x seq_len x output_dim ]
                input_c:  control input with shape [ batch_size x seq_len x input_dim ]
                """
        batch_size = output_y.shape[0]
        dtype = output_y.dtype
        device = output_y.device
        test_len = output_y.shape[1]

        # emission index: indicates which dimensions of latent state are observable
        if emi_idx is None:
            emi_idx = [0]
        assert (len(emi_idx) == self.output_dim)

        ''' -----------------  1.  using recognition network to get latent state 
        can also obtain the first prediction latent state by using the last state got from observable sequence
         -----------------'''
        # construct variational distribution for x0:
        x0_mean, x0_cov = self.recognet(output_y, input_c)  # shape: batch_size x dim_state

        ''' -----------------  3.  data-fit: E_q(xt) [ log(yt | xt) ]   -----------------'''
        # sampling x_t_1 by re-parameterization
        epsilon = torch.randn(torch.Size([self.N_MC, batch_size, self.state_dim]), device=device)
        # shape: N_MC x batch_size x state_dim
        x_t_1 = x0_mean + epsilon * torch.sqrt(torch.diagonal(x0_cov, dim1=-1, dim2=-2))

        log_ll = torch.tensor(0.).to(device)  # initialization
        y_pred = []                           # for visualizations
        for t in range(test_len):

            ''' -------- get x_[t+1] by using variational sparse GP, Eq.(12) in Ref[1]  --------  '''
            gp_input = x_t_1  # shape: N_MC x batch_size x state_dim
            if input_c is not None:
                c_t = input_c[:, t].repeat(self.N_MC, 1, 1)  # shape: N_MC x batch_size x input_dim
                gp_input = torch.cat((c_t, x_t_1), dim=-1)  # shape: N_MC x batch_size x (input_dim + state_dim)
            if self.LMC:
                tmp = gp_input.repeat(self.num_latent_gp, 1, 1, 1)  # shape: num_latent_gp x N_MC x batch_size x state_dim
                tmp = tmp.transpose(0, 1)                           # shape: N_MC x num_latent_gp x batch_size x state_dim
                qf_t = self.transition(tmp)         # get function distribution: N_MC x batch_size x state_dim
                qx_t = self.proc_noise_model(qf_t)  # get state distribution: N_MC x batch_size x state_dim
            else:
                tmp = gp_input.repeat(self.state_dim, 1, 1, 1)  # shape: state_dim x N_MC x batch_size x state_dim
                tmp = tmp.transpose(0, 1)     # shape: N_MC x state_dim x batch_size x state_dim
                qf_t = self.transition(tmp)   # get function distribution: N_MC x batch_size x state_dim
                qx_t = self.proc_noise_model(qf_t)  # get state distribution: N_MC x batch_size x state_dim

            x_t = qx_t.rsample()  # shape: N_MC x batch_size x state_dim

            # emission model
            yt_mean = x_t[:, :, emi_idx]
            pyt = self.obs_noise_model(yt_mean)  # shape:  N_MC x batch_size x output_dim

            y_tmp = output_y[:, t].expand(self.N_MC, batch_size, self.output_dim)
            log_ll = log_ll + pyt.log_prob(y_tmp).mean().div(test_len)   # average over particles and batch

            # update x_t_1
            x_t_1 = x_t

            # save prediction
            # y_pred.append(pyt.sample().view(self.N_MC, batch_size, self.output_dim))
            y_pred.append(yt_mean)

        y_pred = torch.stack(y_pred, dim=0)  # shape: seq_len x N_MC x batch_size x output_dim
        y_pred = y_pred[:, :, 0, :]          # shape: seq_len x N_MC x output_dim
        y_pred_mean = y_pred.mean(dim=1)     # shape: seq_len x output_dim
        y_pred_std =  y_pred.std(dim=1) + torch.sqrt(self.obs_noise_model.task_noises).view(-1, self.output_dim)
        lower, upper = y_pred_mean- 2 * y_pred_std, y_pred_mean + 2*y_pred_std  # shape: seq_len x output_dim
        lower, upper = lower.reshape(-1, ).detach(), upper.reshape(-1, ).detach()

        assert len(output_y[0,:,:].shape)==2
        assert output_y[0,:,:].shape[-1] == y_pred_mean.shape[-1]
        MSE = 1/test_len * torch.norm(output_y[0,:,:] - y_pred_mean) ** 2
        RMSE = MSE.sqrt()

        if plt_save:
            plt_result=False

        # fig, axs = plt.subplots(1, self.output_dim, figsize=(4 * self.output_dim, 3))
        # for task, ax in enumerate(axs.flatten()):
        #     # Plot training data as black stars
        #     ax.plot(np.linspace(1, test_len, test_len), output_y.squeeze()[:, task].detach().cpu().numpy(), 'k*')
        #     # Predictive mean as blue line
        #     ax.plot(np.linspace(1, test_len, test_len), y_pred_mean[:, task].cpu().numpy(), 'b')
        #     # Shade in confidence
        #     ax.fill_between(np.linspace(1, test_len, test_len),
        #                     lower[:, task].detach().cpu().numpy(),
        #                     upper[:, task].detach().cpu().numpy(), alpha=0.5)
        #     # ax.set_ylim([-3, 3])
        #     ax.legend(['Observed Data', 'Mean', 'Confidence'])
        #     ax.set_title(f'Task {task + 1}')
        # fig.tight_layout()
        # if plt_result:
        #     plt.show()
        # if plt_save:
        #     save_dir = result_dir + f"figs/"
        #     if not os.path.exists(save_dir):
        #          os.makedirs(save_dir)
        #     plt.savefig(save_dir + f"{model_str}_iter_{i_iter}.png")
        #     plt.close()

        f, ax = plt.subplots(1, 1)
        plt.plot(range(test_len), output_y[0, :].cpu().numpy(), 'k-', label='true observations')
        plt.plot(range(test_len), y_pred_mean.detach().cpu().numpy(), 'b-', label='predicted observations')
        ax.fill_between(range(test_len), lower.cpu().numpy(), upper.cpu().numpy(), color="b", alpha=0.2, label='95% CI')
        ax.legend(loc=0)  # , fontsize=28)
        if plt_result:
            plt.show()
        if plt_save:
            save_dir = result_dir + f"figs/"
            if not os.path.exists(save_dir):
                 os.makedirs(save_dir)
            plt.savefig(save_dir + f"{model_str}_iter_{i_iter}.png")
            plt.close()

        return   RMSE, log_ll



    def kl_mvn(self, m0, S0, m1, S1):
        """
        https://stackoverflow.com/questions/44549369/kullback-leibler-divergence-from-gaussian-pm-pv-to-gaussian-qm-qv
        The following function computes the KL-Divergence between any two
        multivariate normal distributions
        (no need for the covariance matrices to be diagonal)
        Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
        Also computes KL divergence from a single Gaussian pm,pv to a set
        of Gaussians qm,qv.
        Diagonal covariances are assumed.  Divergence is expressed in nats.
        - accepts stacks of means, but only one S0 and S1
        From wikipedia
        KL( (m0, S0) || (m1, S1))
             = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| +
                      (m1 - m0)^T S1^{-1} (m1 - m0) - N )
        # 'diagonal' is [1, 2, 3, 4]
        tf.diag(diagonal) ==> [[1, 0, 0, 0]
                              [0, 2, 0, 0]
                              [0, 0, 3, 0]
                              [0, 0, 0, 4]]
        # See wikipedia on KL divergence special case.
        #KL = 0.5 * tf.reduce_sum(1 + t_log_var - K.square(t_mean) - K.exp(t_log_var), axis=1)
                    if METHOD['name'] == 'kl_pen':
                    self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                    kl = tf.distributions.kl_divergence(oldpi, pi)
                    self.kl_mean = tf.reduce_mean(kl)
                    self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
        """
        # store inv diag covariance of S1 and diff between means
        N = m0.shape[1]
        iS1 = torch.inverse(S1)
        diff = (m1 - m0).unsqueeze(-1)


        # kl is made of three terms
        tmp = iS1 @ S0
        tr_term = torch.diagonal(tmp, dim1=-1, dim2=-2).sum(dim=-1)
        det_term = torch.log(torch.linalg.det(S1) / torch.linalg.det(S0))
        quad_term = diff.transpose(-1,-2) @ iS1 @ diff
        # print(tr_term,det_term,quad_term)

        KL_batch = .5 * (tr_term + det_term + quad_term - N)
        return KL_batch.mean()
