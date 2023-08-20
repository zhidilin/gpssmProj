"""
Contained within this file are the variational learning and inference techniques tailored for GPSSMs.
The following methodologies are encompassed:

1. PRSSM: An output-independent GPSSM algorithm introduced in [Doerr et al, ICML'2018]
2. ODGPSSM: An output-dependent GPSSM algorithm presented in [Lin et al, ICASSP'2023]

Key Features:
1. Direct parameterization of q(x_0) facilitated through an LSTM-based recognition network.
2. Utilization of stochastic gradient descent.

Author:
    Zhidi Lin
"""
import os
import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import gpytorch
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal
from matplotlib import pyplot as plt
from .GP import IndependentMultitaskGPModel, MultitaskGPModel, SparseGPModel
from .RecogModel import LSTMRecognition
from .utils import dtype, device

class GPSSMs(nn.Module):
    def __init__(self, dim_x, dim_y, seq_len, ips, dim_c = 0, N_MC = 50, process_noise_sd=0.05,
                 emission_noise_sd=0.1, consistentSampling=True):
        """
        Parameters
        ----------
        dim_x:                dimension of state
        dim_y:                dimension of measurement
        seq_len:              sequence length
        ips:                  inducing points:    dim_x x num_ips x (dim_x + dim_c)
        dim_c:                dimension of control input
        N_MC:                 number of particles for variational inference
        process_noise_sd:     transition process noise standard deviation initialization
        emission_noise_sd:    observation noise standard deviation initialization
        consistentSampling:   indication if the sampling is consistent or not
        """
        super().__init__()
        self.output_dim = dim_y
        self.input_dim  = dim_c
        self.state_dim = dim_x
        self.seq_len = seq_len
        self.N_MC = N_MC
        self.consistentSampling = consistentSampling

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([self.state_dim]))
        self.emission_likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([self.output_dim]))

        self.likelihood.noise =  process_noise_sd ** 2
        self.emission_likelihood.noises = emission_noise_sd ** 2

        # recognition network for inferring the initial state x_0
        self.RecNet = LSTMRecognition(dim_outputs=self.output_dim, dim_inputs=self.input_dim,
                                        dim_states=self.state_dim, length=self.seq_len)

    def forward(self, observations, H=None, input_sequence=None):
        """
        Parameters
        ----------
        observations:           Tensor, observation sequence,         [batch_size x seq_len x output_dim]
        H           :           Tensor, emission coefficient matrix,  [output_dim x state_dim]
        input_sequence:         Tensor, control input sequence        [batch_size x seq_len x input_dim]

        Returns
        -------
        """
        raise NotImplementedError("Not Implemented")


class ODGPSSM(GPSSMs):
    """
    Reference:
        Z. Lin et al, output-dependent Gaussian process state-space model, ICASSP 2023
            corresponding to setting:  self. LMC = True,

        A. Doerr et al, Probabilistic recurrent state-space model, ICML 2018
            corresponding to setting:  self. LMC = False,
    """
    def __init__(self, state_dim, output_dim, seq_len, inducing_points, input_dim=0, N_MC = 50,
                 process_noise_sd=0.05, emission_noise_sd=0.1, ARD=False, LMC=True):
        super().__init__(dim_x=state_dim, dim_y=output_dim, seq_len=seq_len, ips=inducing_points, dim_c=input_dim,
                         N_MC=N_MC, process_noise_sd=process_noise_sd, emission_noise_sd=emission_noise_sd)

        self.LMC = LMC
        self.ARD = ARD
        self.num_latent_gp = inducing_points.shape[0]

        # Redefine GP transition
        self.transition = MultitaskGPModel(inducing_points=inducing_points, num_tasks=self.state_dim,
                                           num_latents=self.num_latent_gp, MoDep=self.LMC, ARD=self.ARD)

        # Redefine noise models
        self.likelihood = MultitaskGaussianLikelihood( num_tasks=self.state_dim,
                                                       has_global_noise=True,
                                                       noise_constraint=GreaterThan(1e-6) )
        self.emission_likelihood = MultitaskGaussianLikelihood( num_tasks=self.output_dim, has_global_noise=True)

        # noise initialization
        self.likelihood.noise = process_noise_sd ** 2
        self.emission_likelihood.noise = emission_noise_sd ** 2

    def forward(self, observations, H=None, input_sequence=None):
        """
        observations:           Tensor, observation sequence,         [batch_size x seq_len x output_dim]
        H           :           Tensor, emission coefficient matrix,  [output_dim x state_dim]
        input_sequence:         Tensor, control input sequence        [batch_size x seq_len x input_dim]
        """
        batch_size = observations.shape[0]
        dtype = observations.dtype
        device = observations.device

        # observations.shape == [ batch_size x seq_len x output_dim ]
        assert (observations.shape[-2] == self.seq_len)

        # emission index: indicates which dimensions of latent state are observable
        if H is None:
            # emission matrix
            indices = [i for i in range(self.output_dim)]
            H = torch.eye(self.state_dim, device=device, dtype=observations.dtype)[indices]

        assert (H.shape[0] == self.output_dim)
        assert (H.shape[1] == self.state_dim)

        ''' ---------------  1.  KL[ q(x0) || p(x0) ]  --------------- '''
        # qx0: shape: batch_size x state_dim
        qx0 = self.RecNet(observations, input_sequence)

        # construct px0
        px0_mean = torch.zeros_like(qx0.mean)                  # shape: batch_size x state_dim
        px0_cov = torch.diag_embed(torch.ones_like(qx0.mean))  # shape: batch_size x state_dim x state_dim
        px0 = MultivariateNormal(px0_mean, px0_cov)
        KL_X0 = KL_divergence(qx0, px0).mean()                 # take average over batch_size

        ''' -----------------  2.  data-fit: E_q(xt) [ log(yt | xt) ]   -----------------'''
        # sampling x_t_1 by re-parameterization,  x_t_1 shape: N_MC x batch_size x state_dim
        epsilon = torch.randn(torch.Size([self.N_MC, batch_size, self.state_dim]), device=device)
        x_t_1 = qx0.mean + epsilon * torch.sqrt(qx0.variance)

        log_ll = torch.tensor(0., device=device)             # initialization
        for t in range(self.seq_len):
            ''' -------- get x_[t+1] by using variational sparse GP, Eq.(12) in Ref  --------  '''
            gp_input = x_t_1.mean(dim=0).expand(self.N_MC, batch_size, self.state_dim) # shape: N_MC x batch_size x state_dim
            if input_sequence is not None:
                c_t = input_sequence[:, t].repeat(self.N_MC, 1, 1)  # shape: N_MC x batch_size x input_dim
                gp_input = torch.cat((c_t, x_t_1), dim=-1)          # shape: N_MC x batch_size x (input_dim + state_dim)

            if self.LMC:
                tmp = gp_input.repeat(self.num_latent_gp,1,1,1)   # shape: num_latent_gp x N_MC x batch_size x (input_dim + state_dim)
                tmp = tmp.transpose(0,1)                          # shape: N_MC x num_latent_gp x batch_size x (input_dim + state_dim)
                qf_t = self.transition(tmp)                       # function distribution: N_MC x batch_size x state_dim
                qx_t = self.likelihood(qf_t)                      # state distribution: N_MC x batch_size x state_dim
            else:
                tmp = gp_input.repeat(self.state_dim,1,1,1)       # shape: state_dim x N_MC x batch_size x state_dim
                tmp = tmp.transpose(0,1)                          # shape: N_MC x state_dim x batch_size x state_dim
                qf_t = self.transition(tmp)                       # function distribution: N_MC x batch_size x state_dim
                qx_t = self.likelihood(qf_t)                      # state distribution: N_MC x batch_size x state_dim

            x_t = qx_t.rsample() + x_t_1                          # shape: N_MC x batch_size x state_dim

            # emission model
            yt_mean = Linear(x_t, H)                            # shape: N_MC x batch_size x output_dim
            pyt = self.emission_likelihood(yt_mean)             # shape: N_MC x batch_size x output_dim

            y_tmp = observations[:,t].expand(self.N_MC, batch_size, self.output_dim)
            log_ll = log_ll + pyt.log_prob(y_tmp).mean()#.div(self.seq_len)   # average over particles and batch

            # update x_t_1
            x_t_1 = x_t.mean(dim=0).expand(self.N_MC, batch_size, self.state_dim)

        ''' -----------------  3.  KL[ q(U) || p(U) ]   -----------------'''
        beta = 1
        KL_GP = self.transition.kl_divergence().div(self.seq_len/beta)
        ELBO = -KL_X0 - KL_GP + log_ll
        print(f"\n kl_x0: {KL_X0}")
        print(f" kl_GP: {KL_GP}")
        print(f" log_ll: {log_ll}")
        print(f" ELBO: {ELBO}")
        return ELBO


    def prediction(self, observations, H=None, input_sequence=None):
        """
                observations: observation sequence with shape [ batch_size x seq_len x output_dim ]
                input_sequence:  control input with shape [ batch_size x seq_len x input_dim ]
        """
        #
        batch_size = observations.shape[0]
        dtype = observations.dtype
        device = observations.device
        test_len = observations.shape[1]

        # emission index: indicates which dimensions of latent state are observable
        if H is None:
            # emission matrix
            indices = [i for i in range(self.output_dim)]
            H = torch.eye(self.state_dim, device=device, dtype=observations.dtype)[indices]

        assert (H.shape[0] == self.output_dim)
        assert (H.shape[1] == self.state_dim)

        ''' -----------------  
            1.  using recognition network to get latent state; 
                can also obtain the first prediction latent state by using the last state got from observable sequence
         -----------------'''
        # construct variational distribution for x0:
        qx0 = self.RecNet(observations, input_sequence)  # shape: batch_size x dim_state
        x0_mean, x0_cov = qx0.mean, qx0.covariance_matrix

        ''' -----------------  2.  data-fit: E_q(xt) [ log(yt | xt) ]   -----------------'''
        # sampling x_t_1 by re-parameterization,   shape: N_MC x batch_size x state_dim
        epsilon = torch.randn(torch.Size([self.N_MC, batch_size, self.state_dim]), device=device)
        x_t_1 = x0_mean + epsilon * torch.sqrt(torch.diagonal(x0_cov, dim1=-1, dim2=-2))


        log_ll = torch.tensor(0., device=device)   # initialization
        y_pred = []                                # for visualizations
        for t in range(test_len):
            ''' -------- get x_[t+1] by using variational sparse GP, Eq.(12) in Ref  --------  '''
            gp_input = x_t_1.mean(dim=0).expand(self.N_MC, batch_size, self.state_dim)  # shape: N_MC x batch_size x state_dim
            if input_sequence is not None:
                c_t = input_sequence[:, t].repeat(self.N_MC, 1, 1)  # shape: N_MC x batch_size x input_dim
                gp_input = torch.cat((c_t, x_t_1), dim=-1)          # shape: N_MC x batch_size x (input_dim + state_dim)

            if self.LMC:
                tmp = gp_input.repeat(self.num_latent_gp, 1, 1, 1)  # shape: num_latent_gp x N_MC x batch_size x state_dim
                tmp = tmp.transpose(0, 1)                           # shape: N_MC x num_latent_gp x batch_size x state_dim
                qf_t = self.transition(tmp)                      # function distribution: N_MC x batch_size x state_dim
                qx_t = self.likelihood(qf_t)                     # state distribution: N_MC x batch_size x state_dim
            else:
                tmp = gp_input.repeat(self.state_dim, 1, 1, 1)  # shape: state_dim x N_MC x batch_size x state_dim
                tmp = tmp.transpose(0, 1)                       # shape: N_MC x state_dim x batch_size x state_dim
                qf_t = self.transition(tmp)                     # function distribution: N_MC x batch_size x state_dim
                qx_t = self.likelihood(qf_t)                    # state distribution: N_MC x batch_size x state_dim

            x_t = qx_t.rsample() + x_t_1                       # shape: N_MC x batch_size x state_dim

            # emission model
            yt_mean = Linear(x_t, H)                    # shape:  N_MC x batch_size x output_dim
            pyt = self.emission_likelihood(yt_mean)     # shape:  N_MC x batch_size x output_dim

            y_tmp = observations[:, t].expand(self.N_MC, batch_size, self.output_dim)
            log_ll = log_ll + pyt.log_prob(y_tmp).mean().div(test_len)   # average over particles and batch

            # update x_t_1
            x_t_1 = x_t.mean(dim=0).expand(self.N_MC, batch_size, self.state_dim)

            # save prediction
            y_pred.append(yt_mean)

        # postprocess the results
        y_pred = torch.stack(y_pred, dim=0)       # shape: seq_len x N_MC x batch_size x output_dim
        return   y_pred, log_ll

def KL_divergence(P, Q):
    """
    P: Multivariate
    Q: Multivariate
    return:
        KL( P||Q )
    """
    res = torch.distributions.kl.kl_divergence(P, Q)
    return res

def Linear(x, H):
    """
    Parameters
    ----------

        x: Tensor, latent state.                           batch_size x N_MC x state_dim
        H: Tensor, linear emission coefficient matrix.     output_dim x state_dim

    Returns
    -------
        output: Tensor,   output_dim x batch_size

    """
    out = x @ H.transpose(-1, -2).contiguous()  # shape:  batch_size x N_MC x output_dim
    return out