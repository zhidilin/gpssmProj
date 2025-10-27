"""
This file defines the two modules we used:

EnVI_TGP: which is an EnVI algorithm using TGP as the transition model.

Author: Zhidi Lin
Date:   May 2024
"""

import torch
import torch.nn as nn
import gpytorch
import numpy as np
from gpytorch.distributions import MultivariateNormal
from .GPModels import IndependentMultitaskGPModel
from .InferNet import LSTMRecognition
from .NFs import SAL_Flows, Linear_Flow, Tanh_Flows, RealNVP

def KL_divergence(P, Q):
    """
    P: Multivariate
    Q: Multivariate

    return:
        KL( P||Q )
    """
    res = torch.distributions.kl.kl_divergence(P, Q)
    return res

def construct_Gaspari_Cohn(loc_radius, x_dim, device):
    def G(z):
        if 0 <= z < 1:
            return 1. - 5./3*z**2 + 5./8*z**3 + 1./2*z**4 - 1./4*z**5
        elif 1 <= z < 2:
            return 4. - 5.*z + 5./3*z**2 + 5./8*z**3 - 1./2*z**4 + 1./12*z**5 - 2./(3*z)
        else:
            return 0
    taper = torch.zeros(x_dim, x_dim, device=device)
    for i in range(x_dim):
        for j in range(x_dim):
            dist = min(abs(i-j), x_dim - abs(i-j))
            taper[i, j] = G(dist/loc_radius)
    return taper

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
    out = x @ H.transpose(-1, -2)         # shape:  batch_size x N_MC x output_dim

    return out

class EnVI_TGP(nn.Module):
    def __init__(self, dim_x, dim_y, seq_len, ips, dim_c = 0, N_MC = 50, process_noise_sd=0.05, emission_noise_sd=0.1,
                 consistentSampling=True, learn_emission=False, residual_trans=False, H=None):
        """
        Parameters
        ----------
        dim_x:                dimension of state
        dim_y:                dimension of measurement
        seq_len:              sequence length
        ips:                  inducing points:    dim_x x num_ips x (dim_x + dim_c)
        dim_c:                dimension of control input
        N_MC:                 number of ensemble
        process_noise_sd:     transition process noise standard deviation initialization
        emission_noise_sd:    observation noise standard deviation initialization
        consistentSampling:   indication if the sampling is consistent or not
        learn_emission:       if yes, the emission matrix is learnable
        residual_trans:       if yes, x[t] = f(x[t-1]) + x[t-1]
        H:                    emission matrix initialization
        """
        super().__init__()
        self.output_dim = dim_y
        self.input_dim  = dim_c
        self.state_dim = dim_x
        self.seq_len = seq_len
        self.N_MC = N_MC
        self.consistentSampling = consistentSampling
        self.residual_trans = residual_trans

        # emission index: indicates which dimensions of latent state are observable
        # emission matrix: [output_dim x state_dim]
        if H is None:
            indices = [i for i in range(self.output_dim)]
            H = torch.eye(self.state_dim)[indices]
        self.H = nn.Parameter(H, requires_grad=learn_emission)

        # define GP transition
        self.transition = IndependentMultitaskGPModel(inducing_points=ips,  dim_state=self.state_dim)

        # define RealNVP transformation
        nets = lambda: nn.Sequential(nn.Linear(self.state_dim, 32), nn.LeakyReLU(),
                                     nn.Linear(32, 32), nn.LeakyReLU(),
                                     nn.Linear(32, self.state_dim))

        nett = lambda: nn.Sequential(nn.Linear(self.state_dim, 32), nn.LeakyReLU(),
                                     nn.Linear(32, 32), nn.LeakyReLU(),
                                     nn.Linear(32, self.state_dim))

        mask_arr = np.linspace(1, self.state_dim, self.state_dim)
        mask_idx = mask_arr <= int(self.state_dim / 2)
        mask_arr[mask_idx] = 0
        mask_idx = mask_arr > int(self.state_dim / 2)
        mask_arr[mask_idx] = 1
        num_masks = 2
        # masks = torch.tensor([np.stack((mask_arr, 1-mask_arr))] * num_masks, dtype=torch.float32).reshape(-1, self.state_dim)
        base_mask = np.stack((mask_arr, 1 - mask_arr))  # shape: (2, state_dim)
        full_mask = np.tile(base_mask, (num_masks, 1))  # shape: (2 * num_masks, state_dim)
        masks = torch.tensor(full_mask, dtype=torch.float32)
        # masks = torch.tensor(mask_arr, dtype=torch.float32).reshape(-1, state_dim)
        self.flow = RealNVP(nets=nets, nett=nett, mask=masks)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([self.state_dim]))
        self.emission_likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([self.output_dim]))

        self.likelihood.noise =  process_noise_sd ** 2
        self.emission_likelihood.noise = emission_noise_sd ** 2

        # recognition network for inferring the initial state x_0
        self.RecNet = LSTMRecognition(dim_outputs=self.output_dim, dim_inputs=self.input_dim,
                                        dim_states=self.state_dim, length=self.seq_len)

        # setting EnKF
        self.localize_radius = 5


    def forward(self, observations,  input_sequence=None):
        """

        Parameters
        ----------
        observations:           Tensor, observation sequence,         [batch_size x seq_len x output_dim]
        input_sequence:         Tensor, control input sequence        [batch_size x seq_len x input_dim]

        Returns
        -------
        """
        batch_size = observations.shape[0]

        # qm0_KL = torch.tensor([0.], device=observations.device, dtype=observations.dtype)
        # x_t_1 = torch.zeros(batch_size, self.state_dim, self.N_MC, self.state_dim, device=observations.device, dtype=observations.dtype)

        '''---------------  1. Calculate KL divergence of initial state   ---------------'''
        # qx0: shape: batch_size x state_dim
        qx0, qm0_KL = self.x0_KL(observations, input_sequence)

        '''---------------  2. Calculate likelihood   ---------------'''
        x_t_1 = qx0.rsample(torch.Size([self.N_MC, self.state_dim]))    # N_MC x state_dim x batch_size x state_dim

        likelihood, filtered_mean, filtered_var = self.iterate_sequence(observations=observations, x_0=x_t_1,
                                                                        input_sequence=input_sequence)

        '''---------------  3. Calculate the KL divergence term of variational GP  ---------------'''
        gpKL = self.transition.kl_divergence().div(self.seq_len*batch_size)

        ELBO = -qm0_KL - gpKL  + likelihood
        print(f"\n--x0 KL: {qm0_KL.item()} ")
        print(f"--GP KL term: {gpKL.item()} ")
        print(f"--likelihood: {likelihood.item()}")
        print(f"--ELBO: {ELBO.item()}")

        return ELBO, filtered_mean, filtered_var

    def emission(self, x, P):
        """
        emission from state space m & P to observed space mean & sigma
        """
        pred_mean = self.H @ x
        pred_sigma = self.H @ P @ self.H.transpose(0, 1) + torch.diag_embed(self.emission_likelihood.noise.view(-1))
        return pred_mean.squeeze(-1), pred_sigma


    def iterate_sequence(self, observations, x_0, input_sequence=None):

        """
        Parameters
        ----------
        observations:           Tensor, observation sequence,         [batch_size x seq_len x output_dim]
        x_0:                    Tensor, initial state             [N_MC x state_dim x batch_size x state_dim]
        input_sequence:         Tensor, control input sequence        [batch_size x seq_len x input_dim]

        Returns
        -------
        """
        # batch_size = observations.shape[0]
        device = observations.device
        likelihood = torch.tensor(0., device=device)
        filtered_mean = []
        filtered_var = []

        # observations.shape == [ batch_size x seq_len x output_dim ]
        assert (observations.shape[-2] == self.seq_len)

        x_t_1 = x_0  # initial state,  [N_MC x state_dim x batch_size x state_dim]

        ''' #################    sample U, i.e., U ~ q(U)  ################# '''
        U = torch.randn(1)   # useless initialization
        if self.consistentSampling:
            # shape [state_dim x num_ips x (state_dim + input_dim)]
            ips = self.transition.variational_strategy.inducing_points

            # sample U ~ q(U), MultivariateNorm, shape: N_MC x state_dim x num_ips
            U = self.transition(ips).rsample(torch.Size([self.N_MC]))   # shape: N_MC x state_dim x num_ips

        for t in range(self.seq_len):

            """  ------------------------ GP prediction step  ---------------------  """
            if input_sequence is not None:
                gp_input = input_sequence[:, t]
            else:
                gp_input = None

            # x_t shape:    batch_size x N_MC x state_dim
            # X_mean shape: batch_size x 1 x state_dim
            # P shape:      batch_size x state_dim x state_dim
            x_t, X_mean, P = self.TGP_Predict(x_t_1=x_t_1, U=U, input_sequence=gp_input,
                                             localization_radius=self.localize_radius)

            """  ------------------------ EnKF update step  ---------------------  """
            y_t = observations[:, t]      # shape: batch_size x output_dim

            # x_t_post,         shape: batch_size x N_MC x state_dim
            # X_mean_post,      shape: batch_size x 1 x state_dim
            # P_post,           shape: batch_size x state_dim x state_dim
            x_t_post, log_likelihood, X_mean_post, P_post = self.EnKFUpdate(X=x_t, X_mean=X_mean, P=P, y=y_t)

            # update x_t_1 for next round
            x_t_1 = x_t_post.expand(self.state_dim,-1,-1,-1)                # state_dim x batch_size x N_MC x state_dim
            x_t_1 = x_t_1.permute(2, 0, 1, 3)                               # N_MC x state_dim x batch_size x state_dim

            """  ------------------------ result processing/saving step  ---------------------  """
            likelihood = likelihood + log_likelihood#.div(self.seq_len)
            filtered_mean.append(X_mean_post.squeeze(1))                           # shape: batch_size x state_dim
            filtered_var.append(P_post.diagonal(offset=0, dim1=-1, dim2=-2))       # shape: batch_size x state_dim

        filtered_mean = torch.stack(filtered_mean, dim=1)    # shape: batch_size x seq_len x state_dim
        filtered_var = torch.stack(filtered_var, dim=1)      # shape: batch_size x seq_len x state_dim

        return likelihood, filtered_mean, filtered_var

    def TGP_Predict(self, x_t_1, U, input_sequence=None, localization_radius=None, linear_obs=True, var_inflation = 0.0):
        """

        Parameters
        ----------
        x_t_1:              shape: N_MC x state_dim x batch_size x state_dim
        U:                  shape: N_MC x state_dim x num_ips
        input_sequence:     shape: batch_size x input_dim
        localization_radius
        linear_obs
        var_inflation

        Returns
        -------
        X
        X_mean
        P

        """

        '''  ---------------- GP Prediction step  ---------------- '''
        # x_t_1 shape: N_MC x state_dim x batch_size x state_dim
        gp_input = x_t_1  # shape:  N_MC x state_dim x batch_size x state_dim

        if input_sequence is not None:
            # c_t shape:  N_MC x state_dim x batch_size x input_dim
            c_t = input_sequence.expand(self.N_MC, self.state_dim, -1, -1)
            # gp_input shape:  N_MC x state_dim x batch_size x (input_dim + state_dim)
            gp_input = torch.cat((c_t, x_t_1), dim=-1)

        if self.consistentSampling:
            # 👻 VCDT (Ialongo's paper). i.e., sample from U first
            # GPdynamics, MultivariateNormal, shape: N_MC x state_dim x batch_size
            GPdynamics = self.transition.condition_u(x=gp_input, U=U)
        else:
            # 👻 marginalize from U first
            GPdynamics = self.transition(gp_input)       # MultivariateNormal, shape: N_MC x state_dim x batch_size

        # propagate through normalizing flow model
        _x_t = self.flow.g(GPdynamics.mean.transpose(-1,-2)).transpose(-1,-2)  # shape: N_MC x state_dim x batch_size

        if self.residual_trans:
            bias = x_t_1.data[:, 0].transpose(-1,-2)
            x_t = _x_t + bias                    # shape: N_MC x state_dim x batch_size
        else:
            x_t = _x_t

        if linear_obs and localization_radius is not None:
            taper = construct_Gaspari_Cohn(localization_radius, self.state_dim, x_t.device)
        else:
            taper = torch.tensor(1., device=x_t.device)

        '''  ----------------  Post-processing step  ---------------- '''
        X = x_t.permute(2, 0, 1)                 # shape: batch_size x N_MC x state_dim
        X_mean = X.mean(dim=-2).unsqueeze(-2)    # shape: batch_size x 1 x state_dim
        X_center = X - X_mean                    # shape: batch_size x N_MC x state_dim
        if var_inflation is not None:
            X = (1. + var_inflation) * X_center + X_mean

        P = 1 / (self.N_MC - 1) * X_center.transpose(-1, -2) @ X_center  # shape: batch_size x state_dim x state_dim
        P = taper * P
        if var_inflation is not None:
            P = (1. + var_inflation) * P

        # add the likelihood noise
        P = P + torch.diag_embed(self.likelihood.noise.view(-1))         # shape: batch_size x state_dim x state_dim

        return X, X_mean, P

    def EnKFUpdate(self, X, X_mean, P, y=None):

        """
            This implements the ensemble Kalman filter (EnKF). The EnKF uses an ensemble of hundreds to thousands
            of state vectors that are randomly sampled around the estimate, and adds perturbations at each update
            and predict step. It is useful for extremely large systems such as found in hydrophysics.

            It works with both linear and nonlinear systems.

            There are many versions of this sort of this filter. This formulation is due to:
                Matthias Katzfuss, Jonathan R. Stroud, and Christopher K. Wikle.
                "Understanding the ensemble Kalman filter." The American Statistician 70.4 (2016): 350-357.


            Add a new measurement (y) to the kalman filter. If y is None, nothing is changed.

            Parameters
            ----------

                X: Tensor,  shape: batch_size x N_MC x state_dim
                    ensemble obtained from prediction step

                X_mean: Tensor,  shape: batch_size x 1 x state_dim

                P:  Tensor,  shape: batch_size x state_dim x state_dim

                y : Tensor,  shape:  batch_size x output_dim
                    measurement for this update.
        """


        '''  ----------------  Update Step  ---------------- '''
        noise_R = torch.diag_embed(self.emission_likelihood.noise.view(-1))  # shape: output_dim x output_dim

        if y is None:
            X_post = X
            P_post = P
            return X_post, P_post
        else:
            noise_y = torch.randn_like(y)           # shape: batch_size x output_dim
            chol = torch.cholesky(noise_R)          # shape: output_dim x output_dim
            y_perturb = y + noise_y @ chol.t()      # shape: batch_size x output_dim
            y_perturb = y_perturb.unsqueeze(-2)     # shape: batch_size x 1 x output_dim
            # print(f'\n y is continuous: {y_perturb.is_contiguous()}' )


        # transform ensembles into measurement space
        HX = X @ self.H.transpose(-1, -2)                     # shape:  batch_size x N_MC x output_dim

        HP = self.H @ P                                       # shape:  batch_size x output_dim x state_dim

        HPH_T = HP @ self.H.transpose(-1, -2)                 # shape:  batch_size x output_dim x output_dim

        HPH_TR_chol = torch.linalg.cholesky(HPH_T + noise_R)  # shape: batch_size x output_dim x output_dim, lower-tril

        KalmanGain = torch.cholesky_inverse(HPH_TR_chol) @ HP  # shape:  batch_size x output_dim x state_dim

        pre = (y_perturb - HX) @ KalmanGain                    # shape: batch_size x N_MC x state_dim
        # print(f"\n residual is (y_perturb - HX) @ KalmanGain = {pre.mean()}")

        X_post = X + pre   # shape: batch_size x N_MC x state_dim

        '''  -------------- post-processing and compute log-likelihood ---------- '''

        HX_m = X_mean @ self.H.transpose(-1, -2)                            # shape: batch_size x 1 x output_dim

        X_mean_post = X_mean + (y.unsqueeze(-2) - HX_m) @ KalmanGain        # shape: batch_size x 1 x state_dim

        # shape: batch_size x state_dim x state_dim
        P_post = P - HP.transpose(-1, -2)  @  KalmanGain

        # batch_size x output_dim
        d = torch.distributions.MultivariateNormal(HX_m.squeeze(-2),  scale_tril=HPH_TR_chol)

        log_likelihood = d.log_prob(y).mean().div(self.output_dim)   # normalized log-likelihood

        return X_post, log_likelihood, X_mean_post, P_post

    def x0_KL(self, observations, input_sequence=None):
        """ ---------------  1.  KL[ q(x0) || p(x0) ]  --------------- """
        # qx0: shape: batch_size x state_dim
        qx0 = self.RecNet(observations, input_sequence)

        # construct px0
        px0_mean = torch.zeros_like(qx0.mean)                    # shape: batch_size x state_dim
        px0_cov = torch.diag_embed(torch.ones_like(qx0.mean))    # shape: batch_size x state_dim

        px0 = MultivariateNormal(px0_mean, px0_cov)
        qm0_KL = KL_divergence(qx0, px0).mean()                  # take average over batch_size

        return qx0, qm0_KL

    def Forcasting(self, T, x_0, input_sequence=None, observations=None):
        """
        forecast means and sigmas over given time period

        Keyword arguments:
        T -- observed values (int or torch.Tensor)
        x, P - last states before forecasting window

        Returns:
        pred_means, pred_sigmas
        """
        pred_means = torch.tensor([], device=x_0.device, dtype=x_0.dtype)
        pred_sigmas = torch.tensor([], device=x_0.device, dtype=x_0.dtype)
        assert isinstance(T, int)
        assert T > 0

        if observations is not None:
            # observations shape: batch_size x seq_len x output_dim
            T = observations.shape[1]

        ''' #################    sample U, i.e., U ~ q(U)  ################# '''
        U = torch.randn(1)   # useless initialization
        if self.consistentSampling:
            # shape [state_dim x num_ips x (state_dim + input_dim)]
            ips = self.transition.variational_strategy.inducing_points

            # sample U ~ q(U), MultivariateNorm, shape: batch_size x state_dim x num_ips
            U = self.transition(ips).rsample(torch.Size([self.N_MC]))    # shape: N_MC x state_dim x num_ips

        x = x_0   # shape: [N_MC x state_dim x batch_size x state_dim]
        for i in range(T):
            if input_sequence is not None:
                gp_input = input_sequence[:, i]
            else:
                gp_input = None

            # X shape: batch_size x N_MC x state_dim
            # X_mean shape: batch_size x 1 x state_dim
            # P shape: batch_size x state_dim x state_dim
            X, X_mean, P = self.TGP_Predict(x_t_1=x, U=U, input_sequence=gp_input,localization_radius=self.localize_radius)
            X_variance = torch.diagonal(P, offset=0, dim1=-1, dim2=-2) # shape: batch_size x state_dim

            x = X.expand(self.state_dim, -1, -1, -1)   # state_dim x batch_size x N_MC x state_dim
            x = x.permute(2, 0, 1, 3)                  # N_MC x state_dim x batch_size x state_dim

            pred_means = torch.cat([pred_means, X_mean], dim=1)                          # batch_size x T x state_dim
            pred_sigmas = torch.cat([pred_sigmas, X_variance.unsqueeze(1)], dim=1)       # batch_size x T x state_dim

        if observations is not None:
            y_pred, y_pred_sigma = self.emission(pred_means.unsqueeze(-1), torch.diag_embed(pred_sigmas))
            dist = torch.distributions.MultivariateNormal(y_pred, y_pred_sigma)
            NLL = -dist.log_prob(observations).mean()
            NLL /= observations.shape[-1]

            return NLL, pred_means, pred_sigmas, y_pred, y_pred_sigma
        else:
            return pred_means, pred_sigmas