"""
This document is going to define the following model class:
1. vGPSSM, vanilla GPSSM proposed in  [1]
2. PRSSM, Probabilistic recurrent state-space model in [2]
3. ODGPSSM, Output-dependent GPSSM in [3]
4. VCDT, Variationally coupled dynamics and trajectories in [4]

[1] S. Eleftheriadis, et al. "Identification of Gaussian process state space models." NeurIPS'2017.
[2] A. Doerr et al. "Probabilistic recurrent state-space model." ICML'2018
[3] Z. Lin et al. "Output-Dependent Gaussian Process State-Space Model." ICASSP'2022
[4] A. D. Ialongo et al. "Overcoming mean-field approximations in recurrent Gaussian process Models". ICML'2019

"""
import numpy as np
from torch import Tensor
import math
import torch
import torch.nn as nn
import gpytorch
from gpytorch.distributions import MultivariateNormal
from .GPModels import IndependentMultitaskGPModel
from .InferNet import LSTMRecognition, MFInference, Four_Layer_NN

def KL_divergence(P, Q):
    """
    P: Multivariate
    Q: Multivariate
    return:
        KL( P||Q )
    """
    res = torch.distributions.kl.kl_divergence(P, Q)
    return res

def expected_log_prob(target: Tensor, q: MultivariateNormal, Covariance: Tensor) -> Tensor:
    """

    Parameters
    ----------
    target:  observations, Tensor
    q: MultivariateNormal
    Covariance: Tensor

    Returns
    -------

    """
    mean, variance = q.mean, q.variance
    num_event_dim = len(q.event_shape)

    noise = Covariance.diag()

    # Potentially reshape the noise to deal with the multitask case
    noise = noise.view(*noise.shape[:-1], *q.event_shape)

    res = ((target - mean) ** 2 + variance) / noise + noise.log() + math.log(2 * math.pi)
    res = res.mul(-0.5)
    if num_event_dim > 1:  # Do appropriate summation for multitask Gaussian likelihoods
        res = res.sum(list(range(-1, -num_event_dim, -1)))
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

class GPSSMs(nn.Module):
    def __init__(self, dim_x, dim_y, seq_len, ips, dim_c = 0, N_MC = 50, process_noise_sd=0.05,
                 emission_noise_sd=0.1, consistentSampling=True, learn_emission=False, residual_trans=False):
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
        learn_emission:       if yes, the emission matrix is learnable
        residual_trans:       if yes, x[t] = f(x[t-1]) + x[t-1]
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
        indices = [i for i in range(self.output_dim)]
        H = torch.eye(self.state_dim)[indices]
        self.H = nn.Parameter(H, requires_grad=learn_emission)

        # define GP transition
        self.transition = IndependentMultitaskGPModel(inducing_points=ips,  dim_state=self.state_dim)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([self.state_dim]))
        self.emission_likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([self.output_dim]))
        # self.emission_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.output_dim, rank=0,
        #                                                                             has_global_noise=False,
        #                                                                             has_task_noise=True)

        self.likelihood.noise =  process_noise_sd ** 2
        self.emission_likelihood.noise = emission_noise_sd ** 2

        # recognition network for inferring the initial state x_0
        self.RecNet = LSTMRecognition(dim_outputs=self.output_dim, dim_inputs=self.input_dim,
                                        dim_states=self.state_dim, length=self.seq_len)


    def forward(self, observations, input_sequence=None):
        """
        Parameters
        ----------
        observations:           Tensor, observation sequence,         [batch_size x seq_len x output_dim]
        input_sequence:         Tensor, control input sequence        [batch_size x seq_len x input_dim]

        Returns
        -------
        """
        raise NotImplementedError("Not Implemented")

class vGPSSM(GPSSMs):
    """
    Scalable learning using the structured inference network from paper:
    Eleftheriadis, Stefanos, et al.
            "Identification of Gaussian process state space models."
            Advances in neural information processing systems 30 (2017).

    It turns out that the training results easily fall into a bad local optimum,
    and it is quite hard to train the inference network.

    Joint Gaussian distribution for q(x_{0:T})
    """

    def __init__(self, state_dim, output_dim, seq_len, inducing_points, input_dim=0, N_MC = 50,
                 process_noise_sd=0.05, emission_noise_sd=0.1, residual_trans=False):
        super().__init__(dim_x=state_dim,dim_y=output_dim, seq_len=seq_len, ips=inducing_points, dim_c=input_dim,
                         N_MC=N_MC, process_noise_sd=process_noise_sd, emission_noise_sd=emission_noise_sd,
                         learn_emission=False, residual_trans=residual_trans)

        # define inference network for learning the variational distribution of x_{1:T}
        self.inferNet = MFInference(dim_outputs=self.output_dim,
                                    dim_inputs=self.input_dim,
                                    dim_states=self.state_dim,
                                    length=self.seq_len,
                                    hidden_size=64,
                                    num_layers=2,
                                    batch_first=True,
                                    bd=True)

    def forward(self, observations, input_sequence=None):

        device = observations.device
        batch_size = observations.shape[0]

        # observations.shape == [ batch_size x seq_len x output_dim ]
        assert (observations.shape[-2] == self.seq_len)

        ''' ---------------  1.  KL[ q(x0) || p(x0) ]  --------------- '''
        # qx0: shape: batch_size x state_dim
        qx0 = self.RecNet(observations, input_sequence)

        # construct px0
        px0_mean = torch.zeros_like(qx0.mean)  # shape: batch_size x state_dim
        px0_cov = torch.diag_embed(torch.ones_like(qx0.mean))  # shape: batch_size x state_dim

        px0 = MultivariateNormal(px0_mean, px0_cov)
        qm0_KL = KL_divergence(qx0, px0).mean()  # take average over batch_size

        '''############### --------    get A[1:T] and L[1:T]   -------- ############### '''
        ###  At_all shape: [ batch_size x seq_len x dim_state x dim_state ]
        ###  Lt_all shape: [ batch_size x seq_len x dim_state x dim_state ]
        At_all, Lt_all = self.inferNet(output_sequence=observations, input_sequence=input_sequence)

        '''########## --------   calculate H(x_{1:T})    ------------ ###########'''
        const = self.seq_len * self.state_dim
        Hx = 0.5 * const * math.log(2 * math.pi) + torch.logdet(Lt_all).sum(dim=1).mean() + const * 0.5
        Hx = Hx.div(self.seq_len)

        '''####### ------------  求高斯过程动态转移项 和 数据拟合项  ------------ #########'''
        result_gp_dynamic = torch.tensor(0., device=device)
        data_fit = torch.tensor(0., device=device)

        # sampling  x0:   shape: batch_size x dim_state
        xt_previous = qx0.rsample()
        mt_previous = qx0.mean.unsqueeze(dim=-1)  # shape: batch_size x dim_state x 1
        sigma_t_previous = qx0.covariance_matrix  # shape: batch_size x dim_state x dim_state

        for t in range(self.seq_len):
            ''' -----------   get q(xt) = N(xt | At * m_t-1, At * Sigma_t-1 * At.T + Lt*Lt.T)  --------------'''

            # mean of the q(xt)
            mt =  At_all[:, t] @ mt_previous                                          # mt shape: batch_size x dim_state x 1
            _tmp = At_all[:, t] @ sigma_t_previous @ At_all[:, t].transpose(-1,-2)    # _tmp shape: batch_size x dim_state x dim_state
            sigma_t = _tmp + Lt_all[:,t] @ Lt_all[:,t].transpose(-1,-2)
            # e, v = torch.symeig(sigma_t)
            # print(f"iter: {t}, \n eigenvalue: {e}")
            qxt = MultivariateNormal(mt.squeeze(dim=-1), sigma_t).add_jitter() # shape: batch_size x dim_state
            x_t = qxt.rsample()                                                # shape:  batch_size x dim_state

            # compute E_q(xt) [log p(yt | xt) ]
            yt = observations[:, t]                                                     # shape: batch_size x dim_output
            emission_noise = torch.diag_embed(self.emission_likelihood.noise.view(-1))  # shape: output_dim x output_dim

            Cmt = (self.H @ mt).squeeze(dim=-1)                           # shape:  batch_size x output_dim
            CSigmaC = self.H @ sigma_t @ self.H.T                         # shape:  batch_size x output_dim x output_dim

            qyt = MultivariateNormal(Cmt, emission_noise)                         # shape: batch_size x output_dim

            ''' --------------   calculate the expected log-likelihood   ------------------------'''
            # shape: batch_size x output_dim
            trace_term = (1/emission_noise) @ CSigmaC
            data_fit_tmp = qyt.log_prob(yt) - 0.5 * trace_term.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            data_fit = data_fit + data_fit_tmp.mean()#.div(self.seq_len)

            ''' ---------------------------      GP dynamics term          --------------------------- '''
            # x_t_1 shape: batch_size x dim_state.     GP dynamics, shape: dim_state x batch_size
            gp_input = xt_previous                       # shape:  batch_size x dim_state
            if input_sequence is not None:
                c_t = input_sequence[:, t]  # shape: batch_size x dim_input
                gp_input = torch.cat((c_t, xt_previous), dim=-1)  # shape: batch_size x (dim_input + dim_state)

            # GP dynamics, shape: dim_state x batch_size
            gp_dynamics = self.transition(gp_input.expand(self.state_dim, batch_size, (self.state_dim + self.input_dim)))

            if self.residual_trans:
                _target = x_t.T - xt_previous.T   # shape:  dim_state x  batch_size
            else:
                _target = x_t.T                   # shape:  dim_state x  batch_size
            _result_gp_dynamic = self.likelihood.expected_log_prob(target=_target, input=gp_dynamics)
            result_gp_dynamic = result_gp_dynamic + _result_gp_dynamic.mean().div(self.seq_len)

            # update:
            # shape: batch_size x dim_state
            xt_previous = x_t
            mt_previous = mt
            sigma_t_previous = sigma_t

        ''' -------------    KL divergence:  KL[ q(u) || p(u) ]    --------------------'''
        KL_div = self.transition.kl_divergence().div(self.seq_len*batch_size)
        ELBO = -qm0_KL + Hx - KL_div + data_fit + result_gp_dynamic
        print()
        print(f" Entropy term: {Hx.item()}")
        print(f" x0 KL: {qm0_KL.item()} ")
        print(f" KL term: {KL_div.item()} ")
        print(f" data-fit term: {data_fit.item()}")
        print(f" GP dynamic term: {result_gp_dynamic.item()}")
        print(f" ELBO: {ELBO.item()}")
        print()
        return ELBO

    def Prediction(self, observations, input_sequence=None):
        """
         Parameters
         ----------
             observations    :           Tensor, observation sequence,         [batch_size x seq_len x output_dim]
             input_sequence  :           Tensor, control input sequence        [batch_size x seq_len x input_dim]

         Returns
         -------

         """
        #
        device = observations.device
        test_len = observations.shape[1]

        y_pred = []  # for visualizations
        log_ll = torch.tensor(0., device=device)  # initialization

        # observations.shape == [ batch_size x seq_len x output_dim ]
        assert (observations.shape[-2] == self.seq_len)

        # qx0: shape: batch_size x state_dim
        qx0 = self.RecNet(observations, input_sequence)

        ''' -----------------------------  1.  Prediction Step & Update Step   -------------------------- '''
        x_t_1 = qx0.rsample()  # batch_size x state_dim

        for t in range(self.seq_len):
            """  ------------------------ GP prediction step  ---------------------  """
            # x_t_1 shape: batch_size x state_dim.
            gp_input = x_t_1   # shape: batch_size x state_dim
            if input_sequence is not None:
                c_t = input_sequence[:, t]                  # shape: batch_size x dim_input
                gp_input = torch.cat((c_t, x_t_1), dim=-1)  # shape: batch_size x (dim_input + dim_state)

            # MultivariateNormal, shape: state_dim x batch_size
            GPdynamics = self.transition(gp_input.expand(self.state_dim, -1, -1))

            if self.residual_trans:
                gpMean = GPdynamics.mean.transpose(-1,-2) + x_t_1.detach()   # shape: batch_size x state_dim
            else:
                gpMean = GPdynamics.mean.transpose(-1, -2)                   # shape: batch_size x state_dim
            gpVar = GPdynamics.variance.transpose(-1,-2)             # shape: batch_size x state_dim
            gpCov = torch.diag_embed(gpVar,dim1=-2, dim2=-1)         # shape: batch_size x state_dim x state_dim

            # shape: batch_size x state_dim
            p = MultivariateNormal(gpMean, gpCov + torch.diag_embed(self.likelihood.noise.view(-1)))
            # shape: batch_size x output_dim
            pyt = MultivariateNormal(p.mean @ self.H.T, self.H @ p.covariance_matrix @ self.H.T + torch.diag_embed(self.emission_likelihood.noise.view(-1)))

            y_tmp = observations[:, t]
            log_ll = log_ll + pyt.log_prob(y_tmp).mean().div(test_len)  # average over particles and batch

            # Update x[t-1]:  shape: batch_size x state_dim
            x_t_1 = gpMean

            ''' #------------ save prediction, shape: N_MC x batch_size x output_dim  ------------ '''
            y_pred.append(pyt.mean)

        # postprocess the results
        y_pred = torch.stack(y_pred, dim=0)  # shape: seq_len x N_MC x batch_size x output_dim

        return  y_pred, log_ll

class VCDT(GPSSMs):
    """
    Scalable learning using the structured inference network from paper:
    A. D. Ialongo et al. "Overcoming mean-field approximations in recurrent Gaussian process models." ICML'2019.

    Non-mean-field assumption for the latent states, i.e.,

    q(xt | ft) \approx q(xt | u) = N(xt | At * Mu_t + bt, St + At * Sigma_t * At.T)
    where
        [Mu_t, Sigma_t] are the posterior mean and covariance of GP with input, x_{t-1}, and conditioning on u,
        [At, bt, St] are learnable variational parameters.

    In this implementation, [At, bt, St] are learned by an amortized neural network to solve the problem with
    linear growth in the number of parameters.
    """

    def __init__(self, state_dim, output_dim, seq_len, inducing_points, input_dim=0, N_MC = 1,
                 process_noise_sd=0.05, emission_noise_sd=0.1, consistentSampling=True, hidden_dim=None,
                 learn_emission=False, residual_trans=False):
        super().__init__(dim_x=state_dim,dim_y=output_dim, seq_len=seq_len, ips=inducing_points, dim_c=input_dim,
                         N_MC=N_MC, process_noise_sd=process_noise_sd, emission_noise_sd=emission_noise_sd,
                         consistentSampling=consistentSampling, learn_emission=learn_emission,residual_trans=residual_trans)

        if hidden_dim is None:
            hidden_dim = [256, 128, 64]

        # define inference network for learning the variational distribution of q(xt | ft)
        output_dim_infNet = self.state_dim * self.state_dim + self.state_dim + self.state_dim * self.state_dim
        self.inferNet = Four_Layer_NN(input_dim=self.output_dim, output_dim=output_dim_infNet,
                                      hidden_dim=hidden_dim, residual=False)

    def forward(self, observations, input_sequence=None):

        device = observations.device
        batch_size = observations.shape[0]
        likelihood = torch.tensor(0., device=device)
        KL = torch.tensor(0., device=device)

        # observations.shape == [ batch_size x seq_len x output_dim ]
        assert (observations.shape[-2] == self.seq_len)

        ''' ---------------  1.  KL[ q(x0) || p(x0) ]  --------------- '''
        # qx0: shape: batch_size x state_dim
        qx0 = self.RecNet(observations, input_sequence)

        # construct px0
        px0_mean = torch.zeros_like(qx0.mean)  # shape: batch_size x state_dim
        px0_cov = torch.diag_embed(torch.ones_like(qx0.mean))  # shape: batch_size x state_dim

        px0 = MultivariateNormal(px0_mean, px0_cov)
        qm0_KL = KL_divergence(qx0, px0).mean()                # take average over batch_size

        ''' -----------------------------  2.  Prediction Step & Update Step   -------------------------- '''
        x_t_1 = qx0.rsample(torch.Size([self.N_MC, self.state_dim]))  # N_MC x state_dim x batch_size x state_dim

        ''' #################    sample U, i.e., U ~ q(U)  ################# '''
        if self.consistentSampling:
            # shape [state_dim x num_ips x (state_dim + input_dim)]
            ips = self.transition.variational_strategy.inducing_points

            # sample U ~ q(U), MultivariateNorm, shape: N_MC x state_dim x num_ips
            U = self.transition(ips).rsample(torch.Size([self.N_MC]))  # shape: N_MC x state_dim x num_ips

        for t in range(self.seq_len):
            """  ------------------------ GP prediction step  ---------------------  """
            # x_t_1 shape: N_MC x state_dim x batch_size x state_dim.
            gp_input = x_t_1   # shape:  N_MC x state_dim x batch_size x state_dim
            if input_sequence is not None:
                # c_t shape: N_MC x state_dim x batch_size x input_dim
                c_t = input_sequence[:, t].repeat(self.N_MC, self.state_dim, 1, 1)
                # gp_input shape: N_MC x state_dim x batch_size x (input_dim + state_dim)
                gp_input = torch.cat((c_t, x_t_1), dim=-1)

            if self.consistentSampling:
                # 👻 VCDT (Ialongo's paper). i.e., sample from U first
                # GPdynamics, MultivariateNormal, shape: N_MC x state_dim x batch_size
                GPdynamics = self.transition.condition_u(x=gp_input, U=U)
            else:
                # 👻 marginalize from U first
                GPdynamics = self.transition(gp_input)       # MultivariateNormal, shape: N_MC x state_dim x batch_size

            if self.residual_trans:
                gpMean = GPdynamics.mean.transpose(-1,-2) + x_t_1[:, 0].detach()   # shape: N_MC x batch_size x state_dim
            else:
                gpMean = GPdynamics.mean.transpose(-1, -2)                         # shape: N_MC x batch_size x state_dim
            gpVar = GPdynamics.variance.transpose(-1,-2)             # shape: N_MC x batch_size x state_dim
            gpCov = torch.diag_embed(gpVar)                          # shape: N_MC x batch_size x state_dim x state_dim

            '''-------------------------#        obtain the variational parameters      ----------------'''
            parAll = self.inferNet(observations[:, t])       # shape: batch_size x (params_dim)
            # construct variational distribution of q(xt | f)
            At = parAll[:, 0 : self.state_dim*self.state_dim]         # shape: batch_size x (state_dim x state_dim)
            At = At.view(-1, self.state_dim, self.state_dim)          # shape: batch_size x state_dim x state_dim
            bt = parAll[:, self.state_dim*self.state_dim : self.state_dim*(self.state_dim+1)] # shape: batch_size x state_dim
            _St = parAll[:, self.state_dim*(self.state_dim+1): ]      # shape: batch_size x (state_dim x state_dim)
            _St = _St.view(-1, self.state_dim, self.state_dim)        # shape: batch_size x state_dim x state_dim
            lower_mask = torch.ones(_St.shape[-2:], device=device).tril(0).expand(batch_size, self.state_dim, self.state_dim)
            St_lower = _St.mul(lower_mask)                            # shape: batch_size x state_dim x state_dim
            St = St_lower @ St_lower.transpose(-1, -2)                # shape: batch_size x state_dim x state_dim

            '''-------------------------       obtain the variational distribution      ----------------'''
            # shape: N_MC x batch_size x state_dim x 1
            q_mean = At @ gpMean.unsqueeze(dim=-1) + bt.expand(self.N_MC, batch_size, self.state_dim).unsqueeze(dim=-1)
            # shape: N_MC x batch_size x state_dim
            q_mean = q_mean.squeeze(dim=-1)
            # shape: N_MC x batch_size x state_dim x state_dim
            q_cov = St.expand(self.N_MC, -1, -1, -1) + At @  gpCov @ At.transpose(-1,-2)

            q = MultivariateNormal(q_mean, q_cov)  # shape: N_MC x batch_size x state_dim
            p = MultivariateNormal(gpMean, gpCov + torch.diag_embed(self.likelihood.noise.view(-1)))  # shape: N_MC x batch_size x state_dim

            '''--------------  compute the KL divergence: KL[ q(xt | ft) || p(xt | ft) ]   ------------------------- '''
            KL = KL + KL_divergence(q, p).mean()

            '''--------------  compute the expected_log_prob: E_q(xt | ft) [ log p(yt | xt) ]   ------------------- '''
            yt = observations[:, t].expand(self.N_MC, batch_size, self.output_dim) # shape: N_MC x batch_size x output_dim
            qyt = MultivariateNormal(q_mean @ self.H.T, self.H @ q_cov @ self.H.T) # shape: N_MC x batch_size x output_dim

            likelihood = likelihood + self.emission_likelihood.expected_log_prob(target=yt, input=qyt).sum(-1).mean()

            '''--------------  update xt   ------------------- '''
            # shape:  N_MC x state_dim x batch_size x state_dim
            x_t_1 = q.rsample().expand(self.state_dim, -1, -1, -1).transpose(0,1)


        '''---------------  Calculate the KL divergence term of variational GP  ---------------'''
        gpKL = self.transition.kl_divergence().div(self.seq_len*batch_size)

        ELBO = -qm0_KL - gpKL  + likelihood - KL
        if likelihood>KL:
            ELBO = -qm0_KL - gpKL + likelihood.div(self.seq_len) - KL

        print(f"\n--x0 KL: {qm0_KL.item()} ")
        print(f"--GP KL term: {gpKL.item()} ")
        print(f"--KL term: {KL.item()} ")
        print(f"--likelihood: {likelihood.item()}")
        print(f"--ELBO: {ELBO.item()}")

        return  ELBO

    def Prediction(self, observations, input_sequence=None):
        """
         Parameters
         ----------
             observations    :           Tensor, observation sequence,         [batch_size x seq_len x output_dim]
             input_sequence  :           Tensor, control input sequence        [batch_size x seq_len x input_dim]

         Returns
         -------

         """
        #
        batch_size = observations.shape[0]
        dtype = observations.dtype
        device = observations.device
        test_len = observations.shape[1]

        y_pred = []  # for visualizations
        log_ll = torch.tensor(0., device=device)  # initialization

        # observations.shape == [ batch_size x seq_len x output_dim ]
        assert (observations.shape[-2] == self.seq_len)

        # qx0: shape: batch_size x state_dim
        qx0 = self.RecNet(observations, input_sequence)

        ''' -----------------------------  1.  Prediction Step & Update Step   -------------------------- '''
        x_t_1 = qx0.rsample(torch.Size([self.N_MC, self.state_dim]))  # N_MC x state_dim x batch_size x state_dim

        ''' #################    sample U, i.e., U ~ q(U)  ################# '''
        if self.consistentSampling:
            # shape [state_dim x num_ips x (state_dim + input_dim)]
            ips = self.transition.variational_strategy.inducing_points

            # sample U ~ q(U), MultivariateNorm, shape: N_MC x state_dim x num_ips
            U = self.transition(ips).rsample(torch.Size([self.N_MC]))  # shape: N_MC x state_dim x num_ips

        for t in range(self.seq_len):
            """  ------------------------ GP prediction step  ---------------------  """
            # x_t_1 shape: N_MC x state_dim x batch_size x state_dim.
            gp_input = x_t_1   # shape:  N_MC x state_dim x batch_size x state_dim
            if input_sequence is not None:
                # c_t shape: N_MC x state_dim x batch_size x input_dim
                c_t = input_sequence[:, t].repeat(self.N_MC, self.state_dim, 1, 1)
                # gp_input shape: N_MC x state_dim x batch_size x (input_dim + state_dim)
                gp_input = torch.cat((c_t, x_t_1), dim=-1)

            if self.consistentSampling:
                # 👻 VCDT (Ialongo's paper). i.e., sample from U first
                # GPdynamics, MultivariateNormal, shape: N_MC x state_dim x batch_size
                GPdynamics = self.transition.condition_u(x=gp_input, U=U)

            else:
                # 👻 marginalize from U first
                GPdynamics = self.transition(gp_input)       # MultivariateNormal, shape: N_MC x state_dim x batch_size

            if self.residual_trans:
                gpMean = GPdynamics.mean.transpose(-1,-2) + x_t_1[:, 0].detach()   # shape: N_MC x batch_size x state_dim
            else:
                gpMean = GPdynamics.mean.transpose(-1, -2)                         # shape: N_MC x batch_size x state_dim
            gpVar = GPdynamics.variance.transpose(-1,-2)             # shape: N_MC x batch_size x state_dim
            gpCov = torch.diag_embed(gpVar,dim1=-2, dim2=-1)         # shape: N_MC x batch_size x state_dim x state_dim

            # shape: N_MC x batch_size x state_dim
            p = MultivariateNormal(gpMean, gpCov + torch.diag_embed(self.likelihood.noise.view(-1)))
            # shape: N_MC x batch_size x output_dim
            pyt = MultivariateNormal(p.mean @ self.H.T, self.H @ p.covariance_matrix @ self.H.T
                                     + torch.diag_embed(self.emission_likelihood.noise.view(-1)))

            y_tmp = observations[:, t].expand(self.N_MC, batch_size, self.output_dim)
            log_ll = log_ll + pyt.log_prob(y_tmp).mean().div(test_len)  # average over particles and batch

            # Update x[t-1]:  shape: N_MC x state_dim x batch_size x state_dim
            x_t_1 = gpMean.repeat(self.state_dim, 1, 1, 1).transpose(0,1)

            ''' #------------ save prediction, shape: N_MC x batch_size x output_dim  ------------ '''
            y_pred.append(pyt.mean)

        # postprocess the results
        y_pred = torch.stack(y_pred, dim=0)  # shape: seq_len x N_MC x batch_size x output_dim

        return  y_pred, log_ll


