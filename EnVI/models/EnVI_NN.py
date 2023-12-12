import torch
import torch.nn as nn
import gpytorch
from gpytorch.distributions import MultivariateNormal
from .InferNet import LSTMRecognition,LSTMencoder, Four_Layer_NN, PostNet

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
    out = x @ H.transpose(-1, -2).contiguous()  # shape:  batch_size x N_MC x output_dim

    return out


class AutoEnKF(nn.Module):
    def __init__(self, dim_x, dim_y, seq_len, dim_c = 0, N_MC = 50, process_noise_sd=0.05, emission_noise_sd=0.1,
                 learn_emission=False, residual_trans=False, hidden_dim=None):
        """
        Parameters
        ----------
        dim_x:                dimension of state
        dim_y:                dimension of measurement
        seq_len:              sequence length
        dim_c:                dimension of control input
        N_MC:                 number of ensemble
        process_noise_sd:     transition process noise standard deviation initialization
        emission_noise_sd:    observation noise standard deviation initialization
        learn_emission:       if yes, the emission matrix is learnable
        residual_trans:       if yes, x[t] = f(x[t-1]) + x[t-1]
        hidden_dim:           the hidden dimension specification
        """
        super().__init__()
        self.output_dim = dim_y
        self.input_dim  = dim_c
        self.state_dim = dim_x
        self.seq_len = seq_len
        self.N_MC = N_MC
        self.residual_trans = residual_trans
        self.variance_output = False
        if hidden_dim is None:
            hidden_dim = [128, 64, 32]
        self.hidden_dim = hidden_dim

        # emission index: indicates which dimensions of latent state are observable
        # emission matrix: [output_dim x state_dim]
        indices = [i for i in range(self.output_dim)]
        H = torch.eye(self.state_dim)[indices]
        self.H = nn.Parameter(H, requires_grad=learn_emission)

        # define GP transition
        self.transition = Four_Layer_NN(input_dim=dim_x+dim_c, output_dim=dim_x, hidden_dim=hidden_dim, residual=False, activation='relu')

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

        # qm0_KL = torch.tensor([0.], device=observations.device, dtype=observations.dtype)
        # x_t_1 = torch.zeros(batch_size, self.state_dim, self.N_MC, self.state_dim, device=observations.device, dtype=observations.dtype)

        '''---------------  1. Calculate KL divergence of initial state   ---------------'''
        # qx0: shape: batch_size x state_dim
        qx0, qm0_KL = self.x0_KL(observations, input_sequence)

        '''---------------  2. Calculate likelihood   ---------------'''
        x_t_1 = qx0.rsample(torch.Size([self.N_MC]))    # N_MC x batch_size x state_dim

        likelihood, filtered_mean, filtered_var = self.iterate_sequence(observations=observations, x_0=x_t_1,
                                                                        input_sequence=input_sequence)

        ELBO = -qm0_KL  + likelihood
        print(f"\n--x0 KL: {qm0_KL.item()} ")
        print(f"--likelihood: {likelihood.item()}")
        print(f"--ELBO: {ELBO.item()}")

        return ELBO, filtered_mean, filtered_var

    def iterate_sequence(self, observations, x_0, input_sequence=None):

        """
        Parameters
        ----------
        observations:           Tensor, observation sequence,         [batch_size x seq_len x output_dim]
        x_0:                    Tensor, initial state                 [N_MC x batch_size x state_dim]
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

        x_t_1 = x_0  # initial state,  [N_MC x batch_size x state_dim]

        for t in range(self.seq_len):

            """  ------------------------ prediction step  ---------------------  """
            if input_sequence is not None:
                nn_input = input_sequence[:, t]
            else:
                nn_input = None

            # x_t shape:    batch_size x N_MC x state_dim
            # X_mean shape: batch_size x 1 x state_dim
            # P shape:      batch_size x state_dim x state_dim
            x_t, X_mean, P = self.Prediction(x_t_1=x_t_1, input_sequence=nn_input,
                                             localization_radius=self.localize_radius)

            """  ------------------------ EnKF update step  ---------------------  """
            y_t = observations[:, t]      # shape: batch_size x output_dim

            # x_t_post,         shape: batch_size x N_MC x state_dim
            # X_mean_post,      shape: batch_size x 1 x state_dim
            # P_post,           shape: batch_size x state_dim x state_dim
            x_t_post, log_likelihood, X_mean_post, P_post = self.EnKFUpdate(X=x_t, X_mean=X_mean, P=P, y=y_t)

            # update x_t_1 for next round
            x_t_1 = x_t_post.permute(1, 0, 2)        # N_MC x batch_size x state_dim

            """  ------------------------ result processing/saving step  ---------------------  """
            likelihood = likelihood + log_likelihood#.div(self.seq_len)
            filtered_mean.append(X_mean_post.squeeze(1))                           # shape: batch_size x state_dim
            filtered_var.append(P_post.diagonal(offset=0, dim1=-1, dim2=-2))       # shape: batch_size x state_dim

        filtered_mean = torch.stack(filtered_mean, dim=1)    # shape: batch_size x seq_len x state_dim
        filtered_var = torch.stack(filtered_var, dim=1)      # shape: batch_size x seq_len x state_dim

        return likelihood, filtered_mean, filtered_var


    def emission(self, x, P):
        """
        emission from state space m & P to observed space mean & sigma
        """
        pred_mean = self.H @ x
        pred_sigma = self.H @ P @ self.H.transpose(0, 1) + torch.diag_embed(self.emission_likelihood.noise.view(-1))
        return pred_mean.squeeze(-1), pred_sigma


    def Prediction(self, x_t_1, input_sequence=None, localization_radius=None, linear_obs=True, var_inflation = 0.0):
        """

        Parameters
        ----------
        x_t_1:              shape: N_MC x batch_size x state_dim
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

        '''  ---------------- NN Prediction step  ---------------- '''
        # x_t_1 shape: N_MC x batch_size x state_dim
        nn_input = x_t_1  # shape:  N_MC x batch_size x state_dim

        if input_sequence is not None:
            # c_t shape:  N_MC x batch_size x input_dim
            c_t = input_sequence.expand(self.N_MC, -1, -1)
            # nn_input shape:  N_MC x batch_size x (input_dim + state_dim)
            nn_input = torch.cat((c_t, x_t_1), dim=-1)

        # 👻 marginalize from U first
        x_t = self.transition(nn_input)          # samples, shape: N_MC x batch_size x state_dim
        if self.residual_trans:
            x_t = x_t + x_t_1                    # shape: N_MC x batch_size x state_dim


        if linear_obs and localization_radius is not None:
            taper = construct_Gaspari_Cohn(localization_radius, self.state_dim, x_t.device)
        else:
            taper = torch.tensor(1., device=x_t.device)

        '''  ----------------  Post-processing step  ---------------- '''
        X = x_t.permute(1, 0, 2)                 # shape: batch_size x N_MC x state_dim
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

        x = x_0   # shape: [N_MC x batch_size x state_dim]
        for i in range(T):
            if input_sequence is not None:
                nn_input = input_sequence[:, i]
            else:
                nn_input = None

            # X shape: batch_size x N_MC x state_dim
            # X_mean shape: batch_size x 1 x state_dim
            # P shape: batch_size x state_dim x state_dim
            X, X_mean, P = self.Prediction(x_t_1=x, input_sequence=nn_input,localization_radius=self.localize_radius)
            X_variance = torch.diagonal(P, offset=0, dim1=-1, dim2=-2)                # shape: batch_size x state_dim

            x = X.permute(1, 0, 2)  # N_MC x batch_size x state_dim

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


class DKF(nn.Module):
    def __init__(self, dim_x, dim_y, seq_len, dim_c = 0, process_noise_sd=0.05, emission_noise_sd=0.1,
                 learn_emission=False, residual_trans=False, hidden_dim=None):
        """
        Parameters
        ----------
        dim_x:                dimension of state
        dim_y:                dimension of measurement
        seq_len:              sequence length
        dim_c:                dimension of control input
        process_noise_sd:     transition process noise standard deviation initialization
        emission_noise_sd:    observation noise standard deviation initialization
        learn_emission:       if yes, the emission matrix is learnable
        residual_trans:       if yes, x[t] = f(x[t-1]) + x[t-1]
        hidden_dim:           the hidden dimension specification
        """
        super().__init__()
        self.output_dim = dim_y
        self.input_dim  = dim_c
        self.state_dim = dim_x
        self.seq_len = seq_len
        self.residual_trans = residual_trans
        self.variance_output = False
        if hidden_dim is None:
            hidden_dim = [128, 64, 32]
        self.hidden_dim = hidden_dim

        # emission index: indicates which dimensions of latent state are observable
        # emission matrix: [output_dim x state_dim]
        indices = [i for i in range(self.output_dim)]
        H = torch.eye(self.state_dim)[indices]
        self.H = nn.Parameter(H, requires_grad=learn_emission)

        # define GP transition
        self.transition = Four_Layer_NN(input_dim=dim_x+dim_c, output_dim=dim_x, hidden_dim=hidden_dim, residual=False, activation='relu')

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([self.state_dim]))
        self.emission_likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([self.output_dim]))

        self.likelihood.noise =  process_noise_sd ** 2
        self.emission_likelihood.noise = emission_noise_sd ** 2

        # recognition network for inferring the initial state x_0
        self.RecNet = LSTMRecognition(dim_outputs=self.output_dim, dim_inputs=self.input_dim,
                                        dim_states=self.state_dim, length=self.seq_len)

        # encode y_{1:T} into a series of hidden states (Not to be confused, not latent states x_{1:T})
        self.EncodeNet = LSTMencoder(dim_outputs=self.output_dim,
                                     dim_inputs=self.input_dim,
                                     dim_states=self.state_dim,
                                     length=self.seq_len,
                                     hidden_size=32,
                                     num_layers=2,
                                     batch_first=True,
                                     bd=True)

        # inference network for the latent state distribution,  q(x_{1:T})
        self.postNet = PostNet(x_dim=self.state_dim, h_dim=self.EncodeNet.hidden_size,
                               bd=self.EncodeNet.bd, mf_flag=True)


    def forward(self, observations,  input_sequence=None):
        """

        Parameters
        ----------
        observations:           Tensor, observation sequence,         [batch_size x seq_len x output_dim]
        input_sequence:         Tensor, control input sequence        [batch_size x seq_len x input_dim]

        Returns
        -------
        """

        # qm0_KL = torch.tensor([0.], device=observations.device, dtype=observations.dtype)
        # x_t_1 = torch.zeros(batch_size, self.state_dim, self.N_MC, self.state_dim, device=observations.device, dtype=observations.dtype)

        '''---------------  1. Calculate KL divergence of initial state   ---------------'''
        # qx0: shape: batch_size x state_dim
        qx0, qm0_KL = self.x0_KL(observations, input_sequence)

        '''---------------  2. Calculate likelihood   ---------------'''
        x_t_1 = qx0.rsample()    # batch_size x state_dim

        likelihood, KLD, filtered_mean, filtered_var = self.iterate_sequence(observations=observations, x_0=x_t_1,
                                                                             input_sequence=input_sequence)

        ELBO = -qm0_KL + likelihood - KLD
        print(f"\n--x0 KL: {qm0_KL.item()} ")
        print(f"--xt KL: {KLD.item()}")
        print(f"--likelihood: {likelihood.item()}")
        print(f"--ELBO: {ELBO.item()}")

        return ELBO, filtered_mean, filtered_var

    def iterate_sequence(self, observations, x_0, input_sequence=None):

        """
        Parameters
        ----------
        observations:           Tensor, observation sequence,         [batch_size x seq_len x output_dim]
        x_0:                    Tensor, initial state                 [batch_size x state_dim]
        input_sequence:         Tensor, control input sequence        [batch_size x seq_len x input_dim]

        Returns
        -------
        """
        # batch_size = observations.shape[0]
        device = observations.device
        likelihood = torch.tensor(0., device=device)
        KL_terms = torch.tensor(0., device=device)
        filtered_mean = []
        filtered_var = []

        # observations.shape == [ batch_size x seq_len x output_dim ]
        assert (observations.shape[-2] == self.seq_len)

        '''---------------    encode y_{1:T} using LSTM   --------------- '''
        # shape: batch_size x seq_len x (1+bd)*hidden_size
        hidden_all = self.EncodeNet(output_sequence=observations, input_sequence=input_sequence)

        x_t_1 = x_0  # initial state,  [batch_size x state_dim]

        for t in range(self.seq_len):

            """  ------------------------ prediction step  ---------------------  """
            if input_sequence is not None:
                nn_input = input_sequence[:, t]
            else:
                nn_input = None

            # x_t shape:    batch_size x state_dim
            # PriorDist_xt:  MultivariateNorm,  shape: batch_size x state_dim
            x_t, PriorDist_xt = self.Prediction(x_t_1=x_t_1, input_sequence=nn_input)

            """  ------------------------ EnKF update step  ---------------------  """
            y_t = observations[:, t]      # shape: batch_size x output_dim

            # x_t_post,         shape: batch_size x state_dim
            # P_post,           shape: batch_size x state_dim x state_dim
            x_t_post, log_likelihood, KL_terms, X_mean_post, P_post =self.VariUpdate(X=x_t, P=PriorDist_xt,
                                                                                     H=hidden_all[:, t], y=y_t)

            # update x_t_1 for next round
            x_t_1 = x_t_post

            """  ------------------------ result processing/saving step  ---------------------  """
            likelihood = likelihood + log_likelihood.div(self.seq_len)
            KL_terms = KL_terms + KL_terms.div(self.seq_len)

            filtered_mean.append(X_mean_post.squeeze(1))                           # shape: batch_size x state_dim
            filtered_var.append(P_post.diagonal(offset=0, dim1=-1, dim2=-2))       # shape: batch_size x state_dim

        filtered_mean = torch.stack(filtered_mean, dim=1)    # shape: batch_size x seq_len x state_dim
        filtered_var = torch.stack(filtered_var, dim=1)      # shape: batch_size x seq_len x state_dim

        return likelihood, KL_terms, filtered_mean, filtered_var


    def emission(self, x, P):
        """
        emission from state space m & P to observed space mean & sigma
        """
        pred_mean = self.H @ x
        pred_sigma = self.H @ P @ self.H.transpose(0, 1) + torch.diag_embed(self.emission_likelihood.noise.view(-1))
        return pred_mean.squeeze(-1), pred_sigma


    def Prediction(self, x_t_1, input_sequence=None):
        """

        Parameters
        ----------
        x_t_1:              shape: batch_size x state_dim
        input_sequence:     shape: batch_size x input_dim

        Returns
        -------
        x_t:                    shape: batch_size x state_dim
        prior_dist_xt:          prior distribution if x[t] | x[t-1],     shape: batch_size x state_dim

        """

        '''  ---------------- NN Prediction step  ---------------- '''
        # x_t_1 shape:  batch_size x state_dim
        nn_input = x_t_1  # shape: batch_size x state_dim

        if input_sequence is not None:
            # nn_input shape:  batch_size x (input_dim + state_dim)
            nn_input = torch.cat((input_sequence, x_t_1), dim=-1)

        # 👻 marginalize from U first
        x_t = self.transition(nn_input)          # samples, shape: batch_size x state_dim
        if self.residual_trans:
            x_t = x_t + x_t_1                    # shape:  batch_size x state_dim

        '''  ----------------  Post-processing step  ---------------- '''
        # shape: batch_size x state_dim
        prior_dist_xt = MultivariateNormal(mean=x_t, covariance_matrix=torch.diag_embed(self.likelihood.noise.view(-1)))

        return x_t, prior_dist_xt

    def VariUpdate(self, X, P, H, y):

        """
            Variational distribution update


            Add a new measurement (y) to the kalman filter. If y is None, nothing is changed.

            Parameters
            ----------

                X: Tensor,  shape: batch_size x state_dim
                    latent state obtained from prediction step

                P: MultivariateNormal,  shape: batch_size x state_dim

                H: Hidden state from LSTM encoder,  shape:  batch_size x (1+bd)*hidden_size

                y : Tensor,  shape:  batch_size x output_dim
                    measurement for this update.
        """

        noise_R = torch.diag_embed(self.emission_likelihood.noise.view(-1))  # shape: output_dim x output_dim


        '''  ----------------  Variational distribution Construction   ---------------- '''
        #  x_t: Tensor,  shape: batch_size x state_dim
        X_post, Var_dist_X = self.postNet(x_t_1=X, hidden=H)
        m_t, sigma_t = Var_dist_X.mean, Var_dist_X.covariance_matrix

        Cmt = (self.H @ m_t.unsqueeze(-1)).squeeze(dim=-1)     # shape:  batch_size x output_dim
        CSigmaC = self.H @ sigma_t @ self.H.T                  # shape:  batch_size x output_dim x output_dim

        qyt = MultivariateNormal(Cmt, noise_R)   # shape: batch_size x output_dim

        # E_{q(x_t)} [ log p(y_t | x_t) ]
        # shape: batch_size x output_dim
        trace_term = (1 / noise_R) @ CSigmaC
        data_fit = qyt.log_prob(y) - 0.5 * trace_term.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

        # compute KLD
        KL = KL_divergence(Var_dist_X, P)

        return X_post, data_fit.mean(), KL.mean(), m_t, sigma_t

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

        x = x_0   # shape: [batch_size x state_dim]
        for i in range(T):
            if input_sequence is not None:
                nn_input = input_sequence[:, i]
            else:
                nn_input = None

            # x shape: batch_size x state_dim
            # P shape: batch_size x state_dim
            _, P = self.Prediction(x_t_1=x, input_sequence=nn_input)
            x = P.mean

            pred_means = torch.cat([pred_means, x.unsqueeze(1)], dim=1)                 # batch_size x T x state_dim
            pred_sigmas = torch.cat([pred_sigmas, P.variance.unsqueeze(1)], dim=1)      # batch_size x T x state_dim

        if observations is not None:
            y_pred, y_pred_sigma = self.emission(pred_means.unsqueeze(-1), torch.diag_embed(pred_sigmas))
            dist = torch.distributions.MultivariateNormal(y_pred, y_pred_sigma)
            NLL = -dist.log_prob(observations).mean()
            NLL /= observations.shape[-1]

            return NLL, pred_means, pred_sigmas, y_pred, y_pred_sigma
        else:
            return pred_means, pred_sigmas


class DSSMs(nn.Module):
    def __init__(self, dim_x, dim_y, seq_len, dim_c = 0, N_MC = 50, process_noise_sd=0.05,
                 emission_noise_sd=0.1, hidden_dim=None):
        """
        Parameters
        ----------
        dim_x:                dimension of state
        dim_y:                dimension of measurement
        seq_len:              sequence length
        dim_c:                dimension of control input
        N_MC:                 number of ensemble
        process_noise_sd:     transition process noise standard deviation initialization
        emission_noise_sd:    observation noise standard deviation initialization
        hidden_dim:           hidden dimension of transition neural network
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = [128, 64, 32]
        self.output_dim = dim_y
        self.input_dim  = dim_c
        self.state_dim = dim_x
        self.seq_len = seq_len
        self.N_MC = N_MC
        self.hidden_dim = hidden_dim


        # define GP transition
        self.transition = Four_Layer_NN(input_dim=dim_x+dim_c, output_dim=dim_x, hidden_dim=hidden_dim, residual=False, activation='relu')
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([self.state_dim]))
        # self.emission_likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([self.output_dim]))
        self.emission_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.output_dim, rank=0,
                                                                                    has_global_noise=False,
                                                                                    has_task_noise=True)

        self.likelihood.noise =  process_noise_sd ** 2
        self.emission_likelihood.task_noises = emission_noise_sd ** 2

        # recognition network for inferring the initial state x_0
        self.RecNet = LSTMRecognition(dim_outputs=self.output_dim, dim_inputs=self.input_dim,
                                        dim_states=self.state_dim, length=self.seq_len)


        # setting EnKF
        self.localize_radius = 5


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

        device = observations.device
        batch_size = observations.shape[0]
        likelihood = torch.tensor(0., device=device)
        data = []

        if H is None:
            # emission matrix
            indices = [i for i in range(self.output_dim)]
            H = torch.eye(self.state_dim, device=device, dtype=observations.dtype)[indices]

        assert (H.shape[0] == self.output_dim)
        assert (H.shape[1] == self.state_dim)

        # observations.shape == [ batch_size x seq_len x output_dim ]
        assert (observations.shape[-2] == self.seq_len)


        ''' ---------------  1.  KL[ q(x0) || p(x0) ]  --------------- '''
        # qx0: shape: batch_size x state_dim
        qx0 = self.RecNet(observations, input_sequence)

        # construct px0
        px0_mean = torch.zeros_like(qx0.mean)                    # shape: batch_size x state_dim
        px0_cov = torch.diag_embed(torch.ones_like(qx0.mean))    # shape: batch_size x state_dim

        px0 = MultivariateNormal(px0_mean, px0_cov)
        qm0_KL = KL_divergence(qx0, px0).mean()                  # take average over batch_size


        ''' -----------------------------  2.  Prediction Step & Update Step   -------------------------- '''
        x_t_1 = qx0.rsample(torch.Size([self.N_MC]))     # N_MC x batch_size x state_dim


        for t in range(self.seq_len):
            """  ------------------------ NN prediction step  ---------------------  """
            # x_t_1 shape: N_MC x batch_size x state_dim.
            NN_input = x_t_1   # shape:  N_MC x batch_size x state_dim
            if input_sequence is not None:
                # c_t shape: N_MC x batch_size x input_dim
                c_t = input_sequence[:, t].repeat(self.N_MC, 1, 1)
                # gp_input shape: N_MC x batch_size x (input_dim + state_dim)
                NN_input = torch.cat((c_t, x_t_1), dim=-1)

            # shape: N_MC x batch_size x state_dim  [found that adding x_t_1 worse the performance, so comment it]
            NNdynamics = self.transition(NN_input) # + x_t_1   # N_MC x batch_size x state_dim

            # shape: N_MC x batch_size x state_dim
            std = torch.sqrt(self.likelihood.noise).expand(self.N_MC, batch_size, self.state_dim)

            # shape: N_MC x batch_size x state_dim
            eps = torch.randn(self.N_MC, batch_size, self.state_dim, device=device)

            # shape: N_MC x batch_size x state_dim
            x_t = NNdynamics + eps * std

            y_t = observations[:, t]      # shape: batch_size x output_dim

            # x_t_post, shape:  batch_size x N_MC x state_dim
            x_t_post, log_likelihood, X_mean, P, \
            X_mean_post, P_post = self.EnsembleKalmanFilter(X=x_t, y=y_t, localization_radius=self.localize_radius,
                                                            linear_obs=True, H=H, var_inflation=0.0)
            likelihood = likelihood + log_likelihood.mean()

            data.append(X_mean_post)


            # update x_t_1 for next round
            x_t_1 = x_t_post.transpose(0, 1).contiguous()               # N_MC x batch_size x state_dim


        ELBO = -qm0_KL + likelihood
        print()
        print(f" x0 KL: {qm0_KL.item()} ")
        print(f" likelihood: {likelihood.item()}")
        print(f" ELBO: {ELBO.item()}")
        print()
        return ELBO, data



    def EnsembleKalmanFilter(self, X, y=None,  localization_radius=None,  linear_obs=True, H = None, var_inflation = 0.0):

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
                H : Tensor, shape: output_dim x state_dim
                    Coefficient matrix (Linear emission matrix)

                y : Tensor,  shape:  batch_size x output_dim
                    measurement for this update.

                X: Tensor,  shape: N_MC x batch_size x state_dim
                    ensemble obtained from prediction step

                linear_obs:

                localization_radius:

                var_inflation:  float

            # Return:


        """


        if linear_obs and H is None:
            # emission matrix
            indices = [i for i in range(self.output_dim)]
            H = torch.eye(self.state_dim, device=y.device, dtype=y.dtype)[indices]

        if linear_obs and localization_radius is not None:
            taper = construct_Gaspari_Cohn(localization_radius, self.state_dim, X.device)
        else:
            taper = torch.tensor(1., device=y.device)


        '''  ----------------  Prediction step  ---------------- '''

        X = X.transpose(0, 1).contiguous()    # shape: batch_size x N_MC x state_dim

        X_mean = X.mean(dim=-2).unsqueeze(-2)  # shape: batch_size x 1 x state_dim
        X_center = X - X_mean                  # shape: batch_size x N_MC x state_dim

        if var_inflation is not None:
            X = (1. + var_inflation) * X_center + X_mean

        P = 1 / (self.N_MC - 1) * X_center.transpose(-1, -2) @ X_center  # shape: batch_size x state_dim x state_dim
        P = taper * P.contiguous()

        # if var_inflation is not None:
        #     P = (1. + var_inflation) * P


        '''  ----------------  Update Step  ---------------- '''
        noise_R = torch.diag_embed(self.emission_likelihood.task_noises)  # shape: output_dim x output_dim

        if y is None:
            X_post = X
            P_post = P
            return X_post, P_post
        else:
            noise_y = torch.randn_like(y)       # shape: batch_size x output_dim
            chol = torch.cholesky(noise_R)      # shape: output_dim x output_dim
            y_perturb = y + noise_y @ chol.t()  # shape: batch_size x output_dim
            y_perturb = y_perturb.unsqueeze(-2) # shape: batch_size x 1 x output_dim
            # print(f'\n y is continuous: {y_perturb.is_contiguous()}' )


        # transform ensembles into measurement space
        HX = X @ H.transpose(-1, -2).contiguous()        # shape:  batch_size x N_MC x output_dim

        HP = H @ P                                       # shape:  batch_size x output_dim x state_dim

        HPH_T = HP @ H.transpose(-1, -2).contiguous()    # shape:  batch_size x output_dim x output_dim

        if torch.__version__=='1.7.1+cu110':
            HPH_TR_chol = torch.cholesky(HPH_T + noise_R)  # shape: batch_size x output_dim x output_dim, lower-tril
        else:
            HPH_TR_chol = torch.linalg.cholesky(HPH_T + noise_R)  # shape: batch_size x output_dim x output_dim, lower-tril


        KalmanGain = torch.cholesky_inverse(HPH_TR_chol) @ HP  # shape:  batch_size x output_dim x state_dim

        pre = (y_perturb - HX) @ KalmanGain   # shape: batch_size x N_MC x state_dim
        # print(f"\n residual is (y_perturb - HX) @ KalmanGain = {pre.mean()}")

        X_post = X + pre   # shape: batch_size x N_MC x state_dim

        '''  -------------- post-processing and compute log-likelihood ---------- '''

        HX_m = X_mean @ H.transpose(-1, -2).contiguous()               # shape: batch_size x 1 x output_dim

        X_mean_post = X_mean + (y.unsqueeze(-2) - HX_m) @ KalmanGain   # shape: batch_size x 1 x state_dim

        # shape: batch_size x state_dim x state_dim
        P_post = P - HP.transpose(-1, -2).contiguous()  @  KalmanGain

        # indication if computing EnkF-type log-likelihood
        EnKF_likelihood = True

        if EnKF_likelihood:
            # batch_size x output_dim
            d = torch.distributions.MultivariateNormal(HX_m.squeeze(-2),  scale_tril=HPH_TR_chol)

            log_likelihood = d.log_prob(y)  # shape: batch_size

        else:
            # batch_size x output_dim
            d = MultivariateNormal(HX, noise_R)     # shape:  batch_size x N_MC x output_dim
            # print(f"\n type of d: {type(d)}, \n batch_shape:{d.batch_shape}, \n event_shape:{d.event_shape}")

            log_likelihood = d.log_prob(y.repeat(self.N_MC, 1, 1).transpose(0, 1))      # shape: batch_size x N_MC

        return X_post, log_likelihood, X_mean, P, X_mean_post, P_post

