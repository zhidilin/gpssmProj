#!/usr/bin/env python3
import math
import torch
import gpytorch
from gpytorch import settings
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import DiagLazyTensor, MatmulLazyTensor, RootLazyTensor, SumLazyTensor, TriangularLazyTensor, delazify
from gpytorch.settings import trace_mode
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.variational import CholeskyVariationalDistribution


def _cholesky_factor(induc_induc_covar):
    # Compute Cholesky factor of the inducing points covariance matrix
    L = psd_safe_cholesky(delazify(induc_induc_covar).double())
    return TriangularLazyTensor(L)


class GP_Module(torch.nn.Module):
    def __init__(self, inducing_points, learn_inducing_locations=True, batch_size=1):
        super(GP_Module, self).__init__()

        self.GP_dim = batch_size
        # assert(self.GP_dim == inducing_points.shape[0])
        num_ips = inducing_points.size(-2)

        # Variational distribution for inducing points
        self.variational_distribution = CholeskyVariationalDistribution(num_inducing_points=num_ips,
                                                                        batch_shape=torch.Size([self.GP_dim]))

        # Define kernel and mean functions
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([self.GP_dim]))
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(batch_shape=torch.Size([self.GP_dim])),
                                                   batch_shape=torch.Size([self.GP_dim]))


        # Initialize inducing points
        inducing_points = inducing_points.clone()
        if inducing_points.dim() == 1:
            inducing_points = inducing_points.unsqueeze(-1)

        # Learn inducing locations if specified
        if learn_inducing_locations:
            self.inducing_points = torch.nn.Parameter(inducing_points)
        else:
            self.inducing_points = inducing_points
        self.num_inducing = inducing_points.size(0)


    def prior_distribution(self):
        # Define prior distribution p(u) ~ N(0, I)
        zeros = torch.zeros(self.variational_distribution.shape(),
                            dtype=self.variational_distribution.dtype,
                            device=self.variational_distribution.device
                            )
        ones = torch.ones_like(zeros)
        return MultivariateNormal(zeros, DiagLazyTensor(ones))

    def GP_def(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.kernel(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def kl_divergence(self):

        with settings.max_preconditioner_size(0):
            kl_divergence = torch.distributions.kl.kl_divergence(self.variational_distribution(),
                                                                 self.prior_distribution()).sum()
        return kl_divergence


    def forward(self, x, L=None):
        """
        Predicts the mean and variance of the GP at the given test points
            :param x: Test points
            :param L: Cholesky factor of the inducing points covariance matrix
            :return: Predictive mean and variance
        """
        if L is None:
            induc_induc_covar = self.kernel(self.inducing_points).add_jitter()
            L = _cholesky_factor(induc_induc_covar)

        # evaluation
        test_mean = self.mean_module(x)
        induc_data_covar = self.kernel(self.inducing_points, x).evaluate()  # size: num_inducing x num_test
        data_data_covar = self.kernel(x)

        # Compute the interpolation term: K_ZZ^{-1} K_ZX ,  size: num_inducing x num_test
        interp_term = L.inv_matmul(induc_data_covar.double()).to(x.dtype)

        """ # Compute the mean of q(f):  k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X   """
        predictive_mean = (interp_term.transpose(-1, -2) @ self.variational_distribution().mean.unsqueeze(-1)).squeeze(-1) + test_mean
        # Compute the covariance of q(f):   K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX
        middle_term = self.prior_distribution().lazy_covariance_matrix.mul(-1)
        middle_term = SumLazyTensor(self.variational_distribution().lazy_covariance_matrix, middle_term)
        # Predictive covariance
        predictive_covar = SumLazyTensor(data_data_covar.add_jitter(1e-4),
                                         MatmulLazyTensor(interp_term.transpose(-1, -2), middle_term @ interp_term))

        return MultivariateNormal(predictive_mean, predictive_covar)


    def condition_u(self, x, U, L=None):
        """

        Parameters
        ----------
        x shape: N_MC x state_dim x batch_size x (input_dim + state_dim)
        U shape: N_MC x state_dim x num_ips,
        L:       Cholesky factor of the inducing points covariance matrix

        Returns
        -------

        """

        assert len(x.shape) == 4, 'Bad input x, expected shape: N_MC x state_dim x batch_size x (input_dim + state_dim)'

        N_MC, state_dim, batch_size, x_dim = x.shape   # x shape: N_MC x state_dim x batch_size x (input_dim + state_dim)

        ''' ------------------    get inducing points U and inducing inputs Z    ------------------------  '''
        if L is None:
            induc_induc_covar = self.kernel(self.inducing_points).add_jitter().evaluate()
            L = torch.linalg.cholesky(induc_induc_covar)   # shape: state_dim x num_ips x num_ips

        if U is None:
            # sample U ~ q(U),  shape: N_MC x state_dim x num_ips
            U = self.forward(self.inducing_points).rsample(torch.Size([N_MC]))

        _, num_ips, _ = self.inducing_points.shape

        # shape: state_dim x num_ips x (state_dim + input_dim)
        inducing_points = self.inducing_points

        # shape [N_MC x state_dim x num_ips x (state_dim + input_dim)]
        inducing_points = inducing_points.expand(N_MC, state_dim, num_ips, x_dim)

        induc_mean = self.mean_module(inducing_points)  # shape: N_MC x state_dim x num_ips
        test_mean = self.mean_module(x)                 # shape: N_MC x state_dim x batch_size


        # shape: N_MC x state_dim x batch_size x num_ips
        data_induc_covar = self.kernel(x, self.inducing_points,).evaluate()

        # residue shape: N_MC x state_dim x num_ips
        residue =  U - induc_mean

        # tmp = (LL^T)^{-1} * residue.     shape: N_MC x state_dim x num_ips x 1
        tmp = torch.cholesky_solve(residue.unsqueeze(dim=-1), L)

        # shape: N_MC x state_dim x batch_size x 1
        f_condition_u_mean = test_mean + torch.matmul(data_induc_covar, tmp).squeeze(dim=-1)

        return f_condition_u_mean  # shape: N_MC x state_dim x batch_size



class GP_Module_1D(torch.nn.Module):
    def __init__(self, inducing_points, learn_inducing_locations=True):
        super(GP_Module_1D, self).__init__()

        # Initialize inducing points
        inducing_points = inducing_points.clone()
        if inducing_points.dim() == 1:
            inducing_points = inducing_points.unsqueeze(-1)

        # inducing_points shape: num_ips x (dim_state + dim_input)
        assert len(inducing_points.shape) == 2, 'Wrong initilization of inducing points, expected shape: num_ips x (input_dim + state_dim)'
        num_ips = inducing_points.size(-2)

        # Variational distribution for inducing points
        self.variational_distribution = CholeskyVariationalDistribution(num_inducing_points=num_ips)

        # Define kernel and mean functions
        self.mean_module = gpytorch.means.ConstantMean()
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        # Learn the inducing locations if specified
        if learn_inducing_locations:
            self.inducing_points = torch.nn.Parameter(inducing_points)
        else:
            self.inducing_points = inducing_points
        self.num_inducing = inducing_points.size(0)


    def prior_distribution(self):
        # Define prior distribution p(u) ~ N(0, I)
        zeros = torch.zeros(self.variational_distribution.shape(),
                            dtype=self.variational_distribution.dtype,
                            device=self.variational_distribution.device
                            )
        ones = torch.ones_like(zeros)
        return MultivariateNormal(zeros, DiagLazyTensor(ones))

    def GP_def(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.kernel(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def kl_divergence(self):

        with settings.max_preconditioner_size(0):
            kl_divergence = torch.distributions.kl.kl_divergence(self.variational_distribution(),
                                                                 self.prior_distribution()).sum()
        return kl_divergence


    def forward(self, x, L=None):
        """
        Predicts the mean and variance of the GP at the given test points
            :param x: Test points
            :param L: Cholesky factor of the inducing points covariance matrix
            :return: Predictive mean and variance
        """
        if L is None:
            induc_induc_covar = self.kernel(self.inducing_points).add_jitter()
            L = _cholesky_factor(induc_induc_covar)

        # evaluation
        test_mean = self.mean_module(x)
        induc_data_covar = self.kernel(self.inducing_points, x).evaluate()  # size: num_inducing x num_test
        data_data_covar = self.kernel(x)

        # Compute the interpolation term: K_ZZ^{-1} K_ZX ,  size: num_inducing x num_test
        interp_term = L.inv_matmul(induc_data_covar.double()).to(x.dtype)

        """ # Compute the mean of q(f):  k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X   """
        predictive_mean = (interp_term.transpose(-1, -2) @ self.variational_distribution().mean.unsqueeze(-1)).squeeze(-1) + test_mean
        # Compute the covariance of q(f):   K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX
        middle_term = self.prior_distribution().lazy_covariance_matrix.mul(-1)
        middle_term = SumLazyTensor(self.variational_distribution().lazy_covariance_matrix, middle_term)
        # Predictive covariance
        predictive_covar = SumLazyTensor(data_data_covar.add_jitter(1e-4),
                                         MatmulLazyTensor(interp_term.transpose(-1, -2), middle_term @ interp_term))

        return MultivariateNormal(predictive_mean, predictive_covar)

    def condition_u(self, x, U, L=None):
        """

        Parameters
        ----------
        x shape: N_MC x batch_size x (input_dim + state_dim)
        U shape: N_MC x num_ips,
        L:       Cholesky factor of the inducing points covariance matrix

        Returns
        -------

        """

        assert len(x.shape) == 3, 'Bad input x, expected shape: N_MC x batch_size x (input_dim + state_dim)'

        N_MC, batch_size, x_dim = x.shape   # x shape: N_MC x batch_size x (input_dim + state_dim)

        ''' ------------------    get inducing points U and inducing inputs Z    ------------------------  '''
        if L is None:
            induc_induc_covar = self.kernel(self.inducing_points).add_jitter().evaluate()
            L = torch.linalg.cholesky(induc_induc_covar)   # shape: num_ips x num_ips

        if U is None:
            # sample U ~ q(U),  shape: N_MC x num_ips
            U = self.forward(self.inducing_points).rsample(torch.Size([N_MC]))

        num_ips, _ = self.inducing_points.shape

        # shape: num_ips x (state_dim + input_dim)
        inducing_points = self.inducing_points

        # shape [N_MC x num_ips x (state_dim + input_dim)]
        inducing_points = inducing_points.expand(N_MC, num_ips, x_dim)

        induc_mean = self.mean_module(inducing_points)  # shape: N_MC x num_ips
        test_mean = self.mean_module(x)                 # shape: N_MC x batch_size


        # shape: N_MC x batch_size x num_ips
        data_induc_covar = self.kernel(x, self.inducing_points,).evaluate()

        # residue shape: N_MC x num_ips
        residue =  U - induc_mean

        # tmp = (LL^T)^{-1} * residue.     shape: N_MC x num_ips x 1
        tmp = torch.cholesky_solve(residue.unsqueeze(dim=-1), L)

        # shape: N_MC x batch_size x 1
        f_condition_u_mean = test_mean + torch.matmul(data_induc_covar, tmp).squeeze(dim=-1)

        return f_condition_u_mean  # shape: N_MC x batch_size



class SparseGPModel(torch.nn.Module):
    def __init__(self, inducing_points, learn_inducing_locations=True):
        super(SparseGPModel, self).__init__()
        # Variational distribution for inducing points
        self.variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))

        # Define kernel, mean, and likelihood
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.mean_module = gpytorch.means.ConstantMean()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()


        # Initialize inducing points
        inducing_points = inducing_points.clone()
        if inducing_points.dim() == 1:
            inducing_points = inducing_points.unsqueeze(-1)

        # Learn inducing locations if specified
        if learn_inducing_locations:
            self.inducing_points = torch.nn.Parameter(inducing_points)
        else:
            self.inducing_points = inducing_points
        self.num_inducing = inducing_points.size(0)

    def prior_distribution(self):
        # Define prior distribution p(u) ~ N(0, I)
        zeros = torch.zeros(self.variational_distribution.shape(),
                            dtype=self.variational_distribution.dtype,
                            device=self.variational_distribution.device
                            )
        ones = torch.ones_like(zeros)
        return MultivariateNormal(zeros, DiagLazyTensor(ones))

    def GP_def(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.kernel(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(self, x):
        """
        Predicts the mean and variance of the GP at the given test points
            :param x: Test points
            :return: Predictive mean and variance
        """

        if len(x.shape) == 1:
            x = x.unsqueeze(-1)

        # Get number of inducing points
        num_induc = self.inducing_points.size(-2)

        # Compute full prior distribution
        full_inputs = torch.cat([self.inducing_points, x], dim=-2)
        # full_output = self.GP_def(full_inputs)
        # full_covar = full_output.lazy_covariance_matrix
        # test_mean = full_output.mean[..., num_induc:]
        full_covar = self.kernel(full_inputs)
        test_mean = self.mean_module(x)


        # Split covariance into blocks
        induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter()
        induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate()
        data_data_covar = full_covar[..., num_induc:, num_induc:]

        # Compute interpolation terms
        # K_ZZ^{-1/2} K_ZX
        # K_ZZ^{-1/2} \mu_Z
        L = _cholesky_factor(induc_induc_covar)
        interp_term = L.inv_matmul(induc_data_covar.double()).to(full_inputs.dtype)

        # Compute the mean of q(f):     k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X
        predictive_mean = (interp_term.transpose(-1, -2) @ self.variational_distribution().mean.unsqueeze(-1)).squeeze(-1) + test_mean

        # Compute the covariance of q(f):   K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX
        middle_term = self.prior_distribution().lazy_covariance_matrix.mul(-1)
        middle_term = SumLazyTensor(self.variational_distribution().lazy_covariance_matrix, middle_term)
        # Predictive covariance
        if settings.trace_mode.on():
            predictive_covar = data_data_covar.add_jitter(1e-4).evaluate() + interp_term.transpose(-1, -2) @ middle_term.evaluate() @ interp_term
        else:
            predictive_covar = SumLazyTensor(data_data_covar.add_jitter(1e-4), MatmulLazyTensor(interp_term.transpose(-1, -2), middle_term @ interp_term))

        return MultivariateNormal(predictive_mean, predictive_covar)

    def expected_log_prob(self, target: torch.Tensor, input: MultivariateNormal) -> torch.Tensor:
        """
        Computes the expected log probability of the target given the input distribution
            :param target: Target values
            :param input: Input distribution

            :return: Expected log probability
        """
        mean, variance = input.mean, input.variance
        num_event_dim = len(input.event_shape)

        # Get noise from likelihood
        noise = self.likelihood._shaped_noise_covar(mean.shape).diag().view(*mean.shape[:-1], *input.event_shape)

        # Calculate negative log-likelihood
        res = ((target - mean) ** 2 + variance) / noise + noise.log() + math.log(2 * math.pi)
        res = res.mul(-0.5)
        if num_event_dim > 1:  # Do appropriate summation for multitask Gaussian likelihoods
            res = res.sum(list(range(-1, -num_event_dim, -1)))

        return res.sum(-1)

    def forward(self, train_x, train_y):
        """
        Computes the variational ELBO
            :param train_x: Training inputs
            :param train_y: Training targets
            :return: Variational ELBO
        """
        # Compute the variational ELBO
        beta = 1.0

        # Predictive distribution q(f)
        approximate_dist_f = self.predict(train_x)
        num_batch = approximate_dist_f.event_shape[0]

        # get likelihood
        log_likelihood = self.expected_log_prob(train_y, approximate_dist_f).div(num_batch)

        # KL divergence
        with settings.max_preconditioner_size(0):
            kl_divergence = torch.distributions.kl.kl_divergence(self.variational_distribution(), self.prior_distribution())
        kl_divergence = kl_divergence.div(num_batch / beta)

        return log_likelihood - kl_divergence, kl_divergence
