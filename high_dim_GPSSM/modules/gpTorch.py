import torch
import numpy as np
from scipy.special import roots_hermite
import gpytorch
from torch.distributions import MultivariateNormal
dtype = torch.float32

# Get the Hermite roots and weights using SciPy
n = 10  # number of quadrature points
roots, weights = roots_hermite(n)

# Convert to PyTorch tensors
roots = torch.tensor(roots, dtype=dtype)
weights = torch.tensor(weights, dtype=dtype)

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


class GPModel(torch.nn.Module):
    def __init__(self, likelihood_log_variance=0.0):
        super(GPModel, self).__init__()
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.likelihood_log_variance = torch.nn.Parameter(torch.tensor(likelihood_log_variance))

    def forward(self, train_x, train_y, test_x):
        # Compute the kernel matrix for training data
        K = self.kernel(train_x, train_x).evaluate()
        likelihood_variance = torch.exp(self.likelihood_log_variance)
        K += likelihood_variance * torch.eye(len(train_x))  # Add noise term to the diagonal

        # Cholesky decomposition
        L = torch.cholesky(K)

        # Solve for alpha (the weights)
        alpha = torch.cholesky_solve(train_y.unsqueeze(1), L)

        # Compute the kernel between training and test points
        K_s = self.kernel(train_x, test_x).evaluate()

        # Predictive mean
        pred_mean = K_s.t().matmul(alpha).squeeze()

        # Compute the kernel matrix for the test points
        K_ss = self.kernel(test_x, test_x).evaluate()

        # Compute the covariance of the predictive distribution
        v = torch.cholesky_solve(K_s, L)
        pred_cov = K_ss - K_s.t().matmul(v)
        pred_var = pred_cov.diag()

        return pred_mean, pred_var

    def neg_log_likelihood(self, train_x, train_y):
        """
        Compute the negative log likelihood of the model for training model
            :param train_x: A tensor of shape (n, d) representing the training inputs
            :param train_y: A tensor of shape (n,) representing the training outputs
            :return: A scalar tensor containing the negative log likelihood
        """
        K = self.kernel(train_x, train_x).evaluate()
        likelihood_variance = torch.exp(self.likelihood_log_variance)
        K += likelihood_variance * torch.eye(len(train_x))
        L = torch.cholesky(K)
        alpha = torch.cholesky_solve(train_y.unsqueeze(1), L)
        nll = 0.5 * train_y.unsqueeze(1).t().matmul(alpha).sum() + torch.log(L.diag()).sum() + 0.5 * len(train_x) * np.log(2 * np.pi)
        return nll


# # Example usage
# y = torch.tensor([1.0, 2.0])  # Example observed data
# m = torch.tensor([1.5, 1.8])  # Mean of q(u)
# S = torch.tensor([[0.1, 0.0], [0.0, 0.2]])  # Covariance matrix of q(u)
# sigma_sq = 0.5  # Variance of likelihood
#
# expected_log_likelihood = compute_expected_log_likelihood(y, m, S, sigma_sq)
# print(expected_log_likelihood)


def expected_log_likelihood(y, m, S, sigma_sq):
    """
    Compute E_{q(u)}[log p(y | u)] where
    q(u) = N(u | m, S)
    p(y | u) = N(y | u, sigma^2 * I)

    Parameters:
    y : torch.Tensor -- Observed data, shape (n,)
    m : torch.Tensor -- Mean of q(u), shape (n,)
    S : torch.Tensor -- Covariance matrix of q(u), shape (n, n)
    sigma_sq : float -- Variance of the likelihood

    Returns:
    torch.Tensor -- The expected log-likelihood
    """
    n = y.shape[0]  # number of data points
    diff = y - m

    # (y - m)^T (y - m)
    term1 = diff ** 2

    # Tr(S)
    term2 = S.diag()

    # Log-likelihood expectation
    expected_log_likelihood = -0.5 * (term1 + term2) / sigma_sq - 0.5 * torch.log(2 * np.pi * sigma_sq)

    return expected_log_likelihood.sum().div(n)


class SparseGPModel(torch.nn.Module):
    def __init__(self, inducing_points, likelihood_log_variance=0.0):
        super(SparseGPModel, self).__init__()
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.likelihood_log_variance = torch.nn.Parameter(torch.tensor(likelihood_log_variance))

        self.num_inducing = inducing_points.size(0)
        self.inducing_points = torch.nn.Parameter(inducing_points)

        self.q_mu = torch.nn.Parameter(torch.zeros(inducing_points.size(0), 1))  # Whitened mean of the variational distribution
        self.q_log_var = torch.nn.Parameter(torch.eye(inducing_points.size(0)))  # Log-variance/covariance of the whitened variational distribution

    def predict(self, input):
        """
            Compute the predictive mean and variance of the model for any inputs (i.e., q(f) = N(pred_mean, pred_Cov))
            :param input: A tensor of shape (n, d) representing the inputs
        """
        if len(input.shape) == 1:
            input = input.unsqueeze(-1)

        # Compute the kernel matrices
        K_ff = self.kernel(input, input).evaluate()
        K_uu = self.kernel(self.inducing_points, self.inducing_points).evaluate() + 1e-6 * torch.eye(self.inducing_points.size(0))
        K_uf = self.kernel(self.inducing_points, input).evaluate()

        # Compute the Cholesky decomposition of K_uu, size: m x m
        L_uu = torch.cholesky(K_uu)

        # Compute Lambda = inv(L_uu) * K_uf using torch.triangular_solve()
        Lambda, _ = torch.triangular_solve(input=K_uf, A=L_uu, upper=False)

        # Compute the predictive mean (self.q_mu is whitened variational mean, which is: inv(L_uu) * mu (unwhitened))
        pred_mean = Lambda.t() @ self.q_mu   # K_fu * inv(L_uu) * inv(L_uu) * mu

        # Compute the whitened variational distribution
        q_sqrt = torch.tril(torch.exp(0.5 * self.q_log_var))
        q_cov = q_sqrt.matmul(q_sqrt.t()) + 1e-6 * torch.eye(q_sqrt.size(0))

        """ Compute the predictive covariance """
        # Compute S := K_ff - Lambda.t() @ Lambda = K_ff -  K_fu * inv(K_uu) * K_uu * inv(K_uu) K_uf
        S = K_ff - Lambda.t() @ Lambda

        # Compute V := Lambda.t() @ q_cov, where V @ V.T = K_fu * inv(K_uu) * Sigma * inv(K_uu) K_uf
        V = Lambda.t() @ q_cov

        # Compute the predictive covariance
        pred_cov = S + V @ V.t()

        return pred_mean, pred_cov

    def forward(self, train_x, train_y):
        # Compute the predictive mean and covariance of train_x
        pred_mean, pred_cov = self.predict(train_x)

        # Compute expected log likelihood term
        likelihood_variance = torch.exp(self.likelihood_log_variance)
        log_likelihood = expected_log_likelihood(train_y, pred_mean.squeeze(), pred_cov, likelihood_variance)

        # Compute the whitened variational covariance matrix
        q_sqrt = torch.tril(torch.exp(0.5 * self.q_log_var))
        q_cov = q_sqrt.matmul(q_sqrt.t()) + 1e-6 * torch.eye(q_sqrt.size(0))

        # KL divergence term between q(u) and p(u)
        p_u = MultivariateNormal(torch.zeros(self.num_inducing), torch.eye(self.num_inducing))
        q_u = MultivariateNormal(self.q_mu.squeeze(), q_cov)
        KL = torch.distributions.kl.kl_divergence(q_u, p_u).div(train_x.size(0))

        return log_likelihood - KL