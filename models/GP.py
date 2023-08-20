import gpytorch
import torch
import torch.nn as nn
import torch.distributions as td
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D
from gpytorch.variational import CholeskyVariationalDistribution, LMCVariationalStrategy, IndependentMultitaskVariationalStrategy, VariationalStrategy
from gpytorch.models import  ApproximateGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from . import utils as cg

class SparseGPModel(ApproximateGP):
    def __init__(self, inducing_points, ARD=False):
        state_dim = inducing_points.shape[-1]
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(-2))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution,
                                                   learn_inducing_locations=True)
        super().__init__(variational_strategy)
        self.mean_module = ZeroMean()
        if ARD:
            self.covar_module = ScaleKernel(MaternKernel(ard_num_dims=state_dim))
        else:
            self.covar_module = ScaleKernel(MaternKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def kl_divergence(self):
        """Get the KL-Divergence of the Model."""
        return self.variational_strategy.kl_divergence()


class MultitaskGPModel(ApproximateGP):
    def __init__(self, inducing_points, num_tasks, num_latents=None, MoDep=False, ARD=False):
        # Let's use a different set of inducing points for each latent function
        self.MoDep = MoDep

        if self.MoDep:
            assert (num_latents is not None), 'please specify the number of latent GP for output dependent GP'
            # assert (inducing_points.shape[0] == num_latents), 'please make sure the dimensionality of inducing points is well specified'

            # mark the CholeskyVariationalDistribution as batch so that we learn a variational distribution for each task
            variational_distribution = CholeskyVariationalDistribution(num_inducing_points=inducing_points.size(-2),
                                                                       batch_shape=torch.Size([num_latents])
                                                                       )

            #  wrap the VariationalStrategy in a LMCVariationalStrategy,
            #  so that the output will be a MultitaskMultivariateNormal rather than a batch output
            variational_strategy = LMCVariationalStrategy(
                        VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True),
                        num_tasks=num_tasks,
                        num_latents=num_latents,
                        latent_dim=-1
                    )

            super().__init__(variational_strategy)

            # The mean and covariance modules should be marked as batch, so we learn a different set of hyperparameters
            self.mean_module = ZeroMean(batch_shape=torch.Size([num_latents]))
            # kernel_list = [gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])), gpytorch.kernels.PeriodicKernel(batch_shape=torch.Size([num_latents]))]
            # self.covar_module = gpytorch.kernels.LCMKernel(base_kernels=kernel_list, num_tasks=num_tasks)
            if ARD:
                self.covar_module = ScaleKernel(MaternKernel(batch_shape=torch.Size([num_latents]), ard_num_dims= inducing_points.shape[-1]),
                                            batch_shape=torch.Size([num_latents]))
            else:
                self.covar_module = ScaleKernel(MaternKernel(batch_shape=torch.Size([num_latents])), batch_shape=torch.Size([num_latents]))

        else:
            # Let's use a different set of inducing points for each task
            # inducing_points = torch.rand(num_tasks, 16, 1)
            # assert (inducing_points.shape[0] == num_tasks), 'please make sure the dimensionality of inducing points is well specified'

            # We have to mark the CholeskyVariationalDistribution as batch,
            # so that we learn a variational distribution for each task
            variational_distribution = CholeskyVariationalDistribution(num_inducing_points=inducing_points.size(-2),
                                                                       batch_shape=torch.Size([num_tasks])
                                                                       )

            variational_strategy = IndependentMultitaskVariationalStrategy(
                VariationalStrategy( self, inducing_points, variational_distribution, learn_inducing_locations=True),
                num_tasks=num_tasks,
                task_dim=-1
            )

            super().__init__(variational_strategy)

            # The mean and covariance modules should be marked as batch, so we learn a different set of hyperparameters
            self.mean_module = ZeroMean(batch_shape=torch.Size([num_tasks]))
            if ARD:
                self.covar_module = ScaleKernel(
                    MaternKernel(batch_shape=torch.Size([num_tasks]), ard_num_dims= inducing_points.shape[-1]),
                    batch_shape=torch.Size([num_tasks]))
            else:
                self.covar_module = ScaleKernel(MaternKernel(batch_shape=torch.Size([num_tasks])),batch_shape=torch.Size([num_tasks]))

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def kl_divergence(self):
        """Get the KL-Divergence of the Model."""
        return self.variational_strategy.kl_divergence()  # already sum over dim_state/num_latent


class IndependentMultitaskGPModel(ApproximateGP):
    def __init__(self, state_dim, inducing_points, ARD=False):

        # task 数量等于 latent state 的数量
        self.state_dim = state_dim

        # inducing_points shape: dim_state x num_ips x (state_dim + input_dim)
        assert(self.state_dim == inducing_points.shape[0])
        num_ips = inducing_points.size(-2)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=num_ips,
                                                                   batch_shape=torch.Size([self.state_dim])
                                                                   )

        variational_strategy = VariationalStrategy(self,
                                                   inducing_points=inducing_points,
                                                   variational_distribution=variational_distribution,
                                                   learn_inducing_locations=True
                                                   )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = ZeroMean(batch_shape=torch.Size([self.state_dim]))
        if ARD:
            self.covar_module = ScaleKernel(
                MaternKernel(batch_shape=torch.Size([self.state_dim]), ard_num_dims= inducing_points.shape[-1]),
                batch_shape=torch.Size([self.state_dim]))
        else:
            self.covar_module = ScaleKernel(MaternKernel(batch_shape=torch.Size([self.state_dim])),
                                            batch_shape=torch.Size([self.state_dim]))
        # self.covar_module = ScaleKernel(RBFKernel(batch_shape=torch.Size([self.state_dim])),
        #                                 batch_shape=torch.Size([self.state_dim])
        #                                 )


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def kl_divergence(self):
        """Get the KL-Divergence of the Model."""
        return self.variational_strategy.kl_divergence()


# define likelihood module
class GaussianNonLinearMean(nn.Module):
    """Place a GP over the mean of a Gaussian likelihood $p(y|G(f))$
    with noise variance $\sigma^2$ and with a NON linear transformation $G$ over $f$.
    It supports multi-output (independent) GPs and the possibility of sharing
    the noise between the different outputs. In this case integrations wrt to a
    Gaussian distribution can only be done with quadrature."""

    def __init__(self, out_dim: int, noise_init: float, noise_is_shared: bool, quadrature_points: int):
        super(GaussianNonLinearMean, self).__init__()

        self.out_dim = out_dim
        self.noise_is_shared = noise_is_shared

        if noise_is_shared:  # if noise is shared create one parameter and expand to out_dim shape
            log_var_noise = nn.Parameter(torch.ones(1, 1, dtype=cg.dtype) * torch.log(torch.tensor(noise_init, dtype=cg.dtype)))

        else:  # creates a vector of noise variance parameters.
            log_var_noise = nn.Parameter(torch.ones(out_dim, 1, dtype=cg.dtype) * torch.log(torch.tensor(noise_init, dtype=cg.dtype)))

        self.log_var_noise = log_var_noise

        self.quad_points = quadrature_points
        self.quadrature_distribution = GaussHermiteQuadrature1D(quadrature_points)

    ##  Log Batched Multivariate Gaussian: log N(x|mu,C) ##
    def batched_log_Gaussian(self, obs: torch.tensor, mean: torch.tensor, cov: torch.tensor, diagonal: bool, cov_is_inverse: bool) -> torch.tensor:
        """
        Computes a batched of * log p(obs|mean,cov) where p(y|f) is a  Gaussian distribution, with dimensionality N.
        Returns a vector of shape *.
        -0.5*N log 2pi -0.5*\log|Cov| -0.5[ obs^T Cov^{-1} obs -2 obs^TCov^{-1} mean + mean^TCov^{-1}mean]
                Args:
                        obs            :->: random variable with shape (*,N)
                        mean           :->: mean -> matrix of shape (*,N)
                        cov            :->: covariance -> Matrix of shape (*,N) if diagonal=True else batch of matrix (*,N,N)
                        diagonal       :->: if covariance is diagonal or not
                        cov_is_inverse :->: if the covariance provided is already the inverse

        #TODO: Check argument shapes
        """

        N = mean.size(-1)
        cte = N * torch.log(2 * cg.pi.to(cg.device).type(cg.dtype))

        if diagonal:
            log_det_C = torch.sum(torch.log(cov), -1)
            inv_C = cov
            if not cov_is_inverse:
                inv_C = 1. / cov  # Inversion given a diagonal matrix. Use torch.cholesky_solve for full matrix.
            else:
                log_det_C *= -1  # switch sign

            exp_arg = (obs * inv_C * obs).sum(-1) - 2 * (obs * inv_C * mean).sum(-1) + (mean * inv_C * mean).sum(-1)

        else:
            raise NotImplemented("log_Gaussian for full covariance matrix is not implemented yet.")
        return -0.5 * (cte + log_det_C + exp_arg)


    def log_non_linear(self, f: torch.tensor, Y: torch.tensor, noise_var_: torch.tensor, flow):
        """ Return the log likelihood of S Gaussian distributions, each of this S correspond to a quadrature point.
            The only samples f have to be warped with the composite flow G().
            -> f is assumed to be stacked samples of the same dimension of Y. Here we compute (apply lotus rule):

          \int \log p(y|fK) q(fK) dfK = \int \log p(y|fk) q(f0) df0 \approx 1/sqrt(pi) sum_i w_i { \log[ p( y | G( sqrt(2)\sigma f_i + mu), sigma^2 ) ] };

          where q(f0) is the initial distribution. We just face the problem of computing the expectation under a log Gaussian of a
          non-linear transformation of the mean, given by the flow.

                Args:
                        `f`         (torch.tensor)  :->:  Minibatched - latent function samples in (S,Dy,MB), being S the number of quadrature points and MB the minibatch.
                                                          This is directly given by the gpytorch.GaussHermiteQuadrature1D method in this format and corresponds to
                                                          \sqrt(2)\sigma f_i + mu see https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature
                        `Y`         (torch.tensor)  :->:  Minibatched Observations in Dy x MB.
                        `noise_var` (torch.tensor)  :->:  Observation noise
                        'flow'      (CompositeFlow) :->:  Sequence of flows to be applied to each of the outputs
                        'X'         (torch.tensor)  :->:  Input locations used for input dependent flows. Has shape [Dy,S*MB,Dx] or shape [S*MB,Dx]. N
        """
        # assert len(flow) == self.out_dim, "This likelihood only supports a flow per output_dim. Got {} for Dy {}".format(self.out_dim, len(flow))

        S = self.quad_points
        MB = Y.size(1)
        Dy = self.out_dim

        Y = Y.view(Dy, MB, 1).repeat((S, 1, 1, 1))                      # shape: [S,Dy,MB,1]
        noise_var = noise_var_.view(Dy, MB, 1).repeat((S, 1, 1, 1))     # shape: [S,Dy,MB,1]

        fK = f.clone()
        # for idx, fl in enumerate(flow):
        #     # warp the samples
        #     fK[:, idx, :] = fl(f[:, idx, :])
        fK = flow(f.transpose(-1,-2))   # (S,MB,Dy)
        fK = fK.transpose(-1,-2)        # (S,Dy,MB)
        fK = fK.view(S, Dy, MB, 1)

        # TODO：为什么这玩意还要自己算，用MultivariateNormal.log_prob(y) 不行吗？？？  Done: 可以，但是速度慢了一倍。。。
        log_p_y = self.batched_log_Gaussian(obs=Y, mean=fK, cov=noise_var, diagonal=True, cov_is_inverse=False)  # (S,Dy,MB)
        # noise_cov = noise_var_.view(Dy, MB, 1).repeat((S, 1, 1, 1)).unsqueeze(dim=-1)     # shape: [S,Dy,MB,1, 1]
        # p_y = gpytorch.distributions.MultivariateNormal(mean=fK, covariance_matrix=noise_cov)
        # log_p_y = p_y.log_prob(Y) # (S,Dy,MB)
        # print((log_p_y_2 - log_p_y).sum())

        return log_p_y  # return (S,Dy,MB) so that reduction is done for S.


    def expected_log_prob(self, Y, gauss_mean, gauss_cov, flow):
        """ Expected Log Likelihood

            Computes E_q(f) [\log p(y|G(f))] = \int q(f) \log p(y|G(f)) df \approx with quadrature

                - Acts on batched form. Hence returns vector of shape (Dy,)

            Args:

                `Y`             (torch.tensor)  :->:  Labels representing the mean. Shape (Dy,MB)
                `gauss_mean`    (torch.tensor)  :->:  mean from q(f). Shape (Dy,MB)
                `gauss_cov`     (torch.tensor)  :->:  diagonal covariance from q(f). Shape (Dy,MB)
                `non_linearity` (list)          :->:  List of callable representing the non linearity applied to the mean.

        """

        # assert len(flow) == self.out_dim, "The number of callables representing non linearities is different from out_dim"

        if self.noise_is_shared:
            log_var_noise = self.log_var_noise.expand(self.out_dim, 1)    # shape: [self.out_dim, 1]
        else:
            log_var_noise = self.log_var_noise

        N = Y.size(1)
        C_y = torch.exp(log_var_noise).expand(-1, N)                     # shape: [self.out_dim, N]

        distribution = td.Normal(gauss_mean, gauss_cov.sqrt()) # Distribution of shape (dy, MB). Gpytorch samples from it

        log_likelihood_lambda = lambda f_samples: self.log_non_linear(f_samples, Y, C_y, flow)
        ELL = self.quadrature_distribution(log_likelihood_lambda, distribution)
        # ELL shape is Dy x N

        return ELL


    def marginal_moments(self, gauss_mean, gauss_cov, flow):
        """ Computes the moments of order 1 and non centered 2 of the observation model integrated out w.r.t a Gaussian with means and covariances.
            There is a non linear relation between the mean and integrated variable

            p(y|x) = \int p(y|G(f)) p(f) df

            - Note that p(f) can only be diagonal as this function only supports quadrature integration.
            - Moment1: \widehat{\mu_y} = \frac{1}{\sqrt\pi} \sum^S_{s=1} w_s \mathtt{G}[\sqrt2 \sigma f_s + \mu]
            - Moment2: \sigma^2_o + \frac{1}{\sqrt\pi} \sum^S_{s=1} w_s \mathtt{G}[\sqrt2 \sigma f_s + \mu]^2 - \widehat{\mu_y}^2

            Args:
                `gauss_mean`    (torch.tensor) :->: mean from q(f). Shape (Dy,MB)
                `gauss_cov`     (torch.tensor) :->: covariance from q(f). Shape (Dy,MB) if diagonal is True, else (Dy,MB,MB). For the moment only supports diagonal true
                `non_linearity` (list)         :->: List of callable representing the non linearity applied to the mean.
                `X`             (torch.tensor) :->: Input locations used by input dependent flows
        """

        if self.noise_is_shared:
            log_var_noise = self.log_var_noise.expand(self.out_dim, 1)
        else:
            log_var_noise = self.log_var_noise

        MB = gauss_mean.size(1)
        C_Y = torch.exp(log_var_noise).expand(-1, MB)  # shape (Dy,MB)

        def aux_moment1(f, _flow):
            # f.shape (S,Dy,MB)
            f_ = _flow(f.transpose(-1, -2))
            f = f_.transpose(-1, -2)

            # for idx, fl in enumerate(_flow):
            #     # warp the samples
            #     f[:, idx, :] = fl(f[:, idx, :])
            return f

        def aux_moment2(f, _flow):
            # f.shape (S,Dy,MB)
            # x.shape (Dy,MB) # pytorch automatically broadcast to sum over S inside the flow fl

            # for idx, fl in enumerate(_flow):
            #     # warp the samples
            #     f[:, idx, :] = fl(f[:, idx, :])
            f_ = _flow(f.transpose(-1, -2))
            f = f_.transpose(-1, -2)
            return f ** 2

        aux_moment1_lambda = lambda f_samples: aux_moment1(f_samples, flow)
        aux_moment2_lambda = lambda f_samples: aux_moment2(f_samples, flow)
        distr = td.Normal(gauss_mean, gauss_cov.sqrt())  # Distribution of shape (Dy,MB). Gpytorch samples from it

        m1 = self.quadrature_distribution(aux_moment1_lambda, distr)
        E_square_y = self.quadrature_distribution(aux_moment2_lambda, distr)
        m2 = C_Y + E_square_y - m1 ** 2

        return m1, m2


def ELBO(GP_model, G_matrix, likelihood, X, y, num_data, beta=1):
    """ Define the loss object: Evidence Lower Bound
     Args:
            GP_model: Variational sparse GP module
            G_matrix: flows module
            Likelihood: Likelihood module
            `X` (torch.tensor)  :->:  Inputs, shape: [MB, dx]
            `y` (torch.tensor)  :->:  Targets, shape: [dy, MB]

                ELBO = \int log p(y|f) q(f|u) q(u) df,du -KLD[q||p]

                Returns possitive loss, i.e: ELBO = ELL - KLD; ELL and KLD
            """

    assert len(X.shape) == 2, 'Invalid input X.shape'


    dy = y.shape[0]

    # output is a Gaussian distribution, shape [MB,]
    qf = GP_model(X)
    qf_mean =  qf.mean.expand(dy, -1)
    qf_variance = qf.variance.expand(dy, -1)

    # 2nd argument: mean from q(f). Shape (dy, MB)  #3rd argument: diagonal covariance from q(f). Shape (dy, MB)
    ELL = likelihood.expected_log_prob(y, qf_mean, qf_variance, flow=G_matrix)

    # ELL = ELL.sum(-1).div(y.shape[1])
    kl_divergence = GP_model.variational_strategy.kl_divergence().div(num_data / beta)
    ELBO = ELL.sum() - kl_divergence

    return ELBO


def predictive_distribution(model, likelihood, G_matrix, X: torch.tensor, diagonal: bool = True):
    """ This function computes the moments 1 and 2 from the predictive distribution.
        It also returns the posterior mean and covariance over latent functions.

        p(Y*|X*) = \int p(y*|G(f*)) q(f*,f|u) q(u) df*,df,du

            # Homoceodastic Gaussian observation model p(y|f)
            # GP variational distribution q(f)
            # G() represents a non-linear transformation

            Args:
                    `X`                (torch.tensor)  :->: input locations where the predictive is computed. Can have shape (MB,Dx)
                    `diagonal`         (bool)          :->: if true, samples are drawn independently. For the moment is always true.
                    `S_MC_NNet`        (int)           :->: Number of samples from the dropout distribution is fully_bayesian is true

            Returns:
                    `m1`       (torch.tensor)  :->:  Predictive mean with shape (Dy,MB)
                    `m2`       (torch.tensor)  :->:  Predictive variance with shape (Dy,MB). Takes None for classification likelihoods
                    `mean_q_f` (torch.tensor)  :->:  Posterior mean of q(f) with shape (Dy,MB,1) [same shape as returned by marginal_variational_qf]
                    `cov_q_f`  (torch.tensor)  :->:  Posterior covariance of q(f) with shape (Dy,MB,1) [same shape as returned by marginal_variational_qf]

    """
    assert len(X.shape) == 2, "Bad input specificaton"
    model.eval()
    G_matrix.eval()
    likelihood.eval()  # set parameters for eval mode. Batch normalization, dropout etc
    dy = likelihood.out_dim

    with torch.no_grad():
        if not diagonal:
            raise NotImplemented("This function does not support returning the predictive distribution with correlations")

        qf = model(X)
        qf_mean = qf.mean.expand(dy, -1)
        qf_variance = qf.variance.expand(dy, -1)

        MOMENTS = likelihood.marginal_moments(qf_mean, qf_variance, flow=G_matrix)
        # diagonal True always. Is an element only used by the sparse_MF_GP with SVI. Diag = False is used by standard GP's marginal likelihood

        m1, m2 = MOMENTS
    return m1, m2, qf.mean, qf.variance
