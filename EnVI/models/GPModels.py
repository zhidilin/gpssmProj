import gpytorch
import torch
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import IndependentMultitaskVariationalStrategy


class IndependentMultitaskGPModel(ApproximateGP):
    def __init__(self, dim_state, inducing_points):

        # task 数量等于 latent state 的数量
        self.state_dim = dim_state

        ### inducing_points = torch.rand(num_tasks, 16, 1)

        # Let's use same set of inducing points for each task
        # inducing_points shape: dim_state x num_ips x (dim_state + dim_input)
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
        self.mean_module = ConstantMean(batch_shape=torch.Size([self.state_dim]))
        # self.mean_module = ZeroMean(batch_shape=torch.Size([self.state_dim]))
        # self.covar_module = ScaleKernel(MaternKernel(batch_shape=torch.Size([self.state_dim])),
        #                                 batch_shape=torch.Size([self.state_dim])
        #                                 )
        self.covar_module = ScaleKernel(RBFKernel(batch_shape=torch.Size([self.state_dim])),
                                        batch_shape=torch.Size([self.state_dim])
                                        )


    def forward(self, x):
        # The forward function should be written as if we were dealing with each output dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


    def condition_u(self, x, U):
        """

        Parameters
        ----------
        x shape: N_MC x state_dim x batch_size x (input_dim + state_dim)
        U shape: N_MC x state_dim x num_ips,

        Returns
        -------

        """

        state_dim = self.state_dim

        assert len(x.shape) == 4, 'Bad input x, expected shape: N_MC x state_dim x batch_size x (input_dim + state_dim)'

        N_MC, _, _, x_dim = x.shape   # x shape: N_MC x state_dim x batch_size x (input_dim + state_dim)

        ''' ------------------    get inducing points U and inducing inputs Z    ------------------------  '''
        # U = U.permute(1, 0, 2)    # shape: state_dim x N_MC x num_ips
        _, num_ips, _ = self.variational_strategy.inducing_points.shape
        inducing_points = self.variational_strategy.inducing_points

        # shape [N_MC x state_dim x num_ips x (state_dim + input_dim)]
        inducing_points = inducing_points.expand(N_MC, state_dim, num_ips, x_dim)

        # shape:  N_MC x state_dim x (num_ips + batch_size) x (state_dim + input_dim)
        full_input = torch.cat([inducing_points, x], dim=-2)

        # MultivariateNormal shape:  N_MC x state_dim x (num_ips + batch_size)
        full_output = self.forward(full_input)
        # print(f"\n type of full_output: {type(full_output)}, \n batch_shape:{full_output.batch_shape}, \n event_shape:{full_output.event_shape}")

        full_covar = full_output.lazy_covariance_matrix

        induc_mean = full_output.mean[..., :num_ips] # shape: N_MC x state_dim x num_ips
        test_mean = full_output.mean[..., num_ips:]  # shape: N_MC x state_dim x batch_size

        # Covariance terms
        # shape: N_MC x state_dim x num_ips x num_ips
        induc_induc_covar = full_covar[..., :num_ips, :num_ips].add_jitter().evaluate()

        # shape: N_MC x state_dim x num_ips x batch_size
        induc_data_covar = full_covar[..., :num_ips, num_ips:].evaluate()

        # shape: N_MC x state_dim x batch_size x batch_size
        data_data_covar = full_covar[..., num_ips:, num_ips:].add_jitter().evaluate()

        # result term: tmp shape: N_MC x state_dim x batch_size x num_ips
        # tmp = torch.matmul(induc_data_covar.transpose(-1, -2), torch.inverse(induc_induc_covar))
        L = torch.cholesky(induc_induc_covar)
        tmp = induc_data_covar.transpose(-1, -2) @ torch.cholesky_inverse(L)

        # shape: N_MC x state_dim x num_ips
        residue =  U - induc_mean

        # shape: N_MC x state_dim x batch_size x batch_size
        f_condition_u_mean = test_mean + torch.matmul(tmp, residue.unsqueeze(dim=-1)).squeeze(dim=-1)
        f_condition_u_cov = data_data_covar - torch.matmul(tmp, induc_data_covar)

        var = f_condition_u_cov.diagonal(offset=0,dim1=-2,dim2=-1)  # shape: N_MC x state_dim x batch_size x batch_size

        return MultivariateNormal(f_condition_u_mean, torch.diag_embed(var)).add_jitter()



    def kl_divergence(self):
        """Get the KL-Divergence of the Model."""
        return self.variational_strategy.kl_divergence().sum()
