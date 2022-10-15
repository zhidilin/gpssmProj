""" define the multiple output GP model """
import sys
sys.path.append('../')
import torch
import gpytorch
import tqdm
from gpytorch.variational import CholeskyVariationalDistribution, LMCVariationalStrategy, IndependentMultitaskVariationalStrategy, VariationalStrategy
from gpytorch.models import  ApproximateGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from matplotlib import pyplot as plt
import numpy as np

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
                self.covar_module = ScaleKernel(MaternKernel(batch_shape=torch.Size([num_tasks]), ard_num_dims= inducing_points.shape[-1]),
                                            batch_shape=torch.Size([num_tasks]))
            else:
                self.covar_module = ScaleKernel(MaternKernel(batch_shape=torch.Size([num_tasks])),batch_shape=torch.Size([num_tasks]))

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output dimension in batch
        mean_x = self.mean_module(x) # + x[:, 0].transpose(-1, 1)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def kl_divergence(self):
        """Get the KL-Divergence of the Model."""
        return self.variational_strategy.kl_divergence()  # already sum over dim_state/num_latent



