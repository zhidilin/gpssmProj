"""
This file is to test the performance of the sparse GP model on a simple non-stationary regression task.
"""
import torch
import gpytorch
import torch.optim as optim
import matplotlib.pyplot as plt
# import sys
# directory_to_add = "/home/student2/zhidi/gpbl/PycharmProj/ETGPSSM/high_dim_GPSSM"
# if directory_to_add not in sys.path:
#     # 添加目录到 sys.path
#     sys.path.append(directory_to_add)
from high_dim_GPSSM.modules.gpModel import SparseGPModel
from high_dim_GPSSM.utils_h import reset_seed
reset_seed(0)
torch.set_default_dtype(torch.double)


def train_svgp(model, X_train, y_train, num_epochs=1000, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        elbo, kl_divergence = model(X_train, y_train)
        loss = -elbo  # We minimize the negative ELBO
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, ELBO: {-loss.item():.4f}, KL: {kl_divergence.item():.5f}')


def predict_svgp(model, X_star):
    model.eval()
    with torch.no_grad():
        pred_dist = model.predict(X_star)
    return pred_dist.mean.flatten(), pred_dist.variance.flatten()


class NonStationaryRBFKernel(gpytorch.kernels.Kernel):
    def __init__(self, **kwargs):
        super().__init__(has_lengthscale=True, **kwargs)
        self.base_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.base_kernel.outputscale = 5

    def forward(self, x1, x2, **params):
        # Calculate distance factor with a gentler modulation
        distance_factor_x1 = torch.exp(-0.5 * (x1 ** 2).sum(dim=-1).sqrt())
        distance_factor_x2 = torch.exp(-0.5 * (x2 ** 2).sum(dim=-1).sqrt())

        # Compute the base RBF kernel
        base_covariance = self.base_kernel(x1, x2)

        # Reshape distance factors for broadcasting
        distance_factor_x1 = distance_factor_x1.view(-1, 1)
        distance_factor_x2 = distance_factor_x2.view(1, -1)

        # Apply smoother non-stationary modulation
        return distance_factor_x1 * base_covariance * distance_factor_x2


def NonStationary_data(x):
    mean_module = gpytorch.means.ZeroMean()
    covar_module = NonStationaryRBFKernel()

    mean_x = mean_module(x)
    covar_x = covar_module(x)

    sampled_f = gpytorch.distributions.MultivariateNormal(mean_x, covar_x).sample()

    sampled_y = sampled_f + 0.2 * torch.randn(sampled_f.size())

    return sampled_f, sampled_y


# # Training data
# X_train = torch.linspace(-5, 5, 100)  # 1D inputs from -5 to 5
# # Get GP predictions and sample
# f, y_train = NonStationary_data(X_train)

# # Plot training data and samples
# plt.figure(figsize=(10, 5))
# plt.plot(X_train.numpy(), f.numpy(), 'k')
# plt.plot(X_train.numpy(), y_train.numpy(), 'b')
# plt.title('Samples from a Non-Stationary GP')
# plt.show()


# 定义基础的 kink function
def kink_func(x):
    return 0.8 + (x + 0.2) * (1 - 5 / (1 + torch.exp(-2 * x)))


# 增加局部抖动和变化的非平稳 kink function
def nonstationary_kink_func_with_jitter(x):
    # 原始的 kink function
    f_base = kink_func(x)

    # 减缓 x > 0 区域的下降速度
    # 如果 x > 0，则使用较小的下降系数
    slope_adjustment = torch.where(x > 0, 1 - 0.5 * torch.exp(-0.5 * x), 1.0)
    f_slowed = f_base * slope_adjustment

    # 在 x > 0 的区域增加更明显的抖动
    local_jitter = torch.where(x > 0, 0.5 * torch.sin(8 * x), 0.5 * torch.sin(2 * x))

    # 将抖动和噪声项叠加在调整过的 kink function 上
    f_nonstationary = f_slowed - local_jitter
    return f_nonstationary


X_f = torch.linspace(-6, 2, 100)
X_train = torch.linspace(-6, 2, 10)
# Compute original kink function and non-stationary kink function
f = nonstationary_kink_func_with_jitter(X_train)
y_train = f + .2 * torch.randn_like(X_train)

inducing_points = torch.randn(20)
model = SparseGPModel(inducing_points=inducing_points)

# Train model
model.train()
train_svgp(model, X_train, y_train, num_epochs=1000, lr=1e-2)

# Predict on new data
X_test = torch.linspace(-6, 6, 200)
mean, var = predict_svgp(model, X_test)

plt.figure(figsize=(10, 5))
plt.plot(X_f.numpy(), nonstationary_kink_func_with_jitter(X_f).numpy(), 'r', label='True function')
plt.plot(X_train.numpy(), y_train.numpy(), 'kx', label='Training Data')
plt.plot(X_test.numpy(), mean.numpy(), 'b', label='Stationary GP Mean')
plt.fill_between(X_test.numpy(),
                 (mean - 2 * torch.sqrt(var)).numpy(),
                 (mean + 2 * torch.sqrt(var)).numpy(),
                 color='blue', alpha=0.2, label='95% UI')
plt.legend()
plt.show()
