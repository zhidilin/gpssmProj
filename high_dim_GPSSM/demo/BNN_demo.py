import torch
import torch.nn as nn
import torch.optim as optim
from high_dim_GPSSM.modules import torchbnn as bnn
import matplotlib.pyplot as plt
from high_dim_GPSSM.utils_h import reset_seed
# Ensure reproducibility
reset_seed(0)


# Helper function to create synthetic regression data
def create_synthetic_data(n=100):
    X = torch.linspace(-5, 5, n).unsqueeze(1)
    y1 = torch.sin(X) + 0.2 * torch.randn(X.size())
    y2 = torch.cos(X) + 0.2 * torch.randn(X.size())
    y = torch.cat((y1, y2), dim=1)
    return X, y


# Define function to predict and visualize uncertainty for 2-dimensional output
def predict_and_plot(model, X_test, X_train, y_train, num_samples=100):
    model.eval()
    with torch.no_grad():
        # Perform multiple stochastic forward passes
        preds = torch.stack([model(X_test) for _ in range(num_samples)])  # Shape: [num_samples, batch_size, output_dim]

        # Mean and standard deviation of predictions for each output dimension
        pred_mean = preds.mean(dim=0)  # Shape: [batch_size, output_dim]
        pred_std = preds.std(dim=0)  # Shape: [batch_size, output_dim]

        # Number of output dimensions
        output_dim = y_train.shape[1]

        # Create subplots
        fig, axs = plt.subplots(output_dim, 1, figsize=(10, 6 * output_dim))

        for i in range(output_dim):
            ax = axs[i] if output_dim > 1 else axs  # Handle cases where output_dim == 1 (1D output)

            # Plot training data
            ax.scatter(X_train.numpy(), y_train[:, i].numpy(), c='blue', label='Training Data')

            # Plot predictive mean
            ax.plot(X_test.numpy(), pred_mean[:, i].numpy(), c='red', label='Predictive Mean')

            # Plot uncertainty (2 standard deviations)
            ax.fill_between(X_test.squeeze().numpy(),
                            (pred_mean[:, i] - 2 * pred_std[:, i]).squeeze().numpy(),
                            (pred_mean[:, i] + 2 * pred_std[:, i]).squeeze().numpy(),
                            alpha=0.3, color='orange', label='Uncertainty (2 std dev)')

            # Set plot title and labels
            ax.set_title(f'Bayesian Neural Network Regression (Output Dimension {i + 1})')
            ax.set_xlabel('X')
            ax.set_ylabel(f'y[{i + 1}]')
            ax.legend()

        # Show plot
        plt.tight_layout()
        plt.show()

x, y = create_synthetic_data(n=500)
x_test = torch.linspace(-6, 6, 200).unsqueeze(1)  # Test data

# Define Bayesian Neural Network (BNN) model
model = nn.Sequential(
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.12, in_features=1, out_features=128),
    nn.ReLU(),
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.12, in_features=128, out_features=64),
    nn.ReLU(),
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.12, in_features=64, out_features=2),
)

mse_loss = nn.MSELoss()
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
kl_weight = 1

optimizer = optim.Adam(model.parameters(), lr=0.01)

for step in range(3000):
    pre = model(x)
    mse = mse_loss(pre, y)
    kl = kl_loss(model)
    cost = mse + kl_weight * kl

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # Print progress
    if step % 100 == 0:
        print(f'Epoch {step}: NLL Loss = {mse.item():.4f}, KL Div = {kl.item():.4f}, ELBO Loss = {cost.item():.4f}')
        if step % 500 == 0:
            predict_and_plot(model, x_test, x, y, num_samples=100)

predict_and_plot(model, x_test, x, y, num_samples=100)