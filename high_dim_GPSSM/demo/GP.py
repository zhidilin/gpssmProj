import torch
import numpy as np
from high_dim_GPSSM.modules.gpTorch import GPModel
from matplotlib import pyplot as plt
from high_dim_GPSSM.utils_h import reset_seed
reset_seed(0)

# Generate training data
train_x = torch.linspace(0, 1, 100)
train_y = torch.sin(train_x * (2 * np.pi)) + 0.2 * torch.randn(train_x.size())

# Initialize the model
model = GPModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Training loop
num_iterations = 1000
for i in range(num_iterations):
    optimizer.zero_grad()
    # Negative log marginal likelihood (loss)
    nll = model.neg_log_likelihood(train_x, train_y)
    nll.backward()
    optimizer.step()

    if i % 10 == 0:
        print(f"Iteration {i + 1}/{num_iterations} - Loss: {nll.item()}")

print("Training completed.")

# Test points
test_x = torch.linspace(0, 1.5, 100)

# Make predictions
model.eval()
with torch.no_grad():
    pred_mean, pred_var = model(train_x, train_y, test_x)

# Convert tensors to numpy arrays for plotting
train_x_np = train_x.numpy()
train_y_np = train_y.numpy()
test_x_np = test_x.numpy()
pred_mean_np = pred_mean.detach().numpy()
pred_var_np = pred_var.detach().numpy()


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(train_x_np, train_y_np, 'k*', label='Training data')
plt.plot(test_x_np, pred_mean_np, 'b', label='Mean prediction')
plt.fill_between(test_x_np,
                 pred_mean_np - 2 * np.sqrt(pred_var_np),
                 pred_mean_np + 2 * np.sqrt(pred_var_np),
                 color='blue', alpha=0.2, label='95% confidence interval')
plt.legend()
plt.title('Gaussian Process Regression')
plt.show()