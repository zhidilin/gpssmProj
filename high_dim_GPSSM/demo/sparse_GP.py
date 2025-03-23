import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from high_dim_GPSSM.modules.gpModel import SparseGPModel
from high_dim_GPSSM.utils_h import reset_seed
reset_seed(0)

def train_svgp(model, X_train, y_train, num_epochs=1000, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        elbo = model(X_train, y_train)
        loss = -elbo  # We minimize the negative ELBO
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, ELBO: {-loss.item():.4f}')


def predict_svgp(model, X_star):
    model.eval()
    with torch.no_grad():
        pred_dist = model.predict(X_star)
    return pred_dist.mean.flatten(), pred_dist.variance.flatten()


# Generate synthetic data
X_train = torch.linspace(0, 1, 100)
y_train = torch.sin(X_train * (2 * np.pi)) + 0.2 * torch.randn(X_train.size())

inducing_points = torch.randn(20)
model = SparseGPModel(inducing_points=inducing_points)

# Train model
model.train()
train_svgp(model, X_train, y_train, num_epochs=3000, lr=1e-2)

# Predict on new data
X_test = torch.linspace(0, 1.5, 200)
mean, var = predict_svgp(model, X_test)


plt.figure(figsize=(10, 5))
plt.plot(X_train.numpy(), y_train.numpy(), 'kx', label='Training Data')
plt.plot(X_test.numpy(), mean.numpy(), 'b', label='Predictive Mean')
plt.fill_between(X_test.numpy(),
                 (mean - 2 * torch.sqrt(var)).numpy(),
                 (mean + 2 * torch.sqrt(var)).numpy(),
                 color='blue', alpha=0.2, label='Predictive Variance')
plt.legend()
plt.show()
