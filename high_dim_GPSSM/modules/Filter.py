import numpy as np

def ensemble_kalman_filter(y_obs, particles, A, noise_cov, n_samples, bias=0.0):
    """
    Perform Ensemble Kalman Filter (EnKF) for posterior sampling.

    Args:
        y_obs: Observation (1D array).
        particles: Initial ensemble of particles (n_samples x n_dim).
        A: Observation matrix (2D array).
        noise_cov: Observation noise covariance (2D array).
        n_samples: Number of particles.
        bias: Optional bias term for the observation model.

    Returns:
        posterior_particles: Posterior ensemble of particles (n_samples x n_dim).
    """
    n_dim = particles.shape[1]  # State dimension
    H = A  # Observation operator

    # Step 1: Compute ensemble mean and anomalies
    ensemble_mean = np.mean(particles, axis=0)  # Mean of the ensemble
    anomalies = particles - ensemble_mean  # Ensemble anomalies

    # Step 2: Compute ensemble covariance
    ensemble_cov = anomalies.T @ anomalies / (n_samples - 1)  # Covariance from ensemble

    # Step 3: Compute innovation covariance and Kalman gain
    S = H @ ensemble_cov @ H.T + noise_cov  # Innovation covariance # (n_obs x n_obs)
    K = ensemble_cov @ H.T @ np.linalg.inv(S)  # Kalman gain (using pseudo-inverse for stability) # (n_dim x n_obs)

    # Step 4: Generate observation perturbations
    obs_perturbations = np.random.multivariate_normal(np.zeros_like(y_obs), noise_cov, n_samples).T

    # Step 5: Update ensemble
    y_ensemble = H @ particles.T + bias - obs_perturbations  # Perturbed observations (n_obs x n_samples)
    posterior_particles = particles.T + K @ (y_obs.reshape(-1, 1) - y_ensemble)  # Update particles (n_dim x n_samples)
    posterior_particles = posterior_particles.T  # Transpose back to (n_samples x n_dim)

    return posterior_particles
