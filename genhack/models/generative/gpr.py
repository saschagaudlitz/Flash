import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from torch import nn
from sklearn.gaussian_process import GaussianProcessRegressor
import torch


class GPR(nn.Module):

    def __init__(self, datamodule, n_dim, n_latent_dim, *args, **kwargs) -> None:
        super().__init__()
        self.n_dim = n_dim
        self.n_latent_dim = n_latent_dim

        ssts, positions, times = datamodule.train_dataset[:]
        self.gprs = []

        for sst, position in zip(ssts, positions):
            kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
            gpr = GaussianProcessRegressor(kernel=kernel)
            gpr.fit(position.reshape(2, -1).T, sst)
            self.gprs.append(gpr)

    def sample(self, noise, position, *args, **kwargs):

        y_samples = []

        for i, z in enumerate(noise):
            gpr = self.gprs[i]
            y_mean, y_cov = gpr.predict(position.reshape(2, -1).T, return_cov=True)
            b = np.linalg.cholesky(y_cov)
            y_samples.append(y_mean + np.dot(b, z))

        return torch.tensor(np.array(y_samples))

    def mean_cov(self, noise, position, *args, **kwargs):

        y_samples = []

        for i, z in enumerate(noise):
            gpr = self.gprs[i]
            y_mean, y_cov = gpr.predict(position.reshape(2, -1).T, return_cov=True)
            b = np.linalg.cholesky(y_cov)
            y_samples.append(y_mean + np.dot(b, z))

        return torch.tensor(np.array(y_samples))
