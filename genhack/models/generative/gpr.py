import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, RationalQuadratic
from torch import nn
from sklearn.gaussian_process import GaussianProcessRegressor
import torch
from operator import mul


class GPR(nn.Module):

    def __init__(self, datamodule, n_dim, n_latent_dim, gpr_kernels, *args, **kwargs) -> None:
        super().__init__()
        self.n_dim = n_dim
        self.n_latent_dim = n_latent_dim

        print(gpr_kernels[1])
        self.gpr_kernel = globals()[gpr_kernels[1]]
        print(self.gpr_kernel)
        
        '''
        Maybe we could somehow create a way to compose general kernels together
        One Idea that isn't really working
        self.gpr_kernels = [globals()[i] for i in gpr_kernels]
        self.arguments = [(1, "fixed"), (10, "fixed")]
        self.arg_kernels = np.array([func(val) for func, val in zip(self.gpr_kernels, self.arguments)])
        self.kernel = lambda x : np.prod([k(x) for k in self.arg_kernels])  
        '''
        ssts, positions, times = datamodule.train_dataset[:]
        self.gprs = []


        for sst, position in zip(ssts, positions):
            kernel = ConstantKernel(1., constant_value_bounds="fixed") * self.gpr_kernel(10., length_scale_bounds="fixed")

            #kernel = ConstantKernel(1., constant_value_bounds="fixed") * RationalQuadratic(10., length_scale_bounds="fixed")
            #kernel = ConstantKernel(1., constant_value_bounds="fixed") * RBF(10., length_scale_bounds="fixed")
            #kernel = ConstantKernel(1., constant_value_bounds="fixed") * Matern(10., length_scale_bounds="fixed")
            gpr = GaussianProcessRegressor(kernel=kernel)
            gpr.fit(position.reshape(2, -1).T, sst)
            self.gprs.append(gpr)

    def sample(self, noise, position, *args, **kwargs):
        y_samples = []

        for i, z in enumerate(noise):
            gpr = self.gprs[i]
            y_mean, y_cov = gpr.predict(position.reshape(2, -1).T, return_cov=True)

            # add small perturbation, since matrix often ends up being singular
            y_cov += 1e-4 * np.eye(y_cov.shape[0])
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
