from torch import nn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
import torch


def process_kernels_params(config):
    formula = config['formula']
    for kernel in config['kernels']:
        wrap = lambda x: f'"{x}"' if isinstance(x, str) else str(x)
        kwargs = ", ".join([f"{k}={wrap(v)}" for k, v in kernel['kwargs'].items()])
        string = f"{kernel['class_name']}({kwargs})"
        formula = formula.replace(kernel['name'], string)

    return eval(formula)


class GPR(nn.Module):

    def __init__(self, datamodule, n_dim, n_latent_dim, kernel_params, *args, **kwargs) -> None:
        super().__init__()
        self.n_dim = n_dim
        self.n_latent_dim = n_latent_dim

        ssts, positions, times = datamodule.train_dataset[:]
        self.gprs = []

        for sst, position in zip(ssts, positions):
            kernel = process_kernels_params(kernel_params)
            gpr = GaussianProcessRegressor(kernel=kernel)
            gpr.fit(position.reshape(2, -1).T, sst)
            self.gprs.append(gpr)

    def sample(self, noise, position, start_day=0, *args, **kwargs):

        y_samples = []

        for i, z in zip(range(start_day, start_day + len(noise)), noise):
            gpr = self.gprs[i]
            y_mean, y_cov = gpr.predict(position.reshape(2, -1).T, return_cov=True)

            # add small perturbation, since matrix often ends up being singular
            y_cov += 1e-7 * np.eye(y_cov.shape[0])
            b = np.linalg.cholesky(y_cov)
            y_samples.append(y_mean + np.dot(b, z))

        return torch.tensor(np.array(y_samples))
