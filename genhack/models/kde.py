import torch
from scipy.stats import gaussian_kde
from torch import nn


class KDE(nn.Module):
    def __init__(self, n_dim, bw_method, datamodule, *args, **kwargs):
        super().__init__()

        self.n_dim = n_dim
        self.bw_method = bw_method

        inputs = datamodule.train_dataset[:][0].T
        self.kde = gaussian_kde(inputs, bw_method=self.bw_method)

    def sample(self, n_samples):
        return torch.Tensor(self.kde.resample(n_samples).T)
