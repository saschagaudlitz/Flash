import torch
from torch import nn
from torch.distributions import Normal


class CDF(nn.Module):
    def __init__(self, n_dim, datamodule, *args, **kwargs):
        super().__init__()
        self.n_dim = n_dim

        inputs = datamodule.train_dataset[:][0]
        self.order = torch.sort(inputs, dim=0).values

    def sample(self, n_samples):
        norm = Normal(0, 1)
        noise = norm.cdf(torch.randn(n_samples, 6))
        idx = torch.floor(noise * len(self.order)).int()
        return torch.Tensor([[self.order[idx[i][j]][j] for j in range(idx.shape[1])] for i in range(idx.shape[0])])
