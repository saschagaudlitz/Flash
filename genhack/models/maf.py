"""
https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/09-normalizing-flows.html
"""

from nflows.flows import MaskedAutoregressiveFlow
from scipy.stats import norm
from torch import nn
import torch


class PowerLawWeights(nn.Module):

    def __init__(self, a=0.9, b=0.1, c=1., *args, **kwargs):
        """(a * t + b) ** c"""
        super().__init__()
        self.a = a
        self.b = b
        self.c = c

    def forward(self, input):
        return (self.a * input + self.b) ** self.c


class LearnableWeights(nn.Module):

    def __init__(self, pts=64, n_hidden_units=100, *args, **kwargs):
        """
        Learns non-linear function [0,1]->[0,inf] which integrates to one.

        Parameters
        ----------
        pts : int
            Integration points for calculation of the normalizing constant, so the weights integrate to one
        n_hidden_units
            Number of hidden units in the weight function
        """
        super().__init__()
        self.pts = pts
        self.n_hidden_units = n_hidden_units
        self.model = nn.Sequential(
            nn.Linear(1, n_hidden_units),
            nn.LeakyReLU(),
            nn.Linear(n_hidden_units, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        dt = 1 / self.pts
        normalize = self.model(torch.arange(0, 1, dt)[:, None]).sum() * dt
        return self.model(input[:, None]).reshape(-1) / normalize


weight_models = {
    'LearnableWeights': LearnableWeights,
    'PowerLawWeights': PowerLawWeights,
}


class MAF(nn.Module):

    def __init__(self, trend_factor, n_layers, n_dim, n_latent_dim, n_hidden_features, n_blocks, dropout_probability=0.0, use_batch_norm=False, use_random_permutations=False, use_random_masks=False, weights='LearnedWeights', weights_kwargs=None, *args, **kwargs):
        """Note that you can disable weighting by using PowerLawWeights with c = 0."""
        super().__init__()
        self.trend_factor = trend_factor
        self.n_layers = n_layers
        self.n_dim = n_dim
        self.n_latent_dim = n_latent_dim
        self.n_hidden_features = n_hidden_features
        self.n_blocks = n_blocks
        self.use_batch_norm = use_batch_norm
        self.use_random_permutations = use_random_permutations
        self.use_random_masks = use_random_masks
        self.dropout_probability = dropout_probability

        # initialize flow
        self.flow = MaskedAutoregressiveFlow(features=self.n_dim,
                                             num_layers=self.n_layers,
                                             hidden_features=self.n_hidden_features,
                                             num_blocks_per_layer=self.n_blocks,
                                             use_random_permutations=self.use_random_permutations,
                                             use_random_masks=self.use_random_masks,
                                             dropout_probability=self.dropout_probability,
                                             batch_norm_within_layers=self.use_batch_norm)

        # initialize weights
        self.weights = weight_models[weights](**weights_kwargs)

    def forward(self, inputs):
        inputs, time = inputs
        return inputs, time

    def sample(self, noise, t_min, t_max):
        # noise and time samples
        time = t_min + (t_max - t_min) * norm.cdf(noise[:, 6])
        noise = noise[:, :6]

        # entrend
        intercept = torch.tensor([-0.3, -0.27, -0.41, -0.39, -0.45, -0.68])
        trend = self.trend_factor * torch.tensor([0.56, 0.46, 0.83, 0.78, 0.89, 1.36])
        samples, _ = self.flow._transform.inverse(noise)
        return intercept[None, :] + time[:, None] * trend[None, :] + torch.squeeze(samples)

    def loss(self, *args, **kwargs):
        inputs, time = args
        weights = self.weights(time)
        return {'loss': torch.mean(-weights * self.flow.log_prob(inputs=inputs))}
