"""
https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/09-normalizing-flows.html
"""

from nflows.transforms import MaskedAffineAutoregressiveTransform, CompositeTransform, AffineCouplingTransform
from nflows.transforms.permutations import ReversePermutation, RandomPermutation
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from torch import nn, optim
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

    def __init__(self, n_layers, n_dim, n_latent_dim, n_hidden_features, dropout_probability=0.0, use_batch_norm=False, weights='LearnedWeights', weights_kwargs=None, *args, **kwargs):
        """Note that you can disable weighting by using PowerLawWeights with c = 0."""
        super().__init__()
        self.n_layers = n_layers
        self.n_dim = n_dim
        self.n_latent_dim = n_latent_dim
        self.n_hidden_features = n_hidden_features
        self.use_batch_norm = use_batch_norm
        self.dropout_probability = dropout_probability

        transforms = []

        for _ in range(self.n_layers):
            transforms.append(ReversePermutation(features=self.n_dim))
            transforms.append(MaskedAffineAutoregressiveTransform(
                features=self.n_dim,
                hidden_features=self.n_hidden_features,
                use_batch_norm=self.use_batch_norm,
                dropout_probability=self.dropout_probability,
            ))

        transform = CompositeTransform(transforms)

        # Define a base distribution.
        base_distribution = StandardNormal(shape=[self.n_dim])

        # Combine into a flow. (For normalizing flows, see arXiv:1912.02762)
        self.flow = Flow(transform=transform, distribution=base_distribution)

        # initialize weights
        self.weights = weight_models[weights](**weights_kwargs)

    def forward(self, inputs):
        inputs, time = inputs
        return inputs, time

    def sample(self, noise):
        samples, _ = self.flow._transform.inverse(noise)
        return samples

    def loss(self, *args, **kwargs):
        inputs, time = args
        weights = self.weights(time)
        return {'loss': torch.mean(-weights * self.flow.log_prob(inputs=inputs))}
