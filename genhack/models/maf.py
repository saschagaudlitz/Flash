from nflows.transforms import MaskedAffineAutoregressiveTransform, CompositeTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from torch import nn


class MAF(nn.Module):

    def __init__(self, n_layers, n_dim, n_hidden_features, *args, **kwargs):
        super().__init__()

        self.n_layers = n_layers
        self.n_dim = n_dim
        self.n_hidden_features = n_hidden_features

        transforms = []

        for _ in range(self.n_layers):
            transforms.append(ReversePermutation(features=self.n_dim))
            transforms.append(MaskedAffineAutoregressiveTransform(features=self.n_dim, hidden_features=self.n_hidden_features))
        transform = CompositeTransform(transforms)

        # Define a base distribution.
        base_distribution = StandardNormal(shape=[self.n_dim])

        # Combine into a flow. (For normalizing flows, see arXiv:1912.02762)
        self.flow = Flow(transform=transform, distribution=base_distribution)

    def forward(self, input):
        return input

    def sample(self, n_samples):
        return self.flow.sample(n_samples)

    def loss(self, input):
        return {'loss': -self.flow.log_prob(inputs=input).mean()}
