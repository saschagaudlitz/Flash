import argparse
import numpy as np
import mlflow
from nflows.transforms import MaskedAffineAutoregressiveTransform, CompositeTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from models.trainer import Trainer


class MAF(Trainer):

    def configure_model(self):

        transforms = []

        for _ in range(args.n_layers):
            transforms.append(ReversePermutation(features=args.n_dim))
            transforms.append(MaskedAffineAutoregressiveTransform(features=args.n_dim, hidden_features=args.n_hidden_features))
        transform = CompositeTransform(transforms)

        # Define a base distribution.
        base_distribution = StandardNormal(shape=[args.n_dim])

        # Combine into a flow.
        return Flow(transform=transform, distribution=base_distribution)

    def training_step(self, batch, batch_idx):
        self.optimizer.zero_grad()
        loss = -self.model.log_prob(inputs=batch[0]).mean()
        loss.backward()
        self.optimizer.step()

    def sample(self, model, n_samples):
        return model.sample(n_samples).detach().numpy()


def main(args):

    np.random.seed(1337)

    with mlflow.start_run() as run:
        trainer = MAF(args, run.info.run_id)
        mlflow.log_params(vars(args))
        trainer.train()
        trainer.test_step()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', type=int, default=10)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--n_dim', type=int, default=6)
    parser.add_argument('--n_hidden_features', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)

    args = parser.parse_args()
    main(args)
