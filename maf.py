import argparse
from nflows.transforms import MaskedAffineAutoregressiveTransform, CompositeTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from trainer import Trainer
from utils import main

class MAF(Trainer):
    """
    MAF trainer class - Implements Trainer for Normalizing flows
    
    ...

    Attributes
    ----------
    Attributes Trainer 
    -> (args, run_id, X_train, X_val, X_test, model, optimizer, ri_true, best_kendall, best_ad_mean ) 

    
    Methods
    -------
    -> 
    configure_model : REQUIRED method for model configuration
    sample          : REQUIRED method for sampling from the model
    training_step   : REQUIRED implementing the training step

    """
    def configure_model(self):
        """Create and return a Normalizing flow model
        """
        transforms = []
        for _ in range(self.args['n_layers']):
            transforms.append(ReversePermutation(features=self.args['n_dim']))
            transforms.append(MaskedAffineAutoregressiveTransform(features=self.args['n_dim'], hidden_features=self.args['n_hidden_features']))
        transform = CompositeTransform(transforms)

        # Define a base distribution.
        base_distribution = StandardNormal(shape=[self.args['n_dim']])

        # Combine into a flow. (For normalizing flows, see arXiv:1912.02762)
        return Flow(transform=transform, distribution=base_distribution)

    def training_step(self, batch, batch_idx):
        """Specify the training step for the Normalizing flow model
        """
        self.optimizer.zero_grad()
        loss = -self.model.log_prob(inputs=batch[0]).mean()
        loss.backward()
        self.optimizer.step()

    def sample(self, model, n_samples):
        """sample n samples from the model
        """
        return model.sample(n_samples).detach().numpy()

if __name__ == '__main__':
    """
    Create argument parser in order to allowing user-friendly command-line interface
    """
    parser = argparse.ArgumentParser()
    # Add arguments with default values and types to the parser 
    parser.add_argument('--n_layers', type=int, default=10)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--n_dim', type=int, default=6)
    parser.add_argument('--n_hidden_features', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)

    # Convert to: {'n_layers': 10, 'n_epochs': 10, 'n_dim': 6, 'n_hidden_features': 32, 'batch_size': 32, 'lr': 0.001}
    args = vars(parser.parse_args())
    # Call main routine from utils and pass args and the "class name"
    main(MAF, args)
