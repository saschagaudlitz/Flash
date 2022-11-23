import argparse
from scipy.stats import gaussian_kde
from models.trainer import Trainer


class KDE(Trainer):
    """
    KDE trainer class - Implements Trainer for Kernel density estimation

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

    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--n_dim', type=int, default=6)
        parser.add_argument('--bw_method', type=str, default='silverman')
        return parser

    def configure_model(self):
        return gaussian_kde(self.X_train.T, bw_method=self.args['bw_method'])

    def sample(self, model, n_samples):
        return model.resample(n_samples).T

    def load_best_model(self):
        # since there is no training, the original model is the best model
        return self.model
