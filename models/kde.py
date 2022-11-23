import argparse
from scipy.stats import gaussian_kde
from models.trainer import Trainer


class KDE(Trainer):

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

    # since there is no training, the original model is the best model
    def load_best_model(self):
        return self.model
