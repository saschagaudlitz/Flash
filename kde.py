import argparse
import mlflow
import numpy as np
from scipy.stats import gaussian_kde

from models.trainer import Trainer
from utils import main


class KDE(Trainer):

    def configure_model(self):
        return gaussian_kde(self.X_train.T, bw_method=self.args['bw_method'])

    def sample(self, model, n_samples):
        return model.resample(n_samples).T

    # since there is no training, the original model is the best model
    def load_best_model(self):
        return self.model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_dim', type=int, default=6)
    parser.add_argument('--bw_method', type=str, default='silverman')

    args = vars(parser.parse_args())
    main(KDE, args)
