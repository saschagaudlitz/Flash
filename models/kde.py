import argparse
import mlflow
import numpy as np
from scipy.stats import gaussian_kde

from models.trainer import Trainer


class KDE(Trainer):

    def configure_model(self):
        return gaussian_kde(self.X_train.T, bw_method=self.args.bw_method)

    def sample(self, model, n_samples):
        return model.resample(n_samples).T

    # since there is no training, the original model is the best model
    def load_best_model(self):
        return self.model


def main(args):

    np.random.seed(1337)

    # no need for training step in KDE, since it's trained in the initialization step
    with mlflow.start_run() as run:
        trainer = KDE(args, run.info.run_id)
        mlflow.log_params(vars(args))
        trainer.test_step()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_dim', type=int, default=6)
    parser.add_argument('--bw_method', type=str, default='scott')
    args = parser.parse_args()
    main(args)
