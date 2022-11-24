import argparse
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from utils import COLS, log_test_metrics, log_hist2d


def main(args):

    np.random.seed(1337)

    with mlflow.start_run() as run:

        mlflow.log_params(vars(args))

        X_train = pd.read_csv('data/df_train.csv')[COLS].to_numpy()
        kernel = gaussian_kde(X_train.T)

        X_test = pd.read_csv('data/df_test.csv')[COLS].to_numpy()
        X_test_pred = kernel.resample(len(X_test)).T
        log_test_metrics(X_test, X_test_pred)

        log_hist2d('train_true', X_train)
        log_hist2d('test_true', X_test)
        log_hist2d('test_pred', X_test_pred, X_test)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_dim', type=int, default=6)
    args = parser.parse_args()
    main(args)
