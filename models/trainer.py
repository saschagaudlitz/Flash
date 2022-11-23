import math
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow
from tqdm import tqdm
import torch
from torch import optim
import torch.utils.data as data_utils
import pandas as pd

from utils import calculate_ri, anderson_darling, COLS, log_test_metrics, log_hist2d


class Trainer:

    def __init__(self, args, run_id):

        self.args = args
        self.run_id = run_id

        self.X_train, self.X_val, self.X_test = self.configure_datasets()
        self.model = self.configure_model()
        self.optimizer = self.configure_optimizer()

        # we pre-calculate Anderson-Darling ri for the validation set once outside the training loop, because this operation takes a couple of seconds
        self.ri_true = calculate_ri(self.X_val)
        self.best_kendall = self.best_ad_mean = math.inf

    @staticmethod
    def get_parser():
        return NotImplementedError()

    def configure_model(self):
        return NotImplementedError()

    def training_step(self, batch, batch_idx):
        return NotImplementedError()

    def sample(self, model, n_samples):
        return NotImplementedError()

    def train(self):
        """Training loop.
        """
        train = torch.tensor(self.X_train.astype(np.float32))
        train_dataset = data_utils.TensorDataset(train)
        train_loader = data_utils.DataLoader(dataset=train_dataset, batch_size=self.args['batch_size'], shuffle=True)

        pbar = tqdm(range(self.args['n_epochs']))

        for epoch_idx in pbar:
            for batch_idx, batch in enumerate(train_loader):
                self.training_step(batch, batch_idx)
            self.validation_step(pbar=pbar)

    def configure_datasets(self):
        """Separate dataset into training set and validation set (may vary for every training run).
        """
        X = pd.read_csv('data/df_train.csv')[COLS].to_numpy()
        X_train, X_val = train_test_split(X, test_size=0.2)
        X_test = pd.read_csv('data/df_test.csv')[COLS].to_numpy()
        return X_train, X_val, X_test

    def configure_optimizer(self):
        if hasattr(self.model, 'parameters'):
            return optim.Adam(self.model.parameters(), lr=self.args['lr'])

    def validation_step(self, pbar=None):
        """Record validation metrics and save model, if best.
        """
        X_pred = self.sample(self.model, self.X_val.shape[0])
        ad_ind, ad_mean = anderson_darling(self.X_val, X_pred)

        # calculate Kendall explicitly to avoid the evaluation of ri_true at the end of every epoch
        ri_pred = calculate_ri(X_pred)
        kendall = np.abs(ri_pred - self.ri_true).mean()

        if pbar is not None:
            output = f"Kendall: {kendall:7.4f}; AD: {ad_mean:8.2f}"
            pbar.set_description(output)

        mlflow.log_metric('val_kendall', kendall)
        mlflow.log_metric('val_ad_mean', ad_mean)

        for i in range(self.args['n_dim']):
            mlflow.log_metric(f'val_ad_{i + 1}', ad_ind[i])

        # save best models
        if kendall < self.best_kendall:
            self.best_kendall = kendall
            mlflow.pytorch.log_model(self.model, 'best_kendall')
        if ad_mean < self.best_ad_mean:
            self.best_ad_mean = ad_mean
            mlflow.pytorch.log_model(self.model, 'best_ad_mean')

    def load_best_model(self):
        model_uri = f"runs:/{self.run_id}/best_kendall"
        return mlflow.pytorch.load_model(model_uri)

    def test_step(self):
        """Record metrics on the test set and 2d-histograms.
        """

        best_model = self.load_best_model()

        X_test_pred = self.sample(best_model, len(self.X_test))
        kendall, ad_mean = log_test_metrics(self.X_test, X_test_pred)

        log_hist2d('train_true', self.X_train)
        log_hist2d('val_true', self.X_val)
        log_hist2d('test_true', self.X_test)
        log_hist2d('test_pred', X_test_pred, self.X_test)

        return kendall, ad_mean
