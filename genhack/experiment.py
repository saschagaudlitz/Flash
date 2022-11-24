import math

import mlflow.pytorch
import numpy as np
from torch import optim
import pytorch_lightning as pl

from genhack.utils import calculate_ri, anderson_darling, log_test_metrics, log_hist2d


class Experiment(pl.LightningModule):

    def __init__(self, model, params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.params = params
        self.ri_true = None

        self.best_ad_mean = self.best_kendall = math.inf
        self.best_ad_mean_model_uri = self.best_kendall_model_uri = None

    def configure_optimizers(self):
        if len(list(self.model.parameters())) > 0:
            return optim.Adam(self.model.parameters(), lr=self.params['learning_rate'])

    def training_step(self, batch, batch_idx):
        result = self.model(batch[0])
        loss = self.model.loss(*result)
        self.log_dict({key: val.item() for key, val in loss.items()})
        return loss['loss']

    def validation_step(self, batch, batch_idx):

        X_val = batch[0]

        # calculate Kendall ri for the validation set only once, because this operation takes a couple of seconds
        if self.ri_true is None:
            self.ri_true = calculate_ri(X_val)

        X_pred = self.model.sample(X_val.shape[0])
        ad_ind, ad_mean = anderson_darling(X_val, X_pred)

        # calculate Kendall explicitly to avoid the evaluation of ri_true at the end of every epoch
        ri_pred = calculate_ri(X_pred)
        kendall = np.abs(ri_pred - self.ri_true).mean()

        self.log_dict({'val_kendall': kendall, 'val_ad_mean': ad_mean})

        for i in range(self.model.n_dim):
            self.log(f'val_ad_{i + 1}', ad_ind[i])

        # save best model
        if ad_mean < self.best_ad_mean:
            model_info = mlflow.pytorch.log_model(self.model, 'best_ad_mean')
            self.best_ad_mean = ad_mean
            self.best_ad_mean_model_uri = model_info.model_uri
        if kendall < self.best_kendall:
            model_info = mlflow.pytorch.log_model(self.model, 'best_kendall')
            self.best_kendall = kendall
            self.best_kendall_model_uri = model_info.model_uri

    def test_step(self, batch, batch_idx):

        best_model = mlflow.pytorch.load_model(self.best_ad_mean_model_uri)

        X_test = batch[0]
        X_test_pred = best_model.sample(X_test.shape[0])
        kendall, ad_mean = log_test_metrics(X_test, X_test_pred)

        log_hist2d('test_true', X_test)
        log_hist2d('test_pred', X_test_pred, X_test)

        return {'test_kendall': kendall, 'test_ad_mean': ad_mean}
