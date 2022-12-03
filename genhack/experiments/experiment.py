import math
import mlflow.pytorch
import torch
from torch import optim
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR

from genhack.utils import calculate_ri, anderson_darling, log_hist2d, DEVICE, evaluate_model


class Experiment(pl.LightningModule):

    def __init__(self, model, params, best_ad_mean_model_uri, best_kendall_model_uri, train_start_date, train_end_date, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.params = params
        self.best_ad_mean_model_uri = best_ad_mean_model_uri
        self.best_kendall_model_uri = best_kendall_model_uri
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date

        self.ri_true = None
        self.best_ad_mean = self.best_kendall = math.inf

    def configure_optimizers(self):
        if len(list(self.model.parameters())) > 0:
            adam = optim.Adam(self.model.parameters(), lr=self.params['learning_rate'])
            return adam
            # scheduler = StepLR(adam, step_size=20, gamma=0.1)
            # return [adam], [scheduler]

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        result = self.model(batch)
        loss = self.model.loss(*result)
        self.log_dict({key: val.item() for key, val in loss.items()})
        return loss['loss']

    def validation_step(self, batch, batch_idx):

        X_val = batch[0]
        time = batch[1]

        # calculate Kendall ri for the validation set only once, because this operation takes a couple of seconds
        if self.ri_true is None:
            self.ri_true = calculate_ri(X_val)

        X_val_pred = self.model.sample(torch.randn((len(X_val), self.model.n_latent_dim), device=DEVICE), time)
        ad_ind, ad_mean = anderson_darling(X_val, X_val_pred)

        # calculate Kendall explicitly to avoid the evaluation of ri_true at the end of every epoch
        ri_pred = calculate_ri(X_val_pred)
        kendall = torch.abs(ri_pred - self.ri_true).mean()

        self.log_dict({'val_kendall': kendall, 'val_ad_mean': ad_mean})

        for i in range(self.model.n_dim):
            self.log(f'val_ad_{i + 1}', ad_ind[i])

        # save best model
        if ad_mean < self.best_ad_mean:
            mlflow.pytorch.log_model(self.model, 'best_ad_mean')
            mlflow.log_metric('best_ad_mean_step', self.current_epoch)
            self.best_ad_mean = ad_mean
        if kendall < self.best_kendall:
            mlflow.pytorch.log_model(self.model, 'best_kendall')
            mlflow.log_metric('best_kendall_step', self.current_epoch)
            self.best_kendall = kendall

    def test_step(self, batch, batch_idx):

        X_test = batch[0]
        time = batch[1]

        log_hist2d('test_true', X_test)

        # take bigger number of samples to reduce variance
        N_TEST_SAMPLES = 10

        # log best AD
        best_model = mlflow.pytorch.load_model(self.best_ad_mean_model_uri)
        test_ba_kendall, test_ba_ad_ind, test_ba_ad_mean = \
            evaluate_model(best_model,
                           X_test=X_test,
                           time=time,
                           prefix='ba',
                           n_test_samples=N_TEST_SAMPLES,
                           n_latent_dim=self.model.n_latent_dim,
                           train_start_date=self.train_start_date,
                           train_end_date=self.train_end_date)

        # log best Kendall
        best_model = mlflow.pytorch.load_model(self.best_kendall_model_uri)
        test_bk_kendall, test_bk_ad_ind, test_bk_ad_mean = \
            evaluate_model(best_model,
                           X_test=X_test,
                           time=time,
                           prefix='bk',
                           n_test_samples=N_TEST_SAMPLES,
                           n_latent_dim=self.model.n_latent_dim,
                           train_start_date=self.train_start_date,
                           train_end_date=self.train_end_date)

        return {
            f'test_ba_kendall': test_ba_kendall,
            f'test_ba_mean': test_ba_ad_mean,
            f'test_bk_kendall': test_bk_kendall,
            f'test_bk_mean': test_bk_ad_mean,
        }
