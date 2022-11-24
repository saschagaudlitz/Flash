from pytorch_lightning import LightningDataModule
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from genhack.utils import COLS


class StationsDataset(LightningDataModule):

    def __init__(self, batch_size, *args, **kwargs):
        super().__init__()
        self.batch_size = batch_size

        X = pd.read_csv('data/df_train.csv')[COLS].to_numpy()
        X_train, X_val = train_test_split(X, test_size=0.2)
        X_test = pd.read_csv('data/df_test.csv')[COLS].to_numpy()

        self.train_dataset = TensorDataset(torch.tensor(X_train.astype(np.float32)))
        self.val_dataset = TensorDataset(torch.tensor(X_val.astype(np.float32)))
        self.test_dataset = TensorDataset(torch.tensor(X_test.astype(np.float32)))

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=100000)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=100000)
