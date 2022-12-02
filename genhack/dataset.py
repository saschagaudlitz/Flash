from pytorch_lightning import LightningDataModule
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os
from genhack.utils import COLS, DEVICE
from torch.utils.data.dataloader import default_collate

# https://stackoverflow.com/questions/65932328/pytorch-while-loading-batched-data-using-dataloader-how-to-transfer-the-data-t
collate_fn = lambda x: tuple(y.to(DEVICE) for y in default_collate(x))


class StationsDataset(LightningDataModule):

    def __init__(self, batch_size, val_split_size=0.2, train_val_shuffle=False, *args, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.val_split_size = val_split_size
        self.train_val_shuffle = train_val_shuffle

        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/df_train.csv')
        df = pd.read_csv(filename)
        df['dates'] = pd.to_datetime(df['dates'])
        df = df.set_index('dates')[COLS]
        X = df.to_numpy()

        time = torch.linspace(0, 1, len(X))
        X_train, X_val, date_train, date_val, time_train, time_val = train_test_split(X, df.index, time, test_size=val_split_size, shuffle=train_val_shuffle)

        # rescale training period to 0-1, time_val can be discarded since we don't use time for inference
        time_train /= (date_train.max() - date_train.min()) / (date_val.max() - date_train.min())
        del time_val

        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/df_test.csv')
        df = pd.read_csv(filename)
        df['dates'] = pd.to_datetime(df['dates'])
        df = df.set_index('dates')[COLS]
        X_test = df.to_numpy()

        self.train_start_date = date_train.min()
        self.train_end_date = date_train.max()
        self.val_start_date = date_val.min()
        self.val_end_date = date_val.max()
        self.test_start_date = df.index.min()
        self.test_end_date = df.index.max()

        self.train_dataset = TensorDataset(torch.tensor(X_train.astype(np.float32)), time_train)
        self.val_dataset = TensorDataset(torch.tensor(X_val.astype(np.float32)))
        self.test_dataset = TensorDataset(torch.tensor(X_test.astype(np.float32)))

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, collate_fn=collate_fn, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, collate_fn=collate_fn, batch_size=100000)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, collate_fn=collate_fn, batch_size=100000)
