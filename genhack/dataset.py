from pytorch_lightning import LightningDataModule
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os
from genhack.utils import COLS, DEVICE
from torch.utils.data.dataloader import default_collate

# automatically load all tensor on the device
# https://stackoverflow.com/questions/65932328/pytorch-while-loading-batched-data-using-dataloader-how-to-transfer-the-data-t
collate_fn = lambda x: tuple(y.to(DEVICE) for y in default_collate(x))


class StationsDataset(LightningDataModule):

    def __init__(self, batch_size, val_split_size=0.2, test_split_size=0.2, train_val_shuffle=False, *args, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.val_split_size = val_split_size
        self.test_split_size = test_split_size
        self.train_val_shuffle = train_val_shuffle

        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/df_all.csv')
        df = pd.read_csv(filename)
        df['dates'] = pd.to_datetime(df['dates'])
        df = df.set_index('dates')[COLS]
        X = torch.tensor(df.to_numpy().astype(np.float32))

        # train/val/test split
        # test split is always deterministic (take last period)
        # train/val split can be random (train/val mixed together)
        time = torch.linspace(0, 1, len(X))
        X_train_val, X_test, date_train_val, date_test, time_train_val, time_test = \
            train_test_split(X, df.index, time, test_size=test_split_size, shuffle=False)
        X_train, X_val, date_train, date_val, time_train, time_val = \
            train_test_split(X_train_val, date_train_val, time_train_val, test_size=val_split_size, shuffle=train_val_shuffle)

        # detrend
        # intercept = torch.tensor([-0.3, -0.27, -0.41, -0.39, -0.45, -0.68])
        # trend = torch.tensor([0.56, 0.46, 0.83, 0.78, 0.89, 1.36])
        # X_train = X_train - intercept[None, :] - time_train[:, None] * trend[None, :]

        self.train_start_date = date_train.min()
        self.train_end_date = date_train.max()
        self.val_start_date = date_val.min()
        self.val_end_date = date_val.max()
        self.test_start_date = date_test.min()
        self.test_end_date = date_test.max()

        self.train_dataset = TensorDataset(X_train, time_train / time_train.max())
        self.val_dataset = TensorDataset(X_val)
        self.test_dataset = TensorDataset(X_test)

        self.t_val_min = time_val.min()
        self.t_val_max = time_val.max()
        self.t_test_min = time_test.min()
        self.t_test_max = time_test.max()

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, collate_fn=collate_fn, drop_last=True, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, collate_fn=collate_fn, batch_size=100000)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, collate_fn=collate_fn, batch_size=100000)
