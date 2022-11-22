import argparse
import math

import numpy as np
from sklearn.model_selection import train_test_split
import mlflow
from tqdm import tqdm
import torch
from torch import optim
import torch.utils.data as data_utils
from nflows.transforms import MaskedAffineAutoregressiveTransform, CompositeTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.distributions import StandardNormal
from nflows.flows import Flow
import pandas as pd
from PIL import Image

from utils import calculate_ri, anderson_darling, plot_hist2d, kendall_absolute_error

COLS = ['s1', 's2', 's3', 's4', 's5', 's6']


def build_model(n_dim, n_layers, n_hidden_features):
    transforms = []

    for _ in range(n_layers):
        transforms.append(ReversePermutation(features=n_dim))
        transforms.append(MaskedAffineAutoregressiveTransform(features=n_dim, hidden_features=n_hidden_features))
    transform = CompositeTransform(transforms)

    # Define a base distribution.
    base_distribution = StandardNormal(shape=[n_dim])

    # Combine into a flow.
    flow = Flow(transform=transform, distribution=base_distribution)

    return flow


def main(args):
    np.random.seed(1337)

    # we separate the dataset into three parts: immutable test set (fixed indices all along), training set and validation set (may vary for every training run)
    X = pd.read_csv('data/df_train.csv')[COLS].to_numpy()
    X_train, X_val = train_test_split(X, test_size=0.2)

    n_dim = X_train.shape[1]

    train = torch.tensor(X_train.astype(np.float32))
    train_tensor = data_utils.TensorDataset(train)
    train_loader = data_utils.DataLoader(dataset=train_tensor, batch_size=args.batch_size, shuffle=True)

    flow = build_model(n_dim, args.n_layers, args.n_hidden_features)
    optimizer = optim.Adam(flow.parameters(), lr=args.lr)

    # we do it once outside the loop, because this operation takes a couple of seconds
    ri_true = calculate_ri(X_val)

    with mlflow.start_run() as run:

        mlflow.log_params(vars(args))
        pbar = tqdm(range(args.n_epochs))

        best_kendall = best_ad_mean = math.inf

        for epoch_idx in pbar:
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                loss = -flow.log_prob(inputs=batch[0]).mean()
                loss.backward()
                optimizer.step()

            X_pred = flow.sample(len(X_val)).detach().numpy()
            ad_ind, ad_mean = anderson_darling(X_val, X_pred)

            # calculate Kendall explicitly to avoid the evaluation of ri_true at the end of every epoch
            ri_pred = calculate_ri(X_pred)
            kendall = np.abs(ri_pred - ri_true).mean()

            output = f"Kendall: {kendall:7.4f}; AD: {ad_mean:8.2f}"
            pbar.set_description(output)

            mlflow.log_metric('val_kendall', kendall)
            mlflow.log_metric('val_ad_mean', ad_mean)

            for i in range(n_dim):
                mlflow.log_metric(f'val_ad_{i + 1}', ad_ind[i])

            # save best models
            if kendall < best_kendall:
                best_kendall = kendall
                mlflow.pytorch.log_model(flow, 'best_kendall')
            if ad_mean < best_ad_mean:
                best_ad_mean = ad_mean
                mlflow.pytorch.log_model(flow, 'best_ad_mean')

        # load best model
        model_uri = f"runs:/{run.info.run_id}/best_kendall"
        flow = mlflow.pytorch.load_model(model_uri)

        X_test = pd.read_csv('data/df_test.csv')[COLS].to_numpy()
        log_test_metrics(flow, X_test)
        log_hist2d(flow, X_train, X_val, X_test)


def log_test_metrics(model, X_test):
    n_dim = X_test.shape[1]
    X_pred = model.sample(len(X_test)).detach().numpy()

    kendall = kendall_absolute_error(X_test, X_pred)
    ad_ind, ad_mean = anderson_darling(X_test, X_pred)

    mlflow.log_metric('test_kendall', kendall)
    mlflow.log_metric('test_ad_mean', ad_mean)

    for i in range(n_dim):
        mlflow.log_metric(f'test_ad_{i + 1}', ad_ind[i])


def log_hist2d(model, X_train, X_val, X_test):
    for X_true, label in zip([X_train, X_val, X_test], ['train', 'val', 'test']):
        X_pred = model.sample(len(X_true)).detach().numpy()

        ri_true = calculate_ri(X_true)
        ri_pred = calculate_ri(X_pred)

        fig_true, fig_pred = plot_hist2d(X_true, X_pred, ri_true, ri_pred)

        fig_true.canvas.draw()
        image = Image.frombytes('RGB', fig_true.canvas.get_width_height(), fig_true.canvas.tostring_rgb())
        mlflow.log_image(image, f'hist2d_{label}_true.png')

        fig_pred.canvas.draw()
        image = Image.frombytes('RGB', fig_pred.canvas.get_width_height(), fig_pred.canvas.tostring_rgb())
        mlflow.log_image(image, f'hist2d_{label}_pred.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', type=int, default=10)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--n_hidden_features', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)

    args = parser.parse_args()
    main(args)
