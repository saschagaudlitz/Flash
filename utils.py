import mlflow
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image

COLS = ['s1', 's2', 's3', 's4', 's5', 's6']


def calculate_ri(X):
    """Calculate order statistics for R_{i,n_test}."""
    n_test = X.shape[0]
    # we can ignore j = i, since it's zero anyway
    return np.sort((X[:, None, :] < X[None, :, :]).prod(axis=2).sum(axis=1) / (n_test - 1))


def kendall_absolute_error(X_true, X_pred):
    return np.abs(calculate_ri(X_true) - calculate_ri(X_pred)).mean()


def anderson_darling(X_true, X_pred):
    assert X_true.shape == X_pred.shape
    n_test = X_true.shape[0]

    X_true_sorted = np.sort(X_true, axis=0)
    u = ((X_pred[:, None] <= X_true_sorted[None, :]).sum(axis=1) + 1) / (n_test + 2)

    ad_ind = -n_test - np.sum((2 * np.arange(1, n_test + 1) - 1).reshape(-1, 1) * (np.log(u) + np.log(1 - u[::-1])), axis=0) / n_test
    ad_mean = ad_ind.mean()

    return ad_ind, ad_mean


def plot_hist2d(X_true, X_pred, ri_true, ri_pred):
    assert X_true.shape == X_pred.shape
    n_dim = X_true.shape[1]

    kendall_abs_error = np.abs(ri_pred - ri_true).mean()
    ad_ind, ad_mean = anderson_darling(X_pred, X_true)

    def plot_hists(fig, ax, X):
        for i in range(n_dim):
            for j in range(n_dim):
                if i < j:
                    ax[i][j].hist2d(X[:, i], X[:, j], bins=50, range=[[-7, 7], [-7, 7]])
                if i > j:
                    # delete axes below diagonal
                    fig.delaxes(ax[i][j])

    fig_true, ax = plt.subplots(nrows=n_dim, ncols=n_dim, figsize=(3 * n_dim, 3 * n_dim))
    plot_hists(fig_true, ax, X_true)

    for i in range(n_dim):
        sns.kdeplot(data=X_true[:, i], ax=ax[i][i])

    fig_pred, ax = plt.subplots(nrows=n_dim, ncols=n_dim, figsize=(3 * n_dim, 3 * n_dim))
    plot_hists(fig_pred, ax, X_pred)

    for i in range(n_dim):
        sns.kdeplot(data=pd.DataFrame(np.array([X_true[:, i], X_pred[:, i]]).T, columns=['X_true', 'X_pred']), ax=ax[i][i])
        ax[i][i].set_title(f"AD = {ad_ind[i]:.4f}")
        ax[i][i].get_legend().remove()

    ax[0][1].set_title(f"Kendall = {kendall_abs_error:.6f}")

    return fig_true, fig_pred


def log_test_metrics(X_true, X_pred):

    n_dim = X_true.shape[1]

    kendall = kendall_absolute_error(X_true, X_pred)
    ad_ind, ad_mean = anderson_darling(X_true, X_pred)

    mlflow.log_metric('test_kendall', kendall)
    mlflow.log_metric('test_ad_mean', ad_mean)

    for i in range(n_dim):
        mlflow.log_metric(f'test_ad_{i + 1}', ad_ind[i])


def log_hist2d(X_true, X_pred, label):

    ri_true = calculate_ri(X_true)
    ri_pred = calculate_ri(X_pred)

    fig_true, fig_pred = plot_hist2d(X_true, X_pred, ri_true, ri_pred)

    fig_true.canvas.draw()
    image = Image.frombytes('RGB', fig_true.canvas.get_width_height(), fig_true.canvas.tostring_rgb())
    mlflow.log_image(image, f'hist2d_{label}_true.png')

    fig_pred.canvas.draw()
    image = Image.frombytes('RGB', fig_pred.canvas.get_width_height(), fig_pred.canvas.tostring_rgb())
    mlflow.log_image(image, f'hist2d_{label}_pred.png')
