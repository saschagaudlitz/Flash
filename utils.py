import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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
