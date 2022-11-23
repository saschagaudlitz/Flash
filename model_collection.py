from scipy.stats import anderson_ksamp
from scipy.stats import gaussian_kde

class Models:
    def __init__(self, number_samples, X_train):
        self.number_samples = number_samples
        self.X_train = X_train

    def sample_from_KDE(self):
        kde = gaussian_kde(self.X_train.T)
        samples = kde.resample(self.number_samples)
        return samples