import pandas as pd
import numpy as np

import model_collection
import helpers

def load_training_data():
    data_frame = pd.read_csv('./data/df_train.csv', index_col = 0)
    X_train = data_frame[['s1', 's2', 's3', 's4', 's5', 's6']].to_numpy()
    return X_train

X_train = load_training_data()

models = model_collection.Models(10**3, X_train)
KDE_samples = models.sample_from_KDE()
helpers.show_marginals(X_train, KDE_samples)


