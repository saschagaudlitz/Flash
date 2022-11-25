import mlflow
import numpy as np
from hyperopt import hp, fmin, tpe, space_eval
from genhack.utils import get_config
from train import train

space = {
    'data_params.val_split_size': hp.choice('bw_method', np.arange(0.01, 0.51, 0.01)),
}


def objective(args):

    config = get_config('configs/KDE.yaml')

    for key, value in args.items():
        first, second = key.split('.')
        config[first][second] = value

    result = train(config)
    return float(result['test_ad_mean'])


"""
Check in detail, perhaps try simulated annealing:
https://www.kaggle.com/code/ilialar/hyperparameters-tunning-with-hyperopt/notebook
"""
if __name__ == '__main__':

    mlflow.set_experiment('Tuning MAF n_epochs Kendall')

    # minimize the objective over the space
    best = fmin(objective, space, algo=tpe.suggest, max_evals=50)

    print(best)
    print(space_eval(space, best))
