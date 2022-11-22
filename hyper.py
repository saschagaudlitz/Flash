import mlflow
import numpy as np
from hyperopt import hp, fmin, tpe, space_eval

from models.maf import MAF

space = {
    'n_layers': hp.choice('n_layers', [5, 10, 20]),
    'n_epochs': hp.choice('n_epochs', [1]),
    'n_dim': hp.choice('n_dim', [6]),
    'n_hidden_features': hp.choice('n_hidden_features', [8, 16, 32, 64]),
    'batch_size': hp.choice('batch_size', [32, 128]),
    'lr': hp.choice('lr', [1e-3]),
}


def objective(args):
    with mlflow.start_run() as run:
        trainer = MAF(args, run.info.run_id)
        mlflow.log_params(args)
        trainer.train()
        kendall, ad_mean = trainer.test_step()

    return kendall


if __name__ == '__main__':

    np.random.seed(1337)
    mlflow.set_experiment('Tuning')

    # minimize the objective over the space
    best = fmin(objective, space, algo=tpe.suggest, max_evals=100)

    print(best)
    print(space_eval(space, best))
