import mlflow
import numpy as np
from hyperopt import hp, fmin, tpe, space_eval
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything

from genhack.dataset import StationsDataset
from genhack.experiment import Experiment
from genhack.models import models
from genhack.utils import get_config

space = {
    'model_params.bw_method': hp.choice('bw_method', np.arange(0.1, 1.1, 0.1)),
}


def objective(args):

    config = get_config()

    for key, value in args.items():
        first, second = key.split('.')
        config[first][second] = value

    seed_everything(config['experiment_params']['manual_seed'], True)

    datamodule = StationsDataset(**config['data_params'])
    model = models[config['model_params']['name']](**config['model_params'], datamodule=datamodule)
    experiment = Experiment(model, config.get('experiment_params', None))
    trainer = Trainer(**config['trainer_params'])

    mlflow.pytorch.autolog(log_models=False)

    with mlflow.start_run() as run:
        for name in 'model_params', 'experiment_params', 'data_params', 'trainer_params':
            if name in config:
                mlflow.log_params(config[name])
        trainer.fit(experiment, datamodule=datamodule)
        result = experiment.test_step(datamodule.test_dataset[:], 0)

    return float(result['test_ad_mean'])


if __name__ == '__main__':

    mlflow.set_experiment('Tuning')

    # minimize the objective over the space
    best = fmin(objective, space, algo=tpe.suggest, max_evals=10)

    print(best)
    print(space_eval(space, best))
