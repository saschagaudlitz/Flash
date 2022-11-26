import argparse

import mlflow
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything

from genhack.dataset import StationsDataset
from genhack.models import models
from genhack.experiment import Experiment
from genhack.utils import get_config, DEVICE


def train(config, enable_progress_bar=True, callbacks=None):

    if callbacks is None:
        callbacks = []

    # early_stopping = EarlyStopping(monitor='val_kendall', patience=10)
    # callbacks += [early_stopping]

    seed_everything(config['experiment_params']['manual_seed'], True)

    datamodule = StationsDataset(**config['data_params'])
    model = models[config['model_params']['name']](**config['model_params'], datamodule=datamodule)
    experiment = Experiment(model, config.get('experiment_params', None))
    trainer = Trainer(callbacks=callbacks, enable_progress_bar=enable_progress_bar, **config['trainer_params'])

    mlflow.pytorch.autolog(log_models=False)

    for name in 'model_params', 'experiment_params', 'data_params', 'trainer_params':
        if name in config:
            mlflow.log_params(config[name])
    trainer.fit(experiment, datamodule=datamodule)
    result = experiment.test_step([datamodule.test_dataset[:][0].to(DEVICE)], 0)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', dest="filename", metavar='FILE')
    args = parser.parse_args()
    filename = args.filename
    with mlflow.start_run() as run:
        train(get_config(filename))
