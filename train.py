import argparse

import mlflow
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything

from genhack.dataset import StationsDataset
from genhack.models import models
from genhack.experiment import Experiment
from genhack.utils import get_config


def train(config):
    seed_everything(config['experiment_params']['manual_seed'], True)

    early_stopping = EarlyStopping(monitor='val_ad_mean', patience=10)
    callbacks = [early_stopping]

    datamodule = StationsDataset(**config['data_params'])
    model = models[config['model_params']['name']](**config['model_params'], datamodule=datamodule)
    experiment = Experiment(model, config.get('experiment_params', None))
    trainer = Trainer(callbacks=callbacks, **config['trainer_params'])

    mlflow.pytorch.autolog(log_models=False)

    with mlflow.start_run() as run:
        for name in 'model_params', 'experiment_params', 'data_params', 'trainer_params':
            if name in config:
                mlflow.log_params(config[name])
        trainer.fit(experiment, datamodule=datamodule)
        result = experiment.test_step(datamodule.test_dataset[:], 0)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', dest="filename", metavar='FILE')
    args = parser.parse_args()
    filename = args.filename
    train(get_config(filename))
