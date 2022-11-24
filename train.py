import argparse
import mlflow
import yaml
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.utilities.seed import seed_everything

from genhack.dataset import StationsDataset
from genhack.models import models
from genhack.experiment import Experiment
from genhack.utils import get_config

if __name__ == '__main__':

    config = get_config()
    seed_everything(config['experiment_params']['manual_seed'], True)

    datamodule = StationsDataset(**config['data_params'])
    model = models[config['model_params']['name']](**config['model_params'], datamodule=datamodule)
    experiment = Experiment(model, config.get('experiment_params', None))
    trainer = Trainer(**config['trainer_params'])

    mlflow.pytorch.autolog(log_models=False)

    with mlflow.start_run() as run:
        mlflow.log_params(config)
        trainer.fit(experiment, datamodule=datamodule)
        experiment.test_step(datamodule.test_dataset[:], 0)
