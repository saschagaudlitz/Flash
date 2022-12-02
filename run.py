import argparse

import mlflow
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything

from genhack.dataset import StationsDataset
from genhack.experiments import GANExperiment, Experiment, experiments
from genhack.models import models, TTSGAN
from genhack.utils import get_config, DEVICE


def run(config, mode='train', enable_progress_bar=True, callbacks=None):

    seed_everything(config['experiment_params']['manual_seed'], True)
    datamodule = StationsDataset(**config['data_params'])

    model = models[config['model_params']['name']](**config['model_params'], datamodule=datamodule)

    # initialize experiment

    active_run = mlflow.active_run()
    best_ad_mean_model_uri = f'runs:/{active_run.info.run_id}/best_ad_mean'
    best_kendall_model_uri = f'runs:/{active_run.info.run_id}/best_kendall'

    experiment_class = config['experiment_params'].get('experiment_class', None)
    cls = experiments.get(experiment_class, Experiment)

    # @todo here is potential problem, as actual training dates, and training dates from the config might deviate
    experiment = cls(model, config.get('experiment_params', None),
                     best_ad_mean_model_uri=best_ad_mean_model_uri,
                     best_kendall_model_uri=best_kendall_model_uri,
                     train_start_date=datamodule.train_start_date,
                     train_end_date=datamodule.train_end_date)

    # training

    if mode == 'train':

        mlflow.log_param('train_start_date', datamodule.train_start_date)
        mlflow.log_param('train_end_date', datamodule.train_end_date)
        mlflow.log_param('val_start_date', datamodule.val_start_date)
        mlflow.log_param('val_end_date', datamodule.val_end_date)
        mlflow.log_param('test_start_date', datamodule.test_start_date)
        mlflow.log_param('test_end_date', datamodule.test_end_date)

        if callbacks is None:
            callbacks = []

        # early_stopping = EarlyStopping(monitor='val_kendall', patience=10)
        # callbacks += [early_stopping]

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
    parser.add_argument('--mode', '-m', type=str, choices=['train', 'test'], default='train')
    parser.add_argument('--run_id', '-r', type=str, default=None)
    args = parser.parse_args()
    filename = args.filename
    with mlflow.start_run(run_id=args.run_id) as active_run:
        run(get_config(filename), mode=args.mode)
