import mlflow
from ray import tune, air
from ray.tune.integration.mlflow import mlflow_mixin
from ray.tune.search import BasicVariantGenerator
import os

from genhack.utils import get_config
from run import run

param_space = {
    # "period": tune.grid_search([
    #     ("1982-01-01", "1999-12-31"),
    #     ("1986-01-01", "2003-12-31"),
    #     ("1990-01-01", "2007-12-31"),
    # ]),
    # "model_params.weights_model_params": tune.grid_search([
    #     {"model_name": "LearnableWeights", "kwargs": {"n_hidden_units": 100}},
    #     {"model_name": "PowerLawWeights", "kwargs": {"a": 1., "b": 1., "c": 0.}},
    #     {"model_name": "PowerLawWeights", "kwargs": {"a": -0.9, "b": 1., "c": 1.}},
    #     {"model_name": "PowerLawWeights", "kwargs": {"a": -0.9, "b": 1., "c": 2.}},
    # ]),
    # "model_params.ts_model_params": tune.grid_search([
    #     {"model_name": "TrendModel", "kwargs": {"trend_factor": 1.}},
    #     {"model_name": "TrendModel", "kwargs": {"trend_factor": 1.2}},
    #     {"model_name": "TrendModel", "kwargs": {"trend_factor": 1.5}},
    # ]),
    "data_params.test_split_size": tune.grid_search([0.05, 0.10, 0.15, 0.20, 0.25]),
    # "data_params.train_val_shuffle": tune.grid_search([True, False]),
    # "model_params.n_layers": tune.grid_search([4, 8, 12]),
    # "model_params.n_hidden_features": tune.grid_search([8, 32, 64]),
    # "model_params.n_blocks": tune.grid_search([5, 10, 15]),
    "mlflow": {
        "experiment_id": "2",
        "tracking_uri": mlflow.get_tracking_uri(),
    },
}


@mlflow_mixin
def objective(args):

    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs/maf.yaml')
    config = get_config(filename)

    for key, value in args.items():
        if key == 'mlflow' or key == "period":
            continue
        first, second = key.split('.')
        config[first][second] = value

    # config['data_params']['start_date'] = args['period'][0]
    # config['data_params']['end_date'] = args['period'][1]

    return run(config, enable_progress_bar=False)


if __name__ == '__main__':

    tuner = tune.Tuner(
        objective,
        run_config=air.RunConfig(name="mlflow"),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            search_alg=BasicVariantGenerator(),
            num_samples=1000,
        ),
    )

    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result().config)
