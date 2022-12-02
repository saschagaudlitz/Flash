import mlflow
from ray import tune, air
from ray.tune.integration.mlflow import mlflow_mixin
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch
import os

from genhack.utils import get_config
from run import train

param_space = {
    'model_params.n_blocks': tune.choice([3, 5, 10]),
    'model_params.n_layers': tune.choice([5, 7, 10]),
    'model_params.n_hidden_features': tune.choice([8, 16, 32, 64]),
    'model_params.dropout_probability': tune.choice([0., 0.25, 0.5, 0.75]),
    'model_params.use_batch_norm': tune.choice([True, False]),
    'experiment_params.learning_rate': tune.choice([0.001, 0.0001]),
    "mlflow": {
        "experiment_name": "Tuning MAF",
        "tracking_uri": mlflow.get_tracking_uri(),
    },
}

initial_params = [
    {
        'model_params.n_blocks': 5,
        'model_params.n_layers': 5,
        'model_params.n_hidden_features': 32,
        'model_params.dropout_probability': 0.0,
        'model_params.use_batch_norm': False,
        'experiment_params.learning_rate': 0.001,
    }
]


@mlflow_mixin
def objective(args):
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs/maf.yaml')
    config = get_config(filename)

    for key, value in args.items():
        if key == 'mlflow':
            continue
        first, second = key.split('.')
        config[first][second] = value

    return train(config, enable_progress_bar=False)


if __name__ == '__main__':
    algo = HyperOptSearch(points_to_evaluate=initial_params)
    algo = ConcurrencyLimiter(algo, max_concurrent=16)

    tuner = tune.Tuner(
        objective,
        run_config=air.RunConfig(name="mlflow"),
        tune_config=tune.TuneConfig(
            metric="test_ad_mean",
            mode="min",
            search_alg=algo,
            num_samples=100,
        ),
        param_space=param_space,
    )

    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result().config)
