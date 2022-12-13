from itertools import combinations

import mlflow
from ray import tune, air
from ray.tune.integration.mlflow import mlflow_mixin
from ray.tune.search import BasicVariantGenerator
import os

from genhack.utils import get_config
from run import run

train_dims = [list(x) for x in list(combinations(range(6), 3))]

param_space = {
    "data_params.train_dims": tune.grid_search(train_dims),
    "mlflow": {
        "experiment_id": "11",
        "tracking_uri": mlflow.get_tracking_uri(),
    },
}


@mlflow_mixin
def objective(args):

    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs/gpr.yaml')
    config = get_config(filename)

    for key, value in args.items():
        if key == 'mlflow' or key == "period":
            continue
        first, second = key.split('.')
        config[first][second] = value

    config['data_params']['test_dims'] = [x for x in range(6) if x not in config['data_params']['train_dims']]
    return run(config, enable_progress_bar=False)


if __name__ == '__main__':

    print(f"MLFlow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"MLFlow artifact URI: {mlflow.get_artifact_uri()}")

    tuner = tune.Tuner(
        objective,
        run_config=air.RunConfig(name="mlflow"),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            search_alg=BasicVariantGenerator(constant_grid_search=True, max_concurrent=2),
            num_samples=1,
        ),
    )

    results = tuner.fit()
