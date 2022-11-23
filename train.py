import importlib
import sys
import numpy as np
import mlflow


if __name__ == '__main__':

    model_name = sys.argv[1]
    module = importlib.import_module('models')
    model = getattr(module, model_name)

    parser = model.get_parser()
    parser.add_argument('model', type=str)
    args = vars(parser.parse_args())

    np.random.seed(1337)

    with mlflow.start_run() as run:

        trainer = model(args, run.info.run_id)
        mlflow.log_params(args)

        # some models like KDE might not require training
        if 'n_epochs' in args:
            trainer.train()

        kendall, ad_mean = trainer.test_step()
        print(f"Test Kendall: {kendall:10.4f}")
        print(f"Test AD:      {ad_mean:10.2f}")
