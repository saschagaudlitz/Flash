# GenHack2 - Hackathon for Generative modeling : Simulation of global warming Sea Surface Temperatures ([website](https://www.polytechnique.edu/en/education/academic-and-research-departments/applied-mathematics-department-depmap/student-event/genhack-2-hackathon-generative-modelling))
<img src="https://www.polytechnique.edu/sites/default/files/styles/contenu_detail/public/content/pages/images/2022-10/GenHack%20Challenge%20%28Banni%C3%A8re%20%28paysage%29%29%20%281250%20%C3%97%20350%20px%29_0.png?itok=K1AwTb_0">

## Contact
genhack@polytechnique.fr

# Workflow

## MLFlow

We use MLFlow for model tracking.

    pip install mlflow

Then start the MLFlow from the root directory of the project:

    mlflow server

Then you can access the server under `127.0.0.1:5000`.

Note that in `mlflow` you can run an *experiment* that has several *runs*. For example, for hyperparameter tuning or cross-validation, you would run one experiment with multiple runs.

## Write a model

> See `models/maf.py` for a working example.

Use `argparse` for specifying hyperparameter. Then you can run a model from the root directory of the project:

    python3 models/maf.py --n_epochs=10 --batch_size=32 --n_hidden_features=32

Subclass `Trainer` for each model, and implement the following methods:

    def configure_model(self):
        return NotImplementedError()

    def training_step(self, batch, batch_idx):
        return NotImplementedError()

    def sample(self, n_samples):
        return NotImplementedError()

Trainer exposes the following attributes:

    self.model, self.optimizer
    self.X_train, self.X_val, self.X_test

## Hyperparameter tuning

See `hyper.py` script.
