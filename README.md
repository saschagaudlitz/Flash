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

## Model development

> See `models/maf.py` for a working example.

To create a new model, subclass `Trainer` and implement the following methods:

    # specify model hyperparameters, with argparse
    def get_parser()

    # model definition
    def configure_model(self)

    # model training step
    def training_step(self, batch, batch_idx)

    # sampling from the model
    def sample(self, n_samples)

Trainer exposes the following attributes:

    self.model, self.optimizer
    self.X_train, self.X_val, self.X_test

In the end, import contents of your module in the `models/__init__.py` file, for example:

    from .diffusion import *

# Running models

## From command line

You can run a model from the root directory of the project by specifying the name of the class and passing arguments as follows:

    python3 train.py MAF --n_epochs=10 --batch_size=32 --n_hidden_features=32

## From notebook

You can also run the model from the notebook, see `train.ipynb` for the example. In the notebook you can observe 2d-marginals graphically during training. For this, you need to subclass your model and implement `on_train_epoch_end` hook:

    class MyMAF(MAF):
        def on_train_epoch_end(self):
            X_val_pred = self.sample(self.model, len(self.X_val))
            fig = plot_hist2d(X_val_pred, self.X_val)
            clear_output(wait=True)
            plt.show()

## Hyperparameter tuning

See `hyper.py` script.
