from datetime import datetime
from pathlib import Path
from typing import Iterator
from hyperopt.pyll import scope
from hyperopt import Trials, fmin, hp, tpe

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from loguru import logger
from mads_datasets import DatasetFactoryProvider, DatasetType
from mltrainer import ReportTypes, Trainer, TrainerSettings, metrics
from mltrainer.preprocessors import BasePreprocessor
import mlflow

def get_fashion_streamers(batchsize: int) -> tuple[Iterator, Iterator]:
    fashionfactory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)
    preprocessor = BasePreprocessor()
    streamers = fashionfactory.create_datastreamer(
        batchsize=batchsize, preprocessor=preprocessor
    )
    train = streamers["train"]
    valid = streamers["valid"]
    trainstreamer = train.stream()
    validstreamer = valid.stream()
    return trainstreamer, validstreamer

def get_device() -> str:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
        logger.info("Using MPS")
    elif torch.cuda.is_available():
        device = "cuda:0"
        logger.info("Using cuda")
    else:
        device = "cpu"
        logger.info("Using cpu")
    return device

class CNN(nn.Module):
    def __init__(self, filters, units1, units2, num_blocks=3, use_batchnorm=True, dropout_dense_rate = 0.0, dropout_conv_rate=0.0, use_maxpool=True, input_size=(32, 1, 28, 28)):
        super().__init__()
        self.in_channels = input_size[1]
        self.filters = filters
        in_ch = self.in_channels

        self.conv_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            layers = []
            layers.append(nn.Conv2d(in_ch, filters, kernel_size=3, padding=1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(filters))
            layers.append(nn.ReLU())
            if use_maxpool:
                layers.append(nn.MaxPool2d(2))
            if dropout_conv_rate > 0:
                layers.append(nn.Dropout2d(dropout_conv_rate))
            self.conv_blocks.append(nn.Sequential(*layers))
            in_ch = filters 

        activation_map_size = self._conv_test(input_size)
        self.agg = nn.AvgPool2d(activation_map_size)

        dense_layers = [
            nn.Flatten(),
            nn.Linear(filters, units1),
            nn.ReLU(),
            nn.Linear(units1, units2),
            nn.ReLU(),
            nn.Linear(units2, 10),
        ]

        if dropout_dense_rate > 0.0:
            dense_layers.append(nn.Dropout(dropout_dense_rate))
        self.dense = nn.Sequential(*dense_layers)
    
    def _conv_test(self, input_size):
        with torch.no_grad():
            x = torch.ones(input_size)
            for block in self.conv_blocks:
                x = block(x)
            return x.shape[-2:]

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = self.agg(x)
        return self.dense(x)


def setup_mlflow(experiment_path: str) -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_path)

def objective(params):
    modeldir = Path("models").resolve()
    if not modeldir.exists():
        modeldir.mkdir(parents=True)
        logger.info(f"Created {modeldir}")
    batchsize = 64
    trainstreamer, validstreamer = get_fashion_streamers(batchsize)
    accuracy = metrics.Accuracy()
    settings = TrainerSettings(
        epochs=10,
        metrics=[accuracy],
        logdir=Path("modellog"),
        train_steps=100,
        valid_steps=100,
        reporttypes=[ReportTypes.MLFLOW]
    )
    # Start a new MLflow run for tracking the experiment
    device = get_device()
    with mlflow.start_run():
        # Set MLflow tags to record metadata about the model and developer
        mlflow.set_tag("model", "convnet")
        mlflow.set_tag("dev", "Eline")
        # Log hyperparameters to MLflow
        mlflow.log_params(params)
        mlflow.log_param("batchsize", f"{batchsize}")
        mlflow.log_param("epochs", f"{settings.epochs}")
        mlflow.log_param("train_steps", f"{settings.train_steps}")
        mlflow.log_param("valid_steps", f"{settings.valid_steps}")

        # Initialize the optimizer, loss function, and accuracy metric
        optimizer = optim.Adam
        loss_fn = torch.nn.CrossEntropyLoss()

        # Instantiate the CNN model with the given hyperparameters
        model = CNN(**params)
        model.to(device)
        # Train the model using a custom train loop
        trainer = Trainer(
            model=model,
            settings=settings,
            loss_fn=loss_fn,
            optimizer=optimizer,  # type: ignore
            traindataloader=trainstreamer,
            validdataloader=validstreamer,
            scheduler=optim.lr_scheduler.ReduceLROnPlateau,
            device=device,
        )
        trainer.loop()

        # Save the trained model with a timestamp
        tag = datetime.now().strftime("%Y%m%d-%H%M")
        modelpath = modeldir / (tag + "model.pt")
        logger.info(f"Saving model to {modelpath}")
        torch.save(model, modelpath)

        # Log the saved model as an artifact in MLflow
        mlflow.log_artifact(local_path=str(modelpath), artifact_path="pytorch_models")
        return {"loss": trainer.test_loss, "status": STATUS_OK}
    

def setup_mlflow(experiment_path: str) -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_path)


setup_mlflow("mlflow_database")

search_space = {
    "filters": scope.int(hp.quniform("filters", 16, 128, 8)),
    "units1": scope.int(hp.quniform("units1", 32, 128, 8)),
    "units2": scope.int(hp.quniform("units2", 32, 128, 8)),
    "use_batchnorm": hp.choice("use_batchnorm", [True, False]),
    "use_maxpool": hp.choice("use_maxpool", [True, False]),
    "dropout_conv_rate": 0.0, #hp.uniform("dropout_conve_rate", 0.0, 0.5),
    "dropout_dense_rate": 0.0, #hp.uniform("dropout_dense_rate", 0.0, 0.5),
}

best_result = fmin(
    fn=objective, space=search_space, algo=tpe.suggest, max_evals=2, trials=Trials()
)

logger.info(f"Best result: {best_result}")