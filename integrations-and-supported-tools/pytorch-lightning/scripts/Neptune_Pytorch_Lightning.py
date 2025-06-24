import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers.neptune import NeptuneScaleLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

# define hyper-parameters
params = {
    "batch_size": 32,
    "linear": 32,
    "lr": 0.0005,
    "decay_factor": 0.9,
    "max_epochs": 3,
}


# (neptune) define model with logging (self.log)
class LitModel(pl.LightningModule):
    def __init__(self, linear, learning_rate, decay_factor):
        super().__init__()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.linear = linear
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.train_img_max = 10
        self.train_img = 0
        self.layer_1 = torch.nn.Linear(28 * 28, linear)
        self.layer_2 = torch.nn.Linear(linear, 20)
        self.layer_3 = torch.nn.Linear(20, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = LambdaLR(optimizer, lambda epoch: self.decay_factor**epoch)
        return [optimizer], [scheduler]

    def training_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train/batch/loss", loss, prog_bar=False)

        y_true = y.cpu().detach().numpy()
        y_pred = y_hat.argmax(axis=1).cpu().detach().numpy()
        acc = accuracy_score(y_true, y_pred)
        self.log("train/batch/acc", acc)
        self.training_step_outputs.append({"loss": loss, "y_true": y_true, "y_pred": y_pred})

        return {"loss": loss, "y_true": y_true, "y_pred": y_pred}

    def on_train_epoch_end(self):
        loss = np.array([])
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in self.training_step_outputs:
            loss = np.append(loss, results_dict["loss"].cpu().detach().numpy())
            y_true = np.append(y_true, results_dict["y_true"])
            y_pred = np.append(y_pred, results_dict["y_pred"])
        acc = accuracy_score(y_true, y_pred)

        self.logger.run.log_metrics(
            data={"train/epoch/loss": loss.mean(), "train/epoch/acc": acc}, step=self.global_step
        )

        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        y_true = y.cpu().detach().numpy()
        y_pred = y_hat.argmax(axis=1).cpu().detach().numpy()

        self.validation_step_outputs.append({"loss": loss, "y_true": y_true, "y_pred": y_pred})

        return {"loss": loss, "y_true": y_true, "y_pred": y_pred}

    def on_validation_epoch_end(self):
        loss = np.array([])
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in self.validation_step_outputs:
            loss = np.append(loss, results_dict["loss"].cpu().detach().numpy())
            y_true = np.append(y_true, results_dict["y_true"])
            y_pred = np.append(y_pred, results_dict["y_pred"])
        acc = accuracy_score(y_true, y_pred)

        # You can also use the log_dict() method from PTL
        self.log_dict(
            {
                "val/epoch/loss": loss.mean(),
                "val/epoch/acc": acc,
            }
        )

        self.validation_step_outputs.clear()

    def test_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        y_true = y.cpu().detach().numpy()
        y_pred = y_hat.argmax(axis=1).cpu().detach().numpy()

        """
        # Log misclassified test images to Neptune
        # Currently Neptune Scale does not support file uploads
        # The example will be updated once file support is added
        """

        self.test_step_outputs.append({"loss": loss, "y_true": y_true, "y_pred": y_pred})

        return {"loss": loss, "y_true": y_true, "y_pred": y_pred}

    def on_test_epoch_end(self):
        loss = np.array([])
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in self.test_step_outputs:
            loss = np.append(loss, results_dict["loss"].cpu().detach().numpy())
            y_true = np.append(y_true, results_dict["y_true"])
            y_pred = np.append(y_pred, results_dict["y_pred"])
        acc = accuracy_score(y_true, y_pred)
        self.log("test/loss", loss.mean())
        self.log("test/acc", acc)
        self.validation_step_outputs.clear()


# define DataModule
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, normalization_vector):
        super().__init__()
        self.batch_size = batch_size
        self.normalization_vector = normalization_vector
        self.mnist_train = None
        self.mnist_val = None
        self.mnist_test = None

    def prepare_data(self):
        MNIST(os.getcwd(), train=True, download=True)
        MNIST(os.getcwd(), train=False, download=True)

    def setup(self, stage):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.normalization_vector[0], self.normalization_vector[1]),
            ]
        )
        if stage == "fit":
            mnist_train = MNIST(os.getcwd(), train=True, transform=transform)
            self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
        if stage == "test":
            self.mnist_test = MNIST(os.getcwd(), train=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=0)


def main():

    # (neptune) create NeptuneLogger
    neptune_logger = NeptuneScaleLogger(
        # api_key = "YOUR_API_KEY",
        # project = "YOUR_WORKSPACE_NAME/YOUR_PROJECT_NAME"
        experiment_name="lightning-experiment",
        log_model_checkpoints=True,  # Logs model checkpoint paths to Neptune
    )

    # (neptune) initialize a trainer and pass neptune_logger
    trainer = pl.Trainer(
        logger=neptune_logger,
        max_epochs=params["max_epochs"],
    )

    # init model
    model = LitModel(
        linear=params["linear"],
        learning_rate=params["lr"],
        decay_factor=params["decay_factor"],
    )

    # init datamodule
    dm = MNISTDataModule(
        normalization_vector=((0.1307,), (0.3081,)),
        batch_size=params["batch_size"],
    )

    # (neptune) log model summary
    neptune_logger.log_model_summary(model=model, max_depth=-1)

    # (neptune) log hyper-parameters
    neptune_logger.log_hyperparams(params=params)

    # train and test the model, log metadata to the Neptune run
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
