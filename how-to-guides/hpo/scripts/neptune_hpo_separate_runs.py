import math
import uuid
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from neptune_scale import Run
from torchvision import datasets, transforms
from tqdm.auto import tqdm, trange

ALLOWED_DATATYPES = [int, float, str, datetime, bool, list, set]

parameters = {
    "batch_size": 128,
    "input_size": (1, 28, 28),
    "n_classes": 10,
    "epochs": 3,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

input_size = math.prod(parameters["input_size"])

learning_rates = [0.075, 0.1, 0.25]  # learning rate choices


class BaseModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(BaseModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    model = BaseModel(
        input_size,
        parameters["n_classes"],
    ).to(parameters["device"])

    criterion = nn.CrossEntropyLoss()

    data_tfms = {
        "train": transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    }

    trainset = datasets.MNIST(
        root="mnist",
        train=True,
        download=True,
        transform=data_tfms["train"],
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=parameters["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    sweep_id = str(uuid.uuid4())

    sweep_run = Run(
        family=f"sweep-{sweep_id}",
        run_id=f"sweep-{sweep_id}",
    )

    sweep_run.add_tags(["sweep", "script"])
    sweep_run.add_tags([sweep_id], group_tags=True)

    for key in parameters:
        if type(parameters[key]) not in ALLOWED_DATATYPES:
            sweep_run.log_configs({f"config/{key}": str(parameters[key])})
        else:
            sweep_run.log_configs({f"config/{key}": parameters[key]})

    # Initialize fields for best values across all trials
    best_acc = None

    for trial, lr in tqdm(
        enumerate(learning_rates),
        total=len(learning_rates),
        desc="Trials",
    ):
        # Create a trial-level run
        with Run(
            family=f"sweep-{sweep_id}",
            run_id=f"trial-{sweep_id}-{trial}",
        ) as trial_run:
            trial_run.add_tags(["trial", "script"])

            # Add sweep_id to the trial-level run
            trial_run.add_tags([sweep_id], group_tags=True)

            # Log trial number and hyperparams
            trial_run.log_configs({"trial_num": trial, "parameters/lr": lr})

            optimizer = optim.SGD(model.parameters(), lr=lr)

            step = 0

            for epoch in trange(parameters["epochs"], desc=f"Trial {trial} - lr: {lr}"):
                trial_run.log_metrics(step=epoch, data={"epochs": epoch})

                for x, y in trainloader:
                    x, y = x.to(parameters["device"]), y.to(parameters["device"])
                    optimizer.zero_grad()
                    x = x.view(x.size(0), -1)
                    outputs = model(x)
                    loss = criterion(outputs, y)

                    _, preds = torch.max(outputs, 1)
                    acc = (torch.sum(preds == y.data)) / len(x)

                    # Log trial metrics
                    trial_run.log_metrics(
                        step=step,
                        data={
                            "metrics/batch/loss": float(loss),
                            "metrics/batch/acc": float(acc),
                        },
                    )

                    # Log best values across all trials to Sweep-level run
                    if best_acc is None or acc > best_acc:
                        best_acc = acc
                        sweep_run.log_configs(
                            {
                                "best/trial": trial,
                                "best/metrics/loss": float(loss),
                                "best/metrics/acc": float(acc),
                                "best/parameters/lr": lr,
                            }
                        )

                    loss.backward()
                    optimizer.step()

                    step += 1

    sweep_run.close()
