import torch
import torch.nn as nn
import torch.optim as optim
from neptune_scale import Run
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleModel(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 128,
        output_size: int = 10,
        num_layers: int = 10,
    ):
        super().__init__()

        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]

        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, output_size))

        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_gradient_norms(self) -> dict:
        """
        Calculate the L2 norm of gradients for each layer.

        Returns:
            dict: Dictionary containing gradient norms for each layer
        """
        gradient_norms = {}

        # Iterate through all named parameters
        for name, param in self.named_parameters():
            if param.grad is not None:
                # Calculate L2 norm of gradients
                norm = param.grad.norm(2).item()
                # Store in dictionary with a descriptive key
                gradient_norms[f"debug/L2_grad_norm/{name}"] = norm

        return gradient_norms


def main():

    # Training parameters
    params = {
        "batch_size": 512,
        "epochs": 10,
        "lr": 0.001,
        "num_layers": 20,  # Configurable number of layers
    }

    # Data transformations
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Load MNIST dataset
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Initialize model, loss function, and optimizer
    model = SimpleModel(num_layers=params["num_layers"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])

    # Step 1: Initialize Neptune Run object
    run = Run(
        experiment_name="debugging-gradient-norms",  # Create a run that is the head of an experiment.
    )

    print(run.get_experiment_url())

    # Step 2: Log configuration parameters
    run.log_configs(
        {
            "config/batch_size": params["batch_size"],
            "config/epochs": params["epochs"],
            "config/lr": params["lr"],
        }
    )

    run.add_tags(tags=["debug", "gradient-norm"])

    print(f"See configuration parameters:\n{run.get_experiment_url() + '&detailsTab=metadata'}")

    # Step 3: Track gradient norms during training
    step_counter = 0
    for epoch in range(params["epochs"]):
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)  # Flatten the images

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            gradient_norms = model.get_gradient_norms()
            batch_loss = loss.item()

            run.log_metrics(
                data={
                    "metrics/train/loss": batch_loss,
                    "epoch": epoch,
                    **gradient_norms,
                },
                step=step_counter,
            )
            step_counter += 1

    run.close()

    # Step 4: Analyze training behavior
    # While your model trains, use Neptune's web interface to monitor and analyze metrics in near real-time:
    # 1. Real-time metric visualization
    # 2. Advanced metric filtering
    # 3. Create custom charts and dashboards
    # 4. Dynamic metric analysis


if __name__ == "__main__":
    main()
