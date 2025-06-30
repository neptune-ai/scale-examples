import torch
import torch.nn as nn
import sys
import os
from neptune_scale import Run
import math

from neptune_torchwatcher import TorchWatcher

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def generate_data(n_samples=1000):
    """Generate synthetic data for a regression task."""
    # Generate random input features
    X = torch.randn(n_samples, 10)

    # Create target values with some non-linear relationship
    y = (X[:, 0] ** 2 + 0.5 * X[:, 1] + 0.1 * torch.sum(X[:, 2:], dim=1)).unsqueeze(1)

    # Add some noise
    y += 0.1 * torch.randn_like(y)

    return X, y


def train_model(model, X_train, y_train, X_val, y_val, watcher, n_epochs=50, batch_size=32):
    """Training function demonstrating different ways to use TorchWatcher."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    n_batches = math.ceil(len(X_train) / batch_size)

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        # Training batches
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            x_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]

            # Forward pass
            output = model(x_batch)
            loss = criterion(output, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics during training
            if epoch < 5:  # First 5 epochs: track everything
                watcher.watch(step=epoch * n_batches + i, namespace="train")
            else:  # Later epochs: track only gradients for efficiency
                watcher.watch(
                    step=epoch * n_batches + i,
                    track_activations=False,
                    track_parameters=False,
                    track_gradients=True,
                    namespace="train",
                )

            train_loss += loss.item()

        # Average training loss
        train_loss /= n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)

            # Track metrics during validation
            watcher.watch(
                step=epoch,
                track_activations=True,
                track_gradients=False,
                track_parameters=False,
                namespace="validation",
            )

        # Log metrics
        watcher.run.log_metrics(
            data={"train/loss": train_loss, "val/loss": val_loss.item()}, step=epoch
        )

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{n_epochs}], "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss.item():.4f}"
            )


def main():
    # Initialize Neptune run
    run = Run(
        experiment_name="torch-watcher-example",
    )

    # Generate data
    X_train, y_train = generate_data(n_samples=1000)
    X_val, y_val = generate_data(n_samples=200)

    # Create model and watcher
    model = SimpleNet()

    # Initialize watcher with specific layer types to track
    watcher = TorchWatcher(
        model,
        run,
        tensor_stats=["mean", "norm"],  # Track mean and norm statistics
        base_namespace="model_internals",  # Default namespace for all metrics
    )

    # Train the model
    print("\nTraining with TorchWatcher:")
    print("- Tracking Linear and ReLU layers")
    print("- Computing mean and norm statistics")
    print("- Using 'model_metrics' as base namespace")
    print("- Full tracking during first 5 epochs with 'train/model_metrics' namespace")
    print("- Gradient-only tracking during later epochs with 'train/model_metrics' namespace")
    print("- Activation-only tracking during validation with 'validation/model_metrics' namespace")

    train_model(model, X_train, y_train, X_val, y_val, watcher)
    run.close()


if __name__ == "__main__":
    main()
