# Important: This script can only be run when using multiple GPUS (> 1)

import os
from typing import Dict, Tuple, Optional, Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler


def create_dataloader_minst(rank: int, world_size: int, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Create distributed data loaders for MNIST dataset.
    
    Args:
        rank (int): Process rank
        world_size (int): Total number of processes
        batch_size (int): Batch size per process
        
    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation data loaders
    """
    # Transform to normalize the data and convert it to tensor
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Normalizing the image to range [-1, 1]
        ]
    )

    # Download and load the MNIST dataset
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    return train_loader, val_loader

# Simple Convolutional Neural Network model for MNIST
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=3, padding=1
        )  # Input channels = 1 (grayscale images)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Flattened size of image after convolution layers
        self.fc2 = nn.Linear(128, 10)  # 10 output classes for digits 0-9

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # Pooling layer to downsample
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to evaluate the model during validation
def evaluate(model, data_loader, criterion, device):
    model.eval()  # Ensure model is in training mode if tracking gradients
    correct_preds = 0
    total_preds = 0
    epoch_loss = 0
    with torch.no_grad():  # Disable gradient tracking during evaluation
        for data, target in data_loader:

            data, target = data.to(device), target.to(device)

            # Forward pass (with gradient tracking if specified)
            output = model(data)
            loss = criterion(output, target)  # Correct loss computation
            epoch_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total_preds += target.size(0)
            correct_preds += (predicted == target).sum().item()

    accuracy = correct_preds / total_preds
    return epoch_loss / len(data_loader), accuracy

## Setup distributed environment
def setup_distributed(rank: int, world_size: int, backend: str) -> None:
    """
    Initialize the distributed environment.
    
    Args:
        rank (int): Process rank
        world_size (int): Total number of processes
        backend (str): Distributed backend to use
        
    Raises:
        RuntimeError: If distributed initialization fails
    """
    try:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize distributed environment: {e}")

def train(rank: int, model: nn.Module, params: Dict[str, Any], 
          train_loader: DataLoader, val_loader: DataLoader, 
          run: Optional[Any] = None) -> None:
    """
    Train the model using distributed data parallel.
    
    Args:
        rank (int): Process rank
        model (nn.Module): Model to be trained
        params (Dict[str, Any]): Training parameters
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        run (Optional[Any]): Neptune run object for logging
        
    Raises:
        RuntimeError: If training fails
    """
    # Instantiate the device, loss function, and optimizer
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(f"Rank {rank} using device: {device}")
    
    # Move model to device first
    model = model.to(device)
    
    # Then wrap with DDP
    model = DDP(model, device_ids=[rank])
    
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    try:
        # Training loop
        num_epochs = params["epochs"]
        step_counter = 0
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            correct_preds = 0
            total_preds = 0

            # Training step
            for batch_idx, (data, target) in enumerate(train_loader, 0):
                step_counter += 1
                optimizer.zero_grad()
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(output.data, 1)
                total_preds += target.size(0)
                correct_preds += (predicted == target).sum().item()
                batch_accuracy = correct_preds / total_preds

                # Validation step per training step
                val_loss, val_accuracy = evaluate(
                    model, val_loader, criterion, device
                )  # Evaluate after each step

                if rank == 0:
                    # Log metrics
                    run.log_metrics(
                        data={
                            "metrics/train/loss": loss.item(),
                            "metrics/train/accuracy": batch_accuracy,
                            "metrics/validation/loss": val_loss,
                            "metrics/validation/accuracy": val_accuracy,
                            "epoch_value": epoch,
                        },
                        step=step_counter,
                    )

                dist.barrier()  # synchronize processes before moving to next step

        dist.destroy_process_group()

    except Exception as e:
        raise RuntimeError(f"Error during training process (Rank {rank}): {e}")

def run_ddp(rank: int, world_size: int, params: Dict[str, Any]) -> None:
    """
    Run distributed data parallel training.
    
    Args:
        rank (int): Process rank
        world_size (int): Total number of processes
        params (Dict[str, Any]): Training parameters
        
    Raises:
        RuntimeError: If DDP training fails
    """
    try:
        setup_distributed(rank, world_size, "nccl")
        train_loader, val_loader = create_dataloader_minst(rank, world_size, params["batch_size"])

        # Initialize Neptune logger only on the main process from rank 0
        if rank == 0:
            from uuid import uuid4
            from neptune_scale import Run

            # Initialize Neptune logger
            run = Run(run_id=f"ddp-{uuid4()}", experiment_name="pytorch-ddp-experiment")

            # Log all parameters
            run.log_configs({
                "config/learning_rate": params["learning_rate"],
                "config/batch_size": params["batch_size"],
                "config/num_gpus": params["num_gpus"],
                "config/n_classes": params["n_classes"],
            })

            # Add descriptive tags
            run.add_tags(tags=["Torch-MINST", "ddp", "single-node", params["optimizer"]])

            print(f"View experiment charts:\n{run.get_run_url() + '&detailsTab=charts'}")
        else:
            run = None

        model = SimpleCNN()
        train(rank, model, params, train_loader, val_loader, run)

        # Once training is finished, close the Neptune run from the main process
        if rank == 0:
            run.close()
    except Exception as e:
        raise RuntimeError(f"Failed to run DDP training: {e}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

# Run DDP
if __name__ == "__main__":

    # Set environment variables for DDP setup
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Set environment variables for Neptune
    os.environ["NEPTUNE_PROJECT"] = "your_project_name/your_workspace_name"
    os.environ["NEPTUNE_API_TOKEN"] = "your_api_token"

    # Set parameters
    params = {
        "optimizer": "Adam",
        "batch_size": 512,
        "learning_rate": 0.01,
        "epochs": 5,
        "num_gpus": torch.cuda.device_count(),
        "n_classes": 10,
    }

    # Spawn ddp job to multiple GPU's
    print(f"Example will use {params['num_gpus']} GPU's")
    mp.set_start_method("spawn", force=True)
    mp.spawn(run_ddp, args=(params["num_gpus"], params), nprocs=params["num_gpus"], join=True)
