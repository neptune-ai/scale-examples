# TODO: cleanup model classes for input params

# Important: This script can only be run when using multiple GPUS (> 1)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

def create_dataloader_minst(rank, world_size, batch_size):
    # Transform to normalize the data and convert it to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalizing the image to range [-1, 1]
    ])

    # Download and load the MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)  # Use test set as validation
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# Simple Convolutional Neural Network model for MNIST
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Input channels = 1 (grayscale images)
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

class SimpleNN(nn.Module):
    def __init__(self, params):
        super(SimpleNN, self).__init__()
        # Define layers (increase number of layers)
        self.params = params

        self.fc1 = nn.Linear(params["input_size"], params["input_features"])
        self.fc2 = nn.Linear(params["input_features"], 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, params["n_classes"])      # Output layer (10 classes for MNIST)

    def forward(self, x):
        x = x.view(-1, self.params["input_size"])  # Flatten the input image (28x28)
        x = torch.relu(self.fc1(x))  # Apply ReLU activation
        x = torch.relu(self.fc2(x))  # Apply ReLU activation
        x = torch.relu(self.fc3(x))  # Apply ReLU activation
        x = torch.relu(self.fc4(x))  # Apply ReLU activation
        x = self.fc5(x)  # Output layer
        return x
    
    # Function to evaluate the model (validation/test) with gradients tracked
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
def setup(rank, world_size, backend):

    import os

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12535'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def clean_up():
    dist.destroy_process_group()

def setupModel(rank, model, params):
    # Instantiate the device, loss function, and optimizer
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DDP(model, device_ids=[rank])

    # Select an optimizer
    if params["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
        print(params["optimizer"])
    elif params["optimizer"] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=params["learning_rate"])
        print(params["optimizer"])
    else:
        print("No optimizer selected")

    return model, optimizer, device

def train(rank: int, model, params, train_loader, val_loader, run):

    model, optimizer, device = setupModel(rank, model, params)

    try:
        criterion = nn.CrossEntropyLoss()  # Loss function

        # Training loop
        num_epochs = params["epochs"]
        print(num_epochs)
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

                # Forward pass
                output = model(data)

                # Compute the loss
                loss = criterion(output, target)
 
                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(output.data, 1)
                total_preds += target.size(0)
                correct_preds += (predicted == target).sum().item()

                batch_accuracy = correct_preds / total_preds

                # Validation step per training step
                val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)  # Evaluate after each step

                if rank == 0:
                    print(f"Train loss: {loss.item()}")

                    run.log_metrics(
                        data = {
                            "metrics/train/loss": loss.item(),
                            "metrics/train/accuracy": batch_accuracy,
                            "metrics/validation/loss": val_loss,
                            "metrics/validation/accuracy": val_accuracy,
                            "epoch_value": epoch
                        },
                        step = step_counter
                    )

                dist.barrier() # synchonize processes before moving to next step

        clean_up() # Clean up processes from DDP training

    except Exception as e:
        print(f"Error during training process (Rank {rank}): {e}")

def run_ddp(rank, world_size, params):

    setup(rank, world_size, "nccl")

    # Initialize the Neptune run
    from neptune_scale import Run
    from uuid import uuid4
    if rank == 0: # Create a run object only on the main process
        run = (
            Run(    
                project="leo/pytorch-tutorial",
                api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vc2NhbGUubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3NjYWxlLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMGIyNGUwYzMtMDg2Ni00YTZlLWIyYTctZDUxN2I4ZjE5MzA1In0=",
                run_id=f"ddp-{uuid4()}")
                if rank == 0 else None
        )

        run.log_configs(
        {
            "config/learning_rate": params["learning_rate"],
            "config/optimizer": params["optimizer"],
            "config/batch_size": params["batch_size"],
            "config/epochs": params["epochs"],
            "config/input_size": params["input_size"]
        }
        )

        run.add_tags(tags=[params["optimizer"]], group_tags=True)
        run.add_tags(tags=["Torch-MINST", "ddp", "single-node"])

    else: 
        run = None
    
    model = SimpleNN(params)
    train_loader, val_loader = create_dataloader_minst(rank, world_size, params["batch_size"])
    train(rank, model, params, train_loader, val_loader, run)

# Run DDP
if __name__ == "__main__":

    # Set parameters
    params = {
    "optimizer": "Adam",
    "batch_size": 512,
    "learning_rate": 0.01,
    "epochs": 5,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "num_gpus": torch.cuda.device_count(),
    "input_features": 256,
    "n_classes": 10,
    "input_size": 28 * 28
    }
    
    # Spawn ddp job to multiple devices
    num_gpu = torch.cuda.device_count()
    print(params["num_gpus"])
    mp.set_start_method('spawn', force=True)
    mp.spawn(run_ddp, args=(num_gpu, params), nprocs=num_gpu, join=True)
