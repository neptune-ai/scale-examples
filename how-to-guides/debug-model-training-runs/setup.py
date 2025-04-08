import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from typing import Dict, Tuple

# Model Classes
class SimpleLLM(nn.Module):
    """A simple language model with a single LSTM layer."""
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(SimpleLLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)  # LSTM returns output and hidden/cell state tuple
        out = self.fc1(lstm_out)  # Use the last output from the LSTM
        return out

class MultilayerModel(nn.Module):
    """A larger language model with multiple LSTM and fully connected layers."""
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(MultilayerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Create multiple LSTM layers
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(hidden_size if i > 0 else embed_size, 
                   hidden_size, 
                   num_layers=1, 
                   batch_first=True)
            for i in range(10)  # 10 LSTM layers
        ])
        
        # Create multiple fully connected layers
        self.fc_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(9)  # 9 FC layers
        ])
        
        # Final layer to project back to vocab size
        self.final_layer = nn.Linear(hidden_size, vocab_size)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Embedding layer
        x = self.embedding(x)
        
        # Process through LSTM layers
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
            x = self.dropout(x)
        
        # Process through FC layers
        for fc in self.fc_layers:
            x = fc(x)
            x = self.dropout(x)
            x = torch.relu(x)
        
        # Final projection
        out = self.final_layer(x)
        return out

# Data Loading Function
def load_and_preprocess_data(params: Dict) -> Tuple[DataLoader, DataLoader, int]:
    """
    Load and preprocess the next token prediction dataset from HuggingFace.
    
    Args:
        params (Dict): Dictionary containing parameters for data loading and model configuration
        
    Returns:
        Tuple[DataLoader, DataLoader, int]: Training dataloader, validation dataloader, and vocabulary size
    """
    # Download dataset
    base_url = "https://huggingface.co/datasets/Na0s/Next_Token_Prediction_dataset/resolve/main/data/"
    data_files = {
        "train": base_url + "train-00001-of-00067.parquet",
        "validation": base_url + "validation-00000-of-00001.parquet",
    }

    # Load dataset
    data_subset = load_dataset("parquet", data_files=data_files, num_proc=4)
    validation_subset = data_subset.get("validation").train_test_split(test_size=0.1)

    print(f"Training samples: {data_subset['train'].num_rows}")
    print(f"Validation samples: {validation_subset['test'].num_rows}")

    # Convert to PyTorch format
    train_subset = data_subset["train"].with_format(
        type="torch", columns=["text", "input_ids", "labels"]
    )
    validation_subset = validation_subset["test"].with_format(
        type="torch", columns=["text", "input_ids", "labels"]
    )

    # Create dataloaders
    train_dataloader = DataLoader(train_subset, batch_size=params["batch_size"], shuffle=True)
    val_dataloader = DataLoader(validation_subset, batch_size=params["batch_size"], shuffle=True)

    # Calculate vocabulary size
    vocab_size = max([token for sentence in data_subset["train"]["input_ids"] for token in sentence]) + 1
    print(f"Vocabulary size: {vocab_size}")

    return train_dataloader, val_dataloader, vocab_size

# Helper function to create model
def create_model(params: Dict, use_multilayer: bool = True) -> nn.Module:
    """
    Create a model based on the specified parameters.
    
    Args:
        params (Dict): Dictionary containing model parameters
        use_multilayer (bool): Whether to use the multilayer model
        
    Returns:
        nn.Module: The created model
    """
    if use_multilayer:
        return MultilayerModel(
            params["vocab_size"], 
            params["embed_size"], 
            params["hidden_size"], 
            params["num_lstm_layers"]
        )
    else:
        return SimpleLLM(
            params["vocab_size"], 
            params["embed_size"], 
            params["hidden_size"], 
            params["num_lstm_layers"]
        )

# Helper function to create optimizer
def create_optimizer(model: nn.Module, params: Dict) -> optim.Optimizer:
    """
    Create an optimizer for the model.
    
    Args:
        model (nn.Module): The model to optimize
        params (Dict): Dictionary containing optimizer parameters
        
    Returns:
        optim.Optimizer: The created optimizer
    """
    if params["optimizer"] == "Adam":
        return optim.Adam(model.parameters(), lr=params["learning_rate"])
    elif params["optimizer"] == "SGD":
        return optim.SGD(model.parameters(), lr=params["learning_rate"])
    else:
        raise ValueError(f"Unsupported optimizer: {params['optimizer']}")

# Helper function to create criterion
def create_criterion() -> nn.Module:
    """
    Create a loss function for the model.
    
    Returns:
        nn.Module: The created criterion
    """
    return nn.CrossEntropyLoss(ignore_index=-100)  # Ignore the buffering index of -100 in the dataset

# Helper function to setup training
def setup_training(params: Dict, use_multilayer: bool = True) -> Tuple[nn.Module, optim.Optimizer, nn.Module, DataLoader, DataLoader, int]:
    """
    Setup the complete training environment.
    
    Args:
        params (Dict): Dictionary containing all parameters
        use_multilayer (bool): Whether to use the multilayer model
        
    Returns:
        Tuple[nn.Module, optim.Optimizer, nn.Module, DataLoader, DataLoader, int]: 
            Model, optimizer, criterion, train dataloader, validation dataloader, and vocabulary size
    """
    # Load data
    train_dataloader, val_dataloader, vocab_size = load_and_preprocess_data(params)
    params["vocab_size"] = vocab_size
    
    # Create model
    model = create_model(params, use_multilayer=use_multilayer)
    # Move model to device
    model.to(params["device"])
    print(f"Model created: {model.__class__.__name__}")
    
    # Create optimizer
    optimizer = create_optimizer(model, params)
    print(f"Optimizer: {optimizer.__class__.__name__}")
    
    # Create criterion
    criterion = create_criterion()
    print(f"Criterion: {criterion.__class__.__name__}")
    
    return model, optimizer, criterion, train_dataloader, val_dataloader, vocab_size

# Explicitly define what should be importable
__all__ = [
    'SimpleLLM',
    'MultilayerModel',
    'load_and_preprocess_data',
    'create_model',
    'create_optimizer',
    'create_criterion',
    'setup_training'
]

if __name__ == "__main__":
    # Example usage with all required parameters
    params = {
        "batch_size": 8,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "embed_size": 1000,
        "hidden_size": 256,
        "num_lstm_layers": 3,
        "optimizer": "Adam",
        "learning_rate": 0.01,
        "vocab_size": 50000  # This will be updated by load_and_preprocess_data
    }
    
    # Setup complete training environment
    model, optimizer, criterion, train_dataloader, val_dataloader, vocab_size = setup_training(params, use_multilayer=True) 