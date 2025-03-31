# TorchWatcher

A simple PyTorch hook manager for tracking model metrics during training.

## Features

- Track predefined statistics for specified PyTorch layers
- Automatic metric logging to Neptune
- Support for both training and validation phases
- Selective layer tracking
- Predefined statistics options:
  - `mean`: Mean value
  - `std`: Standard deviation
  - `norm`: L2 norm
  - `min`: Minimum value
  - `max`: Maximum value
  - `var`: Variance
  - `abs_mean`: Mean of absolute values

## Installation

```bash
pip install neptune-client
```

## Usage

### Basic Usage

```python
from neptune_scale import Run
from TorchWatcher import TorchWatcher

# Initialize Neptune run
run = Run(
    experiment_name="my-experiment",
)

# Create your model
model = YourModel()

# Option 1: Track all layers (default)
watcher = TorchWatcher(
    model,
    run,
    tensor_stats=['mean', 'norm']   # Optional: select from predefined statistics
)

# Option 2: Track specific layer types
layers_to_track = [
    nn.Linear,    # Track all linear layers
    nn.Conv2d,    # Track all convolutional layers
    nn.ReLU,      # Track all ReLU layers
]

watcher = TorchWatcher(
    model,
    run,
    track_layers=layers_to_track,  # Specify which layers to track
    tensor_stats=['mean', 'norm']   # Optional: select from predefined statistics
)

# Training loop
for epoch in range(n_epochs):
    for batch in dataloader:
        # Forward pass
        output = model(batch)
        
        # Backward pass
        loss.backward()
        
        # Track metrics
        watcher.watch(step=current_step)
```

### Detailed Examples

#### 1. Using Default Settings

The simplest way to use TorchWatcher is with default settings, which will track all layers in your model:

```python
# Initialize with defaults
watcher = TorchWatcher(
    model,
    run
)

# This will:
# - Track all layers in the model
# - Use only the 'mean' statistic (default)
# - Log all metrics (activations, gradients, parameters)
```

#### 2. Specifying Which Layers to Track

You can choose to track specific layer types in your model:

```python
# Track only convolutional and linear layers
watcher = TorchWatcher(
    model,
    run,
    track_layers=[nn.Conv2d, nn.Linear]
)

# Track only normalization layers
watcher = TorchWatcher(
    model,
    run,
    track_layers=[nn.BatchNorm2d, nn.LayerNorm]
)

# Track only activation layers
watcher = TorchWatcher(
    model,
    run,
    track_layers=[nn.ReLU, nn.LeakyReLU, nn.GELU]
)
```

#### 3. Selecting Statistics to Capture

You can specify which statistics to compute for each layer:

```python
# Track multiple statistics
watcher = TorchWatcher(
    model,
    run,
    tensor_stats=['mean', 'std', 'norm', 'min', 'max']
)

# Track only basic statistics
watcher = TorchWatcher(
    model,
    run,
    tensor_stats=['mean', 'std']
)

# Track only norm-based statistics
watcher = TorchWatcher(
    model,
    run,
    tensor_stats=['norm', 'abs_mean']
)
```

#### 4. Controlling Metric Logging

The `watch()` method allows you to specify which metrics to log:

```python
# Log all metrics (default)
watcher.watch(step=current_step)

# Log only activation metrics
watcher.watch(step=current_step, log=["activations"])

# Log only gradient metrics
watcher.watch(step=current_step, log=["gradients"])

# Log only parameter metrics
watcher.watch(step=current_step, log=["parameters"])

# Log specific combinations
watcher.watch(step=current_step, log=["activations", "gradients"])
```

### Layer Types

You can track any of these PyTorch layer types:
- `nn.Linear`
- `nn.Conv1d`, `nn.Conv2d`, `nn.Conv3d`
- `nn.ConvTranspose1d`, `nn.ConvTranspose2d`, `nn.ConvTranspose3d`
- `nn.BatchNorm1d`, `nn.BatchNorm2d`, `nn.BatchNorm3d`
- `nn.LayerNorm`
- `nn.InstanceNorm1d`, `nn.InstanceNorm2d`, `nn.InstanceNorm3d`
- `nn.GroupNorm`
- `nn.ReLU`, `nn.LeakyReLU`, `nn.PReLU`, `nn.RReLU`, `nn.ELU`, `nn.SELU`, `nn.CELU`, `nn.GELU`
- `nn.Sigmoid`
- `nn.Tanh`
- `nn.Dropout`, `nn.Dropout2d`, `nn.Dropout3d`
- `nn.MaxPool1d`, `nn.MaxPool2d`, `nn.MaxPool3d`
- `nn.AvgPool1d`, `nn.AvgPool2d`, `nn.AvgPool3d`
- `nn.AdaptiveMaxPool1d`, `nn.AdaptiveMaxPool2d`, `nn.AdaptiveMaxPool3d`
- `nn.AdaptiveAvgPool1d`, `nn.AdaptiveAvgPool2d`, `nn.AdaptiveAvgPool3d`
- `nn.LSTM`, `nn.GRU`, `nn.RNN`
- `nn.Embedding`
- `nn.TransformerEncoderLayer`, `nn.TransformerDecoderLayer`

### Available Statistics

The following statistics are available for tracking:

1. Basic Statistics:
   - `mean`: Mean value
   - `std`: Standard deviation
   - `norm`: L2 norm
   - `min`: Minimum value
   - `max`: Maximum value
   - `var`: Variance
   - `abs_mean`: Mean of absolute values

### Example

See `torch_watcher_example.py` for a complete example showing:
- How to track all layers (default behavior)
- How to specify which layers to track
- How to select statistics to monitor
- Integration with a training loop
- Logging metrics to Neptune

## License

MIT License
