# TorchWatcher

TorchWatcher is a powerful tool for monitoring PyTorch models during training. It helps you track activations, gradients, and parameters of your model layers in real-time using neptune.ai for logging.

## Features

- Track layer activations, gradients, and parameters
- Automatic hook management
- Select from predefined tensor statistics
- Flexible tracking configuration
- Support for various layer types (Linear, Conv1d/2d/3d, LSTM, GRU, RNN)

## Installation

Make sure you have the required dependencies:
```bash
pip install torch neptune_scale
```

## Quick Start

1. Initialize your Neptune run:
```python
from neptune_scale import Run
run = Run(
    experiment_name="your-experiment-name",
)
```

2. Create your PyTorch model and initialize TorchWatcher:
```python
from TorchWatcher import TorchWatcher

model = YourModel()
watcher = TorchWatcher(model, run)  # Uses default mean() statistic
```

3. Use the watcher in your training loop:
```python
# Forward pass
output = model(x_batch)
loss = criterion(output, y_batch)

# Backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Track metrics after the forward and backward passes
watcher.watch(step=step)
```

## Advanced Usage

### Tensor Statistics

You can select which predefined statistics to compute:
```python
watcher = TorchWatcher(
    model, 
    run,
    tensor_stats=['mean', 'std', 'norm']  # Select from available statistics
)
```

Available statistics:
- `mean`: Mean value
- `std`: Standard deviation
- `norm`: L2 norm
- `min`: Minimum value
- `max`: Maximum value
- `var`: Variance
- `abs_mean`: Mean of absolute values

By default, only the mean statistic is computed if no statistics are specified.

### Custom Layer Tracking

You can specify which layer types to track:
```python
watcher = TorchWatcher(
    model, 
    run,
    track_layers=[nn.Linear, nn.Conv2d]  # Only track Linear and Conv2d layers
)
```

### Selective Tracking

You can choose which metrics to track:
```python
watcher.watch(
    step=epoch,
    log=["gradients", "parameters"]  # Only track gradients and parameters, not activations
)
```

## Example

See `torch_watcher_example.py` for a complete working example that demonstrates:
- Model definition
- Data generation
- Training loop with metrics tracking
- Validation
- Neptune integration

## Logged Metrics

The following metrics are automatically logged to Neptune:

- `debug/activation/{layer}_{stat_name}`: Layer activation statistics
- `debug/gradient/{layer}_{stat_name}`: Layer gradient statistics
- `debug/parameters/{layer}_{stat_name}`: Parameter statistics

Where `stat_name` corresponds to the statistics you selected. By default, only the mean statistic is computed. 