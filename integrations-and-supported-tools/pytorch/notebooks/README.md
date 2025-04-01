# TorchWatcher

A lightweight PyTorch model monitoring tool that automatically tracks layer activations, gradients, and parameters during training. Built for seamless integration with Neptune.ai.

## Features

- **Automatic Layer Tracking**: Monitors activations, gradients, and parameters of specified PyTorch layers
- **Flexible Layer Selection**: Track all layers or specify which layer types to monitor
- **Comprehensive Statistics**: Predefined tensor statistics including mean, standard deviation, norm, min, max, variance, and absolute mean
- **Configurable Tracking**: Enable/disable tracking of activations, gradients, and parameters as needed
- **Organized Logging**: Structured metric namespacing for better organization in Neptune
- **Memory Efficient**: Clears stored tensors after each logging step
- **Error Handling**: Robust error handling with informative warnings

## Installation

```bash
pip install neptune_scale
```

## Usage

### Basic Usage

```python
from neptune_scale import Run
from TorchWatcher import TorchWatcher

# Initialize Neptune run
run = Run(experiment_name="my-experiment")

# Create your PyTorch model
model = YourModel()

# Initialize TorchWatcher
watcher = TorchWatcher(
    model,
    run,
    track_layers=[nn.Linear, nn.ReLU],  # Specify which layer types to track (default is all layers)
    tensor_stats=['mean', 'norm'],       # Choose which statistics to compute (default = "mean" only)
    base_namespace="model_metrics"       # Set base namespace for all metrics (default = "debug")
)

# Training loop
for epoch in range(n_epochs):
    for batch in train_loader:
        # Forward pass
        output = model(batch)

        # Backward pass
        loss.backward()

        # Track metrics with default namespace
        watcher.watch(step=current_step)
```

### Controlling Metric Logging

The `watch()` method provides flexible control over what metrics to track and how to organize them:

```python
# Track all metrics (default behavior)
watcher.watch(step=current_step)

# Track specific metrics
watcher.watch(
    step=current_step,
    track_activations=True,
    track_gradients=False,
    track_parameters=True
)

# Use a namespace to organize metrics
watcher.watch(
    step=current_step,
    namespace="train"  # Metrics will be under "train/model_metrics/..."
)
```

### Namespace Organization

TorchWatcher provides a hierarchical namespace structure for organizing metrics:

1. **Base Namespace**: Set during initialization
```python
watcher = TorchWatcher(
    model,
    run,
    base_namespace="model_metrics"  # All metrics will be under "model_metrics/"
)
```

2. **Per-Call Namespace**: Prefix for specific tracking calls
```python
# During training
watcher.watch(step=step, namespace="train")  # Metrics under "train/model_metrics/"

# During validation
watcher.watch(step=step, namespace="validation")  # Metrics under "validation/model_metrics/"
```

3. **Metric Structure**: Metrics are organized as:
```
{namespace}/{base_namespace}/{metric_type}/{layer_name}_{statistic}
```

Example metric names:
- `train/model_metrics/activation/fc1_mean`
- `validation/model_metrics/gradient/fc2_norm`
- `train/model_metrics/parameters/fc1_weight_mean`

### Example Use Cases

1. **Training with Full Tracking**:
```python
# Track everything during initial training
watcher.watch(step=step, namespace="train")
```

2. **Validation with Limited Tracking**:
```python
# Track only activations during validation
watcher.watch(
    step=step,
    track_activations=True,
    track_gradients=False,
    track_parameters=False,
    namespace="validation"
)
```

3. **Efficient Training**:
```python
# Track only gradients during later training phases
watcher.watch(
    step=step,
    track_activations=False,
    track_parameters=False,
    track_gradients=True,
    namespace="train"
)
```

## Supported Layer Types

TorchWatcher supports tracking of all common PyTorch layer types, including:
- Linear layers
- Convolutional layers
- Recurrent layers
- Normalization layers
- Activation layers
- Pooling layers
- Dropout layers
- Embedding layers
- Transformer layers
- Attention layers
- And more...

## Available Statistics

Predefined tensor statistics include:
- `mean`: Mean value
- `std`: Standard deviation
- `norm`: L2 norm
- `min`: Minimum value
- `max`: Maximum value
- `var`: Variance
- `abs_mean`: Mean of absolute values

## Example

See `torch_watcher_example.py` for a complete example demonstrating:
- Model definition
- Data generation
- Training loop with different tracking configurations
- Namespace organization
- Integration with Neptune

## License

MIT License
