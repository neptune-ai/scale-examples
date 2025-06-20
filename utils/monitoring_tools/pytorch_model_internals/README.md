# TorchWatcher

A lightweight PyTorch model monitoring tool that automatically tracks layer activations, gradients, and parameters during training. Built for seamless integration with neptune.ai.

## Why TorchWatcher?

Training deep learning models often involves monitoring internal layer behavior to understand model performance, debug issues, and optimize training. However, manually implementing hooks and logging for each layer can be time-consuming and error-prone. TorchWatcher solves this by providing:

- **Zero-configuration monitoring**: Automatically detects and tracks all layers in your PyTorch model
- **Comprehensive insights**: Monitor activations, gradients, and parameters with built-in statistical analysis
- **Seamless Neptune integration**: Direct logging to Neptune with organized namespacing for easy experiment tracking
- **Minimal performance impact**: Optimized for production use with configurable tracking options
- **Memory efficient**: Automatically cleans up tensors to prevent memory leaks during long training runs

Whether you're debugging gradient flow issues, monitoring activation patterns, or tracking parameter changes during training, TorchWatcher provides the insights you need without the overhead of manual implementation.

## Features

- **Automatic Layer Tracking**: Monitors activations, gradients, and parameters of specified PyTorch layers
- **Flexible Layer Selection**: Track all layers or specify which layer types to monitor
- **Comprehensive Statistics**: Predefined tensor statistics including mean, standard deviation, norm, min, max, variance, and absolute mean
- **Configurable Tracking**: Enable/disable tracking of activations, gradients, and parameters as needed
- **Organized Logging**: Structured metric namespacing for better organization in Neptune
- **Memory Efficient**: Clears stored tensors after each logging step
- **Error Handling**: Robust error handling with informative warnings

## Changelog

### [v0.1.0] - 2024-03-19

#### Added
- Initial release of TorchWatcher

## Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA 11.7 or higher (for GPU support)
- Neptune account and API token

## Installation

```bash
pip install neptune_scale
```

## Usage

### Quickstart

See `torch_watcher_example.py` for a complete example demonstrating:
- Model definition
- Data generation
- Training loop with different tracking configurations
- Namespace organization
- Integration with Neptune

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

## Benchmarking and Results

### Benchmarking Methodology

All benchmarks were performed using:
- PyTorch 2.0+
- A single RTX5000 GPU in an isolated cluster
- Various model architecture size with Linear and Relu layers only
- Multiple tracking configurations for TorchWatcher
- Generic numeric dataset
- Training parameters:
    - Samples: 4096
    - Batch size: 512
    - Epochs: 20

A reproduction script can be found called - `benchmark_torchwatcher.py`.

### Performance Impact

TorchWatcher is designed to be lightweight and efficient. Our benchmarks show minimal impact on training batch time while providing comprehensive monitoring capabilities across varying model sizes. The largest contributor to performance degradation is the extraction of the model's named parameters (weights and biases). These are extracted using `model.named_parameters()` and creates increased overhead compared to the hooks that PyTorch models support natively for activations and gradients. Additional profiling showed that the `track_parameters` method created the most overhead and can be explored further for future package optimization.

#### Analysis summary
- Larger models (increase in parameters) have a linearly increasing cumulative time per epoch. 
- Larger models spend more time in training loop per batch, as expected.
- Larger models also require more time to extract model internals such as gradients, activations and parameters.
    - As model size increases, extracting model parameters causes increased time in training loop compared to gradients and activations which are extracted with PyTorch hooks.
    - Larger models have less total overhead between model training time in batch vs. time taken to extract values and is minimal if logging activations and gradients. 
- Benchmarking also showed that the average running time per batch remained constant, indicating that there is no leakage or slowdown between training batches.

![Training time overhead](path/to/training_time_overhead.png)
*Figure 1: Training time overhead comparison between baseline training and training with TorchWatcher enabled. Results shown for different model sizes and tracking configurations.*

### Future benchmarking
- Analyse memory overhead
- Train on multiple GPU's
- Test on different datasets and model techniques

## License

Copyright (c) 2025, Neptune Labs Sp. z o.o.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
