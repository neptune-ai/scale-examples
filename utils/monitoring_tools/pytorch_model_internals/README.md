# Neptune TorchWatcher

[![Explore in Neptune][Explore in Neptune badge]][Neptune dashboard]

A lightweight PyTorch model monitoring tool that automatically tracks layer activations, gradients, and parameters during training. Built for seamless integration with neptune.ai.

## Why TorchWatcher?

Training deep learning models often involves monitoring internal layer behavior to understand model performance, debug issues, and optimize training. However, manually implementing hooks and logging for each layer can be time-consuming and error-prone. TorchWatcher solves this by providing:

- **Zero-configuration monitoring**: Automatically detects and tracks all layers in your PyTorch model
- **Comprehensive insights**: Monitor activations, gradients, and parameters with built-in statistical analysis (mean, std, norm, min, max, variance, abs_mean)
- **Seamless Neptune integration**: Direct logging to Neptune with organized namespacing for easy experiment tracking
- **Flexible configuration**: Track all layers or specify which layer types to monitor, with configurable tracking options
- **Production-ready**: Optimized for minimal performance impact with automatic tensor cleanup to prevent memory leaks
- **Robust error handling**: Informative warnings and graceful error recovery

Whether you're debugging gradient flow issues, monitoring activation patterns, or tracking parameter changes during training, TorchWatcher provides the insights you need without the overhead of manual implementation.

## Changelog

### [v0.1.0] - 2025-07-16

#### Added
- Initial release of TorchWatcher

## Prerequisites
- Neptune account and API token
- `neptune-scale` Python library installed (`pip install neptune-scale`)
- `torch` (PyTorch) 2.6+

## Usage

### Quickstart

See `scripts/torch_watcher_example.py` for a complete example demonstrating:
- Model definition
- Data generation
- Training loop with different tracking configurations
- Namespace organization
- Integration with Neptune

### Basic Usage

Place `neptune_torchwatcher.py` in the CWD and import as another package to your main training script.

```python
from neptune_scale import Run
from neptune_torchwatcher import TorchWatcher

# Initialize Neptune run
run = Run(experiment_name="my-experiment")

# Create your PyTorch model
model = YourModel()

# Initialize TorchWatcher
watcher = TorchWatcher(
    model,
    run,
    track_layers=[nn.Linear, nn.ReLU],  # Specify which layer types to track (default is all layers)
    tensor_stats=['mean', 'norm'],       # Choose which statistics to compute (default = ["mean"])
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

# Use a prefix to organize metrics
watcher.watch(
    step=current_step,
    prefix="train"  # Metrics will be under "train/model_internals/..."
)
```

### Namespace Organization

TorchWatcher provides a hierarchical namespace structure for organizing metrics:

1. **Base Namespace**: Set during initialization
```python
watcher = TorchWatcher(
    model,
    run,
    base_namespace="model_internals"  # All metrics will be under "model_internals/"
)
```

2. **Per-Call Namespace**: Prefix for specific tracking calls
```python
# During training
watcher.watch(step=step, prefix="train")  # Metrics under "train/model_internals/"

# During validation
watcher.watch(step=step, prefix="validation")  # Metrics under "validation/model_internals/"
```

3. **Metric Structure**: Metrics are organized as:
```
{prefix}/{base_namespace}/{metric_type}/{layer_name}_{statistic}
```

Example metric names:
- `train/model_internals/activations/fc1/mean`
- `validation/model_internals/gradients/fc2/norm`
- `train/model_internals/parameters/fc1/weight/mean`

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
    prefix="validation"
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
    prefix="train"
)
```


## Available Statistics

Predefined tensor statistics include:
- `mean`: Mean value
- `std`: Standard deviation
- `norm`: L2 norm
- `min`: Minimum value
- `max`: Maximum value
- `var`: Variance
- `abs_mean`: Mean of absolute values

These can be extended by adding to the `TENSOR_STATS` dictionary.

<details><summary><h2>Performance Benchmarks</h2></summary>

### Benchmarking Methodology

All benchmarks were performed using:
- PyTorch 2.6.0
- A single RTX5000 GPU
- Various model architecture sizes with Linear and Relu layers only
- Multiple tracking configurations for TorchWatcher
- Generic numeric dataset
- Training parameters:
    - Samples: 4096
    - Batch size: 512
    - Epochs: 20

### Performance Impact

TorchWatcher is designed to be lightweight and efficient. Our benchmarks show minimal impact on training batch time while providing comprehensive monitoring capabilities across varying model sizes. The largest contributor to performance degradation is the extraction of the model's named parameters (weights and biases). These are extracted using `model.named_parameters()` and creates increased overhead compared to the hooks that PyTorch models support natively for activations and gradients. Additional profiling showed that the `track_parameters` method created the most overhead and can be explored further for future package optimization.

#### Analysis summary
- Larger models also require more time to extract model internals such as gradients, activations and parameters.
    - As model size increases, extracting model parameters causes increased time in training loop compared to gradients and activations which are extracted with PyTorch hooks.
    - Larger models have less total overhead between model training time in batch vs. time taken to extract values and is minimal if logging activations and gradients.
- Benchmarking also showed that the average running time per batch remained constant, indicating that there is no leakage or slowdown between training batches.

![benchmark_analysis](https://github.com/user-attachments/assets/7981d186-3cf9-4a81-bc5c-d8ee4fb8c689)
*Figure 1: Training time overhead comparison between baseline training and training with TorchWatcher enabled. Results shown for different model sizes and tracking configurations.*

### Future benchmarking
- Analyze memory overhead
- Train on multiple GPUs
- Test on different datasets and model techniques
</details>

## Support and feedback

We welcome your feedback and contributions!
- For issues or feature requests related to this script, please open a [GitHub Issue][Github issues].
- For general Neptune support, visit the [Neptune support center][Support center].

## License

Copyright (c) 2025, Neptune Labs Sp. z o.o.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.


[Explore in Neptune badge]: https://neptune.ai/wp-content/uploads/2024/01/neptune-badge.svg
[Github issues]: https://github.com/neptune-ai/scale-examples/issues/new
[Neptune dashboard]: https://scale.neptune.ai/examples/showcase/runs/details?viewId=standard-view&detailsTab=dashboard&dashboardId=9f67bd03-4080-4d47-83b2-36836b03351c&runIdentificationKey=torch-watcher-example&type=experiment&experimentsOnly=true&runsLineage=FULL&lbViewUnpacked=true&sortBy=%5B%22sys%2Fcreation_time%22%5D&sortFieldType=%5B%22datetime%22%5D&sortFieldAggregationMode=%5B%22auto%22%5D&sortDirection=%5B%22descending%22%5D&experimentOnly=true
[Support center]: https://support.neptune.ai/
