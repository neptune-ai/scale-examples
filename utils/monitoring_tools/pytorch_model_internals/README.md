# Neptune TorchWatcher

[![Explore in Neptune][Explore in Neptune badge]][Neptune dashboard]

A lightweight PyTorch model monitoring tool that automatically tracks layer activations, gradients, and parameters during training and logs them to Neptune.



## Changelog

### [v0.2.0] - 2025-09-08
- Added support for logging distribution histograms
- Added context manager support

### [v0.1.0] - 2025-07-17
- Initial public release

## Prerequisites
- Neptune account and API token
- `neptune-scale>=0.14.0` Python library
- `torch` (PyTorch) 2.6+

## Usage

### Quickstart

See `scripts/torch_watcher_example.py` for a complete example demonstrating:
- Model definition
- Data generation
- Training loop with different tracking configurations
- Namespace organization
- Integration with Neptune

### Basic usage

Place `neptune_torchwatcher.py` in the CWD and import as another package to your main training script.

```python
import torch
import torch.nn as nn
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
    tensor_stats=['mean', 'norm', 'hist'],       # Choose which statistics to compute (default = ["mean"])
    base_namespace="model_internals"     # Set base namespace for all metrics (default = "model_internals")
)

# Training loop
for epoch in range(n_epochs):
    for batch in train_loader:
        # Forward pass
        output = model(batch)

        # Backward pass
        loss.backward()

        # Track metrics with default configuration
        watcher.watch(step=current_step)
```

### Controlling metric logging

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

### Namespace organization

TorchWatcher provides a hierarchical namespace structure for organizing metrics:

1. **Base namespace**: Set during initialization
```python
watcher = TorchWatcher(
    model,
    run,
    base_namespace="model_internals"  # All metrics will be under "model_internals/"
)
```

2. **Per-call namespace**: Prefix for specific tracking calls
```python
# During training
watcher.watch(step=step, prefix="train")  # Metrics under "train/model_internals/"

# During validation
watcher.watch(step=step, prefix="validation")  # Metrics under "validation/model_internals/"
```

3. **Metric structure**: Metrics are organized as:
```
{prefix}/{base_namespace}/{metric_type}/{layer_name}_{statistic}
```

Example metric names:
- `train/model_internals/activations/fc1/mean`
- `validation/model_internals/gradients/fc2/norm`
- `train/model_internals/parameters/fc1/weight/mean`

### Context Manager Support

TorchWatcher supports context manager usage for automatic cleanup:

```python
with TorchWatcher(model, run) as watcher:
    for epoch in range(n_epochs):
        # Your training code here
        watcher.watch(step=epoch)
# Hooks are automatically removed when exiting the context
```

## Example use cases

1. **Training with full tracking**:
```python
# Track everything during initial training
watcher.watch(step=step, prefix="train")
```

2. **Validation with limited tracking**:
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

3. **Efficient training**:
```python
# Track only gradients during later training phases
if epoch >= 50:
    watcher.watch(
        step=step,
        track_activations=False,
        track_parameters=False,
        track_gradients=True,
        prefix="train"
    )
else:
    watcher.watch(
        step=step,
        track_activations=True,
        track_parameters=True,
        track_gradients=True,
        prefix="train"
    )
```

4. **Debugging specific layers**:
```python
# Track only specific layer types
watcher = TorchWatcher(
    model,
    run,
    track_layers=[nn.Linear, nn.Conv2d],  # Only track Linear and Conv2d layers
    tensor_stats=['mean', 'std', 'hist']
)
```

## Available statistics

Predefined tensor statistics include:
- `mean`: Mean value
- `std`: Standard deviation
- `norm`: L2 norm
- `min`: Minimum value
- `max`: Maximum value
- `var`: Variance
- `abs_mean`: Mean of absolute values
- `hist`: Histogram distribution

These can be extended by adding to the `TENSOR_STATS` dictionary.

### Performance Notes

- **Parameter tracking**: Parameters are always extracted fresh to ensure accuracy during training, as they change with each optimizer step
- **Activation/Gradient tracking**: Uses efficient PyTorch hooks with minimal overhead
- **Histogram computation**: More expensive than basic statistics, use selectively

---

## Support and feedback

We welcome your feedback and contributions!
- For issues or feature requests related to this script, please open a [GitHub Issue][Github issues].
- For general Neptune support, visit the [Neptune support center][Support center].

## License:

Copyright (c) 2025, Neptune Labs Sp. z o.o.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.


[Explore in Neptune badge]: https://neptune.ai/wp-content/uploads/2024/01/neptune-badge.svg
[Github issues]: https://github.com/neptune-ai/scale-examples/issues/new
[Neptune dashboard]: https://scale.neptune.ai/o/examples/org/showcase/runs/details?viewId=standard-view&detailsTab=dashboard&dashboardId=9f67bd03-4080-4d47-83b2-36836b03351c&runIdentificationKey=torch-watcher-example&type=experiment&experimentsOnly=true&runsLineage=FULL&lbViewUnpacked=true&sortBy=%5B%22sys%2Fcreation_time%22%5D&sortFieldType=%5B%22datetime%22%5D&sortFieldAggregationMode=%5B%22auto%22%5D&sortDirection=%5B%22descending%22%5D&experimentOnly=true
[Support center]: https://support.neptune.ai/
