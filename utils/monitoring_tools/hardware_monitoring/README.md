# Neptune hardware monitoring

[![Explore in Neptune][Explore in Neptune badge]][Neptune dashboard]


An extensible Python module for logging system and process hardware metrics (CPU, memory, disk, network, GPU, and more) to Neptune.

---

## Changelog

**v0.2.0** (2025-06-06)
- Added `namespace` and `sampling_rate` arguments to `SystemMetricsMonitor` to allow for custom metric namespaces and sampling rates.
- Made `pynvml` and `torch` optional dependencies.
- Added process memory, threads, and file descriptors metrics.
- Updated default namespace to `runtime` from `system`.
- Removed `node_rank` from metric namespaces.
- Improved console logging.

**v0.1.0** (2025-03-27)
- Initial release

---

## Prerequisites

- Python 3.9+
- [Neptune](https://scale.neptune.ai/) account and API token
- `neptune-scale` Python library installed (`pip install neptune-scale`)
- (Optional, for device detection) `torch` (PyTorch)
- (Optional, for GPU metrics) NVIDIA GPU and drivers
- (Optional, for GPU monitoring) `pynvml` (`pip install nvidia-ml-py`)

---

## Instructions

1. **Install dependencies:**
   ```bash
   pip install neptune-scale

   # For GPU monitoring if using NVIDIA GPUs:
   pip install -U 'nvidia-ml-py<13.0.0'
   ```

2. **Set up Neptune:**
   - Set your Neptune API token and project as environment variables:
     ```bash
     export NEPTUNE_API_TOKEN=your_token
     export NEPTUNE_PROJECT=your_workspace/your_project
     ```

3. Download and place `neptune_hardware_monitoring.py` in your project directory.

4. **Integrate the monitoring class in your script:**  
   - Using a context manager:
    ```python
    from neptune_scale import Run
    from neptune_hardware_monitoring import SystemMetricsMonitor

    run = Run(experiment_name=...)

    with SystemMetricsMonitor(run=run):
        # Your training or workload code here
        ...
    ```
    - Using `start()` and `stop()` methods:
    ```python
    from neptune_scale import Run
    from neptune_hardware_monitoring import SystemMetricsMonitor

    run = Run(experiment_name=...)

    monitor = SystemMetricsMonitor(run=run)

    monitor.start()
    # Your training or workload code here
    monitor.stop()
    ```

5. **Metrics will be logged to Neptune automatically** under the specified namespace.

## API reference

```python
class SystemMetricsMonitor(
    run: Run,
    sampling_rate: float = 5.0,
    namespace: str = "runtime",
)
```
- **run**: Neptune Run object for logging metrics.
- **sampling_rate**: How often to sample metrics (in seconds). Default is `5.0`.
- **namespace**: [Namespace][Docs namespaces and attributes] where the metrics will be logged in the Neptune run. Default is `"runtime"`.

---

## Note

- **GPU monitoring requires `pynvml` and a compatible NVIDIA GPU.** If not available, GPU metrics are skipped and a warning is shown.
- **Each process logs its own metrics.** In multi-process scenarios when logging to the same Neptune run, ensure that each process uses a different `namespace`.
- Windows will log `num_handles` to the file descriptor (`process/num_fds`) attribute.

---

## Output
Details and metrics are logged to Neptune in a structured namespace. Example:

```
runtime/details/gpu/0/name
runtime/details/gpu_num
runtime/monitoring/cpu/percent
runtime/monitoring/memory/virtual_used_GiB
runtime/monitoring/gpu/0/memory_used_MiB
runtime/monitoring/process/rss_memory_MiB
...
```

[Explore the structure in Neptune][Neptune attributes]

---
To visualize all important hardware metrics, create [custom dashboards][Docs custom dashboards].

[Explore a sample dashboard in Neptune][Neptune dashboard]

- Add single metric charts, like _CPU Utilization %_
- Add multi-metric charts, like _Network IO_
- [Dynamically select metrics][Docs dynamic metric selection] so that the chart is up-to-date even when the number of metrics is unknown or changes, like _GPU Utilization (%)_ and _GPU Power Usage (W)_

> [!TIP]
> Since hardware metrics are logged in a separate thread from the model training, their steps might not be in sync with the model training steps. We recommend using [Relative time as the x-axis][Docs configure the x-axis] for hardware metrics.


---

## Extending the monitor

To add custom metrics, extend the class with new `_collect_*_metrics` methods and call them from `_collect_metrics`.

---

## Support and feedback

We welcome your feedback and contributions!
- For issues, feature requests, or questions, please open a [GitHub Issue][Github issues].
- For general Neptune support, visit the [Neptune support center][Support center].

---

## License

Copyright (c) 2025, Neptune Labs Sp. z o.o.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.

<!-- Ref links -->
[Docs configure the x-axis]: https://docs.neptune.ai/chart_widget/#configure-the-x-axis
[Docs custom dashboards]: https://docs.neptune.ai/custom_dashboard
[Docs dynamic metric selection]: https://docs.neptune.ai/chart_widget/#dynamic-metric-selection
[Docs namespaces and attributes]: https://docs.neptune.ai/namespaces_and_attributes/
[Explore in Neptune badge]: https://neptune.ai/wp-content/uploads/2024/01/neptune-badge.svg
[Github issues]: https://github.com/neptune-ai/scale-examples/issues/new
[Neptune attributes]: https://scale.neptune.ai/o/examples/org/showcase/runs/details?viewId=9f113328-75aa-4c61-9aa8-5bbdffa90879&detailsTab=attributes&runIdentificationKey=hardware_monitoring&type=experiment&experimentsOnly=true&runsLineage=FULL&lbViewUnpacked=true&sortBy=%5B%22sys%2Fcreation_time%22%5D&sortFieldType=%5B%22datetime%22%5D&sortFieldAggregationMode=%5B%22auto%22%5D&sortDirection=%5B%22descending%22%5D&suggestionsEnabled=false&query=&experimentOnly=true&path=runtime%2F
[Neptune dashboard]: https://scale.neptune.ai/o/examples/org/showcase/runs/details?viewId=9f113328-75aa-4c61-9aa8-5bbdffa90879&detailsTab=dashboard&dashboardId=9f11330c-e4ff-413a-9faa-9e10e5b3f7ee&runIdentificationKey=hardware_monitoring&type=experiment&experimentsOnly=true&runsLineage=FULL&lbViewUnpacked=true&sortBy=%5B%22sys%2Fcreation_time%22%5D&sortFieldType=%5B%22datetime%22%5D&sortFieldAggregationMode=%5B%22auto%22%5D&sortDirection=%5B%22descending%22%5D&suggestionsEnabled=false&query=&experimentOnly=true
[Support center]: https://support.neptune.ai/
