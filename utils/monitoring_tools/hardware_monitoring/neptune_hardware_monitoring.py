import contextlib
import os
import socket
import threading
import time
import traceback
import warnings
from typing import Optional

import psutil
import torch
from neptune_scale import Run

try:
    import pynvml
except ImportError as e:
    raise ImportError("`pynvml` is not installed. Install using `pip install nvidia-ml-py`.") from e


class SystemMetricsMonitor:
    """Monitor system metrics in a background thread and log them to Neptune."""

    def __init__(
        self,
        run: Run,
        sampling_rate: float = 5.0,
        namespace: str = "runtime",
    ):
        """
        Initialize the system metrics monitor.

        Args:
            run: Neptune Run object for logging metrics
            sampling_rate (default=5.0): How often to sample metrics (in seconds)
            namespace (default="runtime"): Namespace where the metrics will be logged in the Neptune run
        """
        self.run = run
        self.namespace = namespace
        self.sampling_rate = sampling_rate
        self._stop_event = threading.Event()
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_step = 0

        self.hostname = socket.gethostname()
        self.node_rank = int(os.environ.get("NODE_RANK", 0))

        try:
            pynvml.nvmlInit()
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            self.has_gpu = True
        except pynvml.NVMLError:
            warnings.warn(
                "No NVIDIA GPU available or driver issues. GPU monitoring will be disabled."
            )
            self.has_gpu = False
            self.gpu_count = 0

        self._log_system_details()

        self.start()

    def _log_system_details(self):
        """Log static system details as configs."""
        system_details = {
            "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            "gpu_num": self.gpu_count,
            "cpu_num": psutil.cpu_count(logical=False),
            "cpu_logical_num": psutil.cpu_count(logical=True),
            "hostname": self.hostname,
        }

        self.run.log_configs(
            {
                f"{self.namespace}/details/node_{self.node_rank}/{key}": value
                for key, value in system_details.items()
            }
        )

        if self.has_gpu:
            try:
                for i in range(self.gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    self.run.log_configs(
                        {f"{self.namespace}/details/node_{self.node_rank}/gpu/name/{i}": name}
                    )
            except pynvml.NVMLError as e:
                warnings.warn(
                    f"Error getting GPU details on node {self.hostname} (rank {self.node_rank}): {e}"
                )

    def _collect_gpu_metrics(self, metrics: dict):
        """Collect GPU metrics if available."""
        if not self.has_gpu:
            return

        prefix = f"{self.namespace}/monitoring/node_{self.node_rank}"

        try:
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                metrics[f"{prefix}/gpu/{i}/memory_used_GB"] = memory.used / (1024**3)
                metrics[f"{prefix}/gpu/{i}/memory_total_GB"] = memory.total / (1024**3)
                metrics[f"{prefix}/gpu/{i}/memory_free_GB"] = memory.free / (1024**3)
                metrics[f"{prefix}/gpu/{i}/memory_utilized_percent"] = (
                    memory.used / memory.total
                ) * 100

                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                metrics[f"{prefix}/gpu/{i}/temperature_celsius"] = temp

                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics[f"{prefix}/gpu/{i}/gpu_utilization_percent"] = utilization.gpu
                metrics[f"{prefix}/gpu/{i}/memory_utilization_percent"] = utilization.memory

                with contextlib.suppress(pynvml.NVMLError):
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                    metrics[f"{prefix}/gpu/{i}/power_usage_watts"] = power

        except pynvml.NVMLError as e:
            warnings.warn(
                f"Error collecting GPU metrics on node {self.hostname} (rank {self.node_rank}): {e}"
            )

    def _collect_metrics(self) -> dict:
        """Collect current system metrics."""

        prefix = f"{self.namespace}/monitoring/node_{self.node_rank}"

        metrics = {f"{prefix}/cpu/percent": psutil.cpu_percent()}

        virtual_memory = psutil.virtual_memory()
        metrics[f"{prefix}/memory/virtual_used_GB"] = virtual_memory.used / (1024**3)
        metrics[f"{prefix}/memory/virtual_utilized_percent"] = virtual_memory.percent

        swap_memory = psutil.swap_memory()
        metrics[f"{prefix}/memory/swap_used_GB"] = swap_memory.used / (1024**3)
        metrics[f"{prefix}/memory/swap_utilized_percent"] = swap_memory.percent

        disk_io = psutil.disk_io_counters()
        metrics[f"{prefix}/disk/read_count"] = disk_io.read_count
        metrics[f"{prefix}/disk/write_count"] = disk_io.write_count
        metrics[f"{prefix}/disk/read_GB"] = disk_io.read_bytes / (1024**3)
        metrics[f"{prefix}/disk/write_GB"] = disk_io.write_bytes / (1024**3)

        network_io = psutil.net_io_counters()
        metrics[f"{prefix}/network/sent_MB"] = network_io.bytes_sent / (1024**2)
        metrics[f"{prefix}/network/recv_MB"] = network_io.bytes_recv / (1024**2)

        self._collect_gpu_metrics(metrics)

        return metrics

    def _monitoring_loop(self):
        """Main monitoring loop that runs in background thread."""
        while not self._stop_event.is_set():
            start_time = time.time()

            try:
                metrics = self._collect_metrics()
                self.run.log_metrics(data=metrics, step=self._monitoring_step)
                self._monitoring_step += 1

            except Exception as e:
                warnings.warn(f"Error collecting metrics: {e}\n{traceback.format_exc()}")

            finally:
                elapsed = time.time() - start_time
                sleep_time = max(0, self.sampling_rate - elapsed)
                time.sleep(sleep_time)

    def start(self):
        """Start the monitoring thread if not already running."""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._stop_event.clear()
            self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._monitoring_thread.start()

    def stop(self):
        """Stop the monitoring thread and wait for it to finish."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._stop_event.set()
            self._monitoring_thread.join(timeout=5)
            self._monitoring_thread = None

            if self.has_gpu:
                with contextlib.suppress(pynvml.NVMLError):
                    pynvml.nvmlShutdown()

    def __enter__(self):
        """Context manager support."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure monitoring is stopped when exiting context."""
        self.stop()
