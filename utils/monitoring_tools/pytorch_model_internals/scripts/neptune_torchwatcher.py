from typing import Any, Dict, List, Literal, Optional, Type, Union

import numpy as np
import torch
import torch.nn as nn
from neptune_scale.types import Histogram
from neptune_scale.util.logger import get_logger

__version__ = "0.2.0"

logger = get_logger()


# Predefined tensor statistics
TENSOR_STATS = {
    "mean": lambda x: x.mean().item(),
    "std": lambda x: x.std().item(),
    "norm": lambda x: x.norm().item(),
    "min": lambda x: x.min().item(),
    "max": lambda x: x.max().item(),
    "var": lambda x: x.var().item(),
    "abs_mean": lambda x: x.abs().mean().item(),
    "hist": lambda x: torch.histogram(x, bins=50),
}


class _HookManager:
    """
    A robust hook management class for PyTorch models to track activations, gradients, and parameters.

    Improvements:
    - More comprehensive error handling
    - Flexible hook registration
    - Support for more layer types
    - Configurable tracking
    """

    def __init__(self, model: nn.Module, track_layers: Optional[List[Type[nn.Module]]] = None):
        """
        Initialize HookManager with layer types to track.

        Args:
            model (nn.Module): The PyTorch model to track
            track_layers (Optional[List[Type[nn.Module]]]): List of PyTorch layer types to track.
                If None, tracks all layers in the model.

        Raises:
            TypeError: If model is not a PyTorch model
            ValueError: If track_layers contains invalid layer types
        """
        if not isinstance(model, nn.Module):
            raise TypeError("The model must be a PyTorch model")

        if track_layers is not None:
            for layer_type in track_layers:
                if not isinstance(layer_type, type) or not issubclass(layer_type, nn.Module):
                    raise ValueError(
                        f"Invalid layer type: {layer_type}. Must be a subclass of nn.Module"
                    )

        self.model = model
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.activations: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}
        self.track_layers = track_layers

    def save_activation(self, name: str):
        """Create a forward hook to save layer activations."""

        def hook(module, input, output):
            try:
                # Handle different output types (tensor or tuple)
                activation = output[0] if isinstance(output, tuple) else output
                self.activations[name] = activation.detach()
            except Exception as e:
                logger.warning(f"Could not save activations for {name}: {e}")

        return hook

    def save_gradient(self, name: str):
        """Create a backward hook to save layer gradients."""

        def hook(module, grad_input, grad_output):
            try:
                # Save the first gradient output
                self.gradients[name] = grad_output[0].detach()
            except Exception as e:
                logger.warning(f"Could not save gradients for {name}: {e}")

        return hook

    def register_hooks(self, track_activations: bool = True, track_gradients: bool = True):
        """
        Register hooks for the model with configurable tracking.

        Args:
            track_activations (bool): Whether to track layer activations
            track_gradients (bool): Whether to track layer gradients
        """
        # Clear existing hooks
        self.remove_hooks()

        # Register forward hooks for activations
        if track_activations:
            for name, module in self.model.named_modules():
                # Skip the model itself
                if name == "":
                    continue
                # Track all layers if track_layers is None, otherwise only specified types
                if self.track_layers is None or any(
                    isinstance(module, layer_type) for layer_type in self.track_layers
                ):
                    hook = module.register_forward_hook(self.save_activation(name))
                    self.hooks.append(hook)

        # Register backward hooks for gradients
        if track_gradients:
            for name, module in self.model.named_modules():
                # Skip the model itself
                if name == "":
                    continue
                # Track all layers if track_layers is None, otherwise only specified types
                if self.track_layers is None or any(
                    isinstance(module, layer_type) for layer_type in self.track_layers
                ):
                    hook = module.register_full_backward_hook(self.save_gradient(name))
                    self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def clear(self):
        """Clear stored activations and gradients."""
        self.activations.clear()
        self.gradients.clear()

    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Get stored activations."""
        return self.activations

    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """Get stored gradients."""
        return self.gradients

    def __del__(self):
        """Ensure hooks are removed when the object is deleted."""
        self.remove_hooks()


class TorchWatcher:
    """
    Tracking mechanism for PyTorch model internals.
    """

    def __init__(
        self,
        model: nn.Module,
        run: Any,  # Made more flexible to support different logging mechanisms
        track_layers: Optional[List[Type[nn.Module]]] = None,
        tensor_stats: Optional[
            List[Literal["mean", "std", "norm", "min", "max", "var", "abs_mean", "hist"]]
        ] = None,
        base_namespace: str = "model_internals",  # Default namespace for all metrics
    ) -> None:
        """
        Initialize TorchWatcher with configuration options.

        Args:
            model (nn.Module): The PyTorch model to watch
            run: Logging mechanism from Neptune
            track_layers (Optional[List[Type[nn.Module]]]): List of PyTorch layer types to track.
                If None, tracks all layers in the model.
                If specified, must contain valid PyTorch layer types.
            tensor_stats (Optional[List[str]]): List of statistics to compute.
                Available options: mean, std, norm, min, max, var, abs_mean, hist.
                Defaults to ['mean'] if not specified.
            base_namespace (str): Base namespace for all logged metrics. Defaults to "model_internals".

        Raises:
            TypeError: If model is not a PyTorch model
            ValueError: If track_layers contains invalid layer types
        """
        if not isinstance(model, nn.Module):
            raise TypeError("The model must be a PyTorch model")

        self.model = model
        self.run = run
        self.hm = _HookManager(model, track_layers)
        self.debug_metrics: Dict[str, float] = {}
        self.base_namespace = base_namespace

        # Validate and set tensor statistics
        if tensor_stats is None:
            tensor_stats = ["mean"]

        if invalid_stats := [stat for stat in tensor_stats if stat not in TENSOR_STATS]:
            raise ValueError(
                f"Invalid statistics requested: {invalid_stats}. "
                f"Available statistics are: {list(TENSOR_STATS.keys())}"
            )

        self.tensor_stats = {stat: TENSOR_STATS[stat] for stat in tensor_stats}

        # Default hook registration
        self.hm.register_hooks()

    def _safe_tensor_stats(
        self, tensor: torch.Tensor
    ) -> Dict[str, Union[float, torch.return_types.histogram]]:
        """
        Safely compute tensor statistics with error handling.

        Args:
            tensor (torch.Tensor): Input tensor

        Returns:
            Dict of statistical metrics. Values can be floats for most stats or
            torch.return_types.histogram for histogram statistics.
        """
        stats = {}
        for stat_name, stat_func in self.tensor_stats.items():
            try:
                stats[stat_name] = stat_func(tensor)
            except Exception as e:
                logger.warning(f"Could not compute {stat_name} statistic: {e}")
        return stats

    def _track_metric(
        self, metric_type: str, data: Dict[str, torch.Tensor], prefix: Optional[str] = None
    ):
        """Track metrics with enhanced statistics for a given metric type.

        Args:
            metric_type (str): Type of metric being tracked (activations/gradients/parameters)
            data (Dict[str, torch.Tensor]): Dictionary mapping layer names to tensors
            prefix (Optional[str]): Optional namespace to prefix the base namespace
        """
        # Construct the full namespace
        full_namespace = f"{prefix}/{self.base_namespace}" if prefix else self.base_namespace

        for layer, tensor in data.items():
            if tensor is not None:
                safe_tensor = tensor.detach().cpu() if tensor.is_cuda else tensor.detach()
                stats = self._safe_tensor_stats(safe_tensor)
                for stat_name, stat_value in stats.items():
                    self.debug_metrics[f"{full_namespace}/{metric_type}/{layer}/{stat_name}"] = (
                        stat_value
                    )

    def track_activations(self, namespace: Optional[str] = None):
        """Track layer activations with enhanced statistics."""
        activations = self.hm.get_activations()
        self._track_metric("activations", activations, namespace)

    def track_gradients(self, namespace: Optional[str] = None):
        """Track layer gradients with enhanced statistics."""
        gradients = self.hm.get_gradients()
        self._track_metric("gradients", gradients, namespace)

    def track_parameters(self, namespace: Optional[str] = None):
        """Track model parameters with enhanced statistics."""
        # TODO: Speed up for extracting parameters
        with torch.no_grad():
            parameters = {
                name.replace(".", "/"): param.data
                for name, param in self.model.named_parameters()
                if param is not None
            }
            self._track_metric("parameters", parameters, namespace)

    def watch(
        self,
        step: Union[int, float],
        track_gradients: bool = True,
        track_parameters: bool = False,
        track_activations: bool = True,
        prefix: Optional[str] = None,
    ):
        """
        Log debug metrics with flexible configuration.

        Args:
            step (int|float): Logging step
            track_gradients (bool): Whether to track gradients. Defaults to True.
            track_parameters (bool): Whether to track parameters. Defaults to False.
            track_activations (bool): Whether to track activations. Defaults to True.
            prefix (Optional[str]): Optional prefix to add to the base namespace.
                If provided, metrics will be logged under {prefix}/{base_namespace}/...
        """
        # Reset metrics
        self.debug_metrics.clear()

        # Track metrics based on boolean flags
        if track_gradients:
            self.track_gradients(prefix)
        if track_parameters:
            self.track_parameters(prefix)
        if track_activations:
            self.track_activations(prefix)

        # Process histograms with proper data type conversion
        histogram_stats = {}
        for attribute_name, torch_hist in self.debug_metrics.items():
            if attribute_name.endswith("/hist"):
                try:
                    # torch_hist is a torch.return_types.histogram object
                    counts_tensor = torch_hist.hist
                    bin_edges_tensor = torch_hist.bin_edges

                    # Convert to numpy arrays
                    counts_np = counts_tensor.numpy(force=True)
                    bin_edges_np = bin_edges_tensor.numpy(force=True)

                    # Check for invalid values using numpy arrays
                    if np.isnan(counts_np).any() or np.isinf(counts_np).any():
                        logger.warning(
                            f"Skipping histogram {attribute_name} due to NaN or Inf values in counts"
                        )
                        continue
                    if np.isnan(bin_edges_np).any() or np.isinf(bin_edges_np).any():
                        logger.warning(
                            f"Skipping histogram {attribute_name} due to NaN or Inf values in bin_edges"
                        )
                        continue

                    # Ensure counts are integers and bin_edges are floats
                    counts_int = counts_np.astype(int)
                    bin_edges_float = bin_edges_np.astype(float)

                    # Convert to Python lists as Neptune expects
                    histogram_stats[attribute_name] = Histogram(
                        bin_edges=bin_edges_float.tolist(),
                        counts=counts_int.tolist(),
                    )
                except Exception as e:
                    logger.warning(f"Could not process histogram {attribute_name}: {e}")
                    continue

        metric_stats = {
            attribute_name: attribute_value
            for attribute_name, attribute_value in self.debug_metrics.items()
            if not attribute_name.endswith("/hist")
        }

        # Log metrics
        try:
            # Batch logging if possible, otherwise ensure atomicity
            if histogram_stats:
                self.run.log_metrics(data=metric_stats, step=step)
                self.run.log_histograms(histograms=histogram_stats, step=step)
            else:
                self.run.log_metrics(data=metric_stats, step=step)
        except Exception as e:
            logger.warning(f"Metric/Histogram logging failed at step {step}: {e}")

        # Clear hooks and cached data
        self.hm.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup hooks."""
        self.hm.remove_hooks()
