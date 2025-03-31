import warnings
from typing import Any, Dict, List, Literal, Optional, Type, Union

import torch
import torch.nn as nn

# Predefined tensor statistics
TENSOR_STATS = {
    "mean": lambda x: x.mean().item(),
    "std": lambda x: x.std().item(),
    "norm": lambda x: x.norm().item(),
    "min": lambda x: x.min().item(),
    "max": lambda x: x.max().item(),
    "var": lambda x: x.var().item(),
    "abs_mean": lambda x: x.abs().mean().item(),
}

# Common PyTorch layer types for validation
PYTORCH_LAYERS = {
    # Linear layers
    nn.Linear,
    # Convolutional layers
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    # Recurrent layers
    nn.LSTM,
    nn.GRU,
    nn.RNN,
    # Normalization layers
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LayerNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    # Activation layers
    nn.ReLU,
    nn.LeakyReLU,
    nn.ELU,
    nn.SELU,
    nn.GELU,
    # Pooling layers
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    # Dropout layers
    nn.Dropout,
    nn.Dropout2d,
    nn.Dropout3d,
    # Embedding layers
    nn.Embedding,
    nn.EmbeddingBag,
    # Transformer layers
    nn.TransformerEncoderLayer,
    nn.TransformerDecoderLayer,
    # Attention layers
    nn.MultiheadAttention,
    # Flatten layers
    nn.Flatten,
    nn.Unflatten,
    # Other common layers
    nn.Sequential,
    nn.ModuleList,
    nn.ModuleDict,
}


class HookManager:
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
                                                          If specified, must contain valid PyTorch layer types.

        Raises:
            TypeError: If model is not a PyTorch model
            ValueError: If track_layers contains invalid layer types
        """
        if not isinstance(model, nn.Module):
            raise TypeError("The model must be a PyTorch model")

        # Validate that all specified layers are valid PyTorch layers if track_layers is provided
        if track_layers is not None:
            invalid_layers = [layer for layer in track_layers if layer not in PYTORCH_LAYERS]
            if invalid_layers:
                raise ValueError(
                    f"Invalid layer types specified: {invalid_layers}. "
                    f"Please use valid PyTorch layer types from torch.nn."
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
                warnings.warn(f"Could not save activation for {name}: {e}")

        return hook

    def save_gradient(self, name: str):
        """Create a backward hook to save layer gradients."""

        def hook(module, grad_input, grad_output):
            try:
                # Save the first gradient output
                self.gradients[name] = grad_output[0].detach()
            except Exception as e:
                warnings.warn(f"Could not save gradient for {name}: {e}")

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
    A comprehensive tracking mechanism for PyTorch models with enhanced logging and context management.
    """

    def __init__(
        self,
        model: nn.Module,
        run: Any,  # Made more flexible to support different logging mechanisms
        track_layers: Optional[List[Type[nn.Module]]] = None,
        tensor_stats: Optional[List[str]] = None,
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
                                              Available options: mean, std, norm, min, max, var, abs_mean.
                                              Defaults to ['mean'] if not specified.

        Raises:
            TypeError: If model is not a PyTorch model
            ValueError: If track_layers contains invalid layer types
        """
        if not isinstance(model, nn.Module):
            raise TypeError("The model must be a PyTorch model")

        self.model = model
        self.run = run
        self.hm = HookManager(model, track_layers)
        self.debug_metrics: Dict[str, float] = {}

        # Validate and set tensor statistics
        if tensor_stats is None:
            tensor_stats = ["mean"]

        # Validate that all requested statistics exist
        invalid_stats = [stat for stat in tensor_stats if stat not in TENSOR_STATS]
        if invalid_stats:
            raise ValueError(
                f"Invalid statistics requested: {invalid_stats}. "
                f"Available statistics are: {list(TENSOR_STATS.keys())}"
            )

        self.tensor_stats = {stat: TENSOR_STATS[stat] for stat in tensor_stats}

        # Default hook registration
        self.hm.register_hooks()

    def _safe_tensor_stats(self, tensor: torch.Tensor) -> Dict[str, float]:
        """
        Safely compute tensor statistics with error handling.

        Args:
            tensor (torch.Tensor): Input tensor

        Returns:
            Dict of statistical metrics
        """
        stats = {}
        for stat_name, stat_func in self.tensor_stats.items():
            try:
                stats[stat_name] = stat_func(tensor)
            except Exception as e:
                warnings.warn(f"Could not compute {stat_name} statistic: {e}")
        return stats

    def track_activations(self):
        """Track layer activations with enhanced statistics."""
        activations = self.hm.get_activations()
        for layer, activation in activations.items():
            if activation is not None:
                stats = self._safe_tensor_stats(activation)
                for stat_name, stat_value in stats.items():
                    self.debug_metrics[f"debug/activation/{layer}_{stat_name}"] = stat_value

    def track_gradients(self):
        """Track layer gradients with enhanced statistics."""
        gradients = self.hm.get_gradients()
        for layer, gradient in gradients.items():
            if gradient is not None:
                stats = self._safe_tensor_stats(gradient)
                for stat_name, stat_value in stats.items():
                    self.debug_metrics[f"debug/gradient/{layer}_{stat_name}"] = stat_value

    def track_parameters(self):
        """Track model parameters with enhanced statistics."""
        for layer, param in self.model.named_parameters():
            if param is not None and param.grad is not None:
                stats = self._safe_tensor_stats(param.grad)
                for stat_name, stat_value in stats.items():
                    self.debug_metrics[f"debug/parameters/{layer}_{stat_name}"] = stat_value

    def watch(
        self,
        step: Union[int, float],
        log: Optional[List[Literal["gradients", "parameters", "activations"]]] = None,
    ):
        """
        Log debug metrics with flexible configuration.

        Args:
            step (int|float): Logging step
            log (Optional[List]): Specific tracking modes.
                                  Defaults to all if None.
        """
        # Reset metrics
        self.debug_metrics.clear()

        # Determine tracking modes
        if log is None or log == "all":
            self.track_gradients()
            self.track_parameters()
            self.track_activations()
        else:
            for mode in log:
                if mode == "gradients":
                    self.track_gradients()
                elif mode == "parameters":
                    self.track_parameters()
                elif mode == "activations":
                    self.track_activations()

        # Log metrics
        try:
            self.run.log_metrics(data=self.debug_metrics, step=step)
        except Exception as e:
            warnings.warn(f"Logging failed: {e}")

        # Clear hooks
        self.hm.clear()
