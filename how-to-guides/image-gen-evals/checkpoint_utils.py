#!/usr/bin/env python3
"""
Utilities for loading and saving model checkpoints in different formats.
"""

import os
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
from diffusers import DDPMScheduler
from net import ClassConditionedUnet, ClassConditionedPipeline


def create_checkpoint_path(run_id: str, global_step: int) -> str:
    """Create checkpoint path based on run ID and global step."""
    return f"checkpoints/{run_id}/step_{global_step:06d}"


def load_checkpoint_by_run_and_step(run_id: str, global_step: int, device=None):
    """Load checkpoint by run ID and global step."""
    checkpoint_path = create_checkpoint_path(run_id, global_step)
    return load_checkpoint_auto(checkpoint_path, device=device)


def load_from_pytorch_checkpoint(
    checkpoint_path: str,
    device: Optional[str] = None,
    num_classes: int = 10,
    class_emb_size: int = 4
) -> Tuple[ClassConditionedPipeline, Dict[str, Any]]:
    """
    Load pipeline from a PyTorch checkpoint file (.pt).
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        device: Device to load the model on
        num_classes: Number of classes for the UNet
        class_emb_size: Size of class embedding
        
    Returns:
        Tuple of (pipeline, checkpoint_info)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
    
    # Create UNet and load weights
    unet = ClassConditionedUnet(num_classes=num_classes, class_emb_size=class_emb_size)
    unet.load_state_dict(checkpoint["model_state_dict"])
    
    # Create scheduler (using same config as training)
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
    
    # Create pipeline
    pipeline = ClassConditionedPipeline(unet=unet, scheduler=scheduler)
    
    if device:
        pipeline = pipeline.to(device)
    
    # Extract training info
    training_info = {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "global_step": checkpoint.get("global_step", 0),
        "optimizer_state_dict": checkpoint.get("optimizer_state_dict", None)
    }
    
    print(f"✓ Loaded pipeline from PyTorch checkpoint: epoch={training_info['epoch']}, step={training_info['step']}")
    
    return pipeline, training_info


def load_from_hf_checkpoint(
    checkpoint_dir: str,
    device: Optional[str] = None
) -> ClassConditionedPipeline:
    """
    Load pipeline from HuggingFace format checkpoint directory.
    
    Args:
        checkpoint_dir: Directory containing the HF-format checkpoint
        device: Device to load the model on
        
    Returns:
        ClassConditionedPipeline
    """
    pipeline = ClassConditionedPipeline.from_pretrained(checkpoint_dir)
    
    if device:
        pipeline = pipeline.to(device)
    
    print(f"✓ Loaded pipeline from HuggingFace checkpoint: {checkpoint_dir}")
    
    return pipeline


def load_checkpoint_auto(
    checkpoint_path: Union[str, Path],
    device: Optional[str] = None,
    **kwargs
) -> Tuple[ClassConditionedPipeline, Optional[Dict[str, Any]]]:
    """
    Automatically detect and load checkpoint from either PyTorch or HuggingFace format.
    
    Args:
        checkpoint_path: Path to checkpoint file or directory
        device: Device to load the model on
        **kwargs: Additional arguments for PyTorch checkpoint loading
        
    Returns:
        Tuple of (pipeline, training_info_or_None)
    """
    checkpoint_path = Path(checkpoint_path)
    
    if checkpoint_path.is_file() and checkpoint_path.suffix == ".pt":
        # PyTorch checkpoint file
        return load_from_pytorch_checkpoint(str(checkpoint_path), device, **kwargs)
    
    elif checkpoint_path.is_dir():
        # Check if it's a HuggingFace checkpoint directory
        if (checkpoint_path / "model_index.json").exists():
            pipeline = load_from_hf_checkpoint(str(checkpoint_path), device)
            return pipeline, None
        
        # Check if it contains a PyTorch checkpoint
        pt_files = list(checkpoint_path.glob("*.pt"))
        if pt_files:
            return load_from_pytorch_checkpoint(str(pt_files[0]), device, **kwargs)
        
        raise ValueError(f"No valid checkpoint found in directory: {checkpoint_path}")
    
    else:
        raise ValueError(f"Invalid checkpoint path: {checkpoint_path}")


def save_unified_checkpoint(
    pipeline: ClassConditionedPipeline,
    save_directory: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    step: int = 0,
    loss: float = 0.0,
    global_step: Optional[int] = None,
    run_id: Optional[str] = None,
    save_hf_format: bool = True,
    save_pytorch_format: bool = True,
    **kwargs
):
    """
    Save checkpoint in both HuggingFace and PyTorch formats.
    
    Args:
        pipeline: Pipeline to save
        save_directory: Directory to save checkpoints
        optimizer: Optimizer to save (for PyTorch format)
        epoch: Current epoch
        step: Current step within epoch
        loss: Current loss
        global_step: Global training step (across all epochs)
        run_id: Neptune run ID for tracking
        save_hf_format: Whether to save in HuggingFace format
        save_pytorch_format: Whether to save in PyTorch format
        **kwargs: Additional state to save
    """
    os.makedirs(save_directory, exist_ok=True)
    
    if save_hf_format:
        # Save HuggingFace format
        hf_path = os.path.join(save_directory, "pipeline")
        pipeline.save_pretrained(hf_path)
        print(f"✓ Saved HuggingFace format to {hf_path}")
    
    if save_pytorch_format:
        # Save PyTorch format
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'global_step': global_step,
            'run_id': run_id,
            'loss': loss,
            'model_state_dict': pipeline.unet.state_dict(),
            **kwargs
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        pt_path = os.path.join(save_directory, "checkpoint.pt")
        torch.save(checkpoint, pt_path)
        print(f"✓ Saved PyTorch format to {pt_path}")


def resume_training(
    checkpoint_path: Union[str, Path],
    optimizer_class: type = torch.optim.Adam,
    optimizer_kwargs: Optional[Dict] = None,
    device: Optional[str] = None,
    **model_kwargs
) -> Tuple[ClassConditionedPipeline, torch.optim.Optimizer, Dict[str, Any]]:
    """
    Resume training from a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        optimizer_class: Optimizer class to create
        optimizer_kwargs: Kwargs for optimizer initialization
        device: Device to load on
        **model_kwargs: Additional model creation kwargs
        
    Returns:
        Tuple of (pipeline, optimizer, training_info)
    """
    # Load pipeline and training info
    pipeline, training_info = load_checkpoint_auto(checkpoint_path, device, **model_kwargs)
    
    # Create optimizer
    optimizer_kwargs = optimizer_kwargs or {"lr": 1e-3}
    optimizer = optimizer_class(pipeline.unet.parameters(), **optimizer_kwargs)
    
    # Load optimizer state if available
    if training_info and "optimizer_state_dict" in training_info and training_info["optimizer_state_dict"]:
        optimizer.load_state_dict(training_info["optimizer_state_dict"])
        print("✓ Loaded optimizer state")
    else:
        print("! No optimizer state found, using fresh optimizer")
    
    return pipeline, optimizer, training_info or {}


def test_checkpoint_loading():
    """Test function to verify checkpoint loading works."""
    print("Testing checkpoint loading...")
    
    # Test current checkpoint structure
    checkpoint_dirs = [
        "checkpoints/epoch=0_step=0",
        "checkpoints/epoch=0_step=100"
    ]
    
    for checkpoint_dir in checkpoint_dirs:
        if os.path.exists(checkpoint_dir):
            print(f"\nTesting {checkpoint_dir}:")
            
            # Test PyTorch checkpoint loading
            pt_file = os.path.join(checkpoint_dir, "checkpoint.pt")
            if os.path.exists(pt_file):
                try:
                    pipeline, info = load_from_pytorch_checkpoint(pt_file)
                    print(f"  ✓ PyTorch loading: epoch={info['epoch']}, step={info['step']}")
                except Exception as e:
                    print(f"  ✗ PyTorch loading failed: {e}")
            
            # Test HF checkpoint loading
            hf_dir = os.path.join(checkpoint_dir, "pipeline")
            if os.path.exists(hf_dir):
                try:
                    pipeline = load_from_hf_checkpoint(hf_dir)
                    print("  ✓ HuggingFace loading successful")
                except Exception as e:
                    print(f"  ✗ HuggingFace loading failed: {e}")
            
            # Test auto loading
            try:
                pipeline, info = load_checkpoint_auto(checkpoint_dir)
                print("  ✓ Auto loading successful")
            except Exception as e:
                print(f"  ✗ Auto loading failed: {e}")


if __name__ == "__main__":
    test_checkpoint_loading()
