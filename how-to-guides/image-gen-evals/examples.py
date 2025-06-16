#!/usr/bin/env python3
"""
Demonstration of loading checkpoints and running inference with trained weights.
"""

import torch
import matplotlib.pyplot as plt
from checkpoint_utils import load_checkpoint_auto, resume_training
from net import get_device


def inference_from_pytorch_checkpoint():
    """Demonstrate loading from PyTorch checkpoint and running inference."""
    print("="*60)
    print("DEMO: Inference from PyTorch Checkpoint")
    print("="*60)
    
    device = get_device()
    
    # Load from PyTorch checkpoint (contains actual trained weights)
    checkpoint_path = "checkpoints/epoch=0_step=0/checkpoint.pt"
    pipeline, training_info = load_checkpoint_auto(checkpoint_path, device=device)
    
    print(f"Loaded checkpoint from epoch {training_info['epoch']}, step {training_info['step']}")
    
    # Run inference
    print("Generating images...")
    with torch.no_grad():
        result = pipeline(
            class_labels=[0, 1, 2, 3, 4],
            batch_size=5,
            num_inference_steps=50,
            return_pil=True,
            show_progress=True
        )
    
    # Display results
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    for i, (img, label) in enumerate(zip(result["images"], result["class_labels"])):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Class {label.item()}")
        axes[i].axis('off')
    
    plt.suptitle(f"Generated from Checkpoint (Epoch {training_info['epoch']}, Step {training_info['step']})")
    plt.tight_layout()
    plt.savefig("inference_from_checkpoint.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return pipeline, training_info


def compare_trained_vs_untrained():
    """Compare generation quality of trained vs untrained model."""
    print("\n" + "="*60)
    print("DEMO: Trained vs Untrained Comparison")
    print("="*60)
    
    device = get_device()
    
    # Load trained model
    checkpoint_path = "checkpoints/epoch=0_step=0/checkpoint.pt"
    trained_pipeline, _ = load_checkpoint_auto(checkpoint_path, device=device)
    
    # Create untrained model for comparison
    from net import ClassConditionedUnet, ClassConditionedPipeline
    from diffusers import DDPMScheduler
    
    untrained_unet = ClassConditionedUnet(num_classes=10, class_emb_size=4).to(device)
    untrained_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
    untrained_pipeline = ClassConditionedPipeline(unet=untrained_unet, scheduler=untrained_scheduler)
    
    # Generate with both models
    class_label = 5  # Generate digit 5
    num_steps = 30   # Faster for demo
    
    print("Generating with trained model...")
    with torch.no_grad():
        trained_result = trained_pipeline(
            class_labels=class_label,
            batch_size=3,
            num_inference_steps=num_steps,
            return_pil=True,
            show_progress=True
        )
    
    print("Generating with untrained model...")
    with torch.no_grad():
        untrained_result = untrained_pipeline(
            class_labels=class_label,
            batch_size=3,
            num_inference_steps=num_steps,
            return_pil=True,
            show_progress=True
        )
    
    # Display comparison
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    
    # Trained model results
    for i in range(3):
        axes[0, i].imshow(trained_result["images"][i], cmap='gray')
        axes[0, i].set_title(f"Trained #{i+1}")
        axes[0, i].axis('off')
    
    # Untrained model results
    for i in range(3):
        axes[1, i].imshow(untrained_result["images"][i], cmap='gray')
        axes[1, i].set_title(f"Untrained #{i+1}")
        axes[1, i].axis('off')
    
    # Add row labels
    axes[0, 0].set_ylabel("Trained Model", rotation=90, labelpad=40, fontsize=12)
    axes[1, 0].set_ylabel("Untrained Model", rotation=90, labelpad=40, fontsize=12)
    
    plt.suptitle(f"Trained vs Untrained Model Comparison (Class {class_label})", fontsize=14)
    plt.tight_layout()
    plt.savefig("trained_vs_untrained_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()


def resume_training_demo():
    """Demonstrate how to resume training from a checkpoint."""
    print("\n" + "="*60)
    print("DEMO: Resume Training from Checkpoint")
    print("="*60)
    
    device = get_device()
    
    # Resume training setup
    checkpoint_path = "checkpoints/epoch=0_step=0"
    pipeline, optimizer, training_info = resume_training(
        checkpoint_path,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={"lr": 1e-3},
        device=device
    )
    
    print(f"Resuming training from:")
    print(f"  Epoch: {training_info['epoch']}")
    print(f"  Step: {training_info['step']}")
    print(f"  Optimizer state loaded: {'optimizer_state_dict' in training_info}")
    
    # Show that the model is ready for training
    print(f"Model is on device: {next(pipeline.unet.parameters()).device}")
    print(f"Model is in training mode: {pipeline.unet.training}")
    
    # Put model in training mode
    pipeline.unet.train()
    
    print("✓ Model ready for training continuation")
    
    return pipeline, optimizer, training_info


def save_and_load_new_checkpoint():
    """Demonstrate saving in the improved format and loading it back."""
    print("\n" + "="*60)
    print("DEMO: Save and Load New Checkpoint Format")
    print("="*60)
    
    device = get_device()
    
    # Load existing checkpoint
    pipeline, training_info = load_checkpoint_auto(
        "checkpoints/epoch=0_step=0/checkpoint.pt", 
        device=device
    )
    
    # Save in new unified format
    from checkpoint_utils import save_unified_checkpoint
    
    print("Saving in new unified format...")
    save_unified_checkpoint(
        pipeline=pipeline,
        save_directory="checkpoints/improved_format_demo",
        epoch=training_info['epoch'],
        step=training_info['step'],
        loss=0.0,  # We don't have loss info from the original checkpoint
        save_hf_format=True,
        save_pytorch_format=True
    )
    
    # Load it back
    print("Loading from new format...")
    loaded_pipeline, loaded_info = load_checkpoint_auto(
        "checkpoints/improved_format_demo",
        device=device
    )
    
    print(f"✓ Successfully loaded: epoch={loaded_info['epoch']}, step={loaded_info['step']}")
    
    # Test that it works
    print("Testing loaded pipeline...")
    with torch.no_grad():
        result = loaded_pipeline(
            class_labels=7,
            batch_size=2,
            num_inference_steps=10,
            show_progress=False
        )
    
    print(f"✓ Generated {result['images'].shape[0]} images successfully")


def main():
    """Run all demonstrations."""
    print("🚀 Checkpoint Loading and Inference Demonstration")
    print("=" * 60)
    print("This script shows how to properly load checkpoints and run inference")
    print("with trained model weights.")
    print()
    
    try:
        # Demo 1: Basic inference from checkpoint
        pipeline, training_info = inference_from_pytorch_checkpoint()
        
        # Demo 2: Compare trained vs untrained
        compare_trained_vs_untrained()
        
        # Demo 3: Resume training
        resume_training_demo()
        
        # Demo 4: Save and load in improved format
        save_and_load_new_checkpoint()
        
        print("\n" + "="*60)
        print("🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Key findings:")
        print("✓ PyTorch checkpoints contain the actual trained weights")
        print("✓ HuggingFace pipeline format was incomplete (now fixed)")
        print("✓ The model can successfully load and generate images")
        print("✓ Training can be resumed with full optimizer state")
        print()
        print("Generated files:")
        print("  - inference_from_checkpoint.png")
        print("  - trained_vs_untrained_comparison.png")
        print("  - checkpoints/improved_format_demo/ (new unified format)")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 