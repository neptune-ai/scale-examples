# Class-Conditioned Diffusion Pipeline

A complete implementation of a class-conditioned diffusion pipeline for MNIST-like image generation, built on top of Hugging Face Diffusers. Generates greyscale images of hand-written digits given a class (digit, 0-9).

## Features

- **Class-Conditioned Generation**: Generate images conditioned on specific class labels (0-9 for MNIST)
- **Training State Management**: Save and resume training with full optimizer state
- **Flexible Inference**: Generate single images, batches, or grids with customizable parameters
- **HuggingFace Compatible**: Follows HF Diffusers best practices for pipeline implementation
- **Multiple Output Formats**: Support for both tensor and PIL image outputs
- **Easy Iteration**: Convenient methods for iterating over generated image grids
- **Comprehensive Neptune Logging**: Advanced experiment tracking with metrics, debugging data, and evaluation results
- **Automated Evaluation**: Gemini AI-powered evaluation of generated samples with statistical analysis

### Components

1. **ClassConditionedUnet**: A U-Net model that takes class labels as additional conditioning
   - Embedding layer maps class labels to conditioning vectors
   - Conditioning vectors are concatenated with input images
   - Based on UNet2DModel with attention layers

2. **ClassConditionedPipeline**: Main pipeline class extending DiffusionPipeline
   - Implements `__call__` method for inference
   - Provides save/load functionality for training states
   - Includes utility methods for grid generation and image handling

3. **TorchWatcher**: Advanced debugging and monitoring system
   - Tracks gradients, activations, and parameters
   - Computes statistical metrics and histograms
   - Integrates seamlessly with Neptune logging

## Prerequisites

Before running the training and evaluation scripts, ensure you have the required API keys and dependencies:

### Environment Variables

```bash
# Required: Neptune AI API token
export NEPTUNE_API_TOKEN="your-neptune-api-token"

# Required: Google Gemini API key for evaluations  
export GEMINI_API_KEY="your-gemini-api-key"

# Optional: Default Neptune project (can be overridden with --neptune-project)
export NEPTUNE_PROJECT="workspace/project-name"
```

### Dependencies

Install required packages:

```bash
pip install torch torchvision diffusers neptune-scale google-genai tqdm matplotlib
```

### API Setup

1. **Neptune AI**: Sign up at [neptune.ai](https://neptune.ai) and get your API token from the user settings
2. **Google Gemini**: Get an API key from [Google AI Studio](https://aistudio.google.com) with access to generative models

## Neptune Logging

This implementation follows Neptune.ai best practices for comprehensive experiment tracking and provides detailed logging across all aspects of the ML pipeline.

### Training Metrics

The training script logs comprehensive metrics using Neptune Scale:

```python
# Core training metrics logged at each step
run.log_metrics({
    "train/loss": loss.item(),                    # Training loss (MSE between predicted and actual noise)
    "train/step_time_s": step_end - step_start,   # Time per training step
    "train/epoch": epoch,                         # Current epoch
    "train/step": step,                          # Step within current epoch  
    "train/global_step": current_global_step,    # Global step across all epochs
}, step=current_global_step)

# Checkpoint timing metrics
run.log_metrics({
    "checkpoints/save_time_s": save_end - save_start,
}, step=current_global_step)
```

### Configuration Logging

All hyperparameters and model configurations are logged for reproducibility:

```python
run.log_configs({
    # Command line arguments
    **{f"config/args/{k}": v for k, v in vars(args).items()},
    
    # Model and training setup
    "config/run_id": run_id,                              # UUID for this training run
    "config/net/repr": str(net),                          # Model architecture string
    "config/net/num_params": sum(p.numel() for p in net.parameters()),
    "config/net/num_trainable_params": sum(p.numel() for p in net.parameters() if p.requires_grad),
    "config/pipeline/repr": str(pipeline),                # Pipeline configuration
    "config/opt/repr": str(opt),                          # Optimizer configuration
    "config/start_epoch": start_epoch,                    # Starting epoch (for resume)
    "config/start_step": start_step,                      # Starting step (for resume)
    "config/start_global_step": start_global_step,        # Starting global step (for resume)
})
```

### Advanced Debugging with TorchWatcher

The [TorchWatcher](https://lightning.ai/docs/pytorch/stable//extensions/generated/lightning.pytorch.loggers.NeptuneLogger.html) integration provides deep insights into model behavior:

```python
# Initialize TorchWatcher for debugging
watcher = TorchWatcher(
    model=net,
    run=run,
    tensor_stats=["hist", "mean", "norm", "std", "min", "max"],  # Statistics to compute
    base_namespace="debug"  # Namespace for all debug metrics
)

# Log debugging metrics at each step
watcher.watch(
    step=current_global_step,
    track_activations=True,    # Log activation statistics and histograms
    track_gradients=True,      # Log gradient statistics and histograms  
    track_parameters=False     # Log parameter statistics (optional)
)
```

**Debug Metrics Logged:**
- **Gradient Statistics**: Mean, std, norm, min, max for each layer's gradients
- **Activation Histograms**: Distribution of activations across layers
- **Parameter Tracking**: Weight and bias statistics (when enabled)
- **Norm Monitoring**: Gradient norms to detect vanishing/exploding gradients

### Evaluation Logging

The evaluation script provides comprehensive logging of generated samples and scoring:

```python
# Evaluation configuration
run.log_configs({
    "evals/config/device": device,
    "evals/config/n_samples": args.n_samples,
    "evals/config/gemini_model": args.gemini_model,
    "evals/config/gemini_api_rpm": args.gemini_api_rpm,
    "evals/config/global_step": args.global_step,
})

# Generated sample logging
run.log_files(
    files={f"evals/samples/digit={digit}/sample={sample_idx:03d}.png": img_path},
    step=args.global_step
)

# Evaluation prompts and system prompts
run.log_string_series(
    data={
        f"evals/{eval_name}/prompt": eval_config["prompt"],
        f"evals/{eval_name}/system_prompt": eval_config["system_prompt"],  
    },
    step=args.global_step
)

# Individual sample scores
run.log_metrics(
    data={f"evals/{eval_name}/scores/digit={digit}/sample={sample_idx:03d}": score},
    step=args.global_step
)

# Statistical aggregations with preview support
run.log_metrics(
    data={
        f"evals/{eval_name}/scores/digit={digit}/avg": statistics.mean(results[eval_name][digit]),
        f"evals/{eval_name}/scores/digit={digit}/max": max(results[eval_name][digit]),
        f"evals/{eval_name}/scores/digit={digit}/min": min(results[eval_name][digit]),
    },
    step=args.global_step,
    preview=True,  # Show live updates during evaluation
    preview_completion=progress  # Progress indicator
)
```

### Neptune Logging Structure

```
project/
├── config/
│   ├── args/                    # All command line arguments
│   ├── net/                     # Model configuration and parameter counts
│   ├── pipeline/                # Pipeline setup information
│   └── run_id                   # Unique UUID for this run
├── train/
│   ├── loss                     # Training loss over time
│   ├── step_time_s             # Training step timing
│   ├── epoch                   # Current epoch
│   ├── step                    # Step within epoch
│   └── global_step             # Global training step
├── debug/
│   ├── activation/             # Layer activation statistics
│   ├── gradient/               # Layer gradient statistics
│   └── parameters/             # Parameter statistics (optional)
├── checkpoints/
│   └── save_time_s             # Checkpoint save timing
└── evals/
    ├── config/                 # Evaluation configuration
    ├── samples/                # Generated image samples
    ├── {eval_name}/
    │   ├── prompt              # Evaluation prompts used
    │   ├── system_prompt       # System prompts used
    │   └── scores/
    │       ├── digit={0-9}/    # Per-digit statistics
    │       │   ├── avg         # Average score for digit
    │       │   ├── min         # Minimum score for digit
    │       │   └── max         # Maximum score for digit
    │       ├── avg             # Overall average score
    │       ├── min             # Overall minimum score
    │       └── max             # Overall maximum score
    └── ...
```

## Training

The training script (`run_training.py`) supports three Neptune modes for different experiment management scenarios:

### Neptune Run Modes

1. **New Mode** (`--neptune-mode new`): Creates a fresh training run
2. **Resume Mode** (`--neptune-mode resume`): Continues logging to an existing run  
3. **Fork Mode** (`--neptune-mode fork`): Creates a new run inheriting history up to a specific step

### Basic Training Examples

#### Start New Training

```bash
# Basic training with default settings
python run_training.py \
    --neptune-mode new \
    --neptune-experiment "mnist-diffusion-baseline" \
    --n-epochs 10 \
    --batch-size 128

# Training with custom checkpoint frequency
python run_training.py \
    --neptune-mode new \
    --neptune-experiment "mnist-diffusion-frequent-checkpoints" \
    --n-epochs 20 \
    --batch-size 64 \
    --checkpoint-interval 50 \
    --save-checkpoints
```

**What happens**: Creates a new Neptune run with a UUID identifier, starts training from scratch, and logs all metrics to Neptune. The run ID is printed for future reference.

#### Resume Existing Training

```bash
# Resume training from a specific checkpoint
python run_training.py \
    --neptune-mode resume \
    --resume-from-run-id "a1b2c3d4-e5f6-7890-abcd-ef1234567890" \
    --resume-from-step 1000 \
    --n-epochs 15
```

**What happens**: Reconnects to the existing Neptune run, loads the checkpoint from the specified step, restores optimizer state, and continues training while logging to the same run.

#### Fork Training for Experimentation

```bash
# Fork from existing run to test different hyperparameters
python run_training.py \
    --neptune-mode fork \
    --resume-from-run-id "a1b2c3d4-e5f6-7890-abcd-ef1234567890" \
    --resume-from-step 500 \
    --neptune-experiment "mnist-diffusion-tuned-lr" \
    --n-epochs 10 \
    --batch-size 256
```

**What happens**: Creates a new Neptune run that inherits the parent run's history up to step 500, then continues with potentially different hyperparameters. Useful for A/B testing or hyperparameter tuning.

### Advanced Training Scenarios

#### Interactive Training with Human Oversight

```bash
# Training that asks for confirmation before each epoch
python run_training.py \
    --neptune-mode new \
    --neptune-experiment "interactive-training" \
    --ask-before-epoch \
    --n-epochs 50 \
    --batch-size 128
```

#### Training with Minimal Logging for Speed

```bash
# Faster training with reduced debugging
python run_training.py \
    --neptune-mode new \
    --neptune-experiment "fast-training" \
    --tensor-stats "mean,norm" \
    --save-checkpoints False \
    --batch-size 256 \
    --n-epochs 5
```

#### High-Frequency Monitoring Setup

```bash
# Training with maximum debugging information
python run_training.py \
    --neptune-mode new \
    --neptune-experiment "detailed-monitoring" \
    --tensor-stats "hist,mean,norm,std,min,max,var,abs_mean" \
    --checkpoint-interval 25 \
    --batch-size 64 \
    --n-epochs 100
```

### Training Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--neptune-mode` | str | "new" | Neptune run behavior: "new", "resume", "fork" |
| `--neptune-project` | str | "neptune/class-conditioned-difussion" | Neptune project name |
| `--neptune-experiment` | str | "debug-001" | Experiment name for organization |
| `--resume-from-run-id` | str | None | Neptune run ID to resume/fork from |
| `--resume-from-step` | int | None | Global step to resume/fork from |
| `--save-checkpoints` | bool | True | Whether to save model checkpoints |
| `--checkpoint-interval` | int | 100 | Save checkpoint every N steps |
| `--ask-before-epoch` | bool | False | Ask for confirmation before each epoch |
| `--tensor-stats` | str | "hist,mean,norm" | Comma-separated debug statistics to log |
| `--batch-size` | int | 128 | Training batch size |
| `--n-epochs` | int | 10 | Number of training epochs |

## Evaluation

The evaluation script (`run_evals.py`) provides comprehensive evaluation of trained models using Google's Gemini AI for image assessment.

### Basic Evaluation

```bash
# Evaluate a checkpoint with default settings
python run_evals.py \
    --neptune-run-id "a1b2c3d4-e5f6-7890-abcd-ef1234567890" \
    --global-step 1000 \
    --n-samples 10
```

**What happens**: Loads the checkpoint, generates 10 samples per digit (0-9), uploads images to Gemini AI, evaluates them across three metrics, and logs results to Neptune with statistical analysis.

### Advanced Evaluation Scenarios

#### Comprehensive Evaluation with High Sample Count

```bash
# Thorough evaluation with many samples
python run_evals.py \
    --neptune-run-id "a1b2c3d4-e5f6-7890-abcd-ef1234567890" \
    --global-step 2000 \
    --n-samples 50 \
    --gemini-model "gemini-1.5-pro-latest" \
    --gemini-api-rpm 30
```

#### Fast Evaluation for Quick Feedback

```bash
# Quick evaluation with fewer samples
python run_evals.py \
    --neptune-run-id "a1b2c3d4-e5f6-7890-abcd-ef1234567890" \
    --global-step 500 \
    --n-samples 5 \
    --gemini-model "gemini-1.5-flash" \
    --gemini-api-rpm 60
```

#### Evaluation with Rate Limiting for API Quotas

```bash
# Conservative evaluation respecting API limits
python run_evals.py \
    --neptune-run-id "a1b2c3d4-e5f6-7890-abcd-ef1234567890" \
    --global-step 1500 \
    --n-samples 20 \
    --gemini-api-rpm 10  # Slower rate for limited quotas
```

### Evaluation Metrics

The evaluation system assesses generated images across three dimensions:

1. **`is_digit`**: Binary classification (0/1) - Whether a recognizable digit is present
   - **Prompt**: "can you see a digit on the attached image?"
   - **Scoring**: 1 if "true", 0 if "false"

2. **`correct_digit`**: Accuracy metric (0/1) - Whether the digit matches the target class
   - **Prompt**: "what digit is it?"  
   - **Scoring**: 1 if predicted digit matches target, 0 otherwise

3. **`hand_writing`**: Human-likeness score (1-5) - How human-like the digit appears
   - **Prompt**: "on scale 1 (false) to 5 (true), how likely do you think this digit was written by a human?"
   - **Scoring**: Integer score from 1-5

### Evaluation Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--neptune-run-id` | str | Required | Neptune run ID containing the checkpoint |
| `--global-step` | int | Required | Global training step of checkpoint to evaluate |
| `--n-samples` | int | 10 | Number of samples to generate per digit (0-9) |
| `--gemini-model` | str | "gemma-3-27b-it" | Gemini model for evaluation |
| `--gemini-api-rpm` | int | 20 | API requests per minute limit |
| `--neptune-project` | str | "neptune/class-conditioned-difussion" | Neptune project for logging |

### Evaluation Output

The evaluation produces:
- **Generated Images**: Saved to Neptune as files under `evals/samples/digit={0-9}/`
- **Individual Scores**: Each sample's score for each metric
- **Statistical Aggregations**: Mean, min, max for each digit and overall
- **Live Progress**: Preview metrics showing real-time evaluation progress

## Custom Usage in Python

```python
import torch
from diffusers import DDPMScheduler
from net import ClassConditionedUnet, ClassConditionedPipeline

# Initialize pipeline
unet = ClassConditionedUnet(num_classes=10, class_emb_size=4)
scheduler = DDPMScheduler(num_train_timesteps=1000)
pipeline = ClassConditionedPipeline(unet=unet, scheduler=scheduler)

# Generate images
result = pipeline(
    class_labels=5,  # Generate digit 5
    batch_size=4,
    num_inference_steps=50,
    return_pil=True
)

# Access generated images
images = result["images"]  # List of PIL Images
```

### Grid Generation

```python
# Generate a grid of images (8 per class, 10 classes)
grid_result = pipeline.generate_grid(
    classes_per_row=8,
    num_rows=10,
    num_inference_steps=50
)

# Save the grid
grid_result["pil_grid"].save("generated_grid.png")

# Iterate over individual images
for i, (img, label) in enumerate(zip(grid_result["pil_images"], grid_result["class_labels"])):
    img.save(f"image_{i}_class_{label.item()}.png")
```

## Training

### Training Loop

```python
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Setup data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # [-1, 1] range
])
dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Setup training
optimizer = optim.AdamW(pipeline.unet.parameters(), lr=1e-4)
mse_loss = torch.nn.MSELoss()

# Training loop
for epoch in range(num_epochs):
    for step, (images, labels) in enumerate(dataloader):
        # Sample noise and timesteps
        noise = torch.randn_like(images)
        timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, 
                                (images.shape[0],), device=device)
        
        # Add noise to images
        noisy_images = pipeline.scheduler.add_noise(images, noise, timesteps)
        
        # Predict noise
        noise_pred = pipeline.unet(noisy_images, timesteps, labels)
        
        # Compute loss and update
        loss = mse_loss(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Save checkpoint periodically
        if step % 100 == 0:
            pipeline.save_training_state(
                "./checkpoint",
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                loss=loss.item()
            )
```

### Save and Load Training State

```python
# Save complete training state
pipeline.save_training_state(
    "./model_checkpoint",
    optimizer=optimizer,
    epoch=10,
    step=1000,
    loss=0.123,
    # Any additional state
    learning_rate=1e-4,
    custom_metric=0.456
)

# Load training state
pipeline, training_state = ClassConditionedPipeline.load_training_state(
    "./model_checkpoint",
    optimizer=optimizer,  # Will restore optimizer state
    device="cuda"
)

print(f"Resumed from epoch {training_state['epoch']}")
```

## API Reference

### ClassConditionedPipeline

#### `__call__(class_labels, batch_size=1, num_inference_steps=1000, **kwargs)`

Generate images conditioned on class labels.

**Parameters:**
- `class_labels`: Class labels for conditioning
  - `int`: Single class (repeated for batch_size)
  - `List[int]`: List of class labels (length must match batch_size)
  - `torch.Tensor`: Tensor of class labels
- `batch_size`: Number of images to generate
- `num_inference_steps`: Number of denoising steps
- `generator`: Random generator for reproducibility
- `return_dict`: Whether to return results as dict (default: True)
- `return_pil`: Whether to return PIL images (default: False)
- `show_progress`: Whether to show progress bar (default: True)

**Returns:**
- Dict with 'images' and 'class_labels' keys

#### `generate_grid(classes_per_row=8, num_rows=10, **kwargs)`

Generate a grid of images with different classes.

**Returns:**
- Dict containing:
  - `images`: Individual images as tensors
  - `grid`: Combined grid as tensor
  - `pil_images`: Individual PIL Images
  - `pil_grid`: Grid as PIL Image
  - `class_labels`: Used class labels

#### `save_training_state(save_directory, optimizer=None, **kwargs)`

Save complete training state.

#### `load_training_state(load_directory, optimizer=None, device=None)`

Load pipeline and training state. Returns `(pipeline, training_state)` tuple.

## Examples

### Multiple Input Formats

```python
# Single class (repeated)
result = pipeline(class_labels=7, batch_size=4)

# Multiple specific classes
result = pipeline(class_labels=[0, 1, 2, 3], batch_size=4)

# Tensor input
labels = torch.tensor([5, 5, 6, 6])
result = pipeline(class_labels=labels, batch_size=4)
```

### Reproducible Generation

```python
# Set seed for reproducible results
generator = torch.Generator().manual_seed(42)
result = pipeline(
    class_labels=3,
    batch_size=2,
    generator=generator
)
```

### Quality vs Speed Trade-off

```python
# Fast generation (lower quality)
fast_result = pipeline(class_labels=1, num_inference_steps=10)

# High quality (slower)
quality_result = pipeline(class_labels=1, num_inference_steps=1000)
```

## File Structure

```
image-gen-evals/
├── run_training.py        # Main training script with Neptune integration
├── run_evals.py          # Evaluation script with Gemini AI assessment
├── net.py                # Pipeline and model implementations
├── checkpoint_utils.py   # Checkpoint save/load utilities
├── torchwatcher.py       # Advanced debugging and monitoring
├── test_pipeline.py      # Test suite
└── README.md            # This file
```

## Running Examples

### Run Training
```bash
python run_training.py --neptune-mode new --neptune-experiment "test-run"
```

### Run Evaluation
```bash
python run_evals.py --neptune-run-id "your-run-id" --global-step 1000
```

### Run Tests
```bash
python test_pipeline.py
```

## Requirements

```txt
torch>=2.0.0
torchvision>=0.15.0
diffusers>=0.21.0
neptune-scale>=0.2.0
google-genai>=0.3.0
tqdm>=4.64.0
matplotlib>=3.5.0
pillow>=9.0.0
```

## Notes

### HuggingFace Compatibility

This implementation follows HuggingFace Diffusers patterns:
- Extends `DiffusionPipeline` base class
- Uses `register_modules()` for component registration
- Implements standard `save_pretrained()` and `from_pretrained()` methods
- Compatible with HF Hub (with custom registration)

### Device Handling

The pipeline automatically detects the appropriate device (CUDA, MPS, CPU) and handles device transfers properly.

### Memory Optimization

For large batches or high-resolution images, consider:
- Using gradient checkpointing
- Enabling attention slicing
- Using fp16 precision
- Processing in smaller chunks

### Extending the Pipeline

To modify for different datasets:
1. Change `num_classes` and `class_emb_size` in `ClassConditionedUnet`
2. Adjust image dimensions in the `__call__` method
3. Update the scheduler configuration as needed

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch_size or num_inference_steps
2. **Slow Generation**: Decrease num_inference_steps for faster results
3. **Poor Quality**: Increase num_inference_steps or train longer
4. **Device Errors**: Ensure all components are on the same device
5. **Neptune Connection**: Verify NEPTUNE_API_TOKEN environment variable
6. **Gemini API Errors**: Check GEMINI_API_KEY and API quotas

### Performance Tips

- Use `torch.compile()` for faster inference (PyTorch 2.0+)
- Enable mixed precision training with `torch.cuda.amp`
- Use `torch.backends.cudnn.benchmark = True` for consistent input sizes
- Monitor Neptune runs at [scale.neptune.ai](https://scale.neptune.ai) for real-time metrics 
