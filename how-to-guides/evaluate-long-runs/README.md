# Long-Running Model Evaluation with Neptune

This project demonstrates how to set up a robust system for evaluating long-running machine learning models with asynchronous evaluation capabilities. It's designed for scenarios where training takes hours or days, and you want to continuously monitor model performance on multiple datasets without interrupting the training process.

## 🎯 Use Cases

- **Long-running training jobs** (hours to days)
- **Continuous model evaluation** during training
- **Multi-dataset evaluation** (in-distribution + out-of-distribution)
- **Real-time performance monitoring**
- **Checkpoint-based evaluation** without stopping training
- **Uncertainty quantification** and OOD detection

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Training      │    │   Checkpoints    │    │   Evaluation    │
│   Script        │───▶│   Directory      │───▶│   Scripts       │
│   (train.py)    │    │                  │    │   (async_*.py)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Neptune       │    │   Model State    │    │   Neptune       │
│   Training Logs │    │   + Metadata     │    │   Eval Logs     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
evaluate-long-runs/
├── train.py                    # Main training script
├── async_evals.py             # Standard evaluation (CIFAR-10)
├── async_eval_cifar101.py     # OOD evaluation (CIFAR-10.1)
├── download_cifar10_1.py      # CIFAR-10.1 dataset downloader
├── load_checkpoint.py         # Checkpoint loading utilities
├── checkpoints/               # Saved model checkpoints
├── data/                      # Training data (CIFAR-10)
├── CIFAR-10.1/               # OOD evaluation data
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision neptune-scale numpy pillow
```

### 2. Download CIFAR-10.1 Dataset

```bash
python download_cifar10_1.py
```

### 3. Start Training

```bash
python train.py
```

### 4. Start Evaluation Scripts (in separate terminals)

```bash
# Standard evaluation
python async_evals.py

# OOD evaluation
python async_eval_cifar101.py
```

## 📊 What Gets Logged

### Training Metrics
- **Loss**: Training loss per batch
- **Accuracy**: Training accuracy per batch
- **Epoch**: Current training epoch

### Evaluation Metrics
- **Standard Evaluation** (`async_evals.py`):
  - `eval/accuracy`: Validation accuracy
  - `eval/loss`: Validation loss

- **OOD Evaluation** (`async_eval_cifar101.py`):
  - `eval/cifar10_ood_c10_1/accuracy`: OOD accuracy
  - `eval/cifar10_ood_c10_1/loss`: OOD loss
  - `eval/cifar10_ood_c10_1/entropy`: Prediction uncertainty

## 💾 Enhanced Checkpoints

The training script saves comprehensive checkpoints containing:

```python
checkpoint = {
    'run_id': run._run_id,              # Neptune run ID
    'epoch': epoch,                     # Current epoch
    'model_state_dict': model.state_dict(),  # Model weights
    'optimizer_state_dict': optimizer.state_dict(),  # Optimizer state
    'loss': running_loss,               # Training loss
    'accuracy': acc,                    # Training accuracy
    'global_step': global_step,         # Total training steps
    'model_config': {                   # Model configuration
        'model_type': 'resnet18',
        'num_classes': 10,
        'device': str(device)
    },
    'training_config': {                # Training configuration
        'learning_rate': 0.001,
        'batch_size': 64,
        'total_epochs': EPOCHS
    }
}
```

## 🔄 How It Works

### Training Process
1. **Model Training**: `train.py` trains a ResNet-18 model on CIFAR-10
2. **Checkpoint Saving**: After each epoch, saves comprehensive checkpoint
3. **Neptune Logging**: Logs training metrics in real-time

### Evaluation Process
1. **Checkpoint Monitoring**: Evaluation scripts poll the checkpoints directory
2. **Model Loading**: Loads model state from checkpoints
3. **Evaluation**: Runs inference on validation/OOD datasets
4. **Neptune Logging**: Logs evaluation metrics to the same run
5. **Continuous Monitoring**: Repeats every 10 seconds

## 🛠️ Key Features

### 1. Asynchronous Evaluation
- Evaluation runs independently of training
- No interruption to training process
- Real-time performance monitoring

### 2. Comprehensive Checkpoints
- Model weights + optimizer state
- Training metadata and configuration
- Neptune run ID for seamless logging

### 3. Multi-Dataset Evaluation
- In-distribution evaluation (CIFAR-10 validation)
- Out-of-distribution evaluation (CIFAR-10.1)
- Uncertainty quantification

### 4. Robust Error Handling
- Graceful handling of missing checkpoints
- Data type compatibility fixes
- Network error recovery

## 🔧 Configuration

### Training Configuration
```python
EPOCHS = 5                    # Number of training epochs
CHECKPOINT_DIR = "checkpoints" # Checkpoint directory
BATCH_SIZE = 64               # Training batch size
LEARNING_RATE = 0.001         # Learning rate
```

### Evaluation Configuration
```python
EVAL_ID = "cifar10_ood_c10_1" # Evaluation identifier
BATCH_SIZE = 64               # Evaluation batch size
POLL_INTERVAL = 10            # Checkpoint polling interval (seconds)
```

## 🚀 Improvements & Extensions

### 1. Additional Evaluation Datasets
```python
# Add more OOD datasets
async_eval_imagenet.py      # ImageNet evaluation
async_eval_svhn.py          # SVHN evaluation
async_eval_texture.py       # Texture dataset evaluation
```

### 2. Advanced Metrics
```python
# Add to evaluation function
def evaluate(model, dataloader):
    # ... existing code ...
    
    # Additional metrics
    calibration_error = calculate_calibration_error(probs, labels)
    ece = expected_calibration_error(probs, labels)
    confidence = probs.max(dim=1)[0].mean().item()
    
    return acc, loss, entropy, calibration_error, ece, confidence
```

### 3. Model Ensembling
```python
# Load multiple checkpoints for ensemble evaluation
checkpoints = glob.glob("checkpoints/model_epoch_*.pt")
ensemble_predictions = []
for ckpt in checkpoints:
    model = load_model(ckpt)
    preds = model(images)
    ensemble_predictions.append(preds)
```

### 4. Early Stopping Integration
```python
# Add to training loop
best_accuracy = 0
patience = 5
patience_counter = 0

if acc > best_accuracy:
    best_accuracy = acc
    patience_counter = 0
    # Save best model
else:
    patience_counter += 1
    
if patience_counter >= patience:
    print("Early stopping triggered")
    break
```

### 5. Learning Rate Scheduling
```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2)

# In training loop
scheduler.step(acc)
```

## 🐛 Troubleshooting

### Common Issues

1. **CIFAR-10.1 Download Fails**
   ```bash
   # The script will automatically create synthetic data
   python download_cifar10_1.py
   ```

2. **Data Type Errors**
   - Fixed in `async_eval_cifar101.py`
   - Labels are now properly converted to `torch.long`

3. **Checkpoint Loading Issues**
   - Use `load_checkpoint.py` utilities
   - Handles both old and new checkpoint formats

4. **Neptune Connection Issues**
   - Check your Neptune API token
   - Verify internet connection
   - Scripts will continue polling even if logging fails

### Performance Tips

1. **GPU Memory**: Use smaller batch sizes if running out of memory
2. **Disk Space**: Monitor checkpoint directory size
3. **Network**: Use local Neptune server for faster logging
4. **Polling**: Adjust polling interval based on training speed

## 📈 Monitoring Dashboard

In Neptune, you'll see:
- **Training curves**: Loss and accuracy over time
- **Evaluation curves**: Validation and OOD performance
- **Model comparison**: Multiple runs side by side
- **Uncertainty analysis**: Entropy and confidence metrics

## 🤝 Contributing

To extend this system:

1. **Add new datasets**: Create new evaluation scripts following the pattern
2. **Add new metrics**: Extend the evaluation functions
3. **Add new models**: Modify the model loading code
4. **Add new experiments**: Create new training configurations

## 📚 References

- [CIFAR-10.1 Dataset](https://github.com/modestyachts/CIFAR-10.1)
- [PyTorch Checkpointing](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- [Neptune Documentation](https://docs.neptune.ai/)
- [Out-of-Distribution Detection](https://arxiv.org/abs/1706.02690)

---

**Note**: This system is designed for research and development. For production use, consider adding proper error handling, logging, and monitoring infrastructure. 