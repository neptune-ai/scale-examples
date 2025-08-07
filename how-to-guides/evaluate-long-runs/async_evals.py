import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import os
import time
import glob
import numpy as np

from neptune_scale import Run

def generate_benchmark_metrics(model_accuracy, benchmark_name):
    """
    Generate realistic benchmark metrics based on model accuracy.
    Returns accuracy and standard error values.
    """
    # Base accuracy ranges for different benchmarks
    base_ranges = {
        'gsm8k': (0.15, 0.85),  # GSM8K typically 15-85%
        'mmlu': (0.25, 0.75),   # MMLU typically 25-75%
        'ifeval': (0.20, 0.80)  # IFEval typically 20-80%
    }
    
    base_min, base_max = base_ranges[benchmark_name]
    
    # Scale the benchmark accuracy based on model performance
    # Higher model accuracy should correlate with higher benchmark scores
    scaled_accuracy = base_min + (base_max - base_min) * model_accuracy
    
    # Add some realistic variation
    noise = np.random.normal(0, 0.05)  # 5% standard deviation
    final_accuracy = np.clip(scaled_accuracy + noise, 0.0, 1.0)
    
    # Calculate standard error based on typical sample sizes
    # Assuming different sample sizes for different benchmarks
    sample_sizes = {
        'gsm8k': 1319,    # GSM8K test set size
        'mmlu': 14042,     # MMLU test set size  
        'ifeval': 500      # IFEval typical size
    }
    
    n = sample_sizes[benchmark_name]
    # Standard error = sqrt(p * (1-p) / n) where p is the proportion
    std_error = np.sqrt(final_accuracy * (1 - final_accuracy) / n)
    
    # Add some realistic variation to the standard error
    std_error_noise = np.random.normal(0, std_error * 0.1)  # 10% variation
    final_std_error = np.clip(std_error + std_error_noise, 0.001, 0.1)
    
    return final_accuracy, final_std_error

def calculate_benchmark_metrics(model_accuracy):
    """
    Calculate all benchmark metrics for a given model accuracy.
    """
    benchmarks = ['gsm8k', 'mmlu', 'ifeval']
    metrics = {}
    
    for benchmark in benchmarks:
        accuracy, std_error = generate_benchmark_metrics(model_accuracy, benchmark)
        
        metrics[f"{benchmark}_accuracy"] = accuracy
        metrics[f"{benchmark}_stderr_plus_1"] = accuracy + std_error
        metrics[f"{benchmark}_stderr_minus_1"] = accuracy - std_error
        
        # For IFEval, also include the standard error itself
        if benchmark == 'ifeval':
            metrics[f"{benchmark}_std_error"] = std_error
    
    return metrics

def main():
    CHECKPOINT_DIR = "checkpoints"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, download=True,
                        transform=transforms.ToTensor()),
        batch_size=64, shuffle=False
    )

    # Evaluation
    def evaluate(model, dataloader):
        model.eval()
        correct, total = 0, 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for imgs, labels in dataloader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                # Calculate accuracy
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        return accuracy, avg_loss

    # Track processed checkpoints
    processed = set()

    while True:
        checkpoint_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "model_epoch_*.pt")))
        for path in checkpoint_files:
            if path in processed:
                continue

            model = models.resnet18(weights=None, num_classes=10)
            checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)

            acc, loss = evaluate(model, val_loader)
            print(f"Evaluated {path}: Accuracy = {acc:.4f}, Loss = {loss:.4f}")

            # Calculate benchmark metrics based on model accuracy
            benchmark_metrics = calculate_benchmark_metrics(acc)
            
            run = Run(
                # experiment_name="evaluate-long-runs", 
                run_id=checkpoint['run_id'],
                resume=True,
            )

            run.log_metrics(
                data={
                    "eval/accuracy": acc,
                    "eval/loss": loss,
                    "eval/benchmark/gsm8k_accuracy": benchmark_metrics['gsm8k_accuracy'],
                    "eval/benchmark/gsm8k_stderr_plus_1": benchmark_metrics['gsm8k_stderr_plus_1'],
                    "eval/benchmark/gsm8k_stderr_minus_1": benchmark_metrics['gsm8k_stderr_minus_1'],
                    "eval/benchmark/mmlu_accuracy": benchmark_metrics['mmlu_accuracy'],
                    "eval/benchmark/mmlu_stderr_plus_1": benchmark_metrics['mmlu_stderr_plus_1'],
                    "eval/benchmark/mmlu_stderr_minus_1": benchmark_metrics['mmlu_stderr_minus_1'],
                    "eval/benchmark/ifeval_accuracy": benchmark_metrics['ifeval_accuracy'],
                    "eval/benchmark/ifeval_std_error": benchmark_metrics['ifeval_std_error'],
                },
                step=checkpoint['global_step'],
            )

            processed.add(path)

        time.sleep(10)  # Poll every 10 seconds

if __name__ == "__main__":
    main()