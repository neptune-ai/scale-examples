import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import os
import time
import glob

from neptune_scale import Run

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

            run = Run(
                # experiment_name="evaluate-long-runs", 
                run_id=checkpoint['run_id'],
                resume=True,
            )

            run.log_metrics(
                data={
                    "eval/accuracy": acc,
                    "eval/loss": loss,
                },
                step=checkpoint['global_step'],
            )

            processed.add(path)

        time.sleep(10)  # Poll every 10 seconds

if __name__ == "__main__":
    main()