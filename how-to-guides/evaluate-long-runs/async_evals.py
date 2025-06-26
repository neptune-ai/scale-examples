import torch
from torchvision import datasets, transforms, models
import os
import time
import glob

from neptune_scale import Run

def main():
    CHECKPOINT_DIR = "checkpoints"
    EVAL_LOG = "eval_log.csv"
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
        with torch.no_grad():
            for imgs, labels in dataloader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total

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

            acc = evaluate(model, val_loader)
            print(f"Evaluated {path}: Accuracy = {acc:.4f}")

            run = Run(
                # experiment_name="evaluate-long-runs", 
                run_id=checkpoint['run_id'],
                resume=True,
            )

            run.log_metrics(
                data={
                    "eval/accuracy": acc,
                },
                step=checkpoint['global_step'],
            )

            processed.add(path)

        time.sleep(10)  # Poll every 10 seconds

if __name__ == "__main__":
    main()