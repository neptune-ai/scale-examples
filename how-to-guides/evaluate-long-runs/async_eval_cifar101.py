# evaluate_cifar10_1.py
## Evaluate OOD CIFAR-10.1

import os
import time
import glob
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

from neptune_scale import Run

# === CIFAR-10.1 Dataset ===
class CIFAR101Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.images = np.load(os.path.join(data_dir, "cifar10.1_v6_data.npy"))
        self.labels = np.load(os.path.join(data_dir, "cifar10.1_v6_labels.npy"))
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray((self.images[idx] * 255).astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        # Convert label to torch.long (int64) to match CrossEntropyLoss expectations
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

def main():
    CHECKPOINT_DIR = "checkpoints"
    EVAL_ID = "cifar10_ood_c10_1"
    DATA_DIR = "./CIFAR-10.1"
    BATCH_SIZE = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed = set()

    # Dataset & loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    dataset = CIFAR101Dataset(DATA_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Evaluation function
    def evaluate(model, dataloader):
        model.eval()
        correct, total = 0, 0
        total_entropy = 0.0
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)

                # Accuracy and loss
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                preds = probs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                # Entropy (uncertainty)
                total_entropy += (-probs * torch.log(probs + 1e-12)).sum(dim=1).sum().item()

        acc = correct / total
        mean_entropy = total_entropy / total
        avg_loss = total_loss / len(dataloader)
        return acc, avg_loss, mean_entropy

    while True:
        checkpoint_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "model_epoch_*.pt")))
        for path in checkpoint_files:
            if path in processed:
                continue

            print(f"[{EVAL_ID}] Evaluating {os.path.basename(path)}...")

            # Load model
            model = models.resnet18(weights=None, num_classes=10)
            checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)

            acc, loss, entropy = evaluate(model, loader)

            # Neptune logging
            run = Run(run_id=checkpoint['run_id'], resume=True)
            run.log_metrics(
                data={
                    f"eval/{EVAL_ID}/accuracy": acc,
                    f"eval/{EVAL_ID}/loss": loss,
                    f"eval/{EVAL_ID}/entropy": entropy,
                },
                step=checkpoint['global_step'],
            )

            print(f"[{EVAL_ID}] Accuracy = {acc:.4f}, Loss = {loss:.4f}, Entropy = {entropy:.4f}")
            processed.add(path)

        time.sleep(10)

if __name__ == "__main__":
    main()
