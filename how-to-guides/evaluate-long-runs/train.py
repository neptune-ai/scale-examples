import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os

from neptune_scale import Run

def main():
    run = Run(
        experiment_name="evaluate-long-runs",
    )

    # Config
    EPOCHS = 5
    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Model, data, optimizer
    model = models.resnet18(weights=None, num_classes=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, download=True,
                        transform=transforms.ToTensor()),
        batch_size=64, shuffle=True
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

# Training loop
    for epoch in range(EPOCHS):
        model.train()
        total, correct, running_loss = 0, 0, 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Calculate global step based on epoch and batch
            global_step = epoch * len(train_loader) + batch_idx

            run.log_metrics(
                data={
                    "loss": loss.item(),
                    "accuracy": correct / total,
                    "epoch": epoch,
                },
                step=global_step,
            )

        acc = correct / total
        print(f"Epoch {epoch}: Loss={running_loss:.2f}, Accuracy={acc:.4f}")

        # Save comprehensive checkpoint with additional state
        checkpoint = {
            'run_id': run._run_id,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss,
            'accuracy': acc,
            'global_step': global_step,
            'model_config': {
                'model_type': 'resnet18',
                'num_classes': 10,
                'device': str(device)
            },
            'training_config': {
                'learning_rate': 0.001,
                'batch_size': 64,
                'total_epochs': EPOCHS
            }
        }
        
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)

if __name__ == "__main__":
    main()