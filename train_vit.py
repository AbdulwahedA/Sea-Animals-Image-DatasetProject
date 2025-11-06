import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
import numpy as np
import time
from logger_config import setup_logger

logger = setup_logger("train_vit")

ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "FishDataset/train"
VAL_DIR = ROOT / "FishDataset/val"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"🔥 Device: {device}")

# --- Transforms ---
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform_train)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_classes = len(train_dataset.classes)
logger.info(f"📊 {num_classes} Klassen erkannt: {train_dataset.classes}")

# --- Vision Transformer (ViT) ---
model = models.vit_b_16(weights="IMAGENET1K_V1")
for p in model.parameters():
    p.requires_grad = False
model.heads = nn.Sequential(
    nn.Linear(model.heads.head.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.heads.parameters(), lr=3e-4)

train_acc, val_acc, val_loss_list = [], [], []
best_acc = 0.0
start_time = time.time()

# --- Training Loop ---
for epoch in range(8):
    model.train()
    correct, total = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_accuracy = 100 * correct / total
    train_acc.append(train_accuracy)

    # --- Validation ---
    model.eval()
    correct, total, val_loss = 0, 0, 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_accuracy = 100 * correct / total
    val_loss /= len(val_loader)
    val_acc.append(val_accuracy)
    val_loss_list.append(val_loss)

    logger.info(f"Epoch {epoch+1}: Train={train_accuracy:.2f}% | Val={val_accuracy:.2f}% | Loss={val_loss:.4f}")

    if val_accuracy > best_acc:
        best_acc = val_accuracy
        torch.save(model.state_dict(), MODELS_DIR / "vit_best.pth")

# --- Confusion Matrix ---
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)
disp.plot(xticks_rotation="vertical", cmap="viridis")
plt.tight_layout()
plt.savefig(MODELS_DIR / "vit_confusion_matrix.png")
plt.close()

# --- Accuracy Plot ---
plt.figure(figsize=(8,5))
plt.plot(train_acc, label="Train Acc")
plt.plot(val_acc, label="Val Acc")
plt.legend()
plt.title("ViT Accuracy Verlauf")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.savefig(MODELS_DIR / "vit_accuracy.png")
plt.close()

# --- Loss Plot ---
plt.figure(figsize=(8,5))
plt.plot(val_loss_list, label="Validation Loss", color="orange")
plt.legend()
plt.title("ViT Loss Verlauf")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig(MODELS_DIR / "vit_loss.png")
plt.close()

duration = (time.time() - start_time) / 60
logger.info(f"✅ Training abgeschlossen in {duration:.1f} Min. Beste Val-Acc: {best_acc:.2f}%")
