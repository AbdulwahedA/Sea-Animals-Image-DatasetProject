import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import time
from logger_config import setup_logger

logger = setup_logger("train_torch")

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
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform_train)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_classes = len(train_dataset.classes)
logger.info(f"📊 {num_classes} Klassen erkannt: {train_dataset.classes}")

# --- Modell ---
model = models.resnet50(weights="IMAGENET1K_V2")
for p in model.parameters():
    p.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)

train_acc, val_acc = [], []
best_acc = 0.0
start_time = time.time()

# --- Training ---
for epoch in range(10):
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
    acc = 100 * correct / total
    train_acc.append(acc)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracy = 100 * correct / total
    val_acc.append(val_accuracy)

    logger.info(f"Epoch {epoch+1}: Train={acc:.2f}% | Val={val_accuracy:.2f}%")

    if val_accuracy > best_acc:
        best_acc = val_accuracy
        torch.save(model.state_dict(), MODELS_DIR / "best_resnet50.pth")

duration = (time.time() - start_time) / 60
logger.info(f"✅ Training abgeschlossen in {duration:.1f} Min. Beste Val-Acc: {best_acc:.2f}%")

# --- Plot speichern ---
plt.figure(figsize=(8, 5))
plt.plot(train_acc, label="Train")
plt.plot(val_acc, label="Val")
plt.legend()
plt.title("Accuracy Verlauf")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.savefig(MODELS_DIR / "accuracy_plot.png")
plt.close()
