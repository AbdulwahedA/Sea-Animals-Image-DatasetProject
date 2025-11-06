import torch
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
VAL_DIR = ROOT / "FishDataset/val"

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# === Modell laden ===
model = models.resnet50(weights=None)
num_classes = len(list((VAL_DIR).glob("*")))
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(512, num_classes)
)
model.load_state_dict(torch.load(MODELS_DIR / "resnet50_best.pth", map_location=device))
model.eval()
model.to(device)
print("✅ Modell geladen!")

# === Beispielbild ===
sample_img = list(VAL_DIR.glob("*/*.jpg"))[0]
print(f"🔍 Testbild: {sample_img}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = Image.open(sample_img).convert("RGB")
x = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    preds = model(x)
    pred_class = preds.argmax(dim=1).item()

classes = [d.name for d in VAL_DIR.iterdir() if d.is_dir()]
print(f"✅ Ergebnis: {classes[pred_class]}")
