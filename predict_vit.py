import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from logger_config import setup_logger

logger = setup_logger("predict_vit")

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "vit_best.pth"
VAL_DIR = ROOT / "FishDataset/val"

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

classes = [d.name for d in VAL_DIR.iterdir() if d.is_dir()]
num_classes = len(classes)

# --- Modell laden ---
model = models.vit_b_16(weights=None)
model.heads = torch.nn.Sequential(
    torch.nn.Linear(model.heads.head.in_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(512, num_classes)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

sample = list(VAL_DIR.glob("*/*.jpg"))[0]
logger.info(f"🔍 Testbild: {sample}")

img = Image.open(sample).convert("RGB")
x = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    preds = model(x)
    probs = torch.nn.functional.softmax(preds, dim=1)
    pred = probs.argmax(dim=1).item()
    conf = probs[0][pred].item() * 100

result = f"✅ Ergebnis: {classes[pred]} ({conf:.2f}% Konfidenz)"
logger.info(result)
print(result)
