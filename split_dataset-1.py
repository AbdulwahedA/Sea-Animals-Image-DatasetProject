import os, shutil
from sklearn.model_selection import train_test_split
from logger_config import setup_logger

logger = setup_logger("split_dataset")

source_dir = "Sea Animals Image Dataset"
base_dir = "FishDataset"

os.makedirs(f"{base_dir}/train", exist_ok=True)
os.makedirs(f"{base_dir}/val", exist_ok=True)

logger.info(f"Start Split aus {source_dir}")

for cls in os.listdir(source_dir):
    cls_path = os.path.join(source_dir, cls)
    if not os.path.isdir(cls_path):
        continue

    images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(images) == 0:
        continue

    train, val = train_test_split(images, test_size=0.2, random_state=42)
    os.makedirs(f"{base_dir}/train/{cls}", exist_ok=True)
    os.makedirs(f"{base_dir}/val/{cls}", exist_ok=True)

    for img in train:
        shutil.copy(os.path.join(cls_path, img), f"{base_dir}/train/{cls}")
    for img in val:
        shutil.copy(os.path.join(cls_path, img), f"{base_dir}/val/{cls}")

    logger.info(f"📂 {cls}: {len(train)} train, {len(val)} val")

logger.info("✅ Split abgeschlossen.")
print("✅ Split abgeschlossen! Daten liegen in FishDataset/train und FishDataset/val.")
