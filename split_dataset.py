import os, shutil
from sklearn.model_selection import train_test_split

source_dir = "F:/Git/Sea Animals Image DatasetProject/Sea Animals Image Dataset"
base_dir = "F:/Git/Sea Animals Image DatasetProject/FishDataset"

os.makedirs(f"{base_dir}/train", exist_ok=True)
os.makedirs(f"{base_dir}/val", exist_ok=True)

for cls in os.listdir(source_dir):
    cls_path = os.path.join(source_dir, cls)
    if not os.path.isdir(cls_path):
        continue

    images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if len(images) == 0:
        continue

    train, val = train_test_split(images, test_size=0.2, random_state=42)

    os.makedirs(f"{base_dir}/train/{cls}", exist_ok=True)
    os.makedirs(f"{base_dir}/val/{cls}", exist_ok=True)

    for img in train:
        shutil.copy(os.path.join(cls_path, img), f"{base_dir}/train/{cls}/{img}")
    for img in val:
        shutil.copy(os.path.join(cls_path, img), f"{base_dir}/val/{cls}/{img}")

print("✅ Split abgeschlossen: FishDataset/train und FishDataset/val erstellt.")
