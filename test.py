import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# Dynamische Pfade
base_path = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(base_path, "FishDataset", "train")
val_dir   = os.path.join(base_path, "FishDataset", "val")

# Überprüfen
print("📂 Trainingsdaten:", train_dir)
print("📂 Validierungsdaten:", val_dir)

# 1️⃣ Datenvorbereitung
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32
)
val_data = datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32
)

# 2️⃣ Modellaufbau (Transfer Learning)
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Feature-Extraktion einfrieren

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(train_data.num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 3️⃣ Training starten
history = model.fit(train_data, validation_data=val_data, epochs=10)

# 4️⃣ Ergebnisse plotten
plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()

# 5️⃣ Modell speichern
model.save("sea_animals_model.h5")
print("✅ Training abgeschlossen! Modell gespeichert als sea_animals_model.h5")

