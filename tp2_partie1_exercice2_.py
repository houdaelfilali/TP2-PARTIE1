# =============================================================
# TP2 - Partie 1 - Exercice 2
# Classification d'images MNIST avec un CNN plus profond
# =============================================================

# ------------------- 1. Importation des bibliothèques -------------------
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Dossier où seront sauvegardées toutes les images (créé automatiquement)
OUTPUT_DIR = "resultats_exercice2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------- 2. Chargement et préparation des données -------------------
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalisation des pixels (0 à 1)
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# Redimensionnement en (28, 28, 1)
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# One-hot encoding des labels
train_labels_cat = to_categorical(train_labels, num_classes=10)
test_labels_cat = to_categorical(test_labels, num_classes=10)

# Séparation train / validation (ex : 90% train, 10% validation)
train_images, val_images, train_labels_cat, val_labels_cat = train_test_split(
    train_images, train_labels_cat, test_size=0.1, random_state=42
)

print("Train shape:", train_images.shape)
print("Val shape  :", val_images.shape)
print("Test shape :", test_images.shape)

# ------------------- 3. Data Augmentation -------------------
# L'augmentation de données crée artificiellement de nouvelles variantes des
# images d'entraînement (légère rotation, translation, zoom). Cela empêche le
# modèle de "mémoriser" les images d'entraînement et le rend plus robuste aux
# variations qu'il rencontrera sur de nouvelles données (moins d'overfitting).
datagen = ImageDataGenerator(
    rotation_range=12,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(train_images)

# ------------------- 4. Définition du modèle CNN -------------------
model = models.Sequential([
    # Bloc 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Bloc 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Bloc 3 (optionnel)
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Classification
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.summary()

# ------------------- 5. Compilation du modèle -------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ------------------- 6. Entraînement et validation -------------------
batch_size = 64
epochs = 12

history = model.fit(
    datagen.flow(train_images, train_labels_cat, batch_size=batch_size),
    steps_per_epoch=len(train_images) // batch_size,
    epochs=epochs,
    validation_data=(val_images, val_labels_cat)
)

# Courbes loss / accuracy train vs validation
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title("Accuracy - Train vs Validation")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Loss - Train vs Validation")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_courbes_accuracy_loss.png"), dpi=150)
plt.show()

# ------------------- 7. Évaluation sur le jeu de test -------------------
test_loss, test_acc = model.evaluate(test_images, test_labels_cat, verbose=0)
print(f"\nPrécision finale sur le jeu de test : {test_acc:.4f}")
print(f"Perte finale sur le jeu de test      : {test_loss:.4f}")

# Prédictions et identification des erreurs
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels_cat, axis=1)

# Matrice de confusion
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Prédiction')
plt.ylabel('Vraie classe')
plt.title('Matrice de confusion - MNIST')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_matrice_confusion.png"), dpi=150)
plt.show()

# Affichage de quelques images mal classées
wrong_idx = np.where(predicted_labels != true_labels)[0]
print(f"\nNombre d'images mal classées : {len(wrong_idx)} / {len(test_images)}")

plt.figure(figsize=(10, 10))
for i, idx in enumerate(wrong_idx[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_images[idx].reshape(28, 28), cmap='gray')
    plt.title(f"Vrai: {true_labels[idx]} / Prédit: {predicted_labels[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_images_mal_classees.png"), dpi=150)
plt.show()

# ------------------- Sauvegarde des résultats texte pour le rapport -------------------
with open(os.path.join(OUTPUT_DIR, "resultats.txt"), "w", encoding="utf-8") as f:
    f.write("TP2 - Partie 1 - Exercice 2 - MNIST\n")
    f.write("=" * 50 + "\n")
    f.write(f"Precision finale sur le jeu de test : {test_acc:.4f}\n")
    f.write(f"Perte finale sur le jeu de test      : {test_loss:.4f}\n")
    f.write(f"Nombre d'images mal classees : {len(wrong_idx)} / {len(test_images)}\n")

print(f"\nToutes les images et résultats ont été sauvegardés dans le dossier : {OUTPUT_DIR}/")

# ------------------- 8. Expérimentation (facultative) -------------------
# Idée d'implémentation : réécrire une fonction qui construit le modèle en
# prenant en paramètres le taux de dropout, la présence ou non de
# BatchNormalization, et le nombre de filtres, puis comparer les résultats.

def build_model(dropout_rate=0.25, use_batchnorm=True, filters=(32, 64)):
    layers_list = []
    for i, f in enumerate(filters):
        layers_list.append(layers.Conv2D(f, (3, 3), activation='relu', padding='same',
                                          input_shape=(28, 28, 1) if i == 0 else None))
        if use_batchnorm:
            layers_list.append(layers.BatchNormalization())
        layers_list.append(layers.MaxPooling2D((2, 2)))
        layers_list.append(layers.Dropout(dropout_rate))

    layers_list += [
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ]
    m = models.Sequential(layers_list)
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m

# Exemple de comparaison (à décommenter pour tester) :
# model_a = build_model(dropout_rate=0.25, use_batchnorm=True, filters=(32, 64))
# model_b = build_model(dropout_rate=0.5, use_batchnorm=False, filters=(64, 128))
# history_a = model_a.fit(train_images, train_labels_cat, epochs=5, batch_size=64,
#                         validation_data=(val_images, val_labels_cat))
# history_b = model_b.fit(train_images, train_labels_cat, epochs=5, batch_size=64,
#                         validation_data=(val_images, val_labels_cat))
