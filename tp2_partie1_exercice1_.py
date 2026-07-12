# =============================================================
# TP2 - Partie 1 - Exercice 1
# Classification d'images Fashion-MNIST avec un CNN
# =============================================================

# ------------------- 1. Importation des bibliothèques -------------------
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Dossier où seront sauvegardées toutes les images (créé automatiquement)
OUTPUT_DIR = "resultats_exercice1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Noms des classes (pour l'affichage)
class_names = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandals', 'Shirt', 'Sneaker', 'Bag', 'Ankle boots']

# ------------------- 2. Chargement et préparation des données -------------------
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

print("Train shape:", train_images.shape)
print("Test shape :", test_images.shape)

# Normalisation des pixels (0 à 1)
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# Redimensionnement pour ajouter le canal (28, 28, 1)
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# One-hot encoding des labels
train_labels_cat = to_categorical(train_labels, num_classes=10)
test_labels_cat = to_categorical(test_labels, num_classes=10)

# Affichage de quelques exemples (optionnel)
plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(train_images[i].reshape(28, 28), cmap="gray")
    plt.title(class_names[train_labels[i]])
    plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_exemples_images.png"), dpi=150)
plt.show()

# ------------------- 3. Construction du modèle CNN -------------------
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()

# ------------------- 4. Compilation du modèle -------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ------------------- 5. Entraînement du modèle -------------------
history = model.fit(
    train_images, train_labels_cat,
    epochs=10,
    batch_size=64,
    validation_data=(test_images, test_labels_cat)
)

# ------------------- 6. Évaluation et analyse -------------------
test_loss, test_acc = model.evaluate(test_images, test_labels_cat, verbose=0)
print(f"\nPrécision (accuracy) sur le jeu de test : {test_acc:.4f}")
print(f"Perte (loss) sur le jeu de test          : {test_loss:.4f}")

# Courbes accuracy et loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title("Accuracy - Entraînement vs Validation")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Loss - Entraînement vs Validation")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_courbes_accuracy_loss.png"), dpi=150)
plt.show()

# Matrice de confusion
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

cm = confusion_matrix(test_labels, predicted_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Prédiction')
plt.ylabel('Vraie classe')
plt.title('Matrice de confusion - Fashion-MNIST')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_matrice_confusion.png"), dpi=150)
plt.show()

# Identification des classes les plus confondues
cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)
i, j = np.unravel_index(np.argmax(cm_no_diag), cm_no_diag.shape)
print(f"\nLes classes les plus confondues sont : '{class_names[i]}' et "
      f"'{class_names[j]}' ({cm_no_diag[i, j]} erreurs)")

# ------------------- Sauvegarde des résultats texte pour le rapport -------------------
with open(os.path.join(OUTPUT_DIR, "resultats.txt"), "w", encoding="utf-8") as f:
    f.write("TP2 - Partie 1 - Exercice 1 - Fashion-MNIST\n")
    f.write("=" * 50 + "\n")
    f.write(f"Precision (accuracy) sur le jeu de test : {test_acc:.4f}\n")
    f.write(f"Perte (loss) sur le jeu de test          : {test_loss:.4f}\n")
    f.write(f"Classes les plus confondues : '{class_names[i]}' et "
            f"'{class_names[j]}' ({cm_no_diag[i, j]} erreurs)\n")

print(f"\nToutes les images et résultats ont été sauvegardés dans le dossier : {OUTPUT_DIR}/")
