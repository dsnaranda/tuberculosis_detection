import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Configuración
IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 30
DATA_DIR = "../data"
MODEL_DIR = "../model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Data augmentation mejorado
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    fill_mode='nearest'
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Mostrar conteo
print("\nCantidad de imágenes por clase (entrenamiento):")
for cls, idx in train_gen.class_indices.items():
    print(f"  {cls}: {sum(train_gen.labels == idx)}")
print("\nCantidad de imágenes por clase (validación):")
for cls, idx in val_gen.class_indices.items():
    print(f"  {cls}: {sum(val_gen.labels == idx)}")

# Construcción del modelo
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3)

# Fase 1 - Congelado
model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=[early_stop, reduce_lr])

# Fase 2 - Fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[early_stop, reduce_lr])

# Guardar modelo
model.save(os.path.join(MODEL_DIR, "tuberculosis_model_finetuned.h5"))

# =======================
# MÉTRICAS Y GRÁFICOS
# =======================

val_probs = model.predict(val_gen, verbose=0).reshape(-1)
val_preds = (val_probs > 0.5).astype(int)
true_labels = val_gen.classes

# 1. Curvas de entrenamiento
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión por Época')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida por Época')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'training_curves.png'))
plt.close()

# 2. Matriz de Confusión
cm = confusion_matrix(true_labels, val_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=val_gen.class_indices.keys(),
            yticklabels=val_gen.class_indices.keys())
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'))
plt.close()

# 3. Curva ROC y AUC
fpr, tpr, _ = roc_curve(true_labels, val_probs)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('Curva ROC')
plt.xlabel('Falsos Positivos')
plt.ylabel('Verdaderos Positivos')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'roc_curve.png'))
plt.close()

# 4. Distribución de Predicciones
plt.figure()
plt.hist(val_probs[true_labels == 0], bins=30, alpha=0.5, label='Normal')
plt.hist(val_probs[true_labels == 1], bins=30, alpha=0.5, label='Tuberculosis')
plt.title('Distribución de Predicciones')
plt.xlabel('Probabilidad')
plt.ylabel('Frecuencia')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'prediction_distribution.png'))
plt.close()

# =======================
# MÉTRICAS ADICIONALES
# =======================
report = classification_report(true_labels, val_preds, target_names=val_gen.class_indices.keys())
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
auc_score = roc_auc_score(true_labels, val_probs)

print("\n📊 Classification Report:\n")
print(report)
print(f"Sensibilidad (Recall clase positiva): {sensitivity:.4f}")
print(f"Especificidad (Recall clase negativa): {specificity:.4f}")
print(f"AUC-ROC: {auc_score:.4f}")

with open(os.path.join(MODEL_DIR, "evaluation_report.txt"), "w") as f:
    f.write("Classification Report:\n")
    f.write(report + "\n")
    f.write(f"Sensibilidad: {sensitivity:.4f}\n")
    f.write(f"Especificidad: {specificity:.4f}\n")
    f.write(f"AUC-ROC: {auc_score:.4f}\n")

# =======================
# FALSOS POSITIVOS Y FALSOS NEGATIVOS
# =======================
false_positives = []
false_negatives = []
filenames = val_gen.filenames

for i, (pred, prob, true) in enumerate(zip(val_preds, val_probs, true_labels)):
    if pred == 1 and true == 0:
        false_positives.append((filenames[i], prob))  # predijo TB pero era normal
    elif pred == 0 and true == 1:
        false_negatives.append((filenames[i], prob))  # predijo normal pero era TB

# Guardar los resultados
with open(os.path.join(MODEL_DIR, "misclassified_samples.txt"), "w") as f:
    f.write("Falsos Positivos (Predijo Tuberculosis, era Normal):\n")
    for fp in false_positives:
        f.write(f"{fp[0]} - prob: {fp[1]:.4f}\n")

    f.write("\nFalsos Negativos (Predijo Normal, era Tuberculosis):\n")
    for fn in false_negatives:
        f.write(f"{fn[0]} - prob: {fn[1]:.4f}\n")   

print("\n✅ Evaluación completa. Resultados guardados en la carpeta 'model'.")
