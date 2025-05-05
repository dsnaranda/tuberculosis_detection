import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, roc_auc_score

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# =======================
# CONFIGURACIONES
# =======================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
DATA_DIR = "../data"
MODEL_DIR = "../model"
MODEL_NAME = "tuberculosis_model.h5"

os.makedirs(MODEL_DIR, exist_ok=True)

# =======================
# GENERADORES DE DATOS
# =======================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.3,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1
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

# =======================
# MODELO
# =======================
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # Inicialmente congelado

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# =======================
# ENTRENAMIENTO FASE 1
# =======================
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[early_stop])

# =======================
# FINE-TUNING FASE 2
# =======================
base_model.trainable = True
for layer in base_model.layers[:-20]:  # Solo entrenar últimas 20 capas
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
history_finetune = model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=[early_stop])

# =======================
# GUARDAR MODELO
# =======================
model.save(os.path.join(MODEL_DIR, MODEL_NAME))

# =======================
# EVALUACIÓN
# =======================
val_probs = model.predict(val_gen, verbose=0).reshape(-1)
val_preds = (val_probs > 0.5).astype(int)
true_labels = val_gen.classes

# Matriz de Confusión
cm = confusion_matrix(true_labels, val_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=val_gen.class_indices.keys(),
            yticklabels=val_gen.class_indices.keys())
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix_eff.png'))
plt.close()

# Curva ROC
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
plt.savefig(os.path.join(MODEL_DIR, 'roc_curve_eff.png'))
plt.close()

# Histograma
plt.figure()
plt.hist(val_probs[true_labels == 0], bins=30, alpha=0.5, label='Normal')
plt.hist(val_probs[true_labels == 1], bins=30, alpha=0.5, label='Tuberculosis')
plt.title('Distribución de Predicciones')
plt.xlabel('Probabilidad')
plt.ylabel('Frecuencia')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'prediction_distribution_eff.png'))
plt.close()

# Accuracy & Loss
def plot_metric(metric, name):
    plt.plot(history.history[metric], label='Entrenamiento')
    plt.plot(history.history[f'val_{metric}'], label='Validación')
    if metric in history_finetune.history:
        plt.plot(history_finetune.history[metric], label='Fine-tuning (train)', linestyle='--')
        plt.plot(history_finetune.history[f'val_{metric}'], label='Fine-tuning (val)', linestyle='--')
    plt.title(name)
    plt.xlabel('Épocas')
    plt.ylabel(name)
    plt.legend()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plot_metric('accuracy', 'Precisión')
plt.subplot(1, 2, 2)
plot_metric('loss', 'Pérdida')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'training_curves_eff.png'))
plt.close()

# Reporte de clasificación
report = classification_report(true_labels, val_preds, target_names=val_gen.class_indices.keys())
print("\n📊 Classification Report:\n")
print(report)

# Sensibilidad y especificidad
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
auc_score = roc_auc_score(true_labels, val_probs)
print(f"Sensibilidad: {sensitivity:.4f}")
print(f"Especificidad: {specificity:.4f}")
print(f"AUC-ROC: {auc_score:.4f}")

# Guardar métricas
with open(os.path.join(MODEL_DIR, "evaluation_report_eff.txt"), "w") as f:
    f.write("Classification Report:\n")
    f.write(report + "\n")
    f.write(f"Sensibilidad: {sensitivity:.4f}\n")
    f.write(f"Especificidad: {specificity:.4f}\n")
    f.write(f"AUC-ROC: {auc_score:.4f}\n")

print("\n✅ Entrenamiento y evaluación con EfficientNet completados.")
