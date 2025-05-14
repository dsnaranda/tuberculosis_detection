import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ======================
# CONFIGURACIONES
# ======================
IMG_SIZE = 224
BATCH_SIZE = 32
INITIAL_EPOCHS = 12
FINETUNE_EPOCHS = 10
DATA_DIR = "../data"
MODEL_DIR = "../model"
MODEL_NAME = "tuberculosis_model_efficientnet.h5"
os.makedirs(MODEL_DIR, exist_ok=True)

# ======================
# VERIFICAR CLASES
# ======================
def contar_imagenes_por_clase(data_dir):
    print("Conteo de imágenes por clase:")
    clases = os.listdir(data_dir)
    for clase in clases:
        clase_path = os.path.join(data_dir, clase)
        if os.path.isdir(clase_path):
            n_imgs = len(os.listdir(clase_path))
            print(f" - {clase}: {n_imgs} imágenes")

contar_imagenes_por_clase(DATA_DIR)

# ======================
# GENERADORES DE DATOS
# ======================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.2,
    width_shift_range=0.05,
    height_shift_range=0.05
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

print("\nClases:", train_gen.class_indices)
print("Train:", dict(zip(*np.unique(train_gen.classes, return_counts=True))))
print("Val:", dict(zip(*np.unique(val_gen.classes, return_counts=True))))

# ======================
# BALANCEO DE CLASES
# ======================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights_dict = dict(enumerate(class_weights))
print("\nPesos por clase:", class_weights_dict)

# ======================
# CONSTRUCCIÓN DEL MODELO
# ======================
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(1e-3))(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(base_model.input, output)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

# ======================
# ENTRENAMIENTO INICIAL
# ======================
history_initial = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=INITIAL_EPOCHS,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights_dict
)

# ======================
# FINE-TUNING
# ======================
base_model.trainable = True
for layer in base_model.layers[:-60]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

history_finetune = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=FINETUNE_EPOCHS,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights_dict
)

# ======================
# GUARDAR MODELO
# ======================
model.save(os.path.join(MODEL_DIR, MODEL_NAME))

# ======================
# EVALUACIÓN
# ======================
val_probs = model.predict(val_gen, verbose=0).ravel()
val_preds = (val_probs >= 0.5).astype(int)
true_labels = val_gen.classes

# Curvas
def plot_metric(hist1, hist2, metric, name, fname):
    plt.figure()
    plt.plot(hist1.history[metric], label='Train init')
    plt.plot(hist1.history['val_' + metric], label='Val init')
    if metric in hist2.history:
        plt.plot(hist2.history[metric], label='Train ft', linestyle='--')
        plt.plot(hist2.history['val_' + metric], label='Val ft', linestyle='--')
    plt.title(name)
    plt.xlabel('Épocas')
    plt.ylabel(name)
    plt.legend()
    plt.savefig(os.path.join(MODEL_DIR, fname))
    plt.close()

plot_metric(history_initial, history_finetune, 'accuracy', 'Precisión', 'curves_eff_accuracy.png')
plot_metric(history_initial, history_finetune, 'loss', 'Pérdida', 'curves_eff_loss.png')

# Matriz de Confusión
dcm = confusion_matrix(true_labels, val_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(dcm, annot=True, fmt='d', cmap='Blues',
            xticklabels=val_gen.class_indices.keys(),
            yticklabels=val_gen.class_indices.keys())
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.savefig(os.path.join(MODEL_DIR, 'confusion_eff.png'))
plt.close()

# ROC
fpr, tpr, _ = roc_curve(true_labels, val_probs)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], '--')
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc='lower right')
plt.savefig(os.path.join(MODEL_DIR, 'roc_eff.png'))
plt.close()

# Distribución de probabilidades
def plot_hist():
    plt.figure()
    plt.hist(val_probs[true_labels == 0], bins=25, alpha=0.6, label='Normal')
    plt.hist(val_probs[true_labels == 1], bins=25, alpha=0.6, label='Tuberculosis')
    plt.title('Distribución de Probabilidades')
    plt.xlabel('Probabilidad')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.savefig(os.path.join(MODEL_DIR, 'hist_eff.png'))
    plt.close()
plot_hist()

# Reporte
report = classification_report(true_labels, val_preds, target_names=val_gen.class_indices.keys())
print(report)
tn, fp, fn, tp = dcm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
auc_score = roc_auc_score(true_labels, val_probs)
metrics_txt = f"Sensibilidad: {sensitivity:.4f}\nEspecificidad: {specificity:.4f}\nAUC-ROC: {auc_score:.4f}\n"
print(metrics_txt)
with open(os.path.join(MODEL_DIR, 'report_eff.txt'), 'w') as f:
    f.write(report + '\n' + metrics_txt)

print("✅ Entrenamiento y evaluación finalizados. Archivos guardados en:", MODEL_DIR)
