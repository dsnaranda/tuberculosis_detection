import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from config import IMG_SIZE, DATA_DIR

# Cargar modelo
model_path = os.path.join("..", "..", "model", "tuberculosis_model_finetuned.h5")
model = load_model(model_path)

# Preparar generador de test
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(
    os.path.join(DATA_DIR, 'test'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

# Obtener predicciones
predictions = model.predict(test_gen, verbose=1)
predicted_labels = (predictions > 0.5).astype(int).flatten()
true_labels = test_gen.classes

# Calcular métricas
correct = np.sum(predicted_labels == true_labels)
total = len(true_labels)
incorrect = total - correct

print(f"\n✅ Evaluación del conjunto de test:")
print(f"Total de imágenes: {total}")
print(f"Aciertos: {correct}")
print(f"Errores: {incorrect}")
print(f"Precisión: {correct / total:.2%}")
