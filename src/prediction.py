import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
p
# Configuración
IMG_SIZE = 224
MODEL_PATH = "../model/tuberculosis_model.h5"
IMAGE_PATH = "Test/TBCONTRASTE.jpg"  # Asegúrate de que este archivo exista en la carpeta src

# Cargar modelo
model = load_model(MODEL_PATH)

# Preprocesamiento de la imagen
def procesar_imagen(ruta):
    img = image.load_img(ruta, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=0)  # Para lotes
    return img_array

# Predicción
def predecir(img_array):
    pred = model.predict(img_array)[0][0]
    if pred >= 0.5:
        print(f"\n🔴 Probabilidad de tuberculosis: {pred:.4f} → Positivo")
    else:
        print(f"\n🟢 Probabilidad de tuberculosis: {1 - pred:.4f} → Negativo")

# Main
if __name__ == "__main__":
    if os.path.exists(IMAGE_PATH):
        print(f"\n📂 Imagen encontrada: {IMAGE_PATH}")
        img_array = procesar_imagen(IMAGE_PATH)
        predecir(img_array)
    else:
        print(f"⚠️ No se encontró la imagen '{IMAGE_PATH}' en la carpeta actual.")
