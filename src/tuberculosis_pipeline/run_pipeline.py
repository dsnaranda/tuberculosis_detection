import os
from config import MODEL_DIR
from data_loader import get_data_generators
from trainer import train_model
from evaluator import evaluate_model

os.makedirs(MODEL_DIR, exist_ok=True)

print("Cargando datos...")
train_gen, val_gen, test_gen = get_data_generators()

print("Entrenando modelo...")
model = train_model(train_gen, val_gen)

print("Evaluando modelo...")
evaluate_model(model, val_gen)

print("\n✅ Proceso completo. Resultados guardados en la carpeta 'model'.")
