import os
from config import MODEL_DIR
from data_loader import get_data_generators
from trainer import train_model
from evaluator import evaluate_model

os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Cargar datos (solo train y val)
train_gen, val_gen = get_data_generators()

# 2. Entrenar y obtener el modelo con gráficas
model = train_model(train_gen, val_gen)

# 3. Evaluar modelo en validación
evaluate_model(model, val_gen)

print("\n✅ Entrenamiento y evaluación en validación completos. El modelo está listo para usar con tus datos de test.")
