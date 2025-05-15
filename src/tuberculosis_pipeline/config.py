IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 40

# Directorios de datos y modelo
DATA_DIR = "../../data"
TRAIN_DIR = f"{DATA_DIR}/train"
VAL_DIR   = f"{DATA_DIR}/val"
MODEL_DIR = "../../model"

# Modo de clases y forma de entrada
CLASS_MODE  = 'binary'
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Tasas de aprendizaje
LEARNING_RATE_FROZEN = 1e-4  # 0.0001 fase congelada
LEARNING_RATE_FINE   = 1e-5  # 0.00001 fase fine-tuning