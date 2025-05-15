import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import IMG_SIZE, BATCH_SIZE, TRAIN_DIR, VAL_DIR, CLASS_MODE


def get_data_generators():
  
    # Devuelve dos generadores: train y val.
    # Aplica aumento de datos solo a train y val.
  
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        brightness_range=(0.8, 1.2),
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_gen = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        shuffle=False
    )

    return train_gen, val_gen