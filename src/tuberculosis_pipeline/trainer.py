import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from config import EPOCHS, MODEL_DIR
from model_builder import build_model


def train_model(train_gen, val_gen):
    # Calcula proporción: 1/4 para fase congelada, el resto para fine-tuning
    epochs_frozen = EPOCHS // 4
    epochs_fine = EPOCHS - epochs_frozen

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3)

    # Fase 1: entrenamiento con capas congeladas
    model = build_model(fine_tune=False)
    model.fit(train_gen, validation_data=val_gen, epochs=epochs_frozen, callbacks=[early_stop, reduce_lr])

    # Fase 2: fine-tuning con capas descongeladas
    model = build_model(fine_tune=True)
    model.fit(train_gen, validation_data=val_gen, epochs=epochs_fine, callbacks=[early_stop, reduce_lr])

    # Guardar modelo final
    model.save(os.path.join(MODEL_DIR, 'tuberculosis_model_finetuned.h5'))

    return model
