import os
import pickle
import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from config import EPOCHS, MODEL_DIR
from model_builder import build_model

# Congelar 80% de las capas de la base
FREEZE_FRACTION = 0.8

def plot_combined_curves(history):
    epochs = range(1, len(history['loss']) + 1)

    # Precisión
    plt.figure()
    plt.plot(epochs, history['accuracy'],     label='Entrenamiento')
    plt.plot(epochs, history['val_accuracy'], label='Validación')
    plt.title('Precisión por Época')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    plt.savefig(os.path.join(MODEL_DIR, 'combined_accuracy_curve.png'))
    plt.close()

    # Pérdida
    plt.figure()
    plt.plot(epochs, history['loss'],     label='Entrenamiento')
    plt.plot(epochs, history['val_loss'], label='Validación')
    plt.title('Pérdida por Época')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.savefig(os.path.join(MODEL_DIR, 'combined_loss_curve.png'))
    plt.close()

def train_model(train_gen, val_gen):
    """
    - Fase 1: entrenar solo la cabeza (25% de EPOCHS), con 80% de la base congelada.
    - Fase 2: fine-tuning, descongelar el 20% restante y continuar hasta EPOCHS totales.
    - Callbacks: ModelCheckpoint, ReduceLROnPlateau, TensorBoard.
    - Guarda modelo final, pkl de historial y gráficas.
    """
    # Parámetros de época
    epochs_head  = EPOCHS // 4     # p.ej. 10/40
    epochs_total = EPOCHS          # 40

    # Crear directorios
    os.makedirs(MODEL_DIR, exist_ok=True)
    log_dir = os.path.join(MODEL_DIR, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'best_model.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    reduce_lr_cb = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=3,
        verbose=1
    )
    tb_cb = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=False,
        write_images=False
    )

    # ─── FASE 1: SOLO CABEZA ───
    model_head = build_model(fine_tune=False, freeze_fraction=FREEZE_FRACTION)
    history_head = model_head.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_head,
        callbacks=[reduce_lr_cb, tb_cb]
    )

    # ─── FASE 2: FINE‑TUNING ───
    model_fine = build_model(fine_tune=True, freeze_fraction=FREEZE_FRACTION)
    model_fine.set_weights(model_head.get_weights())

    history_fine = model_fine.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_total,
        initial_epoch=epochs_head,
        callbacks=[checkpoint_cb, reduce_lr_cb, tb_cb]
    )

    # ─── MERGE & SAVE ───
    merged_history = {
        k: history_head.history[k] + history_fine.history[k]
        for k in history_head.history.keys()
    }

    # Graficar
    plot_combined_curves(merged_history)

    # Guardar modelo final y historial
    final_model_path   = os.path.join(MODEL_DIR, 'tuberculosis_model_finetuned.h5')
    history_path = os.path.join(MODEL_DIR, 'training_history.pkl')

    model_fine.save(final_model_path)
    with open(history_path, 'wb') as f:
        pickle.dump(merged_history, f)

    print(f"\n✔ Entrenamiento completo. Modelos guardados en:\n"
          f" • Mejor modelo: {os.path.join(MODEL_DIR, 'best_model.h5')}\n"
          f" • Modelo final: {final_model_path}\n"
          f" • Historial:    {history_path}\n"
          f" • Logs TensorBoard: {log_dir}\n")

    return model_fine