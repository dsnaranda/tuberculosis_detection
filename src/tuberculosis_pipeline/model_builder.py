from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from config import INPUT_SHAPE, LEARNING_RATE_FROZEN, LEARNING_RATE_FINE


def build_model(fine_tune=False, freeze_fraction=0.8):
    # Construye MobileNetV2 con pesos imagenet.
    # Congela el porcentaje inicial de capas según freeze_fraction en fine-tune.
    # Carga base MobileNetV2
    base_model = MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights='imagenet'
    )

    if not fine_tune:
        # Fase congelada: bloquea todo el base
        base_model.trainable = False
    else:
        # Fine-tuning: solo descongela el último 20% de capas
        total_layers = len(base_model.layers)
        num_to_freeze = int(total_layers * freeze_fraction)
        for i, layer in enumerate(base_model.layers):
            layer.trainable = False if i < num_to_freeze else True

    # Añade la cabeza personalizada
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    # Selección de LR
    lr = LEARNING_RATE_FINE if fine_tune else LEARNING_RATE_FROZEN
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model