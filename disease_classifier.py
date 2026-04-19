# -*- coding: utf-8 -*-
"""
models/disease_classifier.py
Modelo de clasificación de enfermedades en hojas de café (EfficientNet / MobileNetV2 / ResNet50).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from config import (
    IMAGE_SIZE, NUM_CLASSES, ARCHITECTURE,
    FINE_TUNE_LAYERS, LEARNING_RATE, FINE_TUNING_LR,
    INITIAL_EPOCHS, FINE_TUNING_EPOCHS,
    CLASSIFIER_MODEL_PATH, BEST_CLASSIFIER_PATH
)


# ─────────────────────────────────────────────
#  CONSTRUCCIÓN DEL MODELO
# ─────────────────────────────────────────────

def build_disease_classifier(
    num_classes=NUM_CLASSES,
    architecture=ARCHITECTURE,
    fine_tune_layers=FINE_TUNE_LAYERS,
):
    """
    Construye un clasificador transfer-learning.

    Arquitecturas soportadas:
        'efficientnetb0' | 'mobilenetv2' | 'resnet50'

    Retorna: modelo Keras compilado (fase 1 - base congelada).
    """
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

    # ── Backbone ──────────────────────────────
    if architecture == 'efficientnetb0':
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=input_shape, include_top=False, weights='imagenet')
    elif architecture == 'mobilenetv2':
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape, include_top=False, weights='imagenet')
    elif architecture == 'resnet50':
        base_model = tf.keras.applications.ResNet50(
            input_shape=input_shape, include_top=False, weights='imagenet')
    else:
        raise ValueError(
            f"Arquitectura '{architecture}' no soportada. "
            "Usa: 'efficientnetb0', 'mobilenetv2', 'resnet50'"
        )

    # Congelar todo el backbone en fase 1
    base_model.trainable = False

    # ── Cabeza de clasificación ────────────────
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="disease_output")(x)

    model = tf.keras.Model(inputs, outputs, name=f"coffee_disease_{architecture}")

    # Descongelar las últimas N capas para fine-tuning posterior
    if fine_tune_layers > 0:
        for layer in base_model.layers[-fine_tune_layers:]:
            layer.trainable = True

    return model, base_model


def compile_model(model, learning_rate=LEARNING_RATE):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
        ]
    )
    return model


# ─────────────────────────────────────────────
#  ENTRENAMIENTO EN DOS FASES
# ─────────────────────────────────────────────

def get_callbacks(log_dir="logs/fit"):
    import datetime, os
    log_path = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10,
            restore_best_weights=True, min_delta=0.001),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2,
            patience=5, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(
            BEST_CLASSIFIER_PATH,
            monitor='val_accuracy', save_best_only=True,
            mode='max', verbose=1),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_path, histogram_freq=1),
    ]


def train_disease_classifier(
    model, base_model,
    dataset_train, dataset_val,
    class_weights=None,
    initial_epochs=INITIAL_EPOCHS,
    fine_tuning_epochs=FINE_TUNING_EPOCHS,
    learning_rate=LEARNING_RATE,
    fine_tuning_lr=FINE_TUNING_LR,
):
    """
    Entrena en dos fases:
      Fase 1 - backbone congelado, epochs=initial_epochs
      Fase 2 - fine-tuning con LR reducido, epochs adicionales=fine_tuning_epochs

    Retorna: (model, history_dict_combinado)
    """
    callbacks = get_callbacks()

    # ── Fase 1 ────────────────────────────────
    print("\n🔒 Fase 1: entrenamiento con backbone congelado")
    # Aseguramos que el backbone esté congelado en fase 1
    base_model.trainable = False
    model = compile_model(model, learning_rate)
    model.summary(line_length=100)

    h1 = model.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=initial_epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    # ── Fase 2 ────────────────────────────────
    print("\n🔓 Fase 2: fine-tuning con capas descongeladas")
    base_model.trainable = True
    model = compile_model(model, fine_tuning_lr)

    h2 = model.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=initial_epochs + fine_tuning_epochs,
        initial_epoch=h1.epoch[-1] + 1,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    # Combinar historiales
    combined = {}
    for key in h1.history:
        combined[key] = h1.history[key] + h2.history[key]

    model.save(CLASSIFIER_MODEL_PATH)
    print(f"\n✅ Modelo guardado en: {CLASSIFIER_MODEL_PATH}")
    return model, combined


# ─────────────────────────────────────────────
#  INFERENCIA CON EL CLASIFICADOR
# ─────────────────────────────────────────────

def load_disease_classifier(model_path=BEST_CLASSIFIER_PATH):
    print(f"📦 Cargando clasificador desde: {model_path}")
    return tf.keras.models.load_model(model_path)


def preprocess_for_classifier(image_np: np.ndarray) -> np.ndarray:
    """
    Recibe un array HxWx3 (uint8 o float) y devuelve (1, IMAGE_SIZE, IMAGE_SIZE, 3)
    normalizado a [-1, 1].
    """
    img = tf.image.resize(image_np, (IMAGE_SIZE, IMAGE_SIZE))
    img = tf.cast(img, tf.float32) / 127.5 - 1.0
    return tf.expand_dims(img, axis=0).numpy()


def classify_leaf(image_np: np.ndarray, model, class_names) -> dict:
    """
    Clasifica la enfermedad en una hoja recortada.

    Args:
        image_np: array NumPy HxWx3
        model: modelo Keras cargado
        class_names: lista de nombres de clase

    Retorna:
        {'class': str, 'confidence': float, 'probabilities': dict}
    """
    inp = preprocess_for_classifier(image_np)
    probs = model.predict(inp, verbose=0)[0]
    idx   = int(np.argmax(probs))

    return {
        "class":         class_names[idx],
        "confidence":    float(probs[idx]),
        "probabilities": {c: float(p) for c, p in zip(class_names, probs)},
    }
