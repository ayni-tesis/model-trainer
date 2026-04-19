# -*- coding: utf-8 -*-
"""
dataset.py - Carga y preparación de datos para el clasificador de enfermedades
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight

from config import (
    TRAIN_DIR, TEST_DIR, IMAGE_SIZE, BATCH_SIZE,
    VAL_SPLIT, SEED, AUGMENT_STRENGTH
)


# ─────────────────────────────────────────────
#  CARGA DE DATASETS
# ─────────────────────────────────────────────

def load_datasets(train_dir=TRAIN_DIR, test_dir=TEST_DIR, val_split=VAL_SPLIT):
    """
    Carga train / val / test desde directorios con subcarpetas por clase.
    Retorna: (dataset_train, dataset_val, dataset_test, class_names)
    """
    print("📂 Cargando datasets...")

    full_train = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        label_mode='categorical',
    )

    class_names = full_train.class_names
    print(f"   Clases detectadas: {class_names}")

    total_batches = tf.data.experimental.cardinality(full_train).numpy()
    val_batches   = max(1, int(total_batches * val_split))
    train_batches = total_batches - val_batches

    dataset_train = full_train.take(train_batches)
    dataset_val   = full_train.skip(train_batches)

    dataset_test = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=SEED,
        label_mode='categorical',
    )

    analyze_class_distribution(train_dir, test_dir, class_names)
    return dataset_train, dataset_val, dataset_test, class_names


def analyze_class_distribution(train_dir, test_dir, class_names):
    train_counts = [len(os.listdir(os.path.join(train_dir, c))) for c in class_names]
    test_counts  = [len(os.listdir(os.path.join(test_dir,  c))) for c in class_names]

    print(f"\n{'Clase':<15} {'Train':<10} {'Test':<10} {'Total':<10}")
    print("─" * 45)
    for i, c in enumerate(class_names):
        print(f"{c:<15} {train_counts[i]:<10} {test_counts[i]:<10} {train_counts[i]+test_counts[i]:<10}")
    print(f"\n   Total: {sum(train_counts)+sum(test_counts)} imágenes")


# ─────────────────────────────────────────────
#  AUGMENTACIÓN
# ─────────────────────────────────────────────

def create_augmentation_layer(strength=AUGMENT_STRENGTH):
    """Devuelve un tf.keras.Sequential con capas de aumento."""
    if strength == 'light':
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.1),
        ], name="augmentation_light")

    elif strength == 'moderate':
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomFlip("vertical"),
            layers.RandomRotation(0.3),
            layers.RandomZoom(0.2),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomContrast(0.2),
        ], name="augmentation_moderate")

    elif strength == 'strong':
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomFlip("vertical"),
            layers.RandomRotation(0.5),
            layers.RandomZoom(0.3),
            layers.RandomTranslation(0.2, 0.2),
            layers.RandomContrast(0.3),
            layers.RandomBrightness(0.2),
        ], name="augmentation_strong")

    raise ValueError(f"strength debe ser 'light', 'moderate' o 'strong', recibido: '{strength}'")


def normalize_image(image, label):
    """Normaliza píxeles a [-1, 1] (compatible con MobileNetV2/EfficientNet)."""
    return tf.cast(image, tf.float32) / 127.5 - 1.0, label


def prepare_dataset(dataset, augment=False, strength=AUGMENT_STRENGTH):
    """
    Aplica normalización (y opcionalmente augmentación) + prefetch.
    """
    AUTOTUNE = tf.data.AUTOTUNE

    if augment:
        aug_layer = create_augmentation_layer(strength)

        def augment_and_normalize(image, label):
            image = aug_layer(image, training=True)
            return normalize_image(image, label)

        dataset = dataset.map(augment_and_normalize, num_parallel_calls=AUTOTUNE)
    else:
        dataset = dataset.map(normalize_image, num_parallel_calls=AUTOTUNE)

    return dataset.prefetch(AUTOTUNE)


# ─────────────────────────────────────────────
#  PESOS DE CLASE (desbalance)
# ─────────────────────────────────────────────

def calculate_class_weights(dataset_train):
    """
    Calcula pesos por clase para manejar datasets desbalanceados.
    Retorna dict {clase_idx: peso}.
    """
    print("⚖️  Calculando pesos de clase...")
    y_train = []
    for _, labels in dataset_train.unbatch():
        y_train.append(np.argmax(labels.numpy()))

    y_train = np.array(y_train)
    classes = np.unique(y_train)

    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights_dict = dict(zip(classes.tolist(), weights.tolist()))

    print(f"   Pesos: {class_weights_dict}")
    return class_weights_dict
