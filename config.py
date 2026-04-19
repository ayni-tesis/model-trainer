# -*- coding: utf-8 -*-
"""
config.py - Configuración global del proyecto
"""

import os

# ─────────────────────────────────────────────
#  RUTAS
# ─────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
DATA_DIR          = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR         = os.path.join(DATA_DIR, "train")
TEST_DIR          = os.path.join(DATA_DIR, "test")

# Rutas de modelos guardados
MODELS_DIR              = os.path.join(BASE_DIR, "saved_models")
CLASSIFIER_MODEL_PATH   = os.path.join(MODELS_DIR, "disease_classifier.keras")
DETECTOR_MODEL_PATH     = os.path.join(MODELS_DIR, "leaf_detector.keras")
BEST_CLASSIFIER_PATH    = os.path.join(MODELS_DIR, "best_disease_classifier.keras")

os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  CLASIFICADOR DE ENFERMEDADES (EfficientNet)
# ─────────────────────────────────────────────
DISEASE_CLASSES  = ['miner', 'nodisease', 'phoma', 'redspider', 'rust']
NUM_CLASSES      = len(DISEASE_CLASSES)
IMAGE_SIZE       = 480          # píxeles (alto y ancho)
BATCH_SIZE       = 16
VAL_SPLIT        = 0.2

# Entrenamiento
INITIAL_EPOCHS       = 30
FINE_TUNING_EPOCHS   = 15
LEARNING_RATE        = 1e-3
FINE_TUNING_LR       = 1e-4
FINE_TUNE_LAYERS     = 50       # últimas N capas del backbone descongeladas

AUGMENT_STRENGTH     = 'moderate'   # 'light' | 'moderate' | 'strong'
ARCHITECTURE         = 'efficientnetb0'  # 'mobilenetv2' | 'efficientnetb0' | 'resnet50'

# ─────────────────────────────────────────────
#  DETECTOR DE HOJAS (TensorFlow)
# ─────────────────────────────────────────────
YOLO_MODEL_SIZE    = 'small'        # tiny | small | medium
YOLO_IMG_SIZE      = 640
YOLO_EPOCHS        = 50
YOLO_BATCH         = 16
YOLO_CONF_THRESH   = 0.40           # confianza mínima para detectar una hoja
YOLO_IOU_THRESH    = 0.45
YOLO_DATA_YAML     = os.path.join(DATA_DIR, "leaf_detection.yaml")

# ─────────────────────────────────────────────
#  PIPELINE DE INFERENCIA
# ─────────────────────────────────────────────
# Si YOLO no detecta ninguna hoja, se usa la imagen completa como fallback
USE_FULL_IMAGE_AS_FALLBACK = True
CLASSIFIER_CONF_THRESHOLD  = 0.50   # umbral mínimo de confianza para reportar resultado

# ─────────────────────────────────────────────
#  REPRODUCIBILIDAD
# ─────────────────────────────────────────────
SEED = 42
