# -*- coding: utf-8 -*-
"""
leaf_detector.py
Detector de hojas de cafe con TensorFlow/Keras.

Entrenamiento:
- Usa imagenes en dataset/images/{train,val}
- Usa labels en formato YOLO en dataset/labels/{train,val}
- Para mantener compatibilidad, cada imagen se entrena con 1 bbox objetivo
  (se toma la primera anotacion valida del archivo .txt)
"""

from __future__ import annotations
import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from config import (
    YOLO_MODEL_SIZE, YOLO_IMG_SIZE, YOLO_EPOCHS,
    YOLO_BATCH, YOLO_CONF_THRESH, YOLO_DATA_YAML, DETECTOR_MODEL_PATH
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def _parse_yaml_dataset_paths(yaml_path: str) -> dict:
    parsed = {}
    with open(yaml_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = _strip_comment(raw_line)
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key in {"path", "train", "val"}:
                parsed[key] = value
    return parsed


def _resolve_split_dir(base_path: Path, split_path: str) -> Path:
    split = Path(split_path)
    if split.is_absolute():
        return split
    return base_path / split


def _guess_labels_dir(base_path: Path, split_path: str) -> Path:
    normalized = split_path.replace("\\", "/")
    if normalized.startswith("images/"):
        return base_path / normalized.replace("images/", "labels/", 1)
    split_name = Path(normalized).name
    return base_path / "labels" / split_name


def _read_first_valid_yolo_bbox(label_path: Path) -> tuple[np.ndarray, float]:
    if not label_path.exists():
        return np.zeros(4, dtype=np.float32), 0.0

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                _class_id = int(parts[0])
                cx, cy, w, h = [float(x) for x in parts[1:]]
            except ValueError:
                continue
            if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
                continue
            return np.array([cx, cy, w, h], dtype=np.float32), 1.0

    return np.zeros(4, dtype=np.float32), 0.0


def _iter_images(root: Path):
    if not root.exists():
        return
    for file in root.rglob("*"):
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            yield file


def _load_split(
    images_dir: Path,
    labels_dir: Path,
    img_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_data: list[np.ndarray] = []
    bbox_data: list[np.ndarray] = []
    obj_data: list[np.ndarray] = []

    for image_path in _iter_images(images_dir):
        rel = image_path.relative_to(images_dir)
        label_path = labels_dir / rel.with_suffix(".txt")

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)

        bbox, obj = _read_first_valid_yolo_bbox(label_path)

        x_data.append(image_rgb.astype(np.float32) / 255.0)
        bbox_data.append(bbox)
        obj_data.append(np.array([obj], dtype=np.float32))

    if not x_data:
        return (
            np.zeros((0, img_size, img_size, 3), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, 1), dtype=np.float32),
        )

    return (
        np.stack(x_data, axis=0),
        np.stack(bbox_data, axis=0),
        np.stack(obj_data, axis=0),
    )


def _build_detector_model(img_size: int, variant: str) -> tf.keras.Model:
    if variant == "tiny":
        filters = [16, 32, 64]
        dense_units = 128
    elif variant == "medium":
        filters = [32, 64, 128, 256]
        dense_units = 384
    else:
        filters = [32, 64, 128]
        dense_units = 256

    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image")
    x = inputs

    for f in filters:
        x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    bbox = tf.keras.layers.Dense(4, activation="sigmoid", name="bbox")(x)
    obj = tf.keras.layers.Dense(1, activation="sigmoid", name="obj")(x)

    model = tf.keras.Model(inputs=inputs, outputs={"bbox": bbox, "obj": obj}, name="leaf_detector_tf")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            "bbox": tf.keras.losses.MeanSquaredError(),
            "obj": tf.keras.losses.BinaryCrossentropy(),
        },
        loss_weights={"bbox": 5.0, "obj": 1.0},
        metrics={
            "bbox": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
            "obj": [
                tf.keras.metrics.BinaryAccuracy(name="acc"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        },
    )
    return model


class LeafDetector:
    """
    Detector de hojas con TensorFlow.

    API compatible con el pipeline existente:
    - train(...)
    - detect(...)
    - detect_and_crop(...)
    """

    def __init__(self, model_path: str | None = None, model_variant: str = YOLO_MODEL_SIZE):
        self.conf = YOLO_CONF_THRESH
        self.imgsz = YOLO_IMG_SIZE
        self.model_variant = model_variant

        path = model_path or DETECTOR_MODEL_PATH
        if os.path.exists(path):
            print(f"Cargando detector TensorFlow desde: {path}")
            self.model = tf.keras.models.load_model(path)
            input_shape = self.model.input_shape
            if isinstance(input_shape, tuple) and len(input_shape) >= 3 and input_shape[1] is not None:
                self.imgsz = int(input_shape[1])
        else:
            print(
                f"No se encontro modelo en '{path}'. "
                "Se creara un detector TensorFlow nuevo al entrenar."
            )
            self.model = None

    def _ensure_model(self, img_size: int):
        if self.model is None:
            self.model = _build_detector_model(img_size, self.model_variant)
            self.imgsz = img_size
            return

        input_shape = self.model.input_shape
        current_img_size = int(input_shape[1]) if input_shape and input_shape[1] else img_size
        if current_img_size != img_size:
            print(
                f"Tamano de imagen solicitado ({img_size}) distinto al modelo cargado "
                f"({current_img_size}). Se recrea el modelo."
            )
            self.model = _build_detector_model(img_size, self.model_variant)
            self.imgsz = img_size

    def train(
        self,
        data_yaml: str = YOLO_DATA_YAML,
        epochs: int = YOLO_EPOCHS,
        batch: int = YOLO_BATCH,
        img_size: int = YOLO_IMG_SIZE,
        project: str = "runs/leaf_detector",
        name: str = "train",
        device: str = "gpu",
    ):
        """
        Entrena detector TensorFlow con labels en formato YOLO.
        """
        if not os.path.exists(data_yaml):
            raise FileNotFoundError(f"No se encontro YAML del dataset: {data_yaml}")

        parsed = _parse_yaml_dataset_paths(data_yaml)
        required = {"path", "train", "val"}
        missing = required - set(parsed.keys())
        if missing:
            raise ValueError(f"YAML incompleto, faltan claves: {sorted(missing)}")

        base_path = Path(parsed["path"])
        train_images = _resolve_split_dir(base_path, parsed["train"])
        val_images = _resolve_split_dir(base_path, parsed["val"])
        train_labels = _guess_labels_dir(base_path, parsed["train"])
        val_labels = _guess_labels_dir(base_path, parsed["val"])

        x_train, y_bbox_train, y_obj_train = _load_split(train_images, train_labels, img_size)
        x_val, y_bbox_val, y_obj_val = _load_split(val_images, val_labels, img_size)

        if x_train.shape[0] == 0:
            raise RuntimeError("No hay imagenes validas para entrenamiento en el split train.")

        self._ensure_model(img_size)

        output_dir = os.path.join(project, name)
        os.makedirs(output_dir, exist_ok=True)
        best_model_path = os.path.join(output_dir, "best_detector.keras")

        monitor = "val_loss" if x_val.shape[0] > 0 else "loss"
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=8,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                best_model_path,
                monitor=monitor,
                save_best_only=True,
                verbose=1,
            ),
        ]

        validation_data = None
        if x_val.shape[0] > 0:
            validation_data = (
                x_val,
                {
                    "bbox": y_bbox_val,
                    "obj": y_obj_val,
                },
            )

        if device == "gpu":
            tf_device = "/GPU:0"
        elif device == "dml":
            tf_device = "/DML:0"
        else:
            tf_device = "/CPU:0"

        print(f"Entrenando detector en dispositivo TensorFlow: {tf_device}")

        with tf.device(tf_device):
            history = self.model.fit(
                x_train,
                {
                    "bbox": y_bbox_train,
                    "obj": y_obj_train,
                },
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch,
                callbacks=callbacks,
                verbose=1,
            )

        if os.path.exists(best_model_path):
            self.model = tf.keras.models.load_model(best_model_path)

        os.makedirs(os.path.dirname(DETECTOR_MODEL_PATH), exist_ok=True)
        self.model.save(DETECTOR_MODEL_PATH)
        print(f"Detector TensorFlow guardado en: {DETECTOR_MODEL_PATH}")

        return history.history

    def detect(self, image_np: np.ndarray) -> list[dict]:
        if self.model is None:
            raise RuntimeError("Detector no inicializado. Entrena o carga un modelo antes de inferir.")

        h, w = image_np.shape[:2]

        if image_np.dtype != np.uint8:
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)

        resized = cv2.resize(image_np, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)

        preds = self.model.predict(x, verbose=0)

        if isinstance(preds, dict):
            bbox = preds["bbox"][0]
            confidence = float(preds["obj"][0][0])
        elif isinstance(preds, list) and len(preds) == 2:
            bbox = preds[0][0]
            confidence = float(preds[1][0][0])
        else:
            raise RuntimeError("Salida del detector TensorFlow no reconocida.")

        if confidence < self.conf:
            return []

        cx, cy, bw, bh = [float(np.clip(v, 0.0, 1.0)) for v in bbox]

        x1 = int((cx - bw / 2.0) * w)
        y1 = int((cy - bh / 2.0) * h)
        x2 = int((cx + bw / 2.0) * w)
        y2 = int((cy + bh / 2.0) * h)

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return []

        return [{
            "bbox": [x1, y1, x2, y2],
            "confidence": confidence,
            "class_id": 0,
        }]

    def detect_and_crop(
        self,
        image_np: np.ndarray,
        padding: int = 10,
    ):
        detections = self.detect(image_np)
        h, w = image_np.shape[:2]
        crops = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)

            crop = image_np[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)

        return crops, detections

    def draw_detections(
        self,
        image_np: np.ndarray,
        detections: list[dict],
        color: tuple = (0, 255, 0),
    ) -> np.ndarray:
        annotated = image_np.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"leaf {conf:.2f}",
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        return annotated
