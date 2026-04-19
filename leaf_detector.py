# -*- coding: utf-8 -*-
"""
models/leaf_detector.py
Detector de hojas de café usando YOLOv8 (ultralytics).

Instalación:
    pip install ultralytics

Dataset esperado (formato YOLO):
    dataset/
    ├── leaf_detection.yaml
    ├── images/
    │   ├── train/   *.jpg
    │   └── val/     *.jpg
    └── labels/
        ├── train/   *.txt  (formato: class cx cy w h  — normalizado 0-1)
        └── val/     *.txt

leaf_detection.yaml:
    path: /ruta/absoluta/al/dataset
    train: images/train
    val:   images/val
    nc: 1
    names: ['coffee_leaf']
"""

from __future__ import annotations
import os
import numpy as np
import cv2

from config import (
    YOLO_MODEL_SIZE, YOLO_IMG_SIZE, YOLO_EPOCHS,
    YOLO_BATCH, YOLO_CONF_THRESH, YOLO_IOU_THRESH,
    YOLO_DATA_YAML, DETECTOR_MODEL_PATH
)


# ─────────────────────────────────────────────
#  CLASE PRINCIPAL
# ─────────────────────────────────────────────

class LeafDetector:
    """
    Wrapper sobre YOLOv8 para detección de hojas de café.

    Uso básico:
        detector = LeafDetector()               # carga modelo pre-entrenado
        crops    = detector.detect_and_crop(img_np)   # lista de arrays recortados
    """

    def __init__(self, model_path: str | None = None):
        """
        Args:
            model_path: ruta al .pt personalizado.
                        Si es None, usa DETECTOR_MODEL_PATH de config.
                        Si ese archivo tampoco existe, descarga el backbone base.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics no está instalado. Ejecuta:\n"
                "    pip install ultralytics"
            )

        path = model_path or DETECTOR_MODEL_PATH

        if os.path.exists(path):
            print(f"📦 Cargando detector de hojas desde: {path}")
            self.model = YOLO(path)
        else:
            print(f"⚠️  No se encontró modelo en '{path}'. "
                  "Cargando backbone base (requiere entrenamiento propio).")
            self.model = YOLO(YOLO_MODEL_SIZE)

        self.conf  = YOLO_CONF_THRESH
        self.iou   = YOLO_IOU_THRESH
        self.imgsz = YOLO_IMG_SIZE

    # ── Entrenamiento ─────────────────────────

    def train(
        self,
        data_yaml: str   = YOLO_DATA_YAML,
        epochs: int      = YOLO_EPOCHS,
        batch: int       = YOLO_BATCH,
        img_size: int    = YOLO_IMG_SIZE,
        project: str     = "runs/leaf_detector",
        name: str        = "train",
        device: str      = "0",   # '0' para GPU, 'cpu' para CPU
    ):
        """
        Entrena YOLOv8 con el dataset de hojas.

        Requiere dataset en formato YOLO con leaf_detection.yaml.
        """
        print(f"🚀 Entrenando detector de hojas ({epochs} épocas)...")
        results = self.model.train(
            data    = data_yaml,
            epochs  = epochs,
            batch   = batch,
            imgsz   = img_size,
            project = project,
            name    = name,
            device  = device,
            exist_ok = True,
        )

        # Guardar el mejor modelo en la ruta configurada
        best_pt = os.path.join(project, name, "weights", "best.pt")
        if os.path.exists(best_pt):
            import shutil
            os.makedirs(os.path.dirname(DETECTOR_MODEL_PATH), exist_ok=True)
            shutil.copy(best_pt, DETECTOR_MODEL_PATH)
            print(f"✅ Mejor modelo guardado en: {DETECTOR_MODEL_PATH}")

        return results

    # ── Inferencia ────────────────────────────

    def detect(self, image_np: np.ndarray) -> list[dict]:
        """
        Detecta hojas en una imagen.

        Args:
            image_np: array NumPy HxWx3 (BGR o RGB)

        Retorna:
            Lista de dicts con claves:
                'bbox'       : [x1, y1, x2, y2]  (píxeles enteros)
                'confidence' : float
                'class_id'   : int
        """
        results = self.model(
            image_np,
            conf  = self.conf,
            iou   = self.iou,
            imgsz = self.imgsz,
            verbose = False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                detections.append({
                    "bbox":       [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(box.conf[0].cpu()),
                    "class_id":   int(box.cls[0].cpu()),
                })

        return detections

    def detect_and_crop(
        self,
        image_np: np.ndarray,
        padding: int = 10,
    ) -> list[np.ndarray]:
        """
        Detecta hojas y devuelve una lista de recortes (arrays NumPy HxWx3).

        Args:
            image_np : imagen completa
            padding  : píxeles extra alrededor del bounding box

        Retorna:
            Lista de arrays recortados (uno por hoja detectada).
            Lista vacía si no se detecta ninguna hoja.
        """
        detections = self.detect(image_np)
        h, w = image_np.shape[:2]
        crops = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            # Aplicar padding con clamp a bordes
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
        """
        Dibuja bounding boxes sobre la imagen y retorna una copia anotada.
        """
        annotated = image_np.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"leaf {conf:.2f}",
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )
        return annotated
