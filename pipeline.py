# -*- coding: utf-8 -*-
"""
pipeline.py
Pipeline completo: Detector de hojas (TensorFlow) -> Clasificador de enfermedades (EfficientNet)

Flujo:
  imagen
    └─► LeafDetector.detect_and_crop()  →  lista de recortes
          └─► DiseaseClassifier.classify_leaf()  →  resultado por hoja
"""

from __future__ import annotations
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import (
    DISEASE_CLASSES, BEST_CLASSIFIER_PATH, DETECTOR_MODEL_PATH,
    USE_FULL_IMAGE_AS_FALLBACK, CLASSIFIER_CONF_THRESHOLD
)
from leaf_detector import LeafDetector
from disease_classifier import load_disease_classifier, classify_leaf


# Colores por enfermedad (BGR para OpenCV)
DISEASE_COLORS = {
    'miner':     (255, 100,   0),
    'nodisease': ( 50, 200,  50),
    'phoma':     (  0, 100, 255),
    'redspider': ( 50,  50, 255),
    'rust':      (  0, 200, 255),
}


# ─────────────────────────────────────────────
#  CLASE PIPELINE
# ─────────────────────────────────────────────

class CoffeeDiseaseDetectionPipeline:
    """
    Pipeline que integra deteccion de hojas (TensorFlow) +
    clasificación de enfermedades (EfficientNet).

    Uso:
        pipeline = CoffeeDiseaseDetectionPipeline()
        results  = pipeline.run("foto.jpg")
        pipeline.visualize(results, save_path="output.jpg")
    """

    def __init__(
        self,
        detector_path:   str = DETECTOR_MODEL_PATH,
        classifier_path: str = BEST_CLASSIFIER_PATH,
        class_names:    list = DISEASE_CLASSES,
    ):
        self.class_names      = class_names
        self.detector         = LeafDetector(model_path=detector_path)
        self.classifier       = load_disease_classifier(classifier_path)
        print("✅ Pipeline listo.")

    # ── Ejecución principal ───────────────────

    def run(self, image_source) -> dict:
        """
        Procesa una imagen completa y retorna todos los resultados.

        Args:
            image_source: ruta (str) o array NumPy HxWx3 (RGB)

        Retorna dict con:
            'image_path'  : str  (si se proveyó ruta)
            'image_np'    : array original
            'detections'  : lista de dicts YOLO (bbox + conf)
            'leaves'      : lista de dicts por hoja detectada:
                              { 'bbox', 'crop', 'disease_result' }
            'summary'     : resumen agregado
        """
        # ── Cargar imagen ─────────────────────
        if isinstance(image_source, str):
            if not os.path.exists(image_source):
                raise FileNotFoundError(f"Imagen no encontrada: {image_source}")
            image_np = cv2.cvtColor(cv2.imread(image_source), cv2.COLOR_BGR2RGB)
            image_path = image_source
        else:
            image_np   = image_source
            image_path = None

        print(f"\n🔍 Detectando hojas en la imagen...")
        crops, detections = self.detector.detect_and_crop(image_np)

        # ── Fallback si el detector no detecta nada ──
        if len(crops) == 0:
            if USE_FULL_IMAGE_AS_FALLBACK:
                print("⚠️  No se detectaron hojas. Usando imagen completa como fallback.")
                crops      = [image_np]
                detections = [{
                    "bbox": [0, 0, image_np.shape[1], image_np.shape[0]],
                    "confidence": 1.0,
                    "class_id": 0,
                    "fallback": True,
                }]
            else:
                return {
                    "image_path": image_path,
                    "image_np":   image_np,
                    "detections": [],
                    "leaves":     [],
                    "summary":    {"total_leaves": 0},
                }

        # ── Clasificar cada hoja ───────────────
        print(f"🌿 {len(crops)} hoja(s) detectada(s). Clasificando enfermedades...")
        leaves = []
        for i, (crop, det) in enumerate(zip(crops, detections)):
            result = classify_leaf(crop, self.classifier, self.class_names)
            leaves.append({
                "leaf_id":        i + 1,
                "bbox":           det["bbox"],
                "yolo_conf":      det["confidence"],
                "fallback":       det.get("fallback", False),
                "crop":           crop,
                "disease_result": result,
            })

            conf = result['confidence']
            flag = "✅" if conf >= CLASSIFIER_CONF_THRESHOLD else "⚠️ (baja confianza)"
            print(f"   Hoja {i+1}: {result['class']} ({conf*100:.1f}%) {flag}")

        summary = self._build_summary(leaves)
        return {
            "image_path": image_path,
            "image_np":   image_np,
            "detections": detections,
            "leaves":     leaves,
            "summary":    summary,
        }

    def _build_summary(self, leaves: list[dict]) -> dict:
        disease_counts: dict[str, int] = {}
        for leaf in leaves:
            cls = leaf["disease_result"]["class"]
            disease_counts[cls] = disease_counts.get(cls, 0) + 1

        most_common = max(disease_counts, key=disease_counts.get) if disease_counts else None

        return {
            "total_leaves":  len(leaves),
            "disease_counts": disease_counts,
            "most_common":   most_common,
            "healthy_pct":   (disease_counts.get("nodisease", 0) / len(leaves) * 100)
                              if leaves else 0,
        }

    # ── Visualización ────────────────────────

    def visualize(
        self,
        pipeline_result: dict,
        save_path: str | None = None,
        show: bool = False,
    ) -> np.ndarray:
        """
        Genera una imagen anotada con bounding boxes y etiquetas de enfermedad.
        También produce un subplot con las probabilidades de cada hoja.

        Retorna el array de la imagen anotada.
        """
        image_np = pipeline_result["image_np"].copy()
        leaves   = pipeline_result["leaves"]

        if not leaves:
            print("⚠️  Sin hojas para visualizar.")
            return image_np

        # ── Dibujar bounding boxes ─────────────
        for leaf in leaves:
            x1, y1, x2, y2 = leaf["bbox"]
            disease = leaf["disease_result"]["class"]
            conf    = leaf["disease_result"]["confidence"]
            color_rgb = DISEASE_COLORS.get(disease, (200, 200, 200))
            color_bgr = color_rgb[::-1]

            cv2.rectangle(image_np, (x1, y1), (x2, y2), color_bgr, 3)

            label = f"#{leaf['leaf_id']} {disease} {conf*100:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(image_np, (x1, y1 - th - 8), (x1 + tw + 4, y1), color_bgr, -1)
            cv2.putText(image_np, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # ── Figura Matplotlib ──────────────────
        n_leaves = len(leaves)
        fig, axes = plt.subplots(
            1, n_leaves + 1,
            figsize=(6 * (n_leaves + 1), 6),
            gridspec_kw={'width_ratios': [3] + [1] * n_leaves}
        )
        if n_leaves == 0:
            axes = [axes]

        # Panel principal
        axes[0].imshow(image_np)
        axes[0].set_title(
            f"Imagen analizada — {n_leaves} hoja(s)\n"
            f"Más común: {pipeline_result['summary'].get('most_common', 'N/A')}",
            fontsize=13
        )
        axes[0].axis('off')

        # Panel por hoja
        for i, leaf in enumerate(leaves):
            ax    = axes[i + 1]
            probs = leaf["disease_result"]["probabilities"]
            names = list(probs.keys())
            vals  = [probs[n] * 100 for n in names]
            colors = [
                DISEASE_COLORS.get(n, (180, 180, 180))
                for n in names
            ]
            bar_colors = [tuple(c / 255 for c in col) for col in colors]

            bars = ax.barh(names, vals, color=bar_colors)
            ax.set_xlim(0, 105)
            ax.set_xlabel("Probabilidad (%)")
            ax.set_title(f"Hoja #{leaf['leaf_id']}\n"
                         f"{leaf['disease_result']['class']} "
                         f"({leaf['disease_result']['confidence']*100:.1f}%)")

            for bar, val in zip(bars, vals):
                ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", va='center', fontsize=9)

        plt.suptitle("Pipeline: Detección de Hojas + Clasificación de Enfermedades",
                     fontsize=15, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Visualización guardada en: {save_path}")
        if show:
            plt.show()
        plt.close()

        return image_np


# ─────────────────────────────────────────────
#  MODO STANDALONE
# ─────────────────────────────────────────────

def run_pipeline_on_image(image_path: str, save_dir: str = "pipeline_results"):
    """Helper de conveniencia para procesar una sola imagen."""
    os.makedirs(save_dir, exist_ok=True)

    pipeline = CoffeeDiseaseDetectionPipeline()
    results  = pipeline.run(image_path)

    base = os.path.splitext(os.path.basename(image_path))[0]
    out  = os.path.join(save_dir, f"{base}_result.jpg")
    pipeline.visualize(results, save_path=out)

    # Imprimir resumen
    summary = results["summary"]
    print("\n📋 Resumen:")
    print(f"   Hojas detectadas : {summary['total_leaves']}")
    print(f"   Distribución     : {summary['disease_counts']}")
    print(f"   Más frecuente    : {summary['most_common']}")
    print(f"   Hojas sanas      : {summary['healthy_pct']:.1f}%")

    return results


if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else "dataset/test/rust/1120.jpg"
    run_pipeline_on_image(img)
