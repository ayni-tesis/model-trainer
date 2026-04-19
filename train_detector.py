# -*- coding: utf-8 -*-
"""
train_detector.py
Entrena el detector de hojas de café con YOLOv8.

Uso:
    python train_detector.py
    python train_detector.py --epochs 100 --model yolov8s.pt --device cpu

Antes de correr:
    1. Anota tus imágenes con bounding boxes de hojas.
       Herramientas recomendadas: Roboflow, CVAT, LabelImg.
    2. Exporta en formato YOLO (clase cx cy w h  — normalizado).
    3. Asegúrate de que leaf_detection.yaml apunta a tus carpetas.
"""

import argparse
import os
from pathlib import Path

from config import (
    YOLO_MODEL_SIZE, YOLO_IMG_SIZE, YOLO_EPOCHS,
    YOLO_BATCH, YOLO_DATA_YAML, DETECTOR_MODEL_PATH
)
from leaf_detector import LeafDetector


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


YAML_TEMPLATE = """\
# leaf_detection.yaml
# Edita 'path' con la ruta absoluta a tu carpeta de dataset YOLO.

path: {data_path}   # carpeta raíz
train: images/train
val:   images/val

nc: 1
names:
  - coffee_leaf
"""


def create_yaml_if_missing(yaml_path: str):
    if os.path.exists(yaml_path):
        return
    data_dir = os.path.dirname(yaml_path)
    os.makedirs(data_dir, exist_ok=True)
    content = YAML_TEMPLATE.format(data_path=os.path.abspath(data_dir))
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"📄 Archivo YAML de ejemplo creado en: {yaml_path}")
    print("   ⚠️  Edítalo para apuntar a tu dataset real antes de entrenar.")


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def _parse_yaml_dataset_paths(yaml_path: str) -> dict:
    """Parsea un subconjunto mínimo del YAML para validar rutas de dataset YOLO."""
    parsed = {}
    lines = None
    for enc in ("utf-8", "latin-1"):
        try:
            with open(yaml_path, "r", encoding=enc) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue

    if lines is None:
        raise UnicodeDecodeError("utf-8", b"", 0, 1, "No se pudo decodificar YAML")

    for raw_line in lines:
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


def _list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    images = []
    for file in root.rglob("*"):
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(file)
    return images


def _is_label_line_valid(line: str) -> bool:
    parts = line.split()
    if len(parts) != 5:
        return False
    try:
        cls_id = int(parts[0])
        coords = [float(v) for v in parts[1:]]
    except ValueError:
        return False
    if cls_id < 0:
        return False
    return all(0.0 <= c <= 1.0 for c in coords)


def _validate_split(base_path: Path, split_name: str, split_path: str) -> tuple[list[str], dict]:
    errors: list[str] = []
    images_dir = _resolve_split_dir(base_path, split_path)
    labels_dir = _guess_labels_dir(base_path, split_path)

    if not images_dir.exists():
        errors.append(f"- [{split_name}] No existe carpeta de imágenes: {images_dir}")
        return errors, {}

    images = _list_images(images_dir)
    if not images:
        errors.append(f"- [{split_name}] No se encontraron imágenes en: {images_dir}")
        return errors, {}

    missing_labels = 0
    invalid_label_files = 0
    valid_bbox_lines = 0

    for image_path in images:
        rel_image = image_path.relative_to(images_dir)
        label_path = labels_dir / rel_image.with_suffix(".txt")

        if not label_path.exists():
            missing_labels += 1
            continue

        with open(label_path, "r", encoding="utf-8") as lf:
            lines = [ln.strip() for ln in lf if ln.strip()]

        for ln in lines:
            if _is_label_line_valid(ln):
                valid_bbox_lines += 1
            else:
                invalid_label_files += 1
                break

    if missing_labels:
        errors.append(
            f"- [{split_name}] Faltan {missing_labels} labels .txt "
            f"en {labels_dir} (deben emparejar 1:1 con las imágenes)."
        )

    if invalid_label_files:
        errors.append(
            f"- [{split_name}] Hay {invalid_label_files} archivos de label con formato inválido "
            "(esperado: class cx cy w h, normalizado 0-1)."
        )

    stats = {
        "split": split_name,
        "images": len(images),
        "valid_bbox_lines": valid_bbox_lines,
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
    }
    return errors, stats


def validate_yolo_dataset(yaml_path: str) -> bool:
    parsed = _parse_yaml_dataset_paths(yaml_path)
    required = {"path", "train", "val"}
    missing_keys = required - set(parsed.keys())
    if missing_keys:
        print("❌ YAML incompleto.")
        print(f"   Faltan claves: {', '.join(sorted(missing_keys))}")
        return False

    base_path = Path(parsed["path"])
    if not base_path.exists():
        print("❌ La ruta base del dataset no existe:")
        print(f"   {base_path}")
        return False

    all_errors: list[str] = []
    all_stats = []
    for split_name in ("train", "val"):
        split_errors, stats = _validate_split(base_path, split_name, parsed[split_name])
        all_errors.extend(split_errors)
        if stats:
            all_stats.append(stats)

    if all_errors:
        print("\n❌ Dataset YOLO inválido. Corrige estos puntos antes de entrenar:")
        for err in all_errors:
            print(err)
        print("\nSugerencia:")
        print("1) Organiza imágenes en dataset/images/train y dataset/images/val")
        print("2) Crea labels en dataset/labels/train y dataset/labels/val")
        print("3) Anota cada hoja con bbox en formato YOLO")
        return False

    total_bbox_lines = sum(s["valid_bbox_lines"] for s in all_stats)
    if total_bbox_lines == 0:
        print("\n❌ Dataset sin bounding boxes válidas.")
        print("   Se encontraron imágenes y labels, pero no hay anotaciones YOLO utilizables.")
        print("   Anota al menos una hoja por imagen con formato: class cx cy w h")
        return False

    print("\n✅ Validación de dataset YOLO correcta:")
    for s in all_stats:
        print(
            f"   [{s['split']}] imágenes={s['images']} "
            f"labels válidas={s['valid_bbox_lines']}"
        )
    return True


def parse_args():
    p = argparse.ArgumentParser(description="Entrenamiento del detector de hojas (YOLO)")
    p.add_argument("--model",   default=YOLO_MODEL_SIZE,
                   help="Backbone YOLO (yolov8n.pt / yolov8s.pt / yolov8m.pt)")
    p.add_argument("--data",    default=YOLO_DATA_YAML,
                   help="Ruta al archivo .yaml del dataset")
    p.add_argument("--epochs",  type=int, default=YOLO_EPOCHS)
    p.add_argument("--batch",   type=int, default=YOLO_BATCH)
    p.add_argument("--imgsz",   type=int, default=YOLO_IMG_SIZE)
    p.add_argument("--device",  default="0",
                   help="'0' para GPU, 'cpu' para CPU")
    p.add_argument("--project", default="runs/leaf_detector")
    p.add_argument("--name",    default="train")
    return p.parse_args()


def resolve_device(requested_device: str) -> str:
    """Resuelve device para YOLO con fallback a CPU si CUDA no está disponible."""
    if str(requested_device).lower() == "cpu":
        return "cpu"

    if str(requested_device).isdigit():
        try:
            import torch

            if not torch.cuda.is_available():
                print("⚠️  CUDA no disponible. Se usará CPU automáticamente.")
                return "cpu"

            requested_idx = int(requested_device)
            if requested_idx >= torch.cuda.device_count():
                print(
                    f"⚠️  GPU index {requested_idx} no existe "
                    f"(dispositivos disponibles: {torch.cuda.device_count()}). "
                    "Se usará CPU automáticamente."
                )
                return "cpu"
        except Exception:
            print("⚠️  No se pudo validar CUDA. Se usará CPU automáticamente.")
            return "cpu"

    return requested_device


def main():
    args = parse_args()
    selected_device = resolve_device(args.device)

    # Crear YAML de ejemplo si no existe
    create_yaml_if_missing(args.data)

    if not os.path.exists(args.data):
        print(f"❌ No se encontró: {args.data}")
        print("   Crea y edita el archivo .yaml antes de entrenar.")
        return

    if not validate_yolo_dataset(args.data):
        return

    detector = LeafDetector(model_path=None)   # carga backbone base
    detector.model = __import__('ultralytics').YOLO(args.model)

    results = detector.train(
        data_yaml = args.data,
        epochs    = args.epochs,
        batch     = args.batch,
        img_size  = args.imgsz,
        project   = args.project,
        name      = args.name,
        device    = selected_device,
    )

    print(f"\n✅ Entrenamiento finalizado.")
    print(f"   Modelo guardado en: {DETECTOR_MODEL_PATH}")


if __name__ == "__main__":
    main()
