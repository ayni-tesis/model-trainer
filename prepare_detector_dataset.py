# -*- coding: utf-8 -*-
"""
prepare_detector_dataset.py
Prepara estructura de imágenes para detector YOLO a partir de un dataset de clasificación.

Convierte:
    dataset/train/<clase>/*.jpg
    dataset/test/<clase>/*.jpg

En:
    dataset/images/train/*.jpg
    dataset/images/val/*.jpg

Opcionalmente crea labels vacíos correspondientes en:
    dataset/labels/train/*.txt
    dataset/labels/val/*.txt

Uso:
    python prepare_detector_dataset.py
    python prepare_detector_dataset.py --create-empty-labels
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from config import DATA_DIR


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="Preparar estructura de dataset YOLO para detector")
    p.add_argument("--dataset-root", default=DATA_DIR, help="Carpeta raíz del dataset")
    p.add_argument("--train-source", default="train", help="Subcarpeta origen de entrenamiento")
    p.add_argument("--val-source", default="test", help="Subcarpeta origen de validación")
    p.add_argument(
        "--create-empty-labels",
        action="store_true",
        help="Crear .txt vacíos por imagen para facilitar flujo de anotación",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescribir imágenes destino si ya existen",
    )
    return p.parse_args()


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def ensure_dirs(dataset_root: Path):
    for rel in ("images/train", "images/val", "labels/train", "labels/val"):
        (dataset_root / rel).mkdir(parents=True, exist_ok=True)


def iter_class_images(split_dir: Path):
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for file in class_dir.rglob("*"):
            if is_image(file):
                yield class_name, file


def unique_target_path(dest_dir: Path, class_name: str, src_file: Path) -> Path:
    stem = src_file.stem.replace(" ", "_")
    ext = src_file.suffix.lower()
    candidate = dest_dir / f"{class_name}__{stem}{ext}"
    idx = 1
    while candidate.exists():
        candidate = dest_dir / f"{class_name}__{stem}__{idx}{ext}"
        idx += 1
    return candidate


def migrate_split(
    source_split: Path,
    dest_images: Path,
    dest_labels: Path,
    create_empty_labels: bool,
    overwrite: bool,
) -> tuple[int, int]:
    copied = 0
    skipped = 0

    if not source_split.exists():
        print(f"⚠️  No existe origen: {source_split}")
        return copied, skipped

    for class_name, src_file in iter_class_images(source_split):
        dst_file = unique_target_path(dest_images, class_name, src_file)

        if dst_file.exists() and not overwrite:
            skipped += 1
            continue

        shutil.copy2(src_file, dst_file)
        copied += 1

        if create_empty_labels:
            (dest_labels / dst_file.with_suffix(".txt").name).touch(exist_ok=True)

    return copied, skipped


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)

    source_train = dataset_root / args.train_source
    source_val = dataset_root / args.val_source

    ensure_dirs(dataset_root)

    train_images = dataset_root / "images" / "train"
    val_images = dataset_root / "images" / "val"
    train_labels = dataset_root / "labels" / "train"
    val_labels = dataset_root / "labels" / "val"

    print("📦 Preparando dataset de detección YOLO...")
    copied_train, skipped_train = migrate_split(
        source_split=source_train,
        dest_images=train_images,
        dest_labels=train_labels,
        create_empty_labels=args.create_empty_labels,
        overwrite=args.overwrite,
    )
    copied_val, skipped_val = migrate_split(
        source_split=source_val,
        dest_images=val_images,
        dest_labels=val_labels,
        create_empty_labels=args.create_empty_labels,
        overwrite=args.overwrite,
    )

    print("\n✅ Migración de imágenes completada")
    print(f"   train -> copiadas: {copied_train}, omitidas: {skipped_train}")
    print(f"   val   -> copiadas: {copied_val}, omitidas: {skipped_val}")

    print("\nSiguiente paso obligatorio:")
    print("1) Anotar bbox de hojas para cada imagen en labels/train y labels/val")
    print("2) Ejecutar entrenamiento nuevamente con train_detector.py")


if __name__ == "__main__":
    main()
