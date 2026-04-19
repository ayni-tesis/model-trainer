# -*- coding: utf-8 -*-
"""
train_classifier.py
Entrena el clasificador de enfermedades de hojas de café (EfficientNet).

Uso:
    python train_classifier.py
    python train_classifier.py --arch mobilenetv2 --epochs 20
"""

import argparse
import matplotlib.pyplot as plt

from config import (
    ARCHITECTURE, INITIAL_EPOCHS, FINE_TUNING_EPOCHS,
    LEARNING_RATE, FINE_TUNING_LR, FINE_TUNE_LAYERS,
    AUGMENT_STRENGTH, DISEASE_CLASSES
)
from dataset import load_datasets, prepare_dataset, calculate_class_weights
from disease_classifier import build_disease_classifier, train_disease_classifier
from evaluate import evaluate_classifier


def plot_history(history: dict, save_path: str = "training_history.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title in zip(
        axes,
        [('accuracy', 'val_accuracy'), ('loss', 'val_loss')],
        ['Precisión', 'Pérdida']
    ):
        ax.plot(history[metric[0]],     label='Entrenamiento')
        ax.plot(history[metric[1]],     label='Validación')
        ax.set_title(title)
        ax.set_xlabel('Época')
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📈 Curvas de aprendizaje guardadas en: {save_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Entrenamiento del clasificador de enfermedades")
    p.add_argument("--arch",        default=ARCHITECTURE,
                   choices=['efficientnetb0', 'mobilenetv2', 'resnet50'])
    p.add_argument("--epochs",      type=int, default=INITIAL_EPOCHS)
    p.add_argument("--ft-epochs",   type=int, default=FINE_TUNING_EPOCHS)
    p.add_argument("--lr",          type=float, default=LEARNING_RATE)
    p.add_argument("--ft-lr",       type=float, default=FINE_TUNING_LR)
    p.add_argument("--ft-layers",   type=int, default=FINE_TUNE_LAYERS)
    p.add_argument("--aug",         default=AUGMENT_STRENGTH,
                   choices=['light', 'moderate', 'strong'])
    p.add_argument("--no-weights",  action='store_true',
                   help="Desactivar pesos de clase balanceados")
    return p.parse_args()


def main():
    args = parse_args()

    # 1. Datos
    dataset_train, dataset_val, dataset_test, class_names = load_datasets()
    dataset_train_prep = prepare_dataset(dataset_train, augment=True,  strength=args.aug)
    dataset_val_prep   = prepare_dataset(dataset_val,   augment=False)
    dataset_test_prep  = prepare_dataset(dataset_test,  augment=False)

    # 2. Pesos de clase
    class_weights = None
    if not args.no_weights:
        class_weights = calculate_class_weights(dataset_train)

    # 3. Modelo
    model, base_model = build_disease_classifier(
        num_classes      = len(class_names),
        architecture     = args.arch,
        fine_tune_layers = args.ft_layers,
    )

    # 4. Entrenamiento
    model, history = train_disease_classifier(
        model            = model,
        base_model       = base_model,
        dataset_train    = dataset_train_prep,
        dataset_val      = dataset_val_prep,
        class_weights    = class_weights,
        initial_epochs   = args.epochs,
        fine_tuning_epochs = args.ft_epochs,
        learning_rate    = args.lr,
        fine_tuning_lr   = args.ft_lr,
    )

    # 5. Curvas
    plot_history(history)

    # 6. Evaluación en test
    evaluate_classifier(model, dataset_test_prep, class_names)


if __name__ == "__main__":
    main()
