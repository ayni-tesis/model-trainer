# -*- coding: utf-8 -*-
"""
evaluate.py
Evaluación del clasificador de enfermedades: métricas, reporte y matriz de confusión.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf


def evaluate_classifier(model, dataset_test, class_names, save_dir="."):
    """
    Evalúa el clasificador y guarda:
      - Informe de clasificación (consola)
      - Matriz de confusión (imagen PNG)
    """
    print("\n📊 Evaluando clasificador en conjunto de prueba...")

    # Métricas generales
    results = model.evaluate(dataset_test, verbose=1)
    print("\nMétricas generales:")
    for name, val in zip(model.metrics_names, results):
        print(f"  {name:<15}: {val:.4f}")

    # Predicciones
    y_pred, y_true = [], []
    for images, labels in dataset_test:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds,        axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    # Informe detallado
    print("\nInforme de clasificación:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title('Matriz de Confusión — Clasificador de Enfermedades')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()

    cm_path = f"{save_dir}/confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"✅ Matriz de confusión guardada en: {cm_path}")
