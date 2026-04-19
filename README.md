# Coffee Disease Detection — Pipeline de Dos Modelos

## Arquitectura

```
Imagen de entrada
      │
      ▼
┌─────────────────────┐
│  Modelo 1: YOLO v8  │  ← Detecta y recorta hojas de café
│  (leaf_detector.pt) │
└─────────────────────┘
      │  crops (1..N hojas)
      ▼
┌────────────────────────────┐
│  Modelo 2: EfficientNetB0  │  ← Clasifica la enfermedad en cada hoja
│ (disease_classifier.keras) │
└────────────────────────────┘
      │
      ▼
  Resultado por hoja:
  { class, confidence, probabilities }
```

**Clases de enfermedad:** `miner` | `nodisease` | `phoma` | `redspider` | `rust`

---

## Estructura del Proyecto

```
coffee_disease_detection/
├── config.py                   # Todas las constantes y rutas
├── dataset.py                  # Carga y augmentación de datos
├── evaluate.py                 # Métricas y matriz de confusión
├── pipeline.py                 # Pipeline completo (inferencia)
├── train_classifier.py         # Entrena el clasificador EfficientNet
├── train_detector.py           # Entrena el detector YOLO
├── requirements.txt
├── models/
│   ├── __init__.py
│   ├── disease_classifier.py   # Modelo EfficientNet
│   └── leaf_detector.py        # Modelo YOLO
└── dataset/
    ├── train/                  # Subcarpetas por clase (clasificador)
    │   ├── miner/
    │   ├── nodisease/
    │   ├── phoma/
    │   ├── redspider/
    │   └── rust/
    ├── test/                   # Misma estructura que train/
    ├── images/                 # Imágenes para YOLO (detección)
    │   ├── train/
    │   └── val/
    ├── labels/                 # Anotaciones YOLO (bbox)
    │   ├── train/
    │   └── val/
    └── leaf_detection.yaml     # Config del dataset YOLO
```

---

## Instalación

```bash
pip install -r requirements.txt
```

---

## Flujo de Trabajo

### Paso 1 — Entrenar el detector de hojas (YOLO)

Primero anota tus imágenes con bounding boxes usando [Roboflow](https://roboflow.com)
o [LabelImg](https://github.com/HumanSignal/labelImg), exportando en formato YOLO.

```bash
python train_detector.py --epochs 50 --model yolov8n.pt --device 0
```

### Paso 2 — Entrenar el clasificador de enfermedades (EfficientNet)

```bash
python train_classifier.py --arch efficientnetb0 --epochs 30 --ft-epochs 15
```

Opciones:
```
--arch        efficientnetb0 | mobilenetv2 | resnet50
--epochs      épocas fase 1 (backbone congelado)
--ft-epochs   épocas adicionales fase 2 (fine-tuning)
--aug         light | moderate | strong
--no-weights  desactiva balance de clases por peso
```

### Paso 3 — Inferencia con el pipeline completo

```python
from pipeline import CoffeeDiseaseDetectionPipeline

pipeline = CoffeeDiseaseDetectionPipeline()
results  = pipeline.run("mi_foto.jpg")
pipeline.visualize(results, save_path="resultado.jpg", show=True)

print(results["summary"])
# {'total_leaves': 3, 'disease_counts': {'rust': 2, 'nodisease': 1},
#  'most_common': 'rust', 'healthy_pct': 33.3}
```

O desde la línea de comandos:

```bash
python pipeline.py dataset/test/rust/1120.jpg
```

---

## leaf_detection.yaml (ejemplo)

```yaml
path: /ruta/absoluta/al/dataset
train: images/train
val:   images/val

nc: 1
names:
  - coffee_leaf
```

---

## Notas

- Si YOLO no detecta ninguna hoja, el pipeline usa la imagen completa
  como fallback (configurable en `config.py → USE_FULL_IMAGE_AS_FALLBACK`).
- Los modelos entrenados se guardan automáticamente en `saved_models/`.
- Para cambiar el backbone, edita `ARCHITECTURE` en `config.py`.
