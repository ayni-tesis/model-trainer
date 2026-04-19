# Coffee Disease Detection (Detector + Clasificador)

Pipeline de dos etapas para hojas de cafe:

1. Detector de hojas con TensorFlow/Keras (`leaf_detector.keras`).
2. Clasificador de enfermedad con transfer learning (`best_disease_classifier.keras`).

Clases de salida:

- `miner`
- `nodisease`
- `phoma`
- `redspider`
- `rust`

## 1) Regla del proyecto: una sola version de Python

Este proyecto usa una unica version de Python:

- Python `3.10`

No se deben crear entornos con 3.11, 3.12 o 3.13 para este repositorio.

## 2) Estructura real del proyecto

```
model-trainer/
├── config.py
├── dataset.py
├── disease_classifier.py
├── evaluate.py
├── leaf_detector.py
├── pipeline.py
├── prepare_detector_dataset.py
├── train_classifier.py
├── train_detector.py
├── requirements.txt
├── requirements-dml.txt
├── pyproject.toml
├── dataset/
│   ├── train/              # clasificador (subcarpetas por clase)
│   ├── test/               # clasificador (subcarpetas por clase)
│   ├── images/train/       # detector
│   ├── images/val/         # detector
│   ├── labels/train/       # detector (YOLO txt)
│   ├── labels/val/         # detector (YOLO txt)
│   └── leaf_detection.yaml
└── saved_models/
```

## 3) Instalacion con uv (unico entorno)

### 3.1 Instalar uv

```powershell
winget install --id=astral-sh.uv -e
uv --version
```

### 3.2 Crear entorno unico Python 3.10

```powershell
uv python install 3.10
uv venv .venv --python 3.10
uv sync
```

`uv` gestiona el entorno automaticamente. No necesitas activar `.venv` ni guardar variables como `$PY`.

Si prefieres instalación manual (equivalente), este comando también es válido:

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

## 4) Verificar acelerador (GPU/DirectML)

```powershell
uv run --python 3.10 python -c "import tensorflow as tf; print('TF', tf.__version__); print('GPU', tf.config.list_physical_devices('GPU')); print('DML', tf.config.list_physical_devices('DML'))"
```

Nota: con este stack, DirectML puede mostrarse como `GPU`.

## 5) Ejecutar scripts del proyecto

Forma recomendada (sin activar entorno ni definir variables):

```powershell
uv run --python 3.10 <script.py> [args]
```

Ejemplo rápido:

```powershell
uv run --python 3.10 train_detector.py --help
```

## 6) Entrenamiento del detector (hoja)

Script: `train_detector.py`

### 6.1 Formato de dataset del detector

- YAML: `dataset/leaf_detection.yaml`
- Imagenes: `dataset/images/train`, `dataset/images/val`
- Labels YOLO: `dataset/labels/train`, `dataset/labels/val`

Cada `.txt` debe tener lineas tipo:

```
class_id cx cy w h
```

Valores normalizados entre 0 y 1.

### 6.2 Entrenar detector

CPU:

```powershell
uv run --python 3.10 train_detector.py --data dataset/leaf_detection.yaml --epochs 50 --batch 16 --imgsz 640 --model small --device cpu
```

GPU/DirectML (recomendado):

```powershell
uv run --python 3.10 train_detector.py --data dataset/leaf_detection.yaml --epochs 50 --batch 16 --imgsz 640 --model small --device auto --strict-device
```

Opciones principales del detector:

- `--model`: `tiny | small | medium`
- `--device`: `auto | gpu | dml | cpu`
- `--strict-device`: aborta si no hay acelerador cuando se solicita GPU/DML

## 7) Entrenamiento del clasificador de enfermedad

Script: `train_classifier.py`

Dataset esperado (por carpetas de clase):

- `dataset/train/miner`, `dataset/train/nodisease`, ...
- `dataset/test/miner`, `dataset/test/nodisease`, ...

Entrenamiento base:

```powershell
uv run --python 3.10 train_classifier.py --arch efficientnetb0 --epochs 30 --ft-epochs 15 --aug moderate
```

Opciones principales del clasificador:

- `--arch`: `efficientnetb0 | mobilenetv2 | resnet50`
- `--epochs`: etapa 1 (backbone congelado)
- `--ft-epochs`: etapa 2 (fine tuning)
- `--aug`: `light | moderate | strong`
- `--no-weights`: desactiva class weights

## 8) Uso de modelos en inferencia

Script principal: `pipeline.py`

### 8.1 Desde CLI

```powershell
uv run --python 3.10 pipeline.py dataset/test/rust/1120.jpg
```

### 8.2 Desde Python

```python
from pipeline import CoffeeDiseaseDetectionPipeline

pipe = CoffeeDiseaseDetectionPipeline()
result = pipe.run("dataset/test/rust/1120.jpg")
pipe.visualize(result, save_path="resultado.jpg", show=False)

print(result["summary"])
```

## 9) Rutas de modelos generados

Definidas en `config.py`:

- Detector final: `saved_models/leaf_detector.keras`
- Clasificador final: `saved_models/disease_classifier.keras`
- Mejor clasificador (checkpoint): `saved_models/best_disease_classifier.keras`

## 10) Comandos recomendados (tu flujo)

Ejecución recomendada y directa con `uv run` (sin variables y sin activar `.venv`):

```powershell
uv run --python 3.10 train_classifier.py --arch efficientnetb0 --epochs 5 --ft-epochs 2
uv run --python 3.10 pipeline.py dataset/test/rust/1120.jpg
uv run --python 3.10 train_detector.py --data dataset/leaf_detection.yaml --epochs 50 --batch 16 --imgsz 640 --model small --device auto --strict-device
```

## 11) Troubleshooting rapido

### `--strict-device` falla

Significa que TensorFlow no detecto acelerador en ese entorno.

```powershell
uv run --python 3.10 python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU')); print(tf.config.list_physical_devices('DML'))"
```

### Entrena pero muy lento

- Revisa que no este cayendo a CPU.
- Prueba `--imgsz 320` o `--imgsz 160` para smoke tests.
- Reduce `--batch` si hay poca memoria.

---

Si modificas dependencias del proyecto:

```powershell
uv lock
uv sync
```
