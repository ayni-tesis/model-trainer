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

## 1) Estructura real del proyecto

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

## 2) Instalacion con uv (recomendada)

Este repositorio ya esta preparado para trabajar con `uv`.

### 2.1 Instalar uv

Si no lo tienes:

```powershell
winget install --id=astral-sh.uv -e
```

Verifica:

```powershell
uv --version
```

### 2.2 Entorno CPU (Python 3.13)

```powershell
uv python install 3.13
uv venv .venv-py313 --python 3.13
uv pip install --python .venv-py313\Scripts\python.exe -r requirements.txt
```

### 2.3 Entorno GPU Windows (DirectML, validado)

En Windows nativo, esta ruta es la validada para aceleracion:

- Python `3.10`
- `tensorflow-cpu==2.10.0`
- `tensorflow-directml-plugin`

```powershell
uv python install 3.10
uv venv .venv-py310-dml --python 3.10
uv pip install --python .venv-py310-dml\Scripts\python.exe -r requirements-dml.txt
```

Diagnostico rapido:

```powershell
.venv-py310-dml\Scripts\python.exe -c "import tensorflow as tf; print('TF', tf.__version__); print('GPU', tf.config.list_physical_devices('GPU')); print('DML', tf.config.list_physical_devices('DML'))"
```

Nota: en este stack, DirectML puede aparecer como dispositivo `GPU`.

## 3) Ejecutar scripts usando entornos creados por uv

Para evitar activar/desactivar entornos manualmente:

```powershell
$PY_CPU = ".venv-py313\Scripts\python.exe"
$PY_GPU = ".venv-py310-dml\Scripts\python.exe"
```

Luego ejecutas cualquier script con la variable adecuada.

## 4) Entrenamiento del detector (hoja)

Script: `train_detector.py`

### 4.1 Formato de dataset del detector

- YAML: `dataset/leaf_detection.yaml`
- Imagenes: `dataset/images/train`, `dataset/images/val`
- Labels YOLO: `dataset/labels/train`, `dataset/labels/val`

Cada `.txt` debe tener lineas tipo:

```
class_id cx cy w h
```

Valores normalizados entre 0 y 1.

### 4.2 Entrenar detector en CPU

```powershell
$PY_CPU train_detector.py --data dataset/leaf_detection.yaml --epochs 50 --batch 16 --imgsz 640 --model small --device cpu
```

### 4.3 Entrenar detector en GPU/DirectML

```powershell
$PY_GPU train_detector.py --data dataset/leaf_detection.yaml --epochs 50 --batch 16 --imgsz 640 --model small --device auto --strict-device
```

Opciones principales del detector:

- `--model`: `tiny | small | medium`
- `--device`: `auto | gpu | dml | cpu`
- `--strict-device`: aborta si no hay acelerador cuando se solicita GPU/DML

## 5) Entrenamiento del clasificador de enfermedad

Script: `train_classifier.py`

Dataset esperado (por carpetas de clase):

- `dataset/train/miner`, `dataset/train/nodisease`, ...
- `dataset/test/miner`, `dataset/test/nodisease`, ...

Entrenamiento base:

```powershell
$PY_CPU train_classifier.py --arch efficientnetb0 --epochs 30 --ft-epochs 15 --aug moderate
```

Opciones principales del clasificador:

- `--arch`: `efficientnetb0 | mobilenetv2 | resnet50`
- `--epochs`: etapa 1 (backbone congelado)
- `--ft-epochs`: etapa 2 (fine tuning)
- `--aug`: `light | moderate | strong`
- `--no-weights`: desactiva class weights

## 6) Uso de modelos en inferencia

Script principal: `pipeline.py`

### 6.1 Desde CLI

```powershell
$PY_CPU pipeline.py dataset/test/rust/1120.jpg
```

Esto crea salida visual en `pipeline_results/`.

### 6.2 Desde Python

```python
from pipeline import CoffeeDiseaseDetectionPipeline

pipe = CoffeeDiseaseDetectionPipeline()
result = pipe.run("dataset/test/rust/1120.jpg")
pipe.visualize(result, save_path="resultado.jpg", show=False)

print(result["summary"])
```

## 7) Rutas de modelos generados

Definidas en `config.py`:

- Detector final: `saved_models/leaf_detector.keras`
- Clasificador final: `saved_models/disease_classifier.keras`
- Mejor clasificador (checkpoint): `saved_models/best_disease_classifier.keras`

## 8) Ejecucion directa con uv run (opcional)

Para el flujo CPU base del proyecto tambien puedes usar:

```powershell
uv run --python 3.13 train_classifier.py --arch efficientnetb0 --epochs 5 --ft-epochs 2
uv run --python 3.13 pipeline.py dataset/test/rust/1120.jpg
```

Para GPU/DirectML en Windows, usa preferentemente el interprete del entorno `.venv-py310-dml` (seccion 3) porque depende de `requirements-dml.txt`.

## 9) Troubleshooting rapido

### `--strict-device` falla

Significa que TensorFlow no detecto acelerador en ese entorno.

Valida con:

```powershell
$PY_GPU -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU')); print(tf.config.list_physical_devices('DML'))"
```

### Entrena pero muy lento

- Revisa que no este cayendo a CPU.
- Prueba `--imgsz 320` o `--imgsz 160` para smoke tests.
- Reduce `--batch` si hay poca memoria.

---

Si actualizas versiones de dependencias del entorno CPU basado en `pyproject.toml`:

```powershell
uv lock
uv sync
```
