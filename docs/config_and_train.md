# Configuración y Entrenamiento

## Origen del framework

La estructura de este proyecto está basada en [ml_torch_templates (FSU)](https://github.com/fsu-sc/ml_torch_templates), que a su vez se basa en [pytorch-template (victoresque)](https://github.com/victoresque/pytorch-template).

El archivo de configuración principal es: un archivo `config.json` define el experimento (modelo, datos, optimizador, scheduler, loss, métricas), y el framework instancia cada componente dinámicamente vía `ConfigParser.init_obj()`. Esto permite cambiar hiperparámetros modificar gran parte de código.

## Estructura del `config.json`

Referencia: `operativo_files/test_Parallel_all_prev24_heads4_w4_p4_ar8_bootstrapTrue_thresh2_weather4_2_0701_101128.json`

### Secciones principales

```jsonc
{
    "name": "...",          // Nombre del experimento (define carpeta de guardado)
    "n_gpu": 4,             // GPUs a usar (DataParallel si > 1)
    "arch": { ... },        // Arquitectura del modelo
    "data_loader": { ... }, // Datos y DataLoader
    "trainer": { ... },     // Parámetros del loop de entrenamiento
    "optimizer": { ... },   // Optimizador (cualquier torch.optim.*)
    "loss": "...",          // Función de pérdida (nombre en model/loss.py)
    "metrics": [ ... ],     // Métricas (nombres en model/metric.py)
    "lr_scheduler": { ... },// Scheduler (cualquier torch.optim.lr_scheduler.*)
    "test": { ... },        // Configuración para evaluación (5_test.py)
    "analyze": { ... }      // Configuración para análisis
}
```

---

### `name`

```json
"name": "Parallel_all_prev24_heads4_w4_p4_ar8_bootstrapTrue_thresh2_weather4_2"
```

El nombre del experimento. Los checkpoints se guardan en `{save_dir}/models/{name}/{timestamp}/`.

---

### `arch` — Arquitectura del modelo

```json
"arch": {
    "type": "MultiStreamTransformerModel",
    "args": {
        "weather_time_dims": 7,
        "prev_pollutant_hours": 24,
        "weather_fields": 8,
        "input_features": 66,
        "weather_embedding_size": 128,
        "pollution_embedding_size": 64,
        "attention_heads": 4,
        "lat_size": 25,
        "lon_size": 25,
        "dropout": 0.1,
        "weather_transformer_blocks": 4,
        "pollution_transformer_blocks": 4
    }
}
```

| Parámetro | Descripción |
|---|---|
| `type` | Nombre de la clase definida en `model/model.py`. En `4_train.py` se instancia dinámicamente: `config.init_obj('arch', module_arch)` busca la clase con este nombre en el módulo y la crea con los `args` |
| `weather_time_dims` | Pasos temporales de la ventana meteorológica de entrada (`prev_weather_hours + next_weather_hours + 1`) |
| `prev_pollutant_hours` | Horas previas de contaminación como entrada. **Debe coincidir** con `data_loader.args.prev_pollutant_hours` |
| `weather_fields` | Número de campos meteorológicos (variables WRF: T2, RH, PSFC, WS10, RAIN, etc.) |
| `input_features` | Features de contaminación por paso temporal (contaminantes × estaciones + 12 features temporales) |
| `weather_embedding_size` | Dimensión del embedding para cada campo meteorológico (patch → embedding) |
| `pollution_embedding_size` | Dimensión del embedding para la serie de contaminación |
| `attention_heads` | Cabezas de atención en los TransformerEncoder |
| `lat_size`, `lon_size` | Tamaño de la grilla espacial de los datos WRF procesados |
| `dropout` | Dropout en transformers y decoder |
| `weather_transformer_blocks` | Bloques TransformerEncoder apilados por cada campo meteorológico |
| `pollution_transformer_blocks` | Bloques TransformerEncoder apilados para la rama de contaminación |

**Arquitectura del modelo (simplificada):**

```
weather_data (batch, time, fields, lat, lon)
    │
    ├─ Por cada campo meteorológico:
    │   PatchEmbedding → N × TransformerEncoder → concatenar tiempo
    │
    └─ Concatenar todos los campos → weather_combined

pollution_data (batch, time, features)
    │
    └─ Linear embedding + positional encoding → N × TransformerEncoder → concatenar tiempo
       → pollution_combined

[weather_combined | pollution_combined] → Decoder (3 capas Linear + BN + ReLU) → output
```

---

### `data_loader` — Datos

```json
"data_loader": {
    "type": "MLforecastDataLoader",
    "args": {
        "data_folder": "/home/pedro/netcdfs",
        "norm_params_file": "/home/pedro/train_tmp/norm_params_2010_to_2020.yml",
        "years": [2010, 2011, ..., 2022],
        "pollutants_to_keep": ["co", "nodos", "otres", "pmdiez", "pmdoscinco", "nox", "no", "sodos", "pmco"],
        "prev_pollutant_hours": 24,
        "prev_weather_hours": 4,
        "next_weather_hours": 2,
        "auto_regresive_steps": 8,
        "bootstrap_enabled": true,
        "bootstrap_repetition": 20,
        "bootstrap_threshold": 2,
        "batch_size": 1024,
        "shuffle": true,
        "validation_split": 0.1,
        "num_workers": 4
    }
}
```

| Parámetro | Descripción |
|---|---|
| `data_folder` | Carpeta que contiene los netCDFs diarios (`{YYYY-MM-DD}.nc`) y CSVs (`{contaminante}_{estacion}.csv`) generados por `1_MakeNetcdf_From_WRF.py` y `2_MakeCSV_From_DB.py` |
| `norm_params_file` | Archivo YAML con parámetros de normalización (mean, std). **Se genera automáticamente** si no existe al inicializar el DataLoader |
| `years` | Años de datos a cargar para entrenamiento |
| `pollutants_to_keep` | Contaminantes a incluir como features |
| `prev_pollutant_hours` | Horas previas de contaminación como entrada al modelo |
| `prev_weather_hours` | Horas previas de meteorología en la ventana |
| `next_weather_hours` | Horas futuras de meteorología en la ventana |
| `auto_regresive_steps` | Pasos autorregresivos durante entrenamiento (horizonte de pronóstico) |
| `bootstrap_enabled` | Si `true`, sobremuestrea ejemplos con valores altos de contaminación |
| `bootstrap_repetition` | Repeticiones para ejemplos con alta contaminación |
| `bootstrap_threshold` | Umbral (en desviaciones estándar) para definir "alta contaminación" |
| `batch_size` | Tamaño de lote. El Trainer lo reduce automáticamente conforme aumentan los pasos autorregresivos |
| `validation_split` | Fracción de datos para validación (0.1 = 10%) |

**Nota:** `weather_time_dims` en `arch` debe ser `prev_weather_hours + next_weather_hours + 1`.

---

### `trainer` — Entrenamiento

```json
"trainer": {
    "epochs": 3000,
    "save_dir": "/home/pedro/train_tmp/",
    "save_period": 1,
    "verbosity": 2,
    "monitor": "min val_loss",
    "early_stop": 10,
    "tensorboard": true,
    "log_dir": "saved/runs",
    "auto_regresive_steps": 8,
    "epochs_before_increase_auto_regresive_steps": 2
}
```

| Parámetro | Descripción |
|---|---|
| `epochs` | Máximo de épocas |
| `save_dir` | Directorio base. Checkpoints en `{save_dir}/models/{name}/{timestamp}/` |
| `save_period` | Guardar checkpoint cada N épocas |
| `verbosity` | 0=WARNING, 1=INFO, 2=DEBUG |
| `monitor` | Métrica a monitorear para guardar mejor modelo. Formato: `"min val_loss"` o `"max val_accuracy"` |
| `early_stop` | Épocas sin mejora antes de detener. 0 para desactivar |
| `tensorboard` | Activar logging a TensorBoard |
| `auto_regresive_steps` | Máximo de pasos autorregresivos del entrenamiento |
| `epochs_before_increase_auto_regresive_steps` | Cada N épocas se incrementa en 1 el número de pasos autorregresivos usados. Ejemplo: con valor 2, época 1-2 usa 1 paso, época 3-4 usa 2 pasos, etc. |

**Curriculum autoregresivo:** El Trainer incrementa progresivamente los pasos autorregresivos durante el entrenamiento y ajusta el batch_size proporcionalmente para mantener el uso de memoria estable.

---

### `optimizer`

```json
"optimizer": {
    "type": "Adam",
    "args": {
        "lr": 0.0001,
        "weight_decay": 1e-05,
        "amsgrad": true
    }
}
```

Acepta cualquier optimizador de `torch.optim`. Se instancia con `config.init_obj('optimizer', torch.optim, trainable_params)`.

---

### `loss` y `metrics`

```json
"loss": "asymmetric_weighted_mse_loss",
"metrics": ["rmse_metric"]
```

Funciones definidas en `model/loss.py` y `model/metric.py`. Las disponibles son:

**Losses** (`model/loss.py`):
- `asymmetric_weighted_mse_loss` — MSE asimétrico: penaliza subpredicciones ×2 vs sobrepredicciones ×1, con máscara
- `masked_mse_loss` — MSE con máscara de datos faltantes
- `mse_loss` — MSE estándar
- `rmse_loss`, `mae_loss`, `huber_loss`

**Métricas** (`model/metric.py`):
- `rmse_metric` — RMSE
- `masked_rmse_metric` — RMSE con máscara
- `asymmetric_weighted_mse_metric`
- `mae`, `r2_score`, `explained_variance`

---

### `lr_scheduler`

```json
"lr_scheduler": {
    "type": "ReduceLROnPlateau",
    "args": {
        "mode": "min",
        "factor": 0.1,
        "patience": 4,
        "threshold": 0.0001,
        "threshold_mode": "rel",
        "cooldown": 0,
        "min_lr": 1e-08,
        "eps": 1e-08
    }
}
```

Acepta cualquier scheduler de `torch.optim.lr_scheduler`. `ReduceLROnPlateau` se activa automáticamente con `val_loss`.

---

### `test` — Evaluación

```json
"test": {
    "all_models_path": "/ZION/AirPollutionData/pedro_files/models/",
    "model_path": "Parallel_all_prev24_heads4_w4_p4_ar8_bootstrapTrue_thresh2_weather4_2",
    "visualize_batch": false,
    "prediction_path": "/home/pedro/train_tmp/predictions/",
    "denormalization_file": "/home/pedro/train_tmp/norm_params_2010_to_2020.yml",
    "data_loader": {
        "years": [2023, 2024],
        "batch_size": 128,
        "shuffle": false,
        "validation_split": 0.0,
        "num_workers": 2,
        "auto_regresive_steps": 24
    }
}
```

| Parámetro | Descripción |
|---|---|
| `all_models_path` | Directorio base de modelos. Dentro busca `{model_path}/*/model_best.pth` |
| `model_path` | Nombre del experimento (subdirectorio dentro de `all_models_path`) |
| `prediction_path` | Donde guardar los CSVs de predicciones |
| `denormalization_file` | Archivo YAML de normalización (**debe ser el mismo** usado en entrenamiento) |
| `data_loader.years` | Años para evaluación (típicamente diferentes a entrenamiento) |
| `data_loader.auto_regresive_steps` | Pasos autorregresivos para test (en `4_train.py` se fuerza a 24) |

**Nota:** `4_train.py` fuerza `test.data_loader.auto_regresive_steps = 24` para que la validación durante entrenamiento siempre evalúe las 24 horas completas.

---

## Uso de `4_train.py`

```bash
# Entrenamiento básico
python 4_train.py -c config.json

# Reanudar desde checkpoint
python 4_train.py -c config.json -r path/to/checkpoint-epoch42.pth

# Seleccionar GPUs específicas
python 4_train.py -c config.json -d 0,1

# Override de hiperparámetros por CLI
python 4_train.py -c config.json --lr 0.001 --bs 512
```

### Opciones CLI

| Flag | Descripción |
|---|---|
| `-c`, `--config` | Ruta al archivo JSON de configuración (default: `config.json`) |
| `-r`, `--resume` | Ruta a un checkpoint para reanudar entrenamiento |
| `-d`, `--device` | Índices de GPUs (equivale a `CUDA_VISIBLE_DEVICES`) |
| `--lr`, `--learning_rate` | Override del learning rate |
| `--bs`, `--batch_size` | Override del batch size |

### Salidas del entrenamiento

```
{save_dir}/
├── models/{name}/{timestamp}/
│   ├── config.json              # Copia de la configuración usada
│   ├── model_best.pth           # Mejor modelo (según monitor)
│   └── checkpoint-epoch{N}.pth  # Checkpoints periódicos
└── logs/{name}/{timestamp}/
    └── (logs de TensorBoard)
```

### Monitoreo con TensorBoard

```bash
tensorboard --logdir {save_dir}/logs/
```

---

## Cómo crear una nueva configuración

1. Copiar un JSON existente como base
2. Modificar los campos según el experimento deseado
3. Asegurar consistencia entre `arch` y `data_loader`:
   - `arch.args.prev_pollutant_hours` == `data_loader.args.prev_pollutant_hours`
   - `arch.args.weather_time_dims` == `data_loader.args.prev_weather_hours + next_weather_hours + 1`
   - `arch.args.weather_fields` debe coincidir con el número de variables en los netCDFs
   - `arch.args.input_features` debe coincidir con el número de columnas de contaminación del dataset
4. Verificar que las rutas (`data_folder`, `norm_params_file`, `save_dir`) sean correctas para el entorno

### Ejemplo: cambiar horizonte de pronóstico

Para entrenar con 16 pasos autorregresivos en lugar de 8:

```jsonc
// data_loader.args:
"auto_regresive_steps": 16,

// trainer:
"auto_regresive_steps": 16,
"epochs_before_increase_auto_regresive_steps": 2,  // llega a 16 pasos en época 32
```

### Ejemplo: cambiar ventana meteorológica

Para usar 6 horas previas + 3 futuras (10 pasos totales):

```jsonc
// data_loader.args:
"prev_weather_hours": 6,
"next_weather_hours": 3,

// arch.args:
"weather_time_dims": 10,  // 6 + 3 + 1
```

---

## Referencias

- Template original: [victoresque/pytorch-template](https://github.com/victoresque/pytorch-template)
- Fork usado como base: [fsu-sc/ml_torch_templates](https://github.com/fsu-sc/ml_torch_templates)
- Flujo completo: `docs/FLUJO_ENTRENAMIENTO_RESUMEN.md`
- Entrenamiento paralelo: `docs/README_parallel_training.md`
