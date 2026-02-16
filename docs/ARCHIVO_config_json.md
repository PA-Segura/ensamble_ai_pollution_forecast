# Configuración JSON del Modelo

El archivo JSON de configuración define experimentos de modelos de aprendizaje automático: arquitectura del modelo, datos, entrenamiento, optimización e inferencia. Es parseado por `ConfigParser` (`conf/parse_config.py`), que usa el mecanismo `init_obj` para instanciar dinámicamente clases y funciones desde los módulos del proyecto.

Ejemplo de referencia: `operativo_files/test_Parallel_all_prev24_heads4_w4_p4_ar8_bootstrapTrue_thresh2_weather4_2_0701_101128.json`

---

## Estructura

### `name`
Identificador de nombre del experimento. Nombre base de carpeta para checkpoints, logs y predicciones.

### `n_gpu`
Número de GPUs. Si > 1, el modelo se envuelve en `DataParallel` automáticamente.

### `arch`

Arquitectura del modelo. `ConfigParser.init_obj('arch', module_arch)` instancia la clase indicada en `type` con los argumentos de `args`.

| Campo | Descripción |
|---|---|
| `type` | Clase en `model/model.py`. Disponible: `MultiStreamTransformerModel` |
| `weather_time_dims` | Pasos temporales de meteorología: `prev_weather_hours + next_weather_hours + 1` |
| `prev_pollutant_hours` | Horas previas de concentración de contaminantes de entrada |
| `weather_fields` | Número de variables meteorológicas del WRF (T2, RH, U10, V10, etc.) |
| `input_features` | Ccomponente de columnas para input vector: contaminación por paso temporal (54 contaminantes + 12 variables cíclicas de tiempo = 66) |
| `weather_embedding_size` | Dimensión del embedding (codificación) por campo meteorológico (patch → embedding) |
| `pollution_embedding_size` | Dimensión del embedding (codificación) de contaminación |
| `attention_heads` | Cabezas de atención en los bloques Transformer |
| `lat_size`, `lon_size` | Dimensiones espaciales de la malla WRF |
| `weather_transformer_blocks` | Bloques Transformer apilados por campo meteorológico |
| `pollution_transformer_blocks` | Bloques Transformer apilados para la rama de contaminación |

### `data_loader`

Instancia `MLforecastDataLoader` (`data_loader/data_loaders.py`) que internamente crea `MLforecastDataset` (`data_loader/data_sets.py`).

| Campo | Descripción |
|---|---|
| `type` | Clase en `data_loader/data_loaders.py`. Disponible: `MLforecastDataLoader` |
| `data_folder` | Directorio raíz de datos. Contiene: archivos `.nc` preprocesados (uno por año), `PollutionCSV/` (CSVs de contaminantes por año) y `TrainingData/` (pickles generados durante entrenamiento) |
| `norm_params_file` | Archivo YAML con parámetros de normalización (media/std por variable) |
| `years` | Lista de años a cargar para entrenamiento |
| `pollutants_to_keep` | Contaminantes a incluir: `co`, `nodos`, `otres`, `pmdiez`, `pmdoscinco`, `nox`, `no`, `sodos`, `pmco` |
| `prev_pollutant_hours` | Ventana histórica de contaminación (debe coincidir con `arch.args`) |
| `prev_weather_hours` | Horas previas de meteorología |
| `next_weather_hours` | Horas futuras de meteorología |
| `auto_regresive_steps` | Pasos autorregresivos durante entrenamiento (se incrementa progresivamente) |
| `bootstrap_enabled` | Habilita sobremuestreo de episodios con valores altos de contaminación |
| `bootstrap_repetition` | Veces que se repiten los episodios seleccionados por bootstrap |
| `bootstrap_threshold` | Umbral (en desviaciones estándar) para seleccionar episodios de bootstrap |

### `trainer`

Controla la rutina de entrenamiento (`trainer/trainer.py`).

| Campo | Descripción |
|---|---|
| `epochs` | Épocas máximas |
| `save_dir` | Directorio raíz para checkpoints y logs |
| `monitor` | Criterio de early stopping, e.g. `"min val_loss"` |
| `early_stop` | Épocas sin mejora antes de detener |
| `auto_regresive_steps` | Pasos autorregresivos iniciales |
| `epochs_before_increase_auto_regresive_steps` | Épocas entre incrementos del horizonte autorregresivo |
| `tensorboard` | Habilita logging a TensorBoard |

### `optimizer`

Instanciado dinámicamente desde `torch.optim` via `init_obj`.

```json
"optimizer": {
    "type": "Adam",
    "args": { "lr": 0.0001, "weight_decay": 1e-05, "amsgrad": true }
}
```

`type` puede ser cualquier optimizador de `torch.optim`.

### `loss`

Nombre de función en `model/loss.py`. Disponibles:
- `asymmetric_weighted_mse_loss` -- penaliza sub-predicciones con peso 2x (usada en producción)
- `mse_loss`, `masked_mse_loss`, `rmse_loss`, `huber_loss`

### `metrics`

Lista de funciones en `model/metric.py`. Disponibles:
- `rmse_metric` (usada en producción)
- `mae`, `r2_score`, `masked_rmse_metric`, `asymmetric_weighted_mse_metric`

### `lr_scheduler`

Instanciado desde `torch.optim.lr_scheduler` via `init_obj`.

```json
"lr_scheduler": {
    "type": "ReduceLROnPlateau",
    "args": { "mode": "min", "factor": 0.1, "patience": 4 }
}
```

### `test`

Configuración para inferencia y evaluación (`5_test.py`, `7_operativo.py`).

| Campo | Descripción |
|---|---|
| `all_models_path` | Directorio raíz donde se almacenan los modelos entrenados |
| `model_path` | Subdirectorio del modelo específico (contiene `model_best.pth`) |
| `data_loader.years` | Años para evaluación (no deben solapar con entrenamiento) |
| `data_loader.auto_regresive_steps` | Horizonte completo de predicción (típicamente 24h) |
| `denormalization_file` | Archivo YAML para desnormalizar predicciones |
| `prediction_path` | Directorio de salida para predicciones |

### `analyze`

Configuración para scripts de análisis de resultados (`6a_dashboard_singlemodel.py`).

---
