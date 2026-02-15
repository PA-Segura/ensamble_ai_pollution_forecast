# ensamble_ai_pollution_forecast

Sistema de pronóstico de contaminación atmosférica basado en un modelo **MultiStreamTransformerModel** (PyTorch). Tiene dos flujos principales: **entrenamiento** de modelos y **pronóstico operativo** horario.

## Estructura del proyecto

```
├── 1_MakeNetcdf_From_WRF.py        # Genera netCDFs diarios desde archivos WRF históricos
├── 2_MakeCSV_From_DB.py             # Extrae CSVs de contaminantes desde PostgreSQL
├── 4_train.py                       # Entrenamiento del modelo
├── 4b_parallel_training.py          # Entrenamiento paralelo (barrido de hiperparámetros)
├── 5_test.py                        # Evaluación del modelo entrenado
├── 5b_parallel_testing.py           # Evaluación paralela de múltiples modelos
├── 6a_dashboard_singlemodel.py      # Dashboard interactivo (Dash) de predicciones
├── 7_operativo.py                   # Pronóstico operativo (orquestador)
├── run_forecast_hourly.py           # Ejecutor horario del pronóstico operativo
├── config.json                      # Configuración principal (modelo, datos, entrenamiento)
├── MakeWRF_and_DB_CSV_UserConfiguration.py  # Configuración de preprocesamiento de datos
│
├── model/                # Definición del modelo (MultiStreamTransformerModel), loss, métricas
├── data_loader/          # DataLoaders, datasets y preprocesamiento de datos
├── trainer/              # Lógica de entrenamiento (loop de épocas, checkpoints)
├── base/                 # Clases base (BaseTrainer, BaseModel, BaseDataLoader)
├── conf/                 # Parsing de configuración JSON y constantes
├── AI/                   # Utilidades de entrenamiento, métricas, data augmentation
├── operativo_files/      # Módulos del operativo: forecast_utils, procesamiento WRF,
│                         #   imputación, guardado a PostgreSQL/SQLite
├── proj_preproc/         # Preprocesamiento: normalización, WRF, DB, visualización
├── proj_prediction/      # Predicción y métricas de evaluación
├── proj_io/              # Lectura/escritura de archivos
├── io_netcdf/            # I/O específico para netCDF
├── db/                   # Queries y conexiones a PostgreSQL
├── utils/                # Utilidades generales, plotting, naming
├── tests/                # Tests (pytest)
├── saved_confs/          # Configuraciones generadas para entrenamiento paralelo
├── docs/                 # Documentación detallada
└── deprecated/           # Código obsoleto
```

## Flujos de trabajo

### Entrenamiento

```
1_MakeNetcdf_From_WRF.py  →  Genera netCDFs diarios desde WRF
2_MakeCSV_From_DB.py       →  Genera CSVs de contaminantes desde PostgreSQL
4_train.py -c config.json  →  Entrena modelo (DataLoader carga netCDFs + CSVs)
5_test.py  -c config.json  →  Evalúa modelo, genera CSVs de predicciones
```

Para barrido de hiperparámetros: `4b_parallel_training.py` genera variantes de `config.json` y ejecuta entrenamientos en paralelo distribuyendo GPUs.

### Pronóstico operativo

```
run_forecast_hourly.py     →  Ejecutor periódico (cada hora + limpieza diaria)
  └─ 7_operativo.py        →  Orquesta el pronóstico:
       ├─ Procesa WRF actual (misma función que entrenamiento)
       ├─ Carga datos meteorológicos y de contaminación
       ├─ Imputa datos faltantes
       ├─ Ejecuta inferencia autoregresiva (24 horas)
       └─ Guarda resultados en PostgreSQL, SQLite y CSV
```

## Configuración

Todo se controla desde `config.json`:

| Sección | Qué define |
|---------|------------|
| `arch` | Arquitectura del modelo y sus parámetros (attention_heads, transformer_blocks, embeddings) |
| `data_loader` | Ruta a datos, contaminantes, ventanas temporales, batch_size, bootstrap |
| `trainer` | Épocas, early stopping, directorio de guardado, pasos autoregresivos |
| `optimizer` | Optimizador (Adam) y learning rate |
| `loss` | Función de pérdida (`asymmetric_weighted_mse_loss`) |
| `lr_scheduler` | Scheduler (`ReduceLROnPlateau`) |
| `test` | Paths del modelo entrenado, años de test, ruta de predicciones |

## Requisitos

- Python 3.7+
- PyTorch >= 1.12.0
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- pyyaml, tqdm
- PostgreSQL (para datos de contaminantes)
- GPU con CUDA (recomendado)

Instalación de dependencias:

```bash
pip install -r requirements.txt
```

## Uso rápido

```bash
# Entrenamiento
python 4_train.py -c config.json

# Evaluación
python 5_test.py -c config.json

# Entrenamiento paralelo (barrido de hiperparámetros)
python 4b_parallel_training.py --config config.json --max-parallel 4

# Pronóstico operativo
python run_forecast_hourly.py
```

## Documentación

Ver `docs/` para documentación detallada:

- `FLUJO_ENTRENAMIENTO_RESUMEN.md` — Flujo completo de entrenamiento
- `FLUJO_PRONOSTICO_OPERATIVO_RESUMEN.md` — Flujo completo del pronóstico operativo
- `README_parallel_training.md` — Entrenamiento paralelo con barrido de hiperparámetros
