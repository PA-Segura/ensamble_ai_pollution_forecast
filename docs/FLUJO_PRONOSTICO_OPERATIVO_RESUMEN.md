# Flujo de Trabajo del Pronóstico Operativo - Resumen

Este documento describe el flujo resumido del sistema de pronóstico operativo de contaminación del aire.

---

## Resumen del Flujo
En su versión actual se usa un script ejecutor horario, este proceso puede integrarse de forma alternativa con otro tipo de ejecución agendada como cron jobs. 

```
run_forecast_hourly.py (Ejecutor Principal)
    │
    ├─> [Cada Hora] Ejecuta pronóstico
    │   └─> 7_operativo.py
    │
    └─> [Diario 5:30 AM] Limpieza de archivos .nc
        └─> clean_nc_files()
```

---

## Pasos Principales

### 1. Ejecutor Principal: `run_forecast_hourly.py`
- **Función:** Ejecuta pronósticos cada hora y programa limpieza diaria
- **Frecuencia:** Cada 1 hora 
- **Proceso de limpieza de netCDFs disponibles para pronóstico:** Diario a las 5:30 AM
- **Scripts llamados:** `7_operativo.py`

### 2. Sistema Principal: `7_operativo.py`
- **Función:** Orquesta todo el flujo de pronóstico
- **Pasos principales:**
  1. Configuración del sistema (obtiene última hora disponible)
  2. Gestión del modelo (carga modelo preentrenado)
  3. Inicialización del sistema de pronóstico
  4. Ejecución del pronóstico
  5. Guardado de pronósticos en base de datos (PostgreSQL/SQLite)

### 3. Sistema de Pronóstico: `forecast_utils2.py` → `ForecastSystem.run_forecast()`
- **Función:** Prepara y ejecuta datos inferencia de pronóstico
- **Pasos principales:**
  1. Procesamiento de archivos WRF en netCDF
  2. Carga de datos meteorológicos
  3. Obtención de datos de contaminación
  4. Imputación de datos faltantes
  5. Normalización de datos
  6. Alineación temporal
  7. Preparación de tensores
  8. Inferencia autoregresiva
  9. Desnormalización de resultados
  10. Generación de visualizaciones

### 4. Procesamiento WRF: `operativo_files/process_wrf_files_like_in_train.py`
- **Función:** Procesa archivos WRF usando mismas rutinas que en entrenamiento
- **Salida:** Archivos .nc procesados en folder temporal `/dev/shm/tem_ram_forecast/` # En este caso se usa un folder temporal montado en memoria RAM para mayor eficiencia en procesados.

---

## Scripts en Orden de Ejecución

### Nivel 1: Ejecutor Principal
1. **`run_forecast_hourly.py`** - Ejecuta cada hora, programa limpieza diaria

### Nivel 2: Sistema Principal
2. **`7_operativo.py`** - Orquesta todo el flujo de inferencia de pronóstico

### Nivel 3: Módulos de Utilidades
3. **`operativo_files/forecast_utils2.py`**
   - `ForecastSystem.run_forecast()` - Flujo principal de inferencia 
   - `WRFProcessor.process_wrf_files()` - Procesa archivos WRF
   - `WRFDataLoader.load_data()` - Carga datos meteorológicos
   - `PollutionDataManager.get_contaminant_data()` - Obtiene datos de contaminación
   - `ModelInference.run_autoregressive_inference()` - Ejecuta inferencia
   - `ResultsProcessor.denormalize_predictions()` - Desnormaliza resultados

### Nivel 4: Scripts de Procesamiento
4. **`operativo_files/process_wrf_files_like_in_train.py`** - Procesa archivos WRF, genera archivos .nc

### Nivel 5: Scripts de Entrenamiento (Reutilizados)
5. **`1_MakeNetcdf_From_WRF.py`** - `process_single_file()` 

### Nivel 6: Scripts de Base de Datos
6. **`operativo_files/postgres_query_helper.py`** - `get_ozone_target_datetime()` obtiene última hora disponible
7. **`operativo_files/save_predictions_postgres.py`** - `save_predictions_to_postgres()` guarda en PostgreSQL
8. **`operativo_files/save_predictions_sqlite.py`** - `save_predictions_to_sqlite()` guarda en SQLite local

---

## Flujo de Datos Simplificado

```
Archivos WRF Originales
    ↓
operativo_files/process_wrf_files_like_in_train.py
    ↓
Archivos .nc Procesados [/dev/shm/tem_ram_forecast/]
    ↓
WRFDataLoader.load_data() → xarray.Dataset (meteorología)
    ↓
PostgreSQL/SQLite → Datos de contaminación históricos
    ↓
PollutionDataManager → DataFrame (66 columnas)
    ↓
ImputationManager → Rellena valores faltantes
    ↓
Normalización → Datos normalizados
    ↓
Alineación Temporal → Datos alineados en UTC-6
    ↓
Preparación de Tensores → torch.Tensor
    ↓
ModelInference → Predicciones (24 horas)
    ↓
Desnormalización → Predicciones en unidades reales
    ↓
Guardado → CSV, PostgreSQL, SQLite
```

---

## Configuración Clave

### Parámetros del Modelo (desde config JSON):
- `prev_pollutant_hours`: Horas previas de contaminación requeridas
- `prev_weather_hours`: Horas previas de meteorología requeridas
- `next_weather_hours`: Horas futuras de meteorología requeridas
- `auto_regresive_steps`: Pasos autorregresivos (horas a pronosticar)

### Rutas Importantes:
- WRF procesados: `/dev/shm/tem_ram_forecast/`
- Output: `./tem_var/` (configurable)
- Modelo: `{all_models_path}/{model_path}/model_best.pth`

---

## Referencias

- Ver `README_parallel_training.md` para información sobre entrenamiento
- Ver `operativo_files/README_sistema_profesional.md` para documentación del sistema

