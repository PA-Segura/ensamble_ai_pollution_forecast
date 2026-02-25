# Referencia del Modelo Operativo de Pronóstico de Contaminación

Documento de referencia rápida para el modelo de deep learning usado en el sistema operativo de pronóstico de contaminación del aire (ZMVM).

---

## Identificación del Modelo

| Campo | Valor |
|---|---|
| **Nombre del modelo** | `Parallel_all_prev24_heads4_w4_p4_ar8_bootstrapTrue_thresh2_weather4_2` |
| **Arquitectura** | `MultiStreamTransformerModel` |
| **Archivo de checkpoint** | `model_best.pth` |
| **Ruta del checkpoint** | `/ZION/AirPollutionData/pedro_files/models/Parallel_all_prev24_heads4_w4_p4_ar8_bootstrapTrue_thresh2_weather4_2/model_best.pth` |
| **Archivo de configuración** | `operativo_files/test_Parallel_all_prev24_heads4_w4_p4_ar8_bootstrapTrue_thresh2_weather4_2_0701_101128.json` |
| **Archivo de normalización** | `/home/pedro/train_tmp/norm_params_2010_to_2020.yml` |
| **Archivo de nombres de columnas** | `operativo_files/column_names_Parallel_all_prev24_heads4_w4_p4_ar8_bootstrapTrue_thresh2_weather4_2.yml` |
| **Función de pérdida** | `asymmetric_weighted_mse_loss` |
| **Métrica** | `rmse_metric` |

---

## Hiperparámetros de la Arquitectura

| Parámetro | Valor | Descripción |
|---|---|---|
| `weather_time_dims` | 7 | Pasos temporales de meteorología por ventana (prev_weather + next_weather + 1) |
| `prev_pollutant_hours` | 24 | Horas previas de contaminación usadas como entrada |
| `weather_fields` | 8 | Número de variables meteorológicas (campos WRF) |
| `input_features` | 66 | Total de features de entrada del stream de contaminación |
| `weather_embedding_size` | 128 | Dimensión del embedding meteorológico |
| `pollution_embedding_size` | 64 | Dimensión del embedding de contaminación |
| `attention_heads` | 4 | Cabezas de atención en los transformers |
| `lat_size` | 25 | Tamaño de la grilla en latitud |
| `lon_size` | 25 | Tamaño de la grilla en longitud |
| `dropout` | 0.1 | Tasa de dropout |
| `weather_transformer_blocks` | 4 | Bloques transformer para meteorología |
| `pollution_transformer_blocks` | 4 | Bloques transformer para contaminación |

**Tamaño de salida del modelo**: `output_size = input_features - 12 = 54` (se restan las 12 variables cíclicas de tiempo).

---

## Vector de Entrada: 66 Columnas

El modelo recibe un vector de **66 columnas** por cada paso temporal. Estas se dividen en tres grupos:

### Grupo 1: Ozono Individual (30 columnas, índices 0-29)

Valores horarios de O₃ en 30 estaciones de la RAMA:

| # | Columna | Estación |
|---|---|---|
| 1 | `cont_otres_UIZ` | UIZ |
| 2 | `cont_otres_AJU` | AJU |
| 3 | `cont_otres_ATI` | ATI |
| 4 | `cont_otres_CUA` | CUA |
| 5 | `cont_otres_SFE` | SFE |
| 6 | `cont_otres_SAG` | SAG |
| 7 | `cont_otres_CUT` | CUT |
| 8 | `cont_otres_PED` | PED |
| 9 | `cont_otres_TAH` | TAH |
| 10 | `cont_otres_GAM` | GAM |
| 11 | `cont_otres_IZT` | IZT |
| 12 | `cont_otres_CCA` | CCA |
| 13 | `cont_otres_HGM` | HGM |
| 14 | `cont_otres_LPR` | LPR |
| 15 | `cont_otres_MGH` | MGH |
| 16 | `cont_otres_CAM` | CAM |
| 17 | `cont_otres_FAC` | FAC |
| 18 | `cont_otres_TLA` | TLA |
| 19 | `cont_otres_MER` | MER |
| 20 | `cont_otres_XAL` | XAL |
| 21 | `cont_otres_LLA` | LLA |
| 22 | `cont_otres_TLI` | TLI |
| 23 | `cont_otres_UAX` | UAX |
| 24 | `cont_otres_BJU` | BJU |
| 25 | `cont_otres_MPA` | MPA |
| 26 | `cont_otres_MON` | MON |
| 27 | `cont_otres_NEZ` | NEZ |
| 28 | `cont_otres_INN` | INN |
| 29 | `cont_otres_AJM` | AJM |
| 30 | `cont_otres_VIF` | VIF |

### Grupo 2: Variables Cíclicas de Tiempo (12 columnas, índices 30-41)

Codificación cíclica (sin/cos) de la posición temporal:

| # | Columna | Descripción |
|---|---|---|
| 31 | `half_sin_day` | Seno medio del día |
| 32 | `half_cos_day` | Coseno medio del día |
| 33 | `half_sin_week` | Seno medio de la semana |
| 34 | `half_cos_week` | Coseno medio de la semana |
| 35 | `half_sin_year` | Seno medio del año |
| 36 | `half_cos_year` | Coseno medio del año |
| 37 | `sin_day` | Seno del día |
| 38 | `cos_day` | Coseno del día |
| 39 | `sin_week` | Seno de la semana |
| 40 | `cos_week` | Coseno de la semana |
| 41 | `sin_year` | Seno del año |
| 42 | `cos_year` | Coseno del año |

### Grupo 3: Estadísticas de Otros Contaminantes (24 columnas, índices 42-65)

Media, mínimo y máximo espacial (sobre todas las estaciones) para 8 contaminantes:

| # | Columna | Contaminante | Estadística |
|---|---|---|---|
| 43 | `cont_co_mean` | CO | Media |
| 44 | `cont_co_min` | CO | Mínimo |
| 45 | `cont_co_max` | CO | Máximo |
| 46 | `cont_nodos_mean` | NO₂ | Media |
| 47 | `cont_nodos_min` | NO₂ | Mínimo |
| 48 | `cont_nodos_max` | NO₂ | Máximo |
| 49 | `cont_pmdiez_mean` | PM10 | Media |
| 50 | `cont_pmdiez_min` | PM10 | Mínimo |
| 51 | `cont_pmdiez_max` | PM10 | Máximo |
| 52 | `cont_pmdoscinco_mean` | PM2.5 | Media |
| 53 | `cont_pmdoscinco_min` | PM2.5 | Mínimo |
| 54 | `cont_pmdoscinco_max` | PM2.5 | Máximo |
| 55 | `cont_nox_mean` | NOx | Media |
| 56 | `cont_nox_min` | NOx | Mínimo |
| 57 | `cont_nox_max` | NOx | Máximo |
| 58 | `cont_no_mean` | NO | Media |
| 59 | `cont_no_min` | NO | Mínimo |
| 60 | `cont_no_max` | NO | Máximo |
| 61 | `cont_sodos_mean` | SO₂ | Media |
| 62 | `cont_sodos_min` | SO₂ | Mínimo |
| 63 | `cont_sodos_max` | SO₂ | Máximo |
| 64 | `cont_pmco_mean` | PMco | Media |
| 65 | `cont_pmco_min` | PMco | Mínimo |
| 66 | `cont_pmco_max` | PMco | Máximo |

---

## Vector de Salida: 54 Columnas

El modelo predice las **mismas 54 columnas de contaminantes** (sin las 12 de tiempo):
- 30 valores individuales de O₃ (las mismas estaciones del Grupo 1)
- 24 estadísticas de contaminantes (las mismas del Grupo 3)

---

## Entrada Meteorológica (Stream WRF)

El stream meteorológico recibe un tensor con forma `[batch, time, fields, lat, lon]`:

| Dimensión | Valor | Descripción |
|---|---|---|
| `time` | 7 | `prev_weather_hours(4) + next_weather_hours(2) + 1` |
| `fields` | 8 | Variables meteorológicas del WRF |
| `lat` | 25 | Puntos de grilla en latitud |
| `lon` | 25 | Puntos de grilla en longitud |

La ventana meteorológica se desliza junto con cada paso autorregresivo.

---

## Parámetros Temporales de Inferencia

| Parámetro | Entrenamiento | Operativo | Descripción |
|---|---|---|---|
| `prev_pollutant_hours` | 24 | 24 | Ventana histórica de contaminación |
| `prev_weather_hours` | 4 | 4 | Horas previas de meteorología |
| `next_weather_hours` | 2 | 2 | Horas futuras de meteorología |
| `auto_regresive_steps` | 8 | 24 | Pasos de pronóstico autorregresivo |
| `weather_window_size` | 7 | 7 | Total timesteps de weather por paso |

---

## Normalización

- **Archivo**: `/home/pedro/train_tmp/norm_params_2010_to_2020.yml`
- **Período de referencia**: Datos de 2010 a 2020
- **Tipo**: Normalización por z-score (media y desviación estándar) por tipo de contaminante
- **Aplicación**: Se normaliza tanto la entrada (contaminantes + meteorología) como se desnormaliza la salida
- **Función de normalización**: `proj_preproc.normalization.normalize_data`
- **Función de desnormalización**: `proj_preproc.normalization.denormalize_data`

---

## Resumen de la Inferencia Autorregresiva

La predicción de las 24 horas futuras se realiza de forma **autorregresiva** en la función `ModelInference.run_autoregressive_inference()` dentro de `forecast_utils2.py`. El proceso es el siguiente:

### Entradas iniciales

```
x_pollution: [1, 24, 66]  →  1 batch, 24 horas históricas, 66 features
x_weather:   [1, T, 8, 25, 25]  →  1 batch, T timesteps totales, 8 vars, grilla 25x25
```

Donde `T` es suficientemente largo para cubrir los 24 pasos autorregresivos (`prev_weather + next_weather + auto_regressive_steps`).

### Bucle autorregresivo (24 iteraciones)

```
Para step = 0, 1, 2, ..., 23:

  1. EXTRAER VENTANA METEOROLÓGICA
     weather_start = step
     weather_end   = step + 7          # weather_window_size
     current_weather = x_weather[:, weather_start:weather_end, :, :, :]
     → Forma: [1, 7, 8, 25, 25]

  2. PREDICCIÓN
     output = model(current_weather, current_pollution)
     → Forma: [1, 54]   # 54 contaminantes predichos para hora target+step

  3. GUARDAR PREDICCIÓN
     predictions[step] = output       # con datetime = target + step horas

  4. ACTUALIZAR VECTOR DE CONTAMINACIÓN (shift + insert)
     new_pollution[:, :-1, :] = current_pollution[:, 1:, :]   # shift izquierdo
     new_pollution[:, -1, :54] = output                        # insertar predicción
     current_pollution = new_pollution
     → La ventana de 24 horas se recorre: se descarta la hora más antigua
       y se agrega la predicción más reciente al final
```

### Diagrama de la ventana deslizante

```
Step 0:  [h-23, h-22, ..., h-1, h₀ ]  + weather[0:7]   → pred(h+1)
Step 1:  [h-22, h-21, ..., h₀,  p₁ ]  + weather[1:8]   → pred(h+2)
Step 2:  [h-21, h-20, ..., p₁,  p₂ ]  + weather[2:9]   → pred(h+3)
  ...
Step 23: [h-0,  p₁,  ..., p₂₂, p₂₃]  + weather[23:30] → pred(h+24)

Donde:
  h₀     = última observación real (target_datetime)
  pₙ     = predicción del modelo en el paso n
  h-k    = observación real k horas antes del target
```

Cada predicción `pₙ` se retroalimenta como entrada para el paso siguiente, lo que permite generar pronósticos de hasta 24 horas a futuro a partir de una sola ventana de observaciones reales.

---

## Entrenamiento: Referencia Rápida

| Parámetro | Valor |
|---|---|
| **Años de entrenamiento** | 2010-2022 |
| **Años de prueba** | 2023-2024 |
| **Épocas máximas** | 3000 |
| **Early stopping** | 10 épocas sin mejora |
| **Optimizador** | Adam (lr=0.0001, weight_decay=1e-5, amsgrad) |
| **Scheduler** | ReduceLROnPlateau (factor=0.1, patience=4) |
| **Batch size (train)** | 1024 |
| **Batch size (test)** | 128 |
| **Bootstrap** | Activado (repetición=20, threshold=2) |
| **GPUs** | 4 |

---

## Guía para Evaluaciones Independientes

Si se desea construir un vector de entrada para ejecutar una evaluación propia del modelo, se necesita:

1. **Obtener 24 horas consecutivas de datos de contaminación** con las 66 columnas (30 O₃ + 12 tiempo + 24 estadísticas) en el **orden exacto** descrito arriba.

2. **Normalizar** los datos usando el archivo `norm_params_2010_to_2020.yml` con la función `normalize_data`.

3. **Obtener datos meteorológicos WRF** procesados como grillas 25x25 para las 8 variables, cubriendo desde `target - 4h` hasta `target + 2h + 24h` (30 timesteps mínimo).

4. **Construir los tensores**:
   - `x_pollution`: `torch.Tensor` de forma `[1, 24, 66]`
   - `x_weather`: `torch.Tensor` de forma `[1, ≥30, 8, 25, 25]`

5. **Ejecutar el bucle autorregresivo** de 24 pasos como se describe arriba.

6. **Desnormalizar** la salida con `denormalize_data` para obtener concentraciones en unidades reales.

### Archivos necesarios

| Archivo | Propósito |
|---|---|
| `model_best.pth` | Pesos del modelo entrenado |
| `norm_params_2010_to_2020.yml` | Parámetros de normalización (media, std) |
| `column_names_*.yml` | Nombres y orden de columnas de entrada/salida |
| Config JSON | Hiperparámetros de arquitectura y ventanas temporales |
