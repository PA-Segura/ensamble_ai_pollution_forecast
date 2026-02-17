# Imputacion de Datos Faltantes

Documentacion del proceso de imputacion de datos faltantes de contaminacion atmosferica utilizado tanto en el **entrenamiento** como en el **pronostico operativo** del modelo de ML.

---

## Resumen General

El sistema utiliza un pipeline de imputacion secuencial con multiples metodos ordenados por prioridad. Cada metodo intenta rellenar los valores que los metodos anteriores no pudieron resolver. Se mantienen **columnas de banderas** (`i_<columna>`) que registran el metodo utilizado para cada valor imputado.

```
Datos crudos (con NaN)
    |
    v
[Limpieza de calidad]  -->  negativos y outliers extremos -> NaN
    |
    v
[1. Promedio de fila]  -->  promedio de estaciones del mismo contaminante
    |
    v
[2. Persistencia]      -->  valor del dia anterior, misma hora
    |
    v
[3. Climatologia / Promedio historico]  -->  fallback final
    |
    v
Datos completos (sin NaN)
```

---

## Columnas de Banderas de Imputacion

Para cada columna de contaminante `cont_X`, se crea una columna de bandera `i_cont_X` con los siguientes valores posibles:

| Valor de bandera      | Significado                                           |
|---|---|
| `none`                | Valor observado (dato original, sin imputar)          |
| `row_avg`             | Imputado con promedio de fila (estaciones del mismo tipo) |
| `last_day_same_hour`  | Imputado con persistencia (dia anterior, misma hora)  |
| `climatology`         | Imputado con climatologia (solo entrenamiento)        |
| `historical_avg`      | Imputado con promedio historico de BD (solo operativo) |
| `-1`                  | Valor aun faltante (no se pudo imputar)               |

> Las columnas de banderas se eliminan antes de pasar los datos al modelo. Su proposito es diagnostico y trazabilidad.

---

## Imputacion en el Pronostico Operativo

**Archivos involucrados:**
- `operativo_files/imputation_manager.py` — Clase `ImputationManager` con toda la logica
- `operativo_files/forecast_utils2.py` — Invoca la imputacion en `ForecastSystem.run_forecast()`, paso 4

### Punto de invocacion en el pipeline

Dentro de `ForecastSystem.run_forecast()` en `forecast_utils2.py`, la imputacion se ejecuta en el **paso 4**, despues de obtener los datos de contaminacion de la BD y antes de la normalizacion:

```
run_forecast()
    |
    ├─> 1. Procesar archivos WRF
    ├─> 2. Cargar datos meteorologicos
    ├─> 3. Obtener datos de contaminacion (PostgreSQL o SQLite)
    ├─> 4. IMPUTAR DATOS FALTANTES  <--- aqui
    │       ├─> Verificar si hay NaN
    │       ├─> prepare_data_for_imputation()
    │       ├─> apply_imputation_pipeline()
    │       ├─> Generar resumen de imputacion
    │       └─> Eliminar columnas de banderas
    ├─> 5. Normalizar datos
    ├─> 6. Procesar datos de contaminacion (estadisticas, agregaciones)
    ├─> 7. Alinear datos temporalmente
    ├─> 8. Preparar tensores de entrada
    ├─> 9. Ejecutar inferencia autorregresiva
    └─> 10. Desnormalizar resultados
```

### Paso 0: Limpieza de calidad de datos

Antes de iniciar la imputacion propiamente dicha, `prepare_data_for_imputation()` invoca `clean_pollution_data_quality()`, que aplica filtros de calidad:

| Filtro | Criterio | Accion |
|---|---|---|
| Valores negativos | `valor < 0` | Se reemplaza por `NaN` |
| Outliers PM10 | `valor > 1000 ug/m3` | Se reemplaza por `NaN` |
| Outliers PM2.5 | `valor > 500 ug/m3` | Se reemplaza por `NaN` |
| Outliers O3 | `valor > 500 ppb` | Se reemplaza por `NaN` |
| Otros contaminantes | `valor > 1000` | Se reemplaza por `NaN` |

Los valores eliminados por este filtro quedan marcados como `-1` en las banderas y seran candidatos para imputacion.

### Paso 1: Promedio de fila (`row_avg`)

**Metodo:** Promedio de estaciones con observaciones validas del **mismo tipo de contaminante**.

**Implementacion (`ImputationManager.impute_with_row_avg`):**

1. Agrupa las columnas por tipo de contaminante (e.g., todas las `cont_otres_*` juntas, todas las `cont_nodos_*` juntas)
2. Para cada columna faltante, calcula el promedio de las **otras estaciones del mismo tipo** (excluyendo la columna actual para evitar sesgo)
3. Aplica filtro de outliers: solo usa valores positivos y por debajo del percentil 95
4. Requiere al menos **2 valores validos** del mismo tipo de contaminante para imputar

**Ejemplo:** Si `cont_otres_UIZ` tiene NaN a las 14:00, se calcula el promedio de las otras 29 estaciones de ozono que si tienen dato a esa hora.

### Paso 2: Persistencia temporal (`last_day_same_hour`)

**Metodo:** Valor del dia anterior a la misma hora.

**Implementacion (`ImputationManager.impute_with_persistence`):**

1. Para cada valor que sigue faltando (bandera `-1`), busca el valor del mismo contaminante 24 horas antes
2. Solo se aplica si el timestamp del dia anterior existe en el DataFrame
3. Asigna el valor encontrado y actualiza la bandera a `last_day_same_hour`

**Ejemplo:** Si `cont_otres_UIZ` sigue con NaN el 2025-03-15 14:00, se toma el valor de `cont_otres_UIZ` del 2025-03-14 14:00.

### Paso 3: Promedio historico de BD (`historical_avg`)

**Metodo:** Promedio de los ultimos 5 anos para la misma fecha y hora, consultando directamente la base de datos PostgreSQL.

**Implementacion (`ImputationManager.impute_with_historical_avg`):**

1. Solo se ejecuta si el `db_manager` esta disponible (conexion a PostgreSQL activa)
2. Agrupa los valores faltantes por (mes, dia, hora) para hacer queries eficientes
3. Para cada grupo, ejecuta una query SQL que calcula el promedio de los ultimos 5 anos
4. Requiere al menos **5 registros historicos** para considerar el promedio como valido
5. Solo usa valores positivos (`valor > 0`) y no nulos

**Query SQL usada:**
```sql
SELECT AVG(columna) as avg_value, COUNT(*) as data_count
FROM (
    SELECT columna
    FROM pollution_data 
    WHERE EXTRACT(MONTH FROM timestamp) = {mes}
      AND EXTRACT(DAY FROM timestamp) = {dia}
      AND EXTRACT(HOUR FROM timestamp) = {hora}
      AND timestamp >= CURRENT_DATE - INTERVAL '5 years'
      AND timestamp < CURRENT_DATE
      AND columna IS NOT NULL
      AND columna > 0
) subquery
```

> Si la BD no esta disponible, este paso se omite y los valores restantes permanecen con bandera `-1`.

### Manejo post-imputacion

Despues de la imputacion:
1. Se genera un **resumen de imputacion** con conteos por metodo y tasa de imputacion por columna
2. Se **eliminan las columnas de banderas** (`i_cont_*`) del DataFrame
3. En el paso 5 (`_process_pollution_data`), los NaN restantes (si los hay) se rellenan con **0** via `fillna(0)`

---

## Imputacion en el Entrenamiento

**Archivo:** `imputation_7_fixed.py`

La imputacion para entrenamiento se realiza **offline** sobre datos historicos (2010-2024) almacenados en archivos CSV. Los datos imputados se guardan en disco y se usan directamente para el entrenamiento del modelo.

### Flujo completo del script

```
imputation_7_fixed.py
    |
    ├─> 1. Cargar datos originales (CSV por ano: {year}_AllStations.csv)
    ├─> 2. Limpiar columnas WRF residuales
    ├─> 3. Crear indice temporal completo (2010-01-01 a 2024-12-31, frecuencia horaria)
    ├─> 4. Detectar grupos de contaminantes automaticamente
    ├─> 5. Para cada grupo de contaminante:
    │       ├─> Generar climatologia
    │       ├─> Imputar con promedio de fila (row_avg)
    │       ├─> Imputar con persistencia (last_day_same_hour)
    │       └─> Imputar con climatologia (climatology)
    ├─> 6. Actualizar DataFrame principal
    ├─> 7. Guardar datos imputados (completo + por ano)
    ├─> 8. Analisis de clustering por contaminante
    └─> 9. Exportar version limpia final
```

### Paso 1: Promedio de fila (`row_avg`)

Similar al operativo pero con diferencias:

- Requiere al menos **5 valores validos** en la fila (vs. 2 en operativo)
- Usa **todas las columnas de contaminantes** disponibles para el promedio de fila, no solo las del mismo tipo
- No aplica filtro de outliers en el calculo del promedio

### Paso 2: Persistencia temporal (`last_day_same_hour`)

Identico al operativo: valor del dia anterior a la misma hora.

### Paso 3: Climatologia (`climatology`)

**Metodo:** Valor climatologico calculado a partir del promedio historico por (mes, dia, hora), con suavizado temporal.

**Implementacion (`impute_with_climatology` y `generate_climatology`):**

1. **Generacion de climatologia:**
   - Agrupa todos los datos observados por (mes, dia, hora)
   - Calcula el promedio para cada combinacion
   - Aplica suavizado con ventana movil (`rolling window=3`, centrada)
   - Usa ano de referencia 2012 como indice
   - Maneja bordes (inicio/fin del ano) con promedio de 3 valores

2. **Aplicacion:**
   - Para cada valor faltante, busca el valor climatologico correspondiente al (mes, dia, hora) del timestamp
   - La climatologia se guarda en archivos CSV para referencia

### Archivos de salida del entrenamiento

| Archivo | Descripcion |
|---|---|
| `data_imputed_7fix_full.csv` | Datos completos imputados (2010-2024) |
| `data_imputed_7fix_{year}.csv` | Datos imputados por ano individual |
| `climatology_{contaminante}_7fix.csv` | Climatologia por tipo de contaminante |
| `clusters_{contaminante}_7fix.csv` | Resultados de clustering por contaminante |

---

## Diferencias entre Entrenamiento y Operativo

| Aspecto | Entrenamiento (`imputation_7_fixed.py`) | Operativo (`imputation_manager.py`) |
|---|---|---|
| **Fuente de datos** | Archivos CSV historicos | Base de datos en tiempo real (PostgreSQL/SQLite) |
| **Limpieza de calidad** | No incluida explicitamente | Si: negativos y outliers extremos -> NaN |
| **Row avg: umbral minimo** | >5 valores validos por fila | >2 valores validos del mismo tipo |
| **Row avg: agrupacion** | Todas las columnas de contaminantes | Solo columnas del mismo tipo de contaminante |
| **Row avg: filtro outliers** | No | Si: excluye valores > percentil 95 |
| **Metodo 3 (fallback final)** | Climatologia (precalculada y suavizada) | Promedio historico de BD (5 anos) |
| **Ejecucion** | Offline, una sola vez | En tiempo real, cada hora |
| **Datos procesados** | ~131,000 horas (2010-2024) | ~30-60 horas (ventana reciente) |
| **Columnas de banderas** | Se conservan en los archivos de salida | Se eliminan antes de pasar al modelo |

### Nota sobre la consistencia

Aunque los metodos tienen ligeras diferencias en implementacion, el **orden de prioridad** es el mismo en ambos contextos:

1. Promedio de fila (mas confiable, usa datos contemporaneos)
2. Persistencia temporal (segundo mas confiable, patron diario)
3. Fallback final (climatologia o promedio historico, menos especifico)

Esta jerarquia prioriza la informacion espacial contemporanea sobre la temporal, y la informacion temporal reciente sobre la estadistica de largo plazo.

---

## Flujo en el Modo SQLite (Contingencia)

Cuando el sistema opera en modo contingencia SQLite (`ForecastConfig.USE_SQLITE_CONTINGENCY = True`), la obtencion de datos cambia pero la **imputacion se ejecuta de forma identica**. La clase `SQLitePollutionDataManager` en `7_operativo.py` genera los datos de contaminacion con la misma estructura de columnas, y el `ImputationManager` los procesa sin distincion.

La unica diferencia es que en modo SQLite con 66 columnas (30 ozono individuales + 24 estadisticas + 12 tiempo), el paso de imputacion opera sobre las columnas de contaminantes ya presentes en el DataFrame.

---

## Procesamiento Post-Imputacion (Operativo)

Despues de la imputacion, en el paso 5 de `run_forecast()` (metodo `_process_pollution_data`), se realizan transformaciones adicionales:

1. **Relleno final:** `fillna(0)` para cualquier NaN restante
2. **Agregacion de estadisticas:** Para contaminantes distintos a ozono, se calculan estadisticas por fila (mean, min, max) de todas las estaciones
3. **Reduccion de columnas:** Se eliminan las columnas individuales de estaciones (excepto ozono) y se reemplazan por sus estadisticas
4. **Eliminacion de banderas residuales:** Se eliminan columnas `i_cont_*` si quedaron

**Resultado final para el modelo (66 columnas):**
- 30 columnas individuales de ozono (`cont_otres_{estacion}`)
- 24 columnas de estadisticas (8 contaminantes x 3 estadisticas: mean, min, max)
- 12 columnas de variables ciclicas de tiempo

---

## Clase ImputationManager: API

Ubicacion: `operativo_files/imputation_manager.py`

| Metodo | Descripcion |
|---|---|
| `prepare_data_for_imputation(df, cols)` | Limpia datos y crea columnas de banderas |
| `clean_pollution_data_quality(df, cols)` | Filtra negativos y outliers extremos |
| `impute_with_row_avg(df, cols)` | Imputacion por promedio de fila agrupado por tipo |
| `impute_with_persistence(df, cols)` | Imputacion por persistencia (dia anterior) |
| `impute_with_historical_avg(df, cols, years_back)` | Imputacion por promedio historico de BD |
| `apply_imputation_pipeline(df, cols, years_back)` | Pipeline completo en orden de prioridad |
| `check_for_missing_values(df, cols)` | Verifica si hay NaN en columnas |
| `get_missing_values_count(df, cols)` | Conteo de NaN por columna |
| `get_imputation_summary(df, cols)` | Resumen con conteos por metodo y tasas |

### Uso tipico (operativo)

```python
from operativo_files.imputation_manager import ImputationManager

imputation_mgr = ImputationManager(db_manager)

# Verificar si hay faltantes
if imputation_mgr.check_for_missing_values(pollution_data, pollutant_columns):
    # Preparar (limpieza + banderas)
    pollution_data = imputation_mgr.prepare_data_for_imputation(pollution_data, pollutant_columns)
    
    # Aplicar pipeline completo
    pollution_data = imputation_mgr.apply_imputation_pipeline(pollution_data, pollutant_columns, years_back=5)
    
    # Resumen
    summary = imputation_mgr.get_imputation_summary(pollution_data, pollutant_columns)
    
    # Limpiar banderas
    flag_columns = [col for col in pollution_data.columns if col.startswith('i_cont_')]
    pollution_data = pollution_data.drop(columns=flag_columns)
```

### Funciones de conveniencia (uso independiente)

```python
from operativo_files.imputation_manager import create_imputation_columns, apply_imputation_pipeline

# Crear banderas
df_with_flags = create_imputation_columns(df, pollutant_columns)

# Pipeline completo
df_imputed = apply_imputation_pipeline(df, pollutant_columns, db_manager=None, years_back=5)
```
