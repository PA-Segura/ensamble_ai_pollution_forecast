# Descripcion de Archivos NetCDF del Pronostico WRF

Documentacion de la estructura de los archivos NetCDF de salida del pronostico WRF del grupo de Interaccion Oceano Atmosfera (IOA) ICAyCC-UNAM en su ultima version, que se utilizan como insumo para el modelo de Machine Learning de pronostico de contaminacion atmosferica.

---

## 1. Archivo WRF Original (wrfout_d02)

**Archivo de referencia analizado:**

```
/ServerData/WRF_2017_Kraken/2019/04_abril/wrfout_d02_2019-04-10_00.nc
Tamano en disco: 17 GB
```

**Patron de nombres (formato nuevo, anos > 2018):**

```
wrfout_d02_YYYY-MM-DD_00.nc
```

Ubicacion: `{input_folder}/{year}/{MM_mes}/wrfout_d02_YYYY-MM-DD_00.nc`
Ejemplo: `.../2019/04_abril/wrfout_d02_2019-04-10_00.nc`

> Nota: Existe tambien el dominio d01 (padre, menor resolucion), pero el pipeline ML utiliza exclusivamente el dominio d02 (anidado, mayor resolucion).
> Ver patron de busqueda de archivos en `io_netcdf/inout.py:64`:
> `name_pattern = 'wrfout_d02_..._00.nc'`, consistente con los scripts operativos
> (`operativo_files/process_wrf_files_like_in_train.py:119`, `operativo001.py:250`).

### 1.1 Metadata del modelo

| Atributo | Valor |
|---|---|
| Version WRF | V3.9 |
| Tipo de simulacion | REAL-DATA CASE |
| Proyeccion cartografica | Mercator (MAP_PROJ=3) |
| Resolucion horizontal | DX = DY = 5000 m (5 km) |
| Paso de tiempo (DT) | 10 s |
| Dominio | d02 (GRID_ID=2, anidado) |
| Dominio padre | d01 (PARENT_ID=1) |
| Razon de anidamiento | PARENT_GRID_RATIO = 3 |
| Centro del dominio | lat 18.13, lon -99.62 |
| Categorias de uso de suelo | USGS, 24 categorias |

**Esquemas de fisica:**

| Parametrizacion | Opcion | Esquema |
|---|---|---|
| Microfisica (MP_PHYSICS) | 3 | WSM 3-class |
| Radiacion LW (RA_LW_PHYSICS) | 1 | RRTM |
| Radiacion SW (RA_SW_PHYSICS) | 1 | Dudhia |
| Capa superficial (SF_SFCLAY_PHYSICS) | 1 | MM5 similarity |
| Superficie terrestre (SF_SURFACE_PHYSICS) | 2 | Noah LSM |
| Capa limite planetaria (BL_PBL_PHYSICS) | 1 | YSU |
| Cumulus (CU_PHYSICS) | 1 | Kain-Fritsch |

### 1.2 Dimensiones

| Dimension | Tamano | Descripcion |
|---|---|---|
| Time | 121 | Pasos temporales (5 dias horarios, hora 0 a hora 120) |
| south_north | 156 | Puntos de rejilla en latitud |
| west_east | 273 | Puntos de rejilla en longitud |
| bottom_top | 49 | Niveles verticales (half/mass levels) |
| bottom_top_stag | 50 | Niveles verticales (full/w levels) |
| soil_layers_stag | 4 | Capas de suelo |
| south_north_stag | 157 | Puntos staggered en latitud |
| west_east_stag | 274 | Puntos staggered en longitud |

### 1.3 Cobertura espacial

| Coordenada | Min | Max |
|---|---|---|
| XLAT | 14.57 | 21.63 |
| XLONG | -106.14 | -93.10 |

Cobertura temporal del archivo de ejemplo: `2019-04-10_00:00:00` a `2019-04-15_00:00:00` (121 horas).

### 1.4 Resumen de variables

El archivo original contiene **163 variables** en total:
- **86** variables de superficie 2D (Time, south_north, west_east)
- **12** variables de volumen 3D (Time, bottom_top, south_north, west_east)
- Resto: coordenadas, factores de escala, constantes verticales, etc.

### 1.5 Variables extraidas para el pipeline ML

De las 163 variables disponibles, el script `1_MakeNetcdf_From_WRF.py` extrae las siguientes:

| Variable | Dimensiones | Descripcion | Unidades |
|---|---|---|---|
| T2 | (Time, south_north, west_east) | Temperatura a 2 metros | K |
| U10 | (Time, south_north, west_east) | Componente U del viento a 10 metros | m s-1 |
| V10 | (Time, south_north, west_east) | Componente V del viento a 10 metros | m s-1 |
| RAINC | (Time, south_north, west_east) | Precipitacion acumulada convectiva (cumulus) | mm |
| RAINNC | (Time, south_north, west_east) | Precipitacion acumulada de escala de rejilla | mm |
| SWDOWN | (Time, south_north, west_east) | Flujo de onda corta descendente en superficie | W m-2 |
| GLW | (Time, south_north, west_east) | Flujo de onda larga descendente en superficie | W m-2 |

Adicionalmente se leen como insumo para calculos derivados (no se incluyen en la salida final):

| Variable | Descripcion | Unidades | Uso |
|---|---|---|---|
| Q2 | Razon de mezcla de vapor a 2 m | kg kg-1 | Calculo de humedad relativa |
| PSFC | Presion en superficie | Pa | Calculo de humedad relativa |

---

## 2. Procesamiento: de WRF original a NetCDF para ML

**Script:** `1_MakeNetcdf_From_WRF.py`
**Modulos auxiliares:**
- `proj_preproc/wrf.py` — funciones `crop_variables_xr()`, `crop_variable_np()`, `calculate_relative_humidity_metpy()`
- `io_netcdf/inout.py` — funcion `read_wrf_files_names()` (formato nuevo, anos > 2018)

### 2.1 Flujo de transformacion

```
wrfout_d02_YYYY-MM-DD_00.nc (17 GB, 273x156, 121 horas, ~163 vars)
        |
        |  1. Seleccion de variables (T2, U10, V10, RAINC, RAINNC, SWDOWN, GLW, RH)
        |
        |  2. Variables derivadas:
        |     - RAIN = RAINC + RAINNC, luego descumulada (diferencia entre pasos)
        |     - WS10 = sqrt(U10^2 + V10^2)
        |     - RH = relative_humidity_from_mixing_ratio(PSFC, T2, Q2)  [via MetPy]
        |
        |  3. Crop espacial al bbox del Valle de Mexico
        |     bbox = [18.75, 20, -99.75, -98.5]
        |
        |  4. Seleccion temporal: solo las primeras 24 horas (range(24))
        |
        |  5. Interpolacion lineal a rejilla regular
        |     Resolucion: 1/20 grado = 0.05 grados (~5.5 km)
        |     lat: arange(18.75, 20, 0.05) -> 25 puntos
        |     lon: arange(-99.75, -98.5, 0.05) -> 25 puntos
        |
        |  6. Ajuste temporal: UTC -> UTC-6 (hora local CDMX)
        |     Referencia: "hours since YYYY-MM-(DD-1) 18:00:00"
        |
        v
  YYYY-MM-DD.nc (959 KB, 25x25, 24 horas, 8 vars)
```

### 2.2 Detalle de variables derivadas

**RAIN** (precipitacion total no acumulada):
- Se suman RAINC (convectiva) y RAINNC (escala de rejilla)
- Se descumula calculando la diferencia entre pasos temporales consecutivos
- Valores negativos se fuerzan a 0

**WS10** (velocidad del viento a 10 m):
- Magnitud del vector de viento: `sqrt(U10^2 + V10^2)`
- Se conservan tambien U10 y V10 individuales en la salida

**RH** (humedad relativa):
- Calculada con `metpy.calc.relative_humidity_from_mixing_ratio(PSFC, T2, Q2)`
- Resultado en porcentaje (0-100)
- Las variables de entrada Q2 y PSFC no se incluyen en la salida final

---

## 3. Archivo Procesado (YYYY-MM-DD.nc)

**Archivo de referencia analizado:**

```
/home/pedro/netcdfs/2019-04-10.nc
Tamano en disco: 959 KB
```

**Patron de nombres:** `YYYY-MM-DD.nc`

### 3.1 Dimensiones

| Dimension | Tamano |
|---|---|
| time | 24 |
| lat | 25 |
| lon | 25 |

### 3.2 Coordenadas

| Coordenada | Min | Max | Resolucion | Unidades | Atributos CF |
|---|---|---|---|---|---|
| time | 2019-04-09 18:00 (UTC) | 2019-04-10 17:00 (UTC) | 1 hora | hours since YYYY-MM-(DD-1) 18:00:00 | axis=T, calendar=standard |
| lat | 18.75 | 19.95 | 0.05 grados | degrees_north | axis=Y, standard_name=latitude |
| lon | -99.75 | -98.55 | 0.05 grados | degrees_east | axis=X, standard_name=longitude |

> La referencia temporal se ajusta a UTC-6 (hora local CDMX). El primer paso temporal (hora 0 UTC del dia del pronostico) corresponde a las 18:00 UTC del dia anterior.

### 3.3 Variables

| Variable | Dimensiones | Descripcion | Unidades | Rango ejemplo (2019-04-10) |
|---|---|---|---|---|
| T2 | (time, lat, lon) | Temperatura a 2 m | K | 275.70 - 303.52 |
| U10 | (time, lat, lon) | Componente U viento a 10 m | m s-1 | -4.97 - 7.15 |
| V10 | (time, lat, lon) | Componente V viento a 10 m | m s-1 | -6.61 - 8.46 |
| SWDOWN | (time, lat, lon) | Radiacion de onda corta descendente | W m-2 | 0.00 - 1113.45 |
| GLW | (time, lat, lon) | Radiacion de onda larga descendente | W m-2 | 0.00 - 360.67 |
| RH | (time, lat, lon) | Humedad relativa (derivada) | % | 6.39 - 85.59 |
| RAIN | (time, lat, lon) | Precipitacion total no acumulada (derivada) | mm | 0.00 - 0.00 |
| WS10 | (time, lat, lon) | Velocidad del viento a 10 m (derivada) | m s-1 | 0.12 - 9.10 |

Todas las variables tienen dtype `float64` y `_FillValue = NaN`.

### 3.4 Salida de ncdump -h

```
netcdf 2019-04-10 {
dimensions:
    time = 24 ;
    lat = 25 ;
    lon = 25 ;
variables:
    int64 time(time) ;
        time:units = "hours since 2019-04-09 18:00:00" ;
        time:calendar = "standard" ;
        time:axis = "T" ;
        time:long_name = "time" ;
        time:standard_name = "time" ;
    double T2(time, lat, lon) ;
        T2:_FillValue = NaN ;
    double U10(time, lat, lon) ;
        U10:_FillValue = NaN ;
    double V10(time, lat, lon) ;
        V10:_FillValue = NaN ;
    double SWDOWN(time, lat, lon) ;
        SWDOWN:_FillValue = NaN ;
    double GLW(time, lat, lon) ;
        GLW:_FillValue = NaN ;
    double RH(time, lat, lon) ;
        RH:_FillValue = NaN ;
    double RAIN(time, lat, lon) ;
        RAIN:_FillValue = NaN ;
    double WS10(time, lat, lon) ;
        WS10:_FillValue = NaN ;
    double lat(lat) ;
        lat:_FillValue = NaN ;
        lat:units = "degrees_north" ;
        lat:axis = "Y" ;
        lat:long_name = "latitude" ;
        lat:standard_name = "latitude" ;
    double lon(lon) ;
        lon:_FillValue = NaN ;
        lon:units = "degrees_east" ;
        lon:axis = "X" ;
        lon:long_name = "longitude" ;
        lon:standard_name = "longitude" ;
}
```

---

## 4. Resumen Comparativo

| Caracteristica | Original (wrfout_d02) | Procesado (YYYY-MM-DD.nc) |
|---|---|---|
| Tamano en disco | ~17 GB | ~959 KB |
| Rejilla horizontal | 273 x 156 (curvilinea, 5 km) | 25 x 25 (regular, 0.05 grados) |
| Pasos temporales | 121 (5 dias) | 24 (1 dia) |
| Niveles verticales | 49 | Ninguno (solo superficie) |
| Total de variables | 163 | 8 |
| Cobertura lat | 14.57 a 21.63 | 18.75 a 19.95 |
| Cobertura lon | -106.14 a -93.10 | -99.75 a -98.55 |
| Region | Centro-sur de Mexico | Valle de Mexico / ZMVM |
| Proyeccion | Mercator (curvilinea) | Lat-lon regular (CF) |
| Referencia temporal | UTC | UTC-6 (hora local CDMX) |

La reduccion es de aproximadamente **18,000x** en tamano de archivo, enfocando los datos exclusivamente en la region y variables relevantes para el pronostico de calidad del aire en la Zona Metropolitana del Valle de Mexico.

---

## 5. Referencia publicada: configuracion WRF

Segun la Seccion 2.1 del articulo publicado:

> *Atmospheric Environment* (2024).
> [https://www.sciencedirect.com/science/article/pii/S1352231024006927](https://www.sciencedirect.com/science/article/pii/S1352231024006927)

### 5.1 WRF para reananalisis (entrenamiento, validacion y test)

El sistema utiliza datos meteorologicos pronosticados provenientes de un reananalisis regional WRF-ARW durante las fases de entrenamiento, validacion y prueba. La configuracion del modelo WRF se basa en la implementacion descrita en Lopez-Espinoza et al. (2012), con validacion reportada en Meza Carreto (2018).

Las salidas del reananalisis se generan con resolucion temporal de 3 h y resolucion espacial de 10 km. Posteriormente se interpolan para obtener resolucion temporal horaria.

**Tabla: Configuracion del modelo WRF de reananalisis (publicada)**

| Parametro | Descripcion |
|---|---|
| Version WRF | V3.9 |
| Configuracion de dominio | Dominio unico |
| Resolucion espacial | 10 km |
| Dimensiones de rejilla | 617 (west-east) x 348 (south-north) |
| Niveles verticales | 49 (desde superficie hasta 50 hPa) |
| Capa limite planetaria | Yonsei University (YSU) |
| Microfisica | Thompson (MP_PHYSICS = 3) |
| Radiacion onda corta | Dudhia (RA_SW_PHYSICS = 1) |
| Capa superficial | Monin-Obukhov (SF_SFCLAY_PHYSICS = 1) |
| Superficie terrestre | Noah LSM (SF_SURFACE_PHYSICS = 4) |
| Resolucion temporal de salida | Cada 3 h |
| Condiciones iniciales y de frontera | NCEP CFSR (2010-2011) y CFSv2 (2011-2024) |
| Variables de salida WRF | U, V, W, T, QVAPOR, QCLOUD, CLDFRA, PH |
| Proyeccion | Mercator |
| Uso de suelo | INEGI con clasificacion USGS |

### 5.2 WRF operativo (pronostico)

Para el sistema meteorologico operativo se emplea tambien WRF-ARW con **dos dominios**: uno cubriendo la region central de Mexico y otro el pais completo. El sistema produce pronosticos horarios para los siguientes 5 dias y se ejecuta diariamente en un cluster con 110 cores Intel Xeon E5-2670. El dominio nacional abarca latitudes 4.126 N a 38.426 N y longitudes 123.361 W a 74.876 W.



