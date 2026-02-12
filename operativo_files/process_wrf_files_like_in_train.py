#!/usr/bin/env python3
"""
process_wrf_files_like_in_train.py

Script para procesar archivos WRF usando la misma función process_single_file 
que se usó durante el entrenamiento, pero adaptado para procesar solo la ventana 
temporal necesaria para inferencia operativa.

Características principales:
1. Usa process_single_file() del entrenamiento para consistencia
2. Implementa modo forecast_based para usar pronósticos o datos históricos
3. Manejo inteligente de archivos existentes (opcional no reprocesar)
4. Limpieza automática de archivos no necesarios
5. Generación de 1 día extra para garantizar cobertura completa

Autor: Implementado basado en operativo001.py y snippet_correcto_d02.py
"""

# %%
# =============================================================================
# IMPORTS Y CONFIGURACIÓN INICIAL
# =============================================================================

import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
import argparse
import glob
from datetime import datetime, timedelta
import json
import importlib.util
from pathlib import Path



# Agregar path del proyecto si es necesario
project_path = os.path.dirname(os.path.abspath(__file__))
if project_path not in sys.path:
    sys.path.append(project_path)

# Imports del proyecto
from parse_config import ConfigParser
from conf.localConstants import wrfFileType

# %%
# =============================================================================
# CONFIGURACIÓN POR DEFECTO
# =============================================================================

DEFAULT_CONFIG = {
    'config_file': './operativo_files/test_Parallel_all_prev24_heads4_w4_p4_ar8_bootstrapTrue_thresh2_weather4_2_0701_101128.json', #'config22_zion.json',
    'target_datetime': '2023-05-05 09:00:00',  # Esta fecha está en CDMX
    'input_folder': '/ServerData/WRF_2017_Kraken/',
    'output_folder': '/dev/shm/tem_ram_forecast/',
    'bbox': [18.75, 20, -99.75, -98.5],
    'resolution': 1/20,
    'forecast_based': True,
    'skip_existing': False,
    'cleanup_unnecessary': True,
    'extra_days': 1,
    'generate_images': False,
    'variables': ['T2', 'U10', 'V10', 'RAINC', 'RAINNC', 'SWDOWN', 'GLW', 'RH']
}

# %%
# =============================================================================
# CONFIGURACIÓN DE ZONAS HORARIAS (REUTILIZADO DE operativo001.py)
# =============================================================================

TIMEZONE_CONFIG = {
    'UTC_OFFSET': -6,  # CDMX está en GMT-6 (UTC-6)
    'TIMEZONE_NAME': 'CDMX',
    'DESCRIPTION': 'Hora de Ciudad de México (GMT-6)'
}

# %%
# =============================================================================
# FUNCIONES AUXILIARES (REUTILIZADAS DE operativo001.py)
# =============================================================================

def get_month_folder_name(month):
    """Convierte número de mes a formato de carpeta (ej: 06_junio)"""
    month_names = {
        1: '01_enero', 2: '02_febrero', 3: '03_marzo', 4: '04_abril',
        5: '05_mayo', 6: '06_junio', 7: '07_julio', 8: '08_agosto',
        9: '09_septiembre', 10: '10_octubre', 11: '11_noviembre', 12: '12_diciembre'
    }
    return month_names[month]

def convert_cdmx_to_utc(cdmx_datetime):
    """Convierte una fecha/hora de CDMX a UTC."""
    utc_datetime = cdmx_datetime + timedelta(hours=abs(TIMEZONE_CONFIG['UTC_OFFSET']))
    return utc_datetime

def convert_utc_to_cdmx(utc_datetime):
    """Convierte una fecha/hora de UTC a CDMX."""
    cdmx_datetime = utc_datetime + timedelta(hours=TIMEZONE_CONFIG['UTC_OFFSET'])
    return cdmx_datetime

def get_wrf_file_path(target_date, input_folder):
    """Obtiene la ruta del archivo WRF para una fecha específica."""
    year_folder = str(target_date.year)
    month_folder = get_month_folder_name(target_date.month)
    
    # Usar d02 (alta resolución) como en el entrenamiento
    file_pattern = f"wrfout_d02_{target_date.strftime('%Y-%m-%d')}_00.nc"
    file_path = os.path.join(input_folder, year_folder, month_folder, file_pattern)
    
    return file_path

# %%
# =============================================================================
# CÁLCULO DE VENTANA METEOROLÓGICA (REUTILIZADO DE operativo001.py)
# =============================================================================

def calculate_weather_window(config):
    """Calcula la ventana meteorológica necesaria basada en la configuración del modelo."""
    prev_weather_hours = config['data_loader']['args']['prev_weather_hours']
    next_weather_hours = config['data_loader']['args']['next_weather_hours']
    auto_regresive_steps = config['test']['data_loader']['auto_regresive_steps']
    
    weather_window = prev_weather_hours + next_weather_hours + auto_regresive_steps + 1
    
    print(f"📊 CÁLCULO DE VENTANA METEOROLÓGICA:")
    print(f"   - prev_weather_hours: {prev_weather_hours}")
    print(f"   - next_weather_hours: {next_weather_hours}")
    print(f"   - auto_regresive_steps: {auto_regresive_steps}")
    print(f"   - weather_window total: {weather_window} horas")
    
    return weather_window

# %%
# =============================================================================
# NUEVA LÓGICA: DETERMINACIÓN DE ARCHIVOS WRF SEGÚN MODO
# =============================================================================

def get_required_wrf_files_forecast_based(target_datetime_cdmx, weather_window, extra_days=1):
    """
    Determina archivos WRF necesarios en modo forecast_based.
    
    En este modo:
    - Se usa el archivo de pronóstico del día objetivo (o anterior si no existe)
    - Se generan días adicionales para cubrir toda la ventana
    
    Args:
        target_datetime_cdmx: datetime objetivo en CDMX
        weather_window: número de horas necesarias
        extra_days: días extra para generar
    
    Returns:
        list: Lista de tuplas (date_utc, purpose) donde purpose puede ser 'forecast' o 'historical'
    """
    print(f"🔮 MODO FORECAST_BASED ACTIVADO")
    print("-" * 50)
    
    # Convertir a UTC
    target_datetime_utc = convert_cdmx_to_utc(target_datetime_cdmx)
    target_date_utc = target_datetime_utc.date()
    
    print(f"   Fecha objetivo (CDMX): {target_datetime_cdmx}")
    print(f"   Fecha objetivo (UTC): {target_datetime_utc}")
    
    # Calcular días necesarios basado en la ventana + días adicionales de margen
    base_days = int(np.ceil(weather_window / 24))
    required_days = base_days + extra_days + 2  # Margen normal
    print(f"   📊 Días calculados: base={base_days}, extra={extra_days}, margen=+2, total={required_days}")
    
    # El archivo principal es el del día objetivo (pronóstico)
    main_file_date = target_date_utc
    
    # NUEVA LÓGICA: Centrar alrededor del target con días futuros suficientes
    # Para inferencia autorregresiva necesitamos datos futuros
    days_before = required_days // 2  # La mitad antes
    days_after = required_days - days_before - 1  # El resto después (excluyendo día objetivo)
    
    print(f"   📊 Distribución: {days_before} días antes + día objetivo + {days_after} días después")
    
    # Lista de archivos requeridos
    required_files = []
    
    # Agregar días anteriores (historical)
    for i in range(days_before, 0, -1):
        historical_date = target_date_utc - timedelta(days=i)
        required_files.append((historical_date, 'historical'))
    
    # Agregar día objetivo (forecast principal)
    required_files.append((main_file_date, 'forecast'))
    
    # Agregar días posteriores (forecast adicionales)
    for i in range(1, days_after + 1):
        future_date = target_date_utc + timedelta(days=i)
        required_files.append((future_date, 'forecast'))
    
    # Ordenar por fecha
    required_files.sort(key=lambda x: x[0])
    
    print(f"📅 ARCHIVOS REQUERIDOS (FORECAST_BASED):")
    print(f"   - Días totales: {len(required_files)}")
    print(f"   - Archivo principal: {main_file_date} (forecast)")
    
    for date_utc, purpose in required_files:
        date_cdmx = convert_utc_to_cdmx(datetime.combine(date_utc, datetime.min.time())).date()
        print(f"   - {date_utc} UTC ({date_cdmx} CDMX): {purpose}")
    
    return required_files

def get_required_wrf_files_historical(target_datetime_cdmx, weather_window, extra_days=1):
    """
    Determina archivos WRF necesarios en modo histórico.
    
    En este modo:
    - Se usan archivos de días correspondientes sin pronósticos muy a futuro
    - Se centra alrededor del día objetivo
    
    Args:
        target_datetime_cdmx: datetime objetivo en CDMX
        weather_window: número de horas necesarias
        extra_days: días extra para generar
    
    Returns:
        list: Lista de tuplas (date_utc, purpose) todas marcadas como 'historical'
    """
    print(f"📊 MODO HISTÓRICO ACTIVADO")
    print("-" * 50)
    
    # Convertir a UTC
    target_datetime_utc = convert_cdmx_to_utc(target_datetime_cdmx)
    target_date_utc = target_datetime_utc.date()
    
    print(f"   Fecha objetivo (CDMX): {target_datetime_cdmx}")
    print(f"   Fecha objetivo (UTC): {target_datetime_utc}")
    
    # Calcular días necesarios + días adicionales de margen
    base_days = int(np.ceil(weather_window / 24))
    required_days = base_days + extra_days + 2  # Margen normal
    print(f"   📊 Días calculados: base={base_days}, extra={extra_days}, margen=+2, total={required_days}")
    
    # Centrar alrededor del día objetivo
    days_before = required_days // 2
    days_after = required_days - days_before - 1
    
    # Lista de archivos requeridos
    required_files = []
    
    # Agregar días anteriores
    for i in range(days_before, 0, -1):
        historical_date = target_date_utc - timedelta(days=i)
        required_files.append((historical_date, 'historical'))
    
    # Agregar día objetivo
    required_files.append((target_date_utc, 'historical'))
    
    # Agregar días posteriores
    for i in range(1, days_after + 1):
        future_date = target_date_utc + timedelta(days=i)
        required_files.append((future_date, 'historical'))
    
    # Ordenar por fecha
    required_files.sort(key=lambda x: x[0])
    
    print(f"📅 ARCHIVOS REQUERIDOS (HISTÓRICO):")
    print(f"   - Días totales: {len(required_files)}")
    print(f"   - Días antes: {days_before}")
    print(f"   - Días después: {days_after}")
    
    for date_utc, purpose in required_files:
        date_cdmx = convert_utc_to_cdmx(datetime.combine(date_utc, datetime.min.time())).date()
        print(f"   - {date_utc} UTC ({date_cdmx} CDMX): {purpose}")
    
    return required_files

# %%
# =============================================================================
# VERIFICACIÓN DE ARCHIVOS EXISTENTES
# =============================================================================

def check_existing_files(required_files, output_folder):
    """
    Verifica qué archivos ya existen en la carpeta de salida.
    Si dos archivos miden menos de 0.3MB, borra todos los archivos .nc y continúa.
    
    Args:
        required_files: Lista de tuplas (date_utc, purpose)
        output_folder: Carpeta de salida
    
    Returns:
        tuple: (existing_files, missing_files)
    """
    print(f"🔍 VERIFICANDO ARCHIVOS EXISTENTES")
    print("-" * 50)
    print(f"   Carpeta de salida: {output_folder}")
    
    existing_files = []
    missing_files = []
    small_files_count = 0  # Contador de archivos menores a 0.3MB
    
    for date_utc, purpose in required_files:
        date_cdmx = convert_utc_to_cdmx(datetime.combine(date_utc, datetime.min.time())).date()
        expected_output = os.path.join(output_folder, f"{date_cdmx.strftime('%Y-%m-%d')}.nc")
        
        if os.path.exists(expected_output):
            file_size = os.path.getsize(expected_output) / (1024 * 1024)  # MB
            existing_files.append((date_utc, purpose, expected_output, file_size))
            print(f"   ✅ {date_cdmx} CDMX: {file_size:.1f} MB")
            
            # Contar archivos pequeños
            if file_size < 0.3:
                small_files_count += 1
        else:
            missing_files.append((date_utc, purpose))
            print(f"   ❌ {date_cdmx} CDMX: No encontrado")
    
    # Verificar si hay dos o más archivos menores a 0.3MB
    if small_files_count >= 2:
        print(f"\n⚠️  DETECTADOS {small_files_count} ARCHIVOS MENORES A 0.3MB")
        print("🗑️  ELIMINANDO TODOS LOS ARCHIVOS .nc DEL DIRECTORIO...")
        
        # Buscar todos los archivos .nc en el directorio
        nc_files = glob.glob(os.path.join(output_folder, "*.nc"))
        deleted_count = 0
        
        for nc_file in nc_files:
            try:
                os.remove(nc_file)
                deleted_count += 1
                print(f"   🗑️  Eliminado: {os.path.basename(nc_file)}")
            except Exception as e:
                print(f"   ❌ Error eliminando {os.path.basename(nc_file)}: {e}")
        
        print(f"✅ Eliminados {deleted_count} archivos .nc")
        print("🔄 Continuando con el procesamiento...")
        
        # Resetear las listas ya que todos los archivos fueron eliminados
        existing_files = []
        missing_files = required_files.copy()
    
    print(f"\n📊 RESUMEN:")
    print(f"   - Archivos existentes: {len(existing_files)}")
    print(f"   - Archivos faltantes: {len(missing_files)}")
    
    return existing_files, missing_files

# %%
# =============================================================================
# IMPORTACIÓN DE process_single_file
# =============================================================================

def import_process_single_file():
    """Importa la función process_single_file del archivo original."""
    try:
        # Ruta al archivo original
        makenetcdf_path = os.path.join(project_path, "1_MakeNetcdf_From_WRF.py")
        
        if not os.path.exists(makenetcdf_path):
            raise FileNotFoundError(f"No se encontró 1_MakeNetcdf_From_WRF.py en {makenetcdf_path}")
        
        # Importar usando importlib
        spec = importlib.util.spec_from_file_location("makenetcdf_module", makenetcdf_path)
        makenetcdf_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(makenetcdf_module)
        
        # Obtener función
        process_single_file = makenetcdf_module.process_single_file
        
        print(f"✅ process_single_file importada desde: {makenetcdf_path}")
        return process_single_file
        
    except Exception as e:
        print(f"❌ Error importando process_single_file: {str(e)}")
        raise

# %%
# =============================================================================
# PROCESAMIENTO PRINCIPAL
# =============================================================================

def process_wrf_files_with_training_function(files_to_process, config, script_config):
    """
    Procesa archivos WRF usando la función process_single_file del entrenamiento.
    
    Args:
        files_to_process: Lista de tuplas (date_utc, purpose) a procesar
        config: Configuración del modelo
        script_config: Configuración del script
    
    Returns:
        tuple: (success_count, failed_files)
    """
    print(f"\n🔧 PROCESANDO ARCHIVOS WRF CON FUNCIÓN DEL ENTRENAMIENTO")
    print("-" * 60)
    
    # Importar función del entrenamiento
    process_single_file = import_process_single_file()
    
    # Crear directorios de salida
    output_folder = script_config['output_folder']
    output_folder_imgs = os.path.join(output_folder, 'imgs')
    
    os.makedirs(output_folder, exist_ok=True)
    if script_config['generate_images']:
        os.makedirs(output_folder_imgs, exist_ok=True)
    
    success_count = 0
    failed_files = []
    
    for i, (date_utc, purpose) in enumerate(files_to_process):
        date_cdmx = convert_utc_to_cdmx(datetime.combine(date_utc, datetime.min.time())).date()
        
        print(f"\n📅 PROCESANDO ARCHIVO {i+1}/{len(files_to_process)}:")
        print(f"   - Fecha UTC: {date_utc}")
        print(f"   - Fecha CDMX: {date_cdmx}")
        print(f"   - Propósito: {purpose}")
        
        # Obtener ruta del archivo WRF
        wrf_file_path = get_wrf_file_path(date_utc, script_config['input_folder'])
        
        # Verificar si existe el archivo
        if not os.path.exists(wrf_file_path):
            print(f"   ❌ Archivo WRF no encontrado: {os.path.basename(wrf_file_path)}")
            
            # Si es forecast_based y no existe el archivo principal, intentar día anterior
            if purpose == 'forecast' and i == 0:  # Es el archivo principal
                print(f"   🔄 Intentando archivo del día anterior...")
                fallback_date = date_utc - timedelta(days=1)
                fallback_path = get_wrf_file_path(fallback_date, script_config['input_folder'])
                
                if os.path.exists(fallback_path):
                    print(f"   ✅ Usando archivo de respaldo: {os.path.basename(fallback_path)}")
                    wrf_file_path = fallback_path
                    date_utc = fallback_date  # Actualizar fecha
                    date_cdmx = convert_utc_to_cdmx(datetime.combine(date_utc, datetime.min.time())).date()
                else:
                    print(f"   ❌ Archivo de respaldo tampoco encontrado")
                    failed_files.append((date_utc, purpose, "Archivo WRF no encontrado"))
                    continue
            else:
                failed_files.append((date_utc, purpose, "Archivo WRF no encontrado"))
                continue
        
        file_size = os.path.getsize(wrf_file_path) / (1024**3)  # GB
        print(f"   📁 Archivo WRF: {os.path.basename(wrf_file_path)} ({file_size:.1f} GB)")
        
        try:
            # Preparar argumentos para process_single_file
            args = (
                wrf_file_path,                           # file_path
                0,                                       # file_idx (siempre 0 para archivos individuales)
                wrfFileType.new,                         # mode (asumimos archivos nuevos)
                script_config['variables'],              # orig_variable_names
                output_folder,                           # output_folder
                output_folder_imgs,                      # output_folder_imgs
                script_config['bbox'],                   # bbox
                script_config['generate_images'],        # generate_images
                [datetime.combine(date_cdmx, datetime.min.time())],  # result_dates (lista con datetime)
                [None],                                 # result_files_coords (no se usa para archivos nuevos)
                script_config['resolution']             # resolution
            )
            
            print(f"   ⏳ Procesando con process_single_file...")
            
            # Ejecutar función del entrenamiento
            success, result_file = process_single_file(args)
            
            if success:
                print(f"   ✅ Procesamiento exitoso")
                
                # Verificar archivo generado
                expected_output = os.path.join(output_folder, f"{date_cdmx.strftime('%Y-%m-%d')}.nc")
                
                if os.path.exists(expected_output):
                    output_size = os.path.getsize(expected_output) / (1024 * 1024)  # MB
                    print(f"   📁 Archivo generado: {os.path.basename(expected_output)} ({output_size:.1f} MB)")
                    success_count += 1
                else:
                    print(f"   ❌ Archivo esperado no encontrado: {expected_output}")
                    failed_files.append((date_utc, purpose, "Archivo de salida no generado"))
            else:
                print(f"   ❌ Procesamiento falló")
                failed_files.append((date_utc, purpose, "Error en process_single_file"))
                
        except Exception as e:
            print(f"   ❌ Error procesando archivo: {str(e)}")
            failed_files.append((date_utc, purpose, f"Excepción: {str(e)}"))
    
    print(f"\n📊 RESUMEN DEL PROCESAMIENTO:")
    print(f"   - Archivos procesados exitosamente: {success_count}")
    print(f"   - Archivos con errores: {len(failed_files)}")
    
    if failed_files:
        print(f"\n❌ ARCHIVOS CON ERRORES:")
        for date_utc, purpose, error in failed_files:
            date_cdmx = convert_utc_to_cdmx(datetime.combine(date_utc, datetime.min.time())).date()
            print(f"   - {date_cdmx} CDMX ({purpose}): {error}")
    
    return success_count, failed_files

# %%
# =============================================================================
# LIMPIEZA DE ARCHIVOS NO NECESARIOS
# =============================================================================

def cleanup_unnecessary_files(required_files, output_folder):
    """
    Elimina archivos .nc que no son necesarios para la ventana actual.
    
    Args:
        required_files: Lista de tuplas (date_utc, purpose) necesarias
        output_folder: Carpeta de salida
    
    Returns:
        int: Número de archivos eliminados
    """
    print(f"\n🗑️  LIMPIEZA DE ARCHIVOS NO NECESARIOS")
    print("-" * 50)
    
    # Fechas necesarias (en formato CDMX para nombres de archivo)
    required_dates_cdmx = set()
    for date_utc, _ in required_files:
        date_cdmx = convert_utc_to_cdmx(datetime.combine(date_utc, datetime.min.time())).date()
        required_dates_cdmx.add(date_cdmx.strftime('%Y-%m-%d'))
    
    print(f"   Fechas necesarias: {sorted(required_dates_cdmx)}")
    
    # Buscar todos los archivos .nc en la carpeta
    nc_pattern = os.path.join(output_folder, "*.nc")
    existing_nc_files = glob.glob(nc_pattern)
    
    if not existing_nc_files:
        print(f"   ✅ No hay archivos .nc para revisar")
        return 0
    
    print(f"   📊 Archivos .nc encontrados: {len(existing_nc_files)}")
    
    deleted_count = 0
    errors = []
    
    for nc_file in existing_nc_files:
        filename = os.path.basename(nc_file)
        
        # Extraer fecha del nombre del archivo (formato: YYYY-MM-DD.nc)
        if filename.endswith('.nc') and len(filename) == 13:  # YYYY-MM-DD.nc = 13 caracteres
            file_date_str = filename[:-3]  # Remover .nc
            
            if file_date_str not in required_dates_cdmx:
                try:
                    file_size = os.path.getsize(nc_file) / (1024 * 1024)  # MB
                    print(f"   🗑️  Eliminando: {filename} ({file_size:.1f} MB)")
                    os.remove(nc_file)
                    deleted_count += 1
                except Exception as e:
                    error_msg = f"Error eliminando {filename}: {str(e)}"
                    errors.append(error_msg)
                    print(f"   ❌ {error_msg}")
            else:
                print(f"   ✅ Conservando: {filename}")
        else:
            print(f"   ⚠️  Archivo con formato inesperado: {filename}")
    
    print(f"\n📊 RESUMEN DE LIMPIEZA:")
    print(f"   - Archivos eliminados: {deleted_count}")
    print(f"   - Errores: {len(errors)}")
    
    if errors:
        print(f"\n❌ ERRORES EN LA LIMPIEZA:")
        for error in errors:
            print(f"   - {error}")
    
    return deleted_count

# %%
# =============================================================================
# PARSEO DE ARGUMENTOS
# =============================================================================

def parse_arguments():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Procesamiento de archivos WRF usando función del entrenamiento',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--target-datetime',
        type=str,
        help='Fecha objetivo en formato YYYY-MM-DD HH:MM:SS (hora CDMX)'
    )
    
    parser.add_argument(
        '--config-file',
        type=str,
        default=DEFAULT_CONFIG['config_file'],
        help='Archivo de configuración del modelo'
    )
    
    parser.add_argument(
        '--input-folder',
        type=str,
        default=DEFAULT_CONFIG['input_folder'],
        help='Carpeta con archivos WRF originales'
    )
    
    parser.add_argument(
        '--output-folder',
        type=str,
        default=DEFAULT_CONFIG['output_folder'],
        help='Carpeta de salida para archivos procesados'
    )
    
    parser.add_argument(
        '--bbox',
        type=float,
        nargs=4,
        default=DEFAULT_CONFIG['bbox'],
        metavar=('LAT_MIN', 'LAT_MAX', 'LON_MIN', 'LON_MAX'),
        help='Bounding box [lat_min, lat_max, lon_min, lon_max]'
    )
    
    parser.add_argument(
        '--resolution',
        type=float,
        default=DEFAULT_CONFIG['resolution'],
        help='Resolución en grados'
    )
    
    parser.add_argument(
        '--forecast-based',
        action='store_true',
        default=DEFAULT_CONFIG['forecast_based'],
        help='Usar modo forecast-based (pronósticos)'
    )
    
    parser.add_argument(
        '--no-forecast-based',
        action='store_false',
        dest='forecast_based',
        help='Usar modo histórico (no pronósticos)'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=DEFAULT_CONFIG['skip_existing'],
        help='No reprocesar archivos que ya existen'
    )
    
    parser.add_argument(
        '--no-cleanup',
        action='store_false',
        dest='cleanup_unnecessary',
        default=DEFAULT_CONFIG['cleanup_unnecessary'],
        help='No limpiar archivos no necesarios'
    )
    
    parser.add_argument(
        '--extra-days',
        type=int,
        default=DEFAULT_CONFIG['extra_days'],
        help='Días extra a generar para garantizar cobertura'
    )
    
    parser.add_argument(
        '--generate-images',
        action='store_true',
        default=DEFAULT_CONFIG['generate_images'],
        help='Generar imágenes de las variables'
    )
    
    return parser.parse_args()

def setup_configuration():
    """Configura los parámetros del script basado en argumentos y configuración por defecto."""
    args = parse_arguments()
    
    # Crear configuración final
    config = DEFAULT_CONFIG.copy()
    
    # Sobrescribir con argumentos de terminal si están disponibles
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    
    # Mostrar configuración final
    print("🔧 CONFIGURACIÓN DEL SCRIPT")
    print("-" * 50)
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    return config

# %%
# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """Función principal del script."""
    print("🚀 PROCESAMIENTO WRF CON FUNCIÓN DEL ENTRENAMIENTO")
    print("=" * 60)
    
    try:
        # 1. Configuración
        script_config = setup_configuration()
        
        # 2. Cargar configuración del modelo
        print("\n📋 CARGANDO CONFIGURACIÓN DEL MODELO")
        print("-" * 50)
        
        from utils import read_json
        config_dict = read_json(script_config['config_file'])
        config = ConfigParser(config_dict)
        
        print(f"✅ Configuración cargada desde: {script_config['config_file']}")
        
        # 3. Preparar fecha objetivo
        print("\n📅 PREPARANDO FECHA OBJETIVO")
        print("-" * 50)
        
        target_datetime_cdmx = datetime.strptime(script_config['target_datetime'], '%Y-%m-%d %H:%M:%S')
        target_datetime_utc = convert_cdmx_to_utc(target_datetime_cdmx)
        
        print(f"✅ Fecha objetivo (CDMX): {target_datetime_cdmx}")
        print(f"✅ Fecha objetivo (UTC): {target_datetime_utc}")
        
        # 4. Calcular ventana meteorológica
        print("\n📊 CALCULANDO VENTANA METEOROLÓGICA")
        print("-" * 50)
        
        weather_window = calculate_weather_window(config)
        
        # 5. Determinar archivos WRF necesarios según el modo
        print(f"\n🎯 DETERMINANDO ARCHIVOS WRF NECESARIOS")
        print("-" * 50)
        
        if script_config['forecast_based']:
            required_files = get_required_wrf_files_forecast_based(
                target_datetime_cdmx, weather_window, script_config['extra_days']
            )
        else:
            required_files = get_required_wrf_files_historical(
                target_datetime_cdmx, weather_window, script_config['extra_days']
            )
        
        # 6. Verificar archivos existentes
        existing_files, missing_files = check_existing_files(required_files, script_config['output_folder'])
        
        # 7. Determinar qué archivos procesar
        if script_config['skip_existing'] and existing_files:
            print(f"\n⏭️  MODO SKIP_EXISTING ACTIVADO")
            print("-" * 50)
            print(f"   Archivos existentes: {len(existing_files)}")
            print(f"   Archivos faltantes: {len(missing_files)}")
            
            files_to_process = missing_files
            
            if not files_to_process:
                print(f"✅ Todos los archivos ya existen. No hay nada que procesar.")
                
                # Limpieza si está habilitada
                if script_config['cleanup_unnecessary']:
                    cleanup_unnecessary_files(required_files, script_config['output_folder'])
                
                return 0
        else:
            files_to_process = required_files
        
        # 8. Limpiar archivos no necesarios (antes del procesamiento)
        if script_config['cleanup_unnecessary']:
            cleanup_unnecessary_files(required_files, script_config['output_folder'])
        
        # 9. Procesar archivos WRF
        if files_to_process:
            success_count, failed_files = process_wrf_files_with_training_function(
                files_to_process, config, script_config
            )
            
            # 10. Verificar resultado final
            print(f"\n🎉 PROCESAMIENTO COMPLETADO")
            print("-" * 50)
            
            final_existing, final_missing = check_existing_files(required_files, script_config['output_folder'])
            
            print(f"📊 RESULTADO FINAL:")
            print(f"   - Archivos requeridos: {len(required_files)}")
            print(f"   - Archivos disponibles: {len(final_existing)}")
            print(f"   - Archivos faltantes: {len(final_missing)}")
            print(f"   - Tasa de éxito: {(len(final_existing) / len(required_files)) * 100:.1f}%")
            
            if len(final_existing) == len(required_files):
                print(f"✅ TODOS LOS ARCHIVOS DISPONIBLES - LISTO PARA INFERENCIA")
                return 0
            else:
                print(f"⚠️  ALGUNOS ARCHIVOS FALTANTES - REVISAR ERRORES")
                return 1
        else:
            print(f"✅ No hay archivos para procesar")
            return 0
        
    except Exception as e:
        print(f"\n❌ ERROR GENERAL: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

# %%
# =============================================================================
# EJECUCIÓN DEL SCRIPT
# =============================================================================

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
else:
    print("📓 Modo notebook/import detectado")
    print("💡 Para ejecutar procesamiento completo, llama a main()")
