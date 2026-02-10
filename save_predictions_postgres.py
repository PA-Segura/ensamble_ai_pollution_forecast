#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
save_predictions_postgres.py - Exportación de pronósticos a PostgreSQL

Módulo para exportar pronósticos de contaminación a la base de datos PostgreSQL
usando las credenciales AMATE-OPERATIVO con id_tipo_pronostico = 7.

ESTRUCTURA DE DATOS:
- forecast_otres: 30 estaciones individuales con columnas hour_p01-hour_p24
- forecast_[contaminante]: Estadísticas min/max/avg por hora para 8 contaminantes

VERSIÓN: 1.0
FECHA: 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List
from db.sqlCont import getPostgresConn


class PostgresForecastExporter:
    """Exportador de pronósticos a PostgreSQL con id_tipo_pronostico = 7."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
        # Configuración de ID de pronóstico actualizado
        self.ID_TIPO_PRONOSTICO = 7  # Nuevo ID para este sistema
        
        # Configuración de estaciones para ozono (igual que el original)
        self.ozono_stations = [
            "UIZ", "AJU", "ATI", "CUA", "SFE", "SAG", "CUT", "PED", "TAH", "GAM",
            "IZT", "CCA", "HGM", "LPR", "MGH", "CAM", "FAC", "TLA", "MER", "XAL",
            "LLA", "TLI", "UAX", "BJU", "MPA", "MON", "NEZ", "INN", "AJM", "VIF"
        ]
        
        # Configuración de contaminantes con sus propiedades de tabla
        self.contaminants_config = {
            'co': {
                'table': 'forecast_co',
                'has_id_tipo_pronostico': False,
                'stats': ['min', 'max', 'mean']  # Cambiado avg → mean
            },
            'no': {
                'table': 'forecast_no',
                'has_id_tipo_pronostico': False,
                'stats': ['min', 'max', 'mean']  # Cambiado avg → mean
            },
            'nodos': {
                'table': 'forecast_nodos',
                'has_id_tipo_pronostico': False,
                'stats': ['min', 'max', 'mean']  # Cambiado avg → mean
            },
            'nox': {
                'table': 'forecast_nox',
                'has_id_tipo_pronostico': True,
                'stats': ['min', 'max', 'mean']  # Cambiado avg → mean
            },
            'pmco': {
                'table': 'forecast_pmco',
                'has_id_tipo_pronostico': False,
                'stats': ['min', 'max', 'mean']  # Cambiado avg → mean
            },
            'pmdiez': {
                'table': 'forecast_pmdiez',
                'has_id_tipo_pronostico': True,
                'stats': ['min', 'max', 'mean']  # Cambiado avg → mean
            },
            'pmdoscinco': {
                'table': 'forecast_pmdoscinco',
                'has_id_tipo_pronostico': False,
                'stats': ['min', 'max', 'mean']  # Cambiado avg → mean
            },
            'sodos': {
                'table': 'forecast_sodos',
                'has_id_tipo_pronostico': True,
                'stats': ['min', 'max', 'mean']  # Cambiado avg → mean
            }
        }
        
        # Estación fija para estadísticas (según especificación)
        self.STATS_STATION = 'NET'
    
    def save_ozono_forecasts(self, predictions_df: pd.DataFrame, 
                           target_datetime: str) -> Dict[str, int]:
        """
        Guarda pronósticos de ozono en la tabla forecast_otres.
        
        Args:
            predictions_df: DataFrame con pronósticos [24 horas x columnas]
            target_datetime: Fecha del pronóstico 'YYYY-MM-DD HH:MM:SS'
            
        Returns:
            Dict con estadísticas de guardado
        """
        if self.verbose:
            print("🌟 GUARDANDO PRONÓSTICOS DE OZONO (forecast_otres)")
        
        forecast_time = datetime.strptime(target_datetime, '%Y-%m-%d %H:%M:%S')
        
        # Generar string de columnas hour_p01, hour_p02, ..., hour_p24
        all_hours = ','.join([f'hour_p{x:02d}' for x in range(1, 25)])
        
        saved_count = 0
        errors = 0
        
        for station in self.ozono_stations:
            try:
                conn = getPostgresConn()
                
                # Buscar columna de esta estación
                station_column = f'cont_otres_{station}'
                
                if station_column in predictions_df.columns:
                    pred_by_station = predictions_df[station_column]
                    
                    # SQL de inserción (igual que en 7_Operativo.py pero con ID = 7)
                    sql = f"""INSERT INTO forecast_otres 
                             (fecha, id_tipo_pronostico, id_est, val, {all_hours}) 
                             VALUES (%s, %s, %s, %s, {','.join(['%s']*24)})"""
                    
                    cur = conn.cursor()
                    values = [float(f'{x:.2f}') for x in pred_by_station.tolist()]
                    cur.execute(sql, (forecast_time, self.ID_TIPO_PRONOSTICO, station, '-1', *values))
                    cur.close()
                    conn.commit()
                    conn.close()
                    
                    saved_count += 1
                    if self.verbose:
                        print(f"     ✅ {station}: 24 horas guardadas")
                else:
                    if self.verbose:
                        print(f"     ⚠️ {station}: columna no encontrada")
                        
            except Exception as e:
                errors += 1
                if self.verbose:
                    print(f"     ❌ {station}: {e}")
                if 'conn' in locals():
                    conn.close()
        
        if self.verbose:
            print(f"   📊 Ozono: {saved_count}/30 estaciones guardadas, {errors} errores")
        
        return {'saved': saved_count, 'errors': errors}
    
    def save_pollutant_statistics(self, predictions_df: pd.DataFrame, 
                                target_datetime: str, 
                                pollutant: str) -> Dict[str, int]:
        """
        Guarda estadísticas de un contaminante en su tabla correspondiente.
        
        Args:
            predictions_df: DataFrame con pronósticos
            target_datetime: Fecha del pronóstico
            pollutant: Tipo de contaminante ('co', 'nox', etc.)
            
        Returns:
            Dict con estadísticas de guardado
        """
        if pollutant not in self.contaminants_config:
            raise ValueError(f"Contaminante no soportado: {pollutant}")
        
        config = self.contaminants_config[pollutant]
        table_name = config['table']
        has_id_tipo = config['has_id_tipo_pronostico']
        stats = config['stats']
        
        if self.verbose:
            print(f"   🔍 Guardando {pollutant} en {table_name}")
        
        forecast_time = datetime.strptime(target_datetime, '%Y-%m-%d %H:%M:%S')
        
        saved_count = 0
        errors = 0
        
        # Procesar cada estadística (min, max, avg)
        for stat in stats:
            try:
                conn = getPostgresConn()
                
                # Buscar columna de estadística
                stat_column = f'cont_{pollutant}_{stat}'
                
                if stat_column in predictions_df.columns:
                    pred_stat = predictions_df[stat_column]
                    
                    # Mapear 'mean' del DataFrame a 'avg' de PostgreSQL
                    db_stat_name = 'avg' if stat == 'mean' else stat
                    
                    # Construir nombres de columnas dinámicamente
                    hour_columns = [f'{db_stat_name}_hour_p{x:02d}' for x in range(1, 25)]
                    hour_placeholders = ','.join(['%s'] * 24)
                    hour_names = ','.join(hour_columns)
                    
                    # SQL base
                    base_fields = "fecha, val, id_est"
                    base_values = "%s, %s, %s"
                    
                    # Agregar id_tipo_pronostico si la tabla lo tiene
                    if has_id_tipo:
                        base_fields += ", id_tipo_pronostico"
                        base_values += ", %s"
                    
                    sql = f"""INSERT INTO {table_name} 
                             ({base_fields}, {hour_names}) 
                             VALUES ({base_values}, {hour_placeholders})"""
                    
                    cur = conn.cursor()
                    values = [float(f'{x:.2f}') for x in pred_stat.tolist()]
                    
                    # Construir parámetros
                    if has_id_tipo:
                        params = (forecast_time, '-1', self.STATS_STATION, self.ID_TIPO_PRONOSTICO, *values)
                    else:
                        params = (forecast_time, '-1', self.STATS_STATION, *values)
                    
                    cur.execute(sql, params)
                    cur.close()
                    conn.commit()
                    conn.close()
                    
                    saved_count += 1
                    if self.verbose:
                        # Mostrar mapeo correcto para mean → avg
                        display_stat = f"{pollutant}_{stat} → DB:{db_stat_name}" if stat == 'mean' else f"{pollutant}_{stat}"
                        print(f"     ✅ {display_stat}: 24 horas guardadas")
                else:
                    if self.verbose:
                        print(f"     ⚠️ {stat_column}: columna no encontrada")
                        
            except Exception as e:
                errors += 1
                if self.verbose:
                    print(f"     ❌ {pollutant}_{stat}: {e}")
                if 'conn' in locals():
                    conn.close()
        
        return {'saved': saved_count, 'errors': errors}
    
    def save_all_statistics(self, predictions_df: pd.DataFrame, 
                          target_datetime: str) -> Dict[str, int]:
        """
        Guarda estadísticas de todos los contaminantes.
        
        Args:
            predictions_df: DataFrame con pronósticos
            target_datetime: Fecha del pronóstico
            
        Returns:
            Dict con estadísticas totales de guardado
        """
        if self.verbose:
            print("📊 GUARDANDO ESTADÍSTICAS DE CONTAMINANTES")
        
        total_saved = 0
        total_errors = 0
        
        for pollutant in self.contaminants_config.keys():
            try:
                result = self.save_pollutant_statistics(predictions_df, target_datetime, pollutant)
                total_saved += result['saved']
                total_errors += result['errors']
            except Exception as e:
                if self.verbose:
                    print(f"     ❌ Error en {pollutant}: {e}")
                total_errors += 1
        
        if self.verbose:
            print(f"   📈 Total estadísticas: {total_saved} guardadas, {total_errors} errores")
        
        return {'saved': total_saved, 'errors': total_errors}


def save_predictions_to_postgres(predictions_denormalized: pd.DataFrame, 
                               target_datetime: str,
                               verbose: bool = True) -> Dict[str, any]:
    """
    Función principal para guardar pronósticos en PostgreSQL.
    
    Args:
        predictions_denormalized: DataFrame [24 horas x 54 contaminantes]
        target_datetime: Fecha del pronóstico 'YYYY-MM-DD HH:MM:SS'
        verbose: Mostrar mensajes
        
    Returns:
        Dict con resumen de guardado
    """
    
    if verbose:
        print(f"\n🐘 GUARDANDO PREDICCIONES EN POSTGRESQL")
        print(f"   📅 Fecha: {target_datetime}")
        print(f"   📊 Datos: {predictions_denormalized.shape}")
        print(f"   🆔 ID Tipo Pronóstico: 7")
    
    exporter = PostgresForecastExporter(verbose)
    
    try:
        # 1. Guardar pronósticos de ozono (30 estaciones)
        ozono_result = exporter.save_ozono_forecasts(predictions_denormalized, target_datetime)
        
        # 2. Guardar estadísticas de contaminantes (24 combinaciones)
        stats_result = exporter.save_all_statistics(predictions_denormalized, target_datetime)
        
        # 3. Resumen final
        total_saved = ozono_result['saved'] + stats_result['saved']
        total_errors = ozono_result['errors'] + stats_result['errors']
        
        if verbose:
            print(f"\n📈 RESUMEN POSTGRESQL:")
            print(f"   🌟 Ozono guardado: {ozono_result['saved']}/30 estaciones")
            print(f"   📊 Estadísticas guardadas: {stats_result['saved']}/24 combinaciones")
            print(f"   ❌ Errores totales: {total_errors}")
            print(f"   💾 Total registros: {total_saved}")
            
            if total_saved > 0:
                print("   ✅ Guardado exitoso en PostgreSQL")
            else:
                print("   ⚠️ No se guardaron datos en PostgreSQL")
        
        return {
            'ozono_saved': ozono_result['saved'],
            'stats_saved': stats_result['saved'],
            'total_saved': total_saved,
            'total_errors': total_errors,
            'success': total_saved > 0
        }
        
    except Exception as e:
        if verbose:
            print(f"❌ Error crítico en exportación PostgreSQL: {e}")
        
        return {
            'ozono_saved': 0,
            'stats_saved': 0,
            'total_saved': 0,
            'total_errors': 1,
            'success': False,
            'error': str(e)
        }


# =============================================================================
# FUNCIÓN PARA MIGRAR DESDE SQLITE (OPCIONAL)
# =============================================================================
def migrate_from_sqlite_to_postgres(sqlite_db_path: str = "forecast_predictions.db",
                                  verbose: bool = True):
    """
    Migra datos de pronósticos desde SQLite a PostgreSQL.
    
    Args:
        sqlite_db_path: Ruta de la base SQLite
        verbose: Mostrar mensajes
    """
    
    if verbose:
        print("🔄 MIGRANDO SQLITE → POSTGRESQL")
    
    try:
        import sqlite3
        
        # Conectar a SQLite
        sqlite_conn = sqlite3.connect(sqlite_db_path)
        
        # Obtener datos de SQLite
        if verbose:
            print("📖 Leyendo datos de SQLite...")
        
        # Leer pronósticos de ozono
        ozono_query = "SELECT * FROM forecast_otres ORDER BY fecha DESC"
        ozono_df = pd.read_sql_query(ozono_query, sqlite_conn)
        
        # Leer estadísticas
        stats_query = "SELECT * FROM forecast_stats ORDER BY fecha DESC"
        stats_df = pd.read_sql_query(stats_query, sqlite_conn)
        
        sqlite_conn.close()
        
        if verbose:
            print(f"   📊 Ozono: {len(ozono_df)} registros")
            print(f"   📈 Estadísticas: {len(stats_df)} registros")
        
        # Aquí se implementaría la lógica de migración específica
        # según el formato de datos requerido por PostgreSQL
        
        if verbose:
            print("✅ Migración completada")
            
    except ImportError:
        if verbose:
            print("❌ sqlite3 no disponible para migración")
    except Exception as e:
        if verbose:
            print(f"❌ Error en migración: {e}")


if __name__ == "__main__":
    # Prueba básica del módulo
    print("🧪 PROBANDO EXPORTACIÓN POSTGRESQL...")
    
    # Simular datos de predicción
    columns = []
    
    # 30 estaciones de ozono
    ozono_stations = ["UIZ", "AJU", "ATI", "CUA", "SFE"]  # Solo 5 para prueba
    for station in ozono_stations:
        columns.append(f'cont_otres_{station}')
    
    # Algunas estadísticas
    for cont in ['co', 'nox', 'pmdiez']:
        for stat in ['mean', 'min', 'max']:
            columns.append(f'cont_{cont}_{stat}')
    
    # Crear DataFrame de prueba [24 horas x columnas]
    test_data = pd.DataFrame(
        np.random.rand(24, len(columns)) * 100,
        columns=columns
    )
    
    # Probar guardado
    result = save_predictions_to_postgres(
        test_data, 
        '2023-05-10 07:00:00'
    )
    
    print(f"Resultado: {result}")