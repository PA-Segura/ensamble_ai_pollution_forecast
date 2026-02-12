#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
save_predictions_sqlite.py - Función para guardar predicciones (estilo 7_Operativo.py)
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

def getLocalSQLiteConn(db_path="forecast_predictions.db"):
    """Conexión a SQLite local (equivalente a getPostgresConn() del sistema anterior)."""
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except Exception as e:
        print(f"Failed to connect to SQLite: {e}")
        return None

def save_predictions_to_sqlite(predictions_denormalized: pd.DataFrame, 
                              target_datetime: str,
                              db_path: str = "forecast_predictions.db",
                              verbose: bool = True):
    """
    Guarda predicciones en SQLite siguiendo el patrón de 7_Operativo.py
    
    Args:
        predictions_denormalized: DataFrame [24 horas x 54 contaminantes]
        target_datetime: Fecha del pronóstico 'YYYY-MM-DD HH:MM:SS'
        db_path: Ruta de la base SQLite
        verbose: Mostrar mensajes
    """
    
    if verbose:
        print(f"\n💾 GUARDANDO PREDICCIONES EN SQLITE")
        print(f"   📅 Fecha: {target_datetime}")
        print(f"   📊 Datos: {predictions_denormalized.shape}")
        print(f"   🗄️ Base: {db_path}")
    
    forecast_time = datetime.strptime(target_datetime, '%Y-%m-%d %H:%M:%S')
    
    # Generar string de columnas hour_p01, hour_p02, ..., hour_p24 (igual que en 7_Operativo.py)
    all_hours = ','.join([f'hour_p{x:02d}' for x in range(1, 25)])
    
    # Contadores
    saved_ozono = 0
    saved_stats = 0
    errors = 0
    
    # =============================================================================
    # 1. GUARDAR OZONO (30 ESTACIONES) - IGUAL QUE EN 7_Operativo.py
    # =============================================================================
    if verbose:
        print("   🌟 Guardando ozono por estaciones...")
    
    # Lista de estaciones de ozono (igual que all_stations en 7_Operativo.py)
    ozono_stations = ["UIZ","AJU" ,"ATI" ,"CUA" ,"SFE" ,"SAG" ,"CUT" ,"PED" ,"TAH" ,"GAM" ,"IZT" ,"CCA" ,"HGM" ,"LPR" ,
                      "MGH" ,"CAM" ,"FAC" ,"TLA" ,"MER" ,"XAL" ,"LLA" ,"TLI" ,"UAX" ,"BJU" ,"MPA" ,
                      "MON" ,"NEZ" ,"INN" ,"AJM" ,"VIF"]
    
    for c_station in ozono_stations:
        try:
            conn = getLocalSQLiteConn(db_path)
            if not conn:
                continue
                
            # Filtrar predicciones de esta estación (igual que en 7_Operativo.py)
            station_column = f'cont_otres_{c_station}'
            
            if station_column in predictions_denormalized.columns:
                pred_by_station = predictions_denormalized[station_column]
                
                # SQL de inserción (adaptado de 7_Operativo.py)
                sql = f"""INSERT OR REPLACE INTO forecast_otres 
                         (fecha, id_tipo_pronostico, id_est, val, {all_hours}) 
                         VALUES (?, ?, ?, ?, {','.join(['?']*24)})"""
                
                cur = conn.cursor()
                values = [float(f'{x:.2f}') for x in pred_by_station.tolist()]
                cur.execute(sql, (forecast_time, 6, c_station, '-1', *values))
                cur.close()
                conn.commit()
                conn.close()
                
                saved_ozono += 1
                if verbose:
                    print(f"     ✅ {c_station}: {len(values)} horas")
            else:
                if verbose:
                    print(f"     ⚠️ {c_station}: columna no encontrada")
                    
        except Exception as e:
            errors += 1
            if verbose:
                print(f"     ❌ {c_station}: {e}")
            if conn:
                conn.close()
    
    # =============================================================================
    # 2. GUARDAR ESTADÍSTICAS DE OTROS CONTAMINANTES (24 ESTADÍSTICAS)
    # =============================================================================
    if verbose:
        print("   📊 Guardando estadísticas de contaminantes...")
    
    # Tipos de contaminantes con sus estadísticas
    contaminant_types = ['co', 'nodos', 'pmdiez', 'pmdoscinco', 'nox', 'no', 'sodos', 'pmco']
    statistics = ['mean', 'min', 'max']
    
    for contaminant in contaminant_types:
        for stat in statistics:
            try:
                conn = getLocalSQLiteConn(db_path)
                if not conn:
                    continue
                
                # Buscar columna correspondiente
                stat_column = f'cont_{contaminant}_{stat}'
                
                if stat_column in predictions_denormalized.columns:
                    pred_by_stat = predictions_denormalized[stat_column]
                    
                    # SQL de inserción para estadísticas
                    sql = f"""INSERT OR REPLACE INTO forecast_stats 
                             (fecha, id_tipo_pronostico, contaminante, estadistica, val, {all_hours}) 
                             VALUES (?, ?, ?, ?, ?, {','.join(['?']*24)})"""
                    
                    cur = conn.cursor()
                    values = [float(f'{x:.2f}') for x in pred_by_stat.tolist()]
                    cur.execute(sql, (forecast_time, 6, contaminant, stat, '-1', *values))
                    cur.close()
                    conn.commit()
                    conn.close()
                    
                    saved_stats += 1
                    if verbose:
                        print(f"     ✅ {contaminant}_{stat}: {len(values)} horas")
                else:
                    if verbose:
                        print(f"     ⚠️ {stat_column}: columna no encontrada")
                        
            except Exception as e:
                errors += 1
                if verbose:
                    print(f"     ❌ {contaminant}_{stat}: {e}")
                if conn:
                    conn.close()
    
    # =============================================================================
    # RESUMEN FINAL
    # =============================================================================
    if verbose:
        print(f"\n📈 RESUMEN DEL GUARDADO:")
        print(f"   🌟 Ozono guardado: {saved_ozono}/30 estaciones")
        print(f"   📊 Estadísticas guardadas: {saved_stats}/24 combinaciones")
        print(f"   ❌ Errores: {errors}")
        print(f"   💾 Total registros: {saved_ozono + saved_stats}")
        
        if saved_ozono + saved_stats > 0:
            print("   ✅ Guardado exitoso")
        else:
            print("   ⚠️ No se guardaron datos")
    
    return {
        'ozono_saved': saved_ozono,
        'stats_saved': saved_stats,
        'errors': errors,
        'total_saved': saved_ozono + saved_stats
    }

# =============================================================================
# FUNCIÓN PARA MIGRAR A POSTGRESQL (CUANDO ESTÉ DISPONIBLE)
# =============================================================================
def migrate_to_postgresql(sqlite_db_path="forecast_predictions.db"):
    """Migra datos de SQLite a PostgreSQL usando el mismo formato que 7_Operativo.py"""
    
    print("🔄 MIGRANDO SQLITE → POSTGRESQL")
    
    try:
        # Importar función de PostgreSQL del sistema anterior
        from db.sqlCont import getPostgresConn
        
        # Conectar a ambas bases
        sqlite_conn = sqlite3.connect(sqlite_db_path)
        pg_conn = getPostgresConn()
        
        # Migrar ozono (tabla forecast_otres)
        print("🌟 Migrando tabla ozono...")
        sqlite_cursor = sqlite_conn.cursor()
        sqlite_cursor.execute("SELECT * FROM forecast_otres")
        
        all_hours = ','.join([f'hour_p{x:02d}' for x in range(1, 25)])
        
        for row in sqlite_cursor.fetchall():
            try:
                pg_cursor = pg_conn.cursor() 
                sql = f"""INSERT INTO forecast_otres (fecha, id_tipo_pronostico, id_est, val, {all_hours}) 
                         VALUES (%s, %s, %s, %s, {','.join(['%s']*24)})"""
                
                # row[0]=id, row[1]=fecha, row[2]=id_tipo, row[3]=id_est, row[4]=val, row[5:29]=hours
                values = (row[1], row[2], row[3], row[4], *row[5:29])
                pg_cursor.execute(sql, values)
                pg_cursor.close()
                pg_conn.commit()
                
            except Exception as e:
                print(f"Error migrando ozono {row[3]}: {e}")
        
        sqlite_conn.close()
        pg_conn.close()
        
        print("✅ Migración completada")
        
    except ImportError:
        print("❌ No se puede importar getPostgresConn - PostgreSQL no disponible")
    except Exception as e:
        print(f"❌ Error en migración: {e}")

if __name__ == "__main__":
    # Crear datos de prueba
    print("🧪 PROBANDO GUARDADO...")
    
    # Simular datos de predicción
    columns = []
    # 30 estaciones de ozono
    for station in ["UIZ", "AJU", "ATI"]:  # Solo 3 para prueba
        columns.append(f'cont_otres_{station}')
    # Algunas estadísticas
    for cont in ['co', 'nox']:
        for stat in ['mean', 'min', 'max']:
            columns.append(f'cont_{cont}_{stat}')
    
    # Crear DataFrame de prueba [24 horas x columnas]
    test_data = pd.DataFrame(
        np.random.rand(24, len(columns)) * 100,
        columns=columns
    )
    
    # Probar guardado
    result = save_predictions_to_sqlite(
        test_data, 
        '2023-05-10 07:00:00',
        'test_forecast.db'
    )
    
    print(f"Resultado: {result}") 