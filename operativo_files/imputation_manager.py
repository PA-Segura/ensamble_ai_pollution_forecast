#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
imputation_manager.py - Gestor de imputación de datos faltantes para operativo

Módulo que implementa métodos de imputación para datos de contaminación:
1. Promedio de estaciones con observaciones (row_avg)
2. Persistencia temporal (día anterior)
3. Promedio histórico de 5 años (alternativa a climatología)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImputationManager:
    """Gestor de imputación de datos faltantes para contaminantes."""
    
    def __init__(self, db_manager=None):
        """
        Inicializa el gestor de imputación.
        
        Args:
            db_manager: Gestor de base de datos (PostgreSQL)
        """
        self.db_manager = db_manager
        self.imputation_methods = {
            'row_avg': self.impute_with_row_avg,
            'persistence': self.impute_with_persistence,
            'historical_avg': self.impute_with_historical_avg
        }
        
        logger.info("🔄 ImputationManager inicializado")
    
    def prepare_data_for_imputation(self, df: pd.DataFrame, pollutant_columns: List[str]) -> pd.DataFrame:
        """
        Prepara datos para imputación creando columnas de banderas.
        
        Args:
            df: DataFrame original
            pollutant_columns: Lista de columnas de contaminantes a procesar
        
        Returns:
            DataFrame con columnas de banderas agregadas
        """
        df_imputed = df.copy()
        
        # PASO 1: Limpiar datos de calidad (valores negativos, outliers, etc.)
        df_imputed = self.clean_pollution_data_quality(df_imputed, pollutant_columns)
        
        # PASO 2: Crear columnas de banderas
        for col in pollutant_columns:
            new_col_name = f"i_{col}"
            df_imputed[new_col_name] = df_imputed[col].apply(
                lambda x: 'none' if pd.notna(x) else -1
            )
        
        logger.info(f"✅ Preparadas {len(pollutant_columns)} columnas de banderas")
        return df_imputed
    
    def impute_with_row_avg(self, df: pd.DataFrame, pollutant_columns: List[str]) -> pd.DataFrame:
        """
        Imputa valores usando promedio de fila (estaciones con observaciones).
        Agrupa por tipo de contaminante para evitar mezclar diferentes tipos.
        
        Args:
            df: DataFrame con datos y columnas de banderas
            pollutant_columns: Lista de columnas de contaminantes a imputar
        
        Returns:
            DataFrame con valores imputados y banderas actualizadas
        """
        df_imputed = df.copy()
        
        # PASO 1: Agrupar columnas por tipo de contaminante
        pollutant_groups = {}
        for col in pollutant_columns:
            # Extraer el tipo de contaminante (ej: "cont_otres", "cont_nodos", etc.)
            if col.startswith('cont_'):
                parts = col.split('_')
                if len(parts) >= 2:
                    pollutant_type = f"{parts[0]}_{parts[1]}"  # "cont_otres", "cont_nodos"
                    if pollutant_type not in pollutant_groups:
                        pollutant_groups[pollutant_type] = []
                    pollutant_groups[pollutant_type].append(col)
        
        logger.info(f"   🔍 Agrupados {len(pollutant_groups)} tipos de contaminantes:")
        for pollutant_type, cols in pollutant_groups.items():
            logger.info(f"      {pollutant_type}: {len(cols)} estaciones")
        
        # PASO 2: Procesar cada tipo de contaminante por separado
        for pollutant_type, type_columns in pollutant_groups.items():
            logger.info(f"   🚀 Procesando {pollutant_type} ({len(type_columns)} estaciones)")
            
            # Crear mini DataFrame solo para este tipo de contaminante
            mini_df = df_imputed[type_columns].copy()
            
            # PASO 3: Para cada columna de este tipo de contaminante
            for col in type_columns:
                flag_col_name = f"i_{col}"
                
                # Identificar valores que aún necesitan imputación
                mask_values_still_missing = (df_imputed[flag_col_name] == -1)
                
                if mask_values_still_missing.any():
                    # Verificar que hay suficientes datos válidos en la fila (>2 para este tipo específico)
                    # Usar solo las columnas del mismo tipo de contaminante
                    mask_enough_valid_data = (mini_df.notna().sum(axis=1) > 2)
                    
                    # Combinar condiciones para determinar qué valores imputar
                    mask_can_impute_with_row_avg = mask_values_still_missing & mask_enough_valid_data
                    
                    if mask_can_impute_with_row_avg.any():
                        # Calcular promedio SOLO de las columnas del mismo tipo de contaminante
                        # Excluir la columna actual para evitar sesgo
                        other_columns_same_type = [c for c in type_columns if c != col]
                        
                        # Para cada fila que necesita imputación
                        for idx in df_imputed.index[mask_can_impute_with_row_avg]:
                            # Obtener valores válidos de otras estaciones del mismo tipo de contaminante
                            valid_values = mini_df.loc[idx, other_columns_same_type].dropna()
                            
                            # Solo imputar si hay al menos 2 valores válidos del mismo tipo
                            if len(valid_values) >= 2:
                                # Filtrar valores extremos (opcional pero recomendado)
                                valid_values_filtered = valid_values[
                                    (valid_values > 0) &  # Solo valores positivos
                                    (valid_values < valid_values.quantile(0.95))  # Excluir outliers superiores
                                ]
                                
                                if len(valid_values_filtered) >= 1:  # Al menos 1 valor después del filtrado
                                    row_avg = valid_values_filtered.mean()
                                    df_imputed.loc[idx, col] = row_avg
                                    df_imputed.loc[idx, flag_col_name] = 'row_avg'
                                    
                                    logger.debug(f"      {col} en {idx}: imputado con {row_avg:.2f} (de {len(valid_values_filtered)} valores)")
                        
                        logger.info(f"   📊 {col}: {mask_can_impute_with_row_avg.sum()} valores imputados con promedio de {pollutant_type}")
                    else:
                        logger.debug(f"   ⚠️ {col}: No se pueden imputar valores (datos insuficientes)")
                else:
                    logger.debug(f"   ✅ {col}: No necesita imputación")
        
        return df_imputed
    
    def impute_with_persistence(self, df: pd.DataFrame, pollutant_columns: List[str]) -> pd.DataFrame:
        """
        Imputa valores usando persistencia (día anterior).
        
        Args:
            df: DataFrame con datos y columnas de banderas
            pollutant_columns: Lista de columnas de contaminantes a imputar
        
        Returns:
            DataFrame con valores imputados y banderas actualizadas
        """
        df_imputed = df.copy()
        
        for col in pollutant_columns:
            flag_col_name = f"i_{col}"
            
            # PASO 1: Identificar valores que aún necesitan imputación
            mask_values_still_missing = (df_imputed[flag_col_name] == -1)
            
            if mask_values_still_missing.any():
                # PASO 2: Calcular índices del día anterior para cada valor faltante
                current_timestamps = df_imputed.index[mask_values_still_missing]
                previous_day_timestamps = current_timestamps - pd.Timedelta(days=1)
                
                # PASO 3: Verificar que los índices del día anterior existan
                valid_previous_day_timestamps = df_imputed.index.intersection(previous_day_timestamps)
                
                # PASO 4: Crear máscara para valores que pueden ser imputados
                mask_previous_day_exists = df_imputed.index.isin(valid_previous_day_timestamps)
                mask_can_impute_with_persistence = mask_values_still_missing & mask_previous_day_exists
                
                if mask_can_impute_with_persistence.any():
                    # PASO 5: Obtener valores del día anterior
                    current_indices = df_imputed.index[mask_can_impute_with_persistence]
                    previous_day_indices = current_indices - pd.Timedelta(days=1)
                    
                    # Obtener valores del día anterior solo para los índices válidos
                    valid_previous_values = df_imputed.loc[previous_day_indices, col]
                    
                    # PASO 6: Asignar valores imputados y actualizar banderas
                    df_imputed.loc[mask_can_impute_with_persistence, col] = valid_previous_values.values
                    df_imputed.loc[mask_can_impute_with_persistence, flag_col_name] = 'last_day_same_hour'
                    
                    logger.info(f"   ⏰ {col}: {mask_can_impute_with_persistence.sum()} valores imputados con persistencia")
        
        return df_imputed
    
    def impute_with_historical_avg(self, df: pd.DataFrame, pollutant_columns: List[str], 
                                  years_back: int = 5) -> pd.DataFrame:
        """
        Imputa valores usando promedio histórico de 5 años (alternativa a climatología).
        
        Args:
            df: DataFrame con datos y columnas de banderas
            pollutant_columns: Lista de columnas de contaminantes a imputar
            years_back: Años hacia atrás para calcular el promedio histórico
        
        Returns:
            DataFrame con valores imputados y banderas actualizadas
        """
        if self.db_manager is None:
            logger.warning("⚠️ Gestor de BD no disponible - omitiendo promedio histórico")
            return df
        
        df_imputed = df.copy()
        
        for col in pollutant_columns:
            flag_col_name = f"i_{col}"
            
            # PASO 1: Identificar valores que aún necesitan imputación
            mask_values_still_missing = (df_imputed[flag_col_name] == -1)
            
            if mask_values_still_missing.any():
                logger.info(f"   🗄️ {col}: Imputando {mask_values_still_missing.sum()} valores con promedio histórico...")
                
                # PASO 2: Agrupar por fecha/hora para hacer queries eficientes
                missing_dates = df_imputed.index[mask_values_still_missing]
                date_hour_groups = self._group_missing_values_by_date_hour(missing_dates)
                
                # PASO 3: Para cada grupo de fecha/hora, hacer una query eficiente
                for (month, day, hour), timestamps in date_hour_groups.items():
                    try:
                        # PASO 4: Obtener promedio histórico para esta fecha/hora
                        historical_avg = self._get_historical_average_efficient(
                            col, month, day, hour, years_back
                        )
                        
                        if historical_avg is not None:
                            # PASO 5: Aplicar a todos los timestamps de este grupo
                            for timestamp in timestamps:
                                df_imputed.loc[timestamp, col] = historical_avg
                                df_imputed.loc[timestamp, flag_col_name] = 'historical_avg'
                            
                            logger.info(f"      ✅ {month:02d}-{day:02d} {hour:02d}:00 → {historical_avg:.2f}")
                        else:
                            logger.warning(f"      ⚠️ {month:02d}-{day:02d} {hour:02d}:00 → Sin datos históricos")
                            
                    except Exception as e:
                        logger.error(f"      ❌ Error en {month:02d}-{day:02d} {hour:02d}:00: {str(e)}")
                        continue
        
        return df_imputed
    
    def _group_missing_values_by_date_hour(self, timestamps: pd.DatetimeIndex) -> Dict[Tuple[int, int, int], List[pd.Timestamp]]:
        """
        Agrupa timestamps faltantes por fecha/hora para hacer queries eficientes.
        
        Args:
            timestamps: Índice de timestamps con valores faltantes
        
        Returns:
            Diccionario agrupando por (mes, día, hora)
        """
        groups = {}
        for timestamp in timestamps:
            key = (timestamp.month, timestamp.day, timestamp.hour)
            if key not in groups:
                groups[key] = []
            groups[key].append(timestamp)
        
        return groups
    
    def _get_historical_average_efficient(self, column_name: str, month: int, day: int, 
                                        hour: int, years_back: int) -> Optional[float]:
        """
        Obtiene el promedio histórico de manera eficiente para una fecha/hora específica.
        
        Args:
            column_name: Nombre de la columna
            month: Mes (1-12)
            day: Día (1-31)
            hour: Hora (0-23)
            years_back: Años hacia atrás
        
        Returns:
            Promedio histórico o None si no hay datos
        """
        try:
            # Query SQL optimizada para PostgreSQL
            query = f"""
            SELECT AVG({column_name}) as avg_value, COUNT(*) as data_count
            FROM (
                SELECT {column_name}
                FROM pollution_data 
                WHERE EXTRACT(MONTH FROM timestamp) = {month}
                  AND EXTRACT(DAY FROM timestamp) = {day}
                  AND EXTRACT(HOUR FROM timestamp) = {hour}
                  AND timestamp >= CURRENT_DATE - INTERVAL '{years_back} years'
                  AND timestamp < CURRENT_DATE
                  AND {column_name} IS NOT NULL
                  AND {column_name} > 0
            ) subquery
            """
            
            # Ejecutar query
            result = self.db_manager.execute_query(query)
            
            if result and len(result) > 0:
                avg_value, data_count = result[0]
                
                # Verificar que hay suficientes datos históricos (mínimo 5 registros)
                if data_count >= 5 and avg_value is not None:
                    return float(avg_value)
                else:
                    logger.debug(f"      ⚠️ Datos insuficientes para {column_name}: {data_count} registros")
                    return None
            else:
                return None
                
        except Exception as e:
            logger.error(f"      ❌ Error en query histórica para {column_name}: {str(e)}")
            return None
    
    def apply_imputation_pipeline(self, df: pd.DataFrame, pollutant_columns: List[str], 
                                 years_back: int = 5) -> pd.DataFrame:
        """
        Aplica el pipeline completo de imputación en orden de prioridad.
        
        Args:
            df: DataFrame con datos y columnas de banderas
            pollutant_columns: Lista de columnas de contaminantes a imputar
            years_back: Años hacia atrás para promedio histórico
        
        Returns:
            DataFrame con todos los valores imputados
        """
        df_imputed = df.copy()
        
        logger.info("🔄 Aplicando pipeline de imputación...")
        
        # PASO 1: Promedio de fila (más confiable)
        logger.info("   📊 1. Imputación con promedio de fila...")
        df_imputed = self.impute_with_row_avg(df_imputed, pollutant_columns)
        
        # PASO 2: Persistencia temporal
        logger.info("   ⏰ 2. Imputación con persistencia...")
        df_imputed = self.impute_with_persistence(df_imputed, pollutant_columns)
        
        # PASO 3: Promedio histórico de BD (si está disponible)
        if self.db_manager is not None:
            logger.info("   🗄️ 3. Imputación con promedio histórico de BD...")
            df_imputed = self.impute_with_historical_avg(df_imputed, pollutant_columns, years_back)
        else:
            logger.warning("   ⚠️ 3. Gestor de BD no disponible - omitiendo promedio histórico")
        
        return df_imputed
    
    def get_imputation_summary(self, df: pd.DataFrame, pollutant_columns: List[str]) -> Dict:
        """
        Genera resumen de la imputación aplicada.
        
        Args:
            df: DataFrame con datos imputados
            pollutant_columns: Lista de columnas de contaminantes
        
        Returns:
            Diccionario con resumen de imputación por contaminante
        """
        summary = {}
        
        for col in pollutant_columns:
            flag_col = f"i_{col}"
            if flag_col in df.columns:
                flag_counts = df[flag_col].value_counts()
                summary[col] = flag_counts.to_dict()
                
                # Calcular estadísticas adicionales
                total_imputed = flag_counts.get('row_avg', 0) + flag_counts.get('last_day_same_hour', 0) + flag_counts.get('historical_avg', 0)
                total_original = flag_counts.get('none', 0)
                total_missing = flag_counts.get(-1, 0)
                
                summary[f"{col}_stats"] = {
                    'original': total_original,
                    'imputed': total_imputed,
                    'still_missing': total_missing,
                    'imputation_rate': f"{(total_imputed / (total_original + total_imputed + total_missing) * 100):.1f}%"
                }
        
        return summary
    
    def check_for_missing_values(self, df: pd.DataFrame, pollutant_columns: List[str]) -> bool:
        """
        Verifica si hay valores faltantes en las columnas de contaminantes.
        
        Args:
            df: DataFrame a verificar
            pollutant_columns: Lista de columnas de contaminantes
        
        Returns:
            True si hay valores faltantes, False en caso contrario
        """
        for col in pollutant_columns:
            if col in df.columns and df[col].isna().any():
                return True
        return False
    
    def get_missing_values_count(self, df: pd.DataFrame, pollutant_columns: List[str]) -> Dict[str, int]:
        """
        Cuenta valores faltantes por columna de contaminante.
        
        Args:
            df: DataFrame a verificar
            pollutant_columns: Lista de columnas de contaminantes
        
        Returns:
            Diccionario con conteo de valores faltantes por columna
        """
        missing_counts = {}
        for col in pollutant_columns:
            if col in df.columns:
                missing_counts[col] = df[col].isna().sum()
        return missing_counts
    
    def clean_pollution_data_quality(self, df: pd.DataFrame, pollutant_columns: List[str]) -> pd.DataFrame:
        """
        Limpia datos de contaminación aplicando filtros de calidad.
        
        Args:
            df: DataFrame con datos de contaminación
            pollutant_columns: Lista de columnas de contaminantes
        
        Returns:
            DataFrame con datos limpios
        """
        df_clean = df.copy()
        
        logger.info("🧹 Aplicando filtros de calidad de datos de contaminación...")
        
        total_cleaned = 0
        
        for col in pollutant_columns:
            if col in df_clean.columns:
                initial_count = df_clean[col].isna().sum()
                
                # 1. Valores negativos (inconsistentes)
                negative_mask = df_clean[col] < 0
                negative_count = negative_mask.sum()
                
                if negative_count > 0:
                    logger.info(f"   ⚠️ {col}: {negative_count} valores negativos → NaN")
                    df_clean.loc[negative_mask, col] = np.nan
                
                # 2. Valores extremadamente altos (outliers)
                # Para PM10: > 1000 μg/m³ es extremadamente alto
                # Para PM2.5: > 500 μg/m³ es extremadamente alto
                # Para O3: > 500 ppb es extremadamente alto
                if 'pmdiez' in col.lower() or 'pm10' in col.lower():
                    outlier_mask = df_clean[col] > 1000
                elif 'pmdoscinco' in col.lower() or 'pm2.5' in col.lower() or 'pm25' in col.lower():
                    outlier_mask = df_clean[col] > 500
                elif 'otres' in col.lower() or 'o3' in col.lower():
                    outlier_mask = df_clean[col] > 500
                else:
                    # Para otros contaminantes, usar un umbral más conservador
                    outlier_mask = df_clean[col] > 1000
                
                outlier_count = outlier_mask.sum()
                if outlier_count > 0:
                    logger.info(f"   🚨 {col}: {outlier_count} valores extremadamente altos → NaN")
                    df_clean.loc[outlier_mask, col] = np.nan
                
                # 3. Contar total de valores limpiados
                final_count = df_clean[col].isna().sum()
                cleaned_in_col = final_count - initial_count
                total_cleaned += cleaned_in_col
                
                if cleaned_in_col > 0:
                    logger.info(f"   📊 {col}: {cleaned_in_col} valores limpiados")
        
        if total_cleaned > 0:
            logger.info(f"   🎯 Total de valores limpiados: {total_cleaned}")
        else:
            logger.info("   ✅ No se requirió limpieza de datos")
        
        return df_clean


# Funciones de utilidad para uso independiente
def create_imputation_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Crea columnas de imputación con banderas.
    
    Args:
        df: DataFrame original
        columns: Lista de columnas a procesar
    
    Returns:
        DataFrame con columnas de banderas agregadas
    """
    imputation_mgr = ImputationManager()
    return imputation_mgr.prepare_data_for_imputation(df, columns)


def apply_imputation_pipeline(df: pd.DataFrame, pollutant_columns: List[str], 
                            db_manager=None, years_back: int = 5) -> pd.DataFrame:
    """
    Aplica el pipeline completo de imputación.
    
    Args:
        df: DataFrame con datos y columnas de banderas
        pollutant_columns: Lista de columnas de contaminantes a imputar
        db_manager: Gestor de base de datos
        years_back: Años hacia atrás para promedio histórico
    
    Returns:
        DataFrame con todos los valores imputados
    """
    imputation_mgr = ImputationManager(db_manager)
    return imputation_mgr.apply_imputation_pipeline(df, pollutant_columns, years_back) 