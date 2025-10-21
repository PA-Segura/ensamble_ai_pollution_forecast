# %%
# Imports y configuración
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from os.path import join
from proj_io.inout import create_folder

# %%
# Configuración de directorios y constantes
os.chdir('/home/pedro/git2/gitflow/air_pollution_forecast')

START_YEAR = 2010
END_YEAR = 2024
INPUT_FOLDER = '/home/pedro/DATA_bkup/16/'
OUTPUT_FOLDER = '/home/pedro/DATA_bkup/16/Imputed/'
CLIMATOLOGY_FOLDER = '/home/pedro/DATA_bkup/16/Climatology/'
PLOTS_FOLDER = '/home/pedro/DATA_bkup/16/Imputed/Plots/'

# Crear directorios
for folder in [OUTPUT_FOLDER, CLIMATOLOGY_FOLDER, PLOTS_FOLDER]:
    create_folder(folder)

# %%
# Funciones de utilidad
def create_folder(folder_path: str) -> None:
    """Crea una carpeta si no existe."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def read_merged_files(input_folder: str, start_year: int, end_year: int) -> pd.DataFrame:
    """Lee y combina los archivos de datos de contaminación."""
    for c_year in range(start_year, end_year+1):
        db_file_name = join(input_folder, f"{c_year}_AllStations.csv")
        if c_year == start_year:
            data = pd.read_csv(db_file_name, index_col=0)
        else:
            data = pd.concat([data, pd.read_csv(db_file_name, index_col=0)])
    return data

# %%
# Funciones de climatología
def generate_climatology(df: pd.DataFrame, value_columns: list) -> pd.DataFrame:
    """Genera climatología para las columnas especificadas usando solo datos observados."""
    climatology = pd.DataFrame(index=pd.date_range(start='2012-01-01 00:00:00', 
                                                 end='2012-12-31 23:00:00', 
                                                 freq='h'))
    
    for col in value_columns:
        hourly_means = df.groupby([df.index.month, df.index.day, df.index.hour])[col].mean()
        
        hourly_means_dict = {}
        for (month, day, hour), value in hourly_means.items():
            try:
                date = pd.Timestamp(f'2012-{month:02d}-{day:02d} {hour:02d}:00:00')
                if not pd.isna(value):
                    hourly_means_dict[date] = value
            except ValueError:
                continue
        
        climatology[col] = pd.Series(hourly_means_dict)
        climatology[col] = climatology[col].rolling(window=3, center=True, min_periods=1).mean()
        
        if not climatology[col].isna().all():
            first_valid = climatology[col].first_valid_index()
            last_valid = climatology[col].last_valid_index()
            
            if first_valid is not None and last_valid is not None:
                climatology.loc[climatology.index[0], col] = (climatology.loc[climatology.index[-1], col] + climatology.loc[climatology.index[0], col] + climatology.loc[climatology.index[1], col]) / 3
                climatology.loc[climatology.index[-1], col] = (climatology.loc[climatology.index[-2], col] + climatology.loc[climatology.index[-1], col] + climatology.loc[climatology.index[0], col]) / 3
    
    return climatology

def create_complete_time_index(start_year: int, end_year: int) -> pd.DatetimeIndex:
    """Crea un índice temporal completo para todos los años especificados."""
    return pd.date_range(
        start=f'{start_year}-01-01 00:00:00',
        end=f'{end_year}-12-31 23:00:00',
        freq='h'
    )

# %%
# Funciones de imputación
def create_imputation_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Crea columnas de imputación."""
    df_imputed = df.copy()
    for col in columns:
        new_col_name = f"i_{col}"
        df_imputed[new_col_name] = df_imputed[col].apply(lambda x: 'none' if pd.notna(x) else -1)
    return df_imputed

def impute_with_row_avg(df: pd.DataFrame, value_columns: list) -> pd.DataFrame:
    """Imputa valores usando promedio de fila."""
    df_imputed = df.copy()
    
    for col in value_columns:
        flag_col_name = f"i_{col}"
        
        # PASO 1: Identificar valores que aún necesitan imputación
        mask_values_still_missing = (df_imputed[flag_col_name] == -1)
        
        # PASO 2: Verificar que hay suficientes datos válidos en la fila (>5)
        mask_enough_valid_data = (df[value_columns].notna().sum(axis=1) > 5)
        
        # PASO 3: Combinar condiciones para determinar qué valores imputar
        mask_can_impute_with_row_avg = mask_values_still_missing & mask_enough_valid_data
        
        if mask_can_impute_with_row_avg.any():
            # PASO 4: Calcular promedio de fila para valores faltantes
            row_averages = df_imputed.loc[mask_can_impute_with_row_avg, value_columns].mean(axis=1, skipna=True)
            
            # PASO 5: Asignar valores imputados y actualizar banderas
            df_imputed.loc[mask_can_impute_with_row_avg, col] = row_averages
            df_imputed.loc[mask_can_impute_with_row_avg, flag_col_name] = 'row_avg'
    
    return df_imputed

def impute_with_persistence(df: pd.DataFrame, value_columns: list) -> pd.DataFrame:
    """Imputa valores usando persistencia (mismo día y hora del día anterior)."""
    df_imputed = df.copy()
    
    for col in value_columns:
        flag_col_name = f"i_{col}"
        
        # PASO 1: Identificar valores que aún necesitan imputación después del promedio de fila
        mask_values_still_missing = (df_imputed[flag_col_name] == -1)
        
        if mask_values_still_missing.any():
            # PASO 2: Para cada valor faltante, buscar su valor del día anterior
            for idx in df_imputed.index[mask_values_still_missing]:
                # PASO 3: Calcular índice del día anterior
                previous_day_idx = idx - pd.Timedelta(days=1)
                
                # PASO 4: Verificar que el índice del día anterior existe en el DataFrame
                if previous_day_idx in df_imputed.index:
                    # PASO 5: Obtener valor del día anterior
                    previous_day_value = df_imputed.loc[previous_day_idx, col]
                    
                    # PASO 6: Solo imputar si el valor del día anterior no es NaN
                    if pd.notna(previous_day_value):
                        df_imputed.loc[idx, col] = previous_day_value
                        df_imputed.loc[idx, flag_col_name] = 'last_day_same_hour'
    
    return df_imputed

def impute_with_climatology(df: pd.DataFrame, climatology: pd.DataFrame, value_columns: list) -> pd.DataFrame:
    """Imputa valores usando climatología."""
    df_imputed = df.copy()
    
    for col in value_columns:
        flag_col_name = f"i_{col}"
        
        # PASO 1: Identificar valores que aún necesitan imputación después de métodos anteriores
        mask_values_still_missing = (df_imputed[flag_col_name] == -1)
        
        # PASO 2: Para cada valor faltante, buscar su valor climatológico correspondiente
        for idx in df_imputed.index[mask_values_still_missing]:
            # PASO 3: Extraer mes, día y hora del timestamp actual
            month = idx.month
            day = idx.day
            hour = idx.hour
            
            # PASO 4: Crear índice para buscar en climatología (usando año 2012 como referencia)
            climatology_idx = pd.Timestamp(f'2012-{month:02d}-{day:02d} {hour:02d}:00:00')
            
            # PASO 5: Obtener valor climatológico y asignarlo
            climatology_value = climatology.loc[climatology_idx, col]
            df_imputed.loc[idx, col] = climatology_value
            
            # PASO 6: Actualizar bandera de imputación
            df_imputed.loc[idx, flag_col_name] = 'climatology'
    
    return df_imputed

# %%
# Funciones de análisis y visualización
def prepare_clustering_data(climatology_df: pd.DataFrame) -> tuple:
    """Prepara los datos para el clustering."""
    X = climatology_df.T
    X = pd.DataFrame(X)
    X = X.fillna(X.mean())
    
    if np.isinf(X.values).any():
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if not np.all(np.isfinite(X_scaled)):
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X_scaled, scaler

def analizar_contaminante(df: pd.DataFrame, prefix: str, output_folder: str) -> tuple:
    """Analiza un contaminante específico usando clustering."""
    columns = [col for col in df.columns if col.startswith(prefix)]
    climatology = generate_climatology(df, columns)
    X_scaled, scaler = prepare_clustering_data(climatology)
    
    # Clustering jerárquico
    linkage_matrix = linkage(X_scaled, method='ward')
    
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix, labels=climatology.columns, leaf_rotation=90)
    plt.title(f'Dendrograma de Estaciones {prefix.replace("cont_", "").replace("_", "")}')
    plt.xlabel('Estaciones')
    plt.ylabel('Distancia')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'dendrograma_{prefix.replace("cont_", "").replace("_", "")}.png'))
    plt.close()
    
    # K-means clustering
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    cluster_results = pd.DataFrame({
        'Estacion': climatology.columns,
        'Cluster': clusters
    })
    cluster_results.to_csv(os.path.join(output_folder, f'clusters_{prefix.replace("cont_", "").replace("_", "")}_7fix.csv'))
    
    # Visualizar perfiles medios por cluster
    plt.figure(figsize=(15, 6))
    for i in range(n_clusters):
        estaciones_cluster = cluster_results[cluster_results['Cluster'] == i]['Estacion']
        perfil_medio = climatology[estaciones_cluster].mean(axis=1)
        plt.plot(climatology.index, perfil_medio, label=f'Cluster {i}')
    
    plt.title(f'Perfiles Medios por Cluster - {prefix.replace("cont_", "").replace("_", "")}')
    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'perfiles_cluster_{prefix.replace("cont_", "").replace("_", "")}.png'))
    plt.close()
    
    return climatology, cluster_results, None

def plot_time_series_imputed(df: pd.DataFrame, prefix: str, title_suffix: str = "(Imputados)") -> None:
    """Visualiza las series temporales de los datos imputados."""
    columns = [col for col in df.columns if col.startswith(prefix) and not col.startswith(f"i_{prefix}")]
    
    plt.figure(figsize=(20, 6))
    for col in columns:
        plt.plot(df.index, df[col], label=col, alpha=0.5)
    
    plt.title(f'Series Temporales de {prefix.replace("cont_", "").replace("_", "")} {title_suffix}')
    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(os.path.join(PLOTS_FOLDER, f'series_temporales_{prefix.replace("cont_", "").replace("_", "")}.png'))
    plt.close()

def plot_comparison_original_imputed(original_df: pd.DataFrame, 
                                  imputed_df: pd.DataFrame, 
                                  prefix: str, 
                                  year: int = 2023) -> None:
    """Visualiza la comparación entre datos originales e imputados."""
    columns = [col for col in original_df.columns if col.startswith(prefix) and not col.startswith(f"i_{prefix}")]
    
    original_year = original_df[original_df.index.year == year]
    imputed_year = imputed_df[imputed_df.index.year == year]
    
    n_stations = len(columns)
    n_cols = 2
    n_rows = (n_stations + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(columns):
        ax = axes[idx]
        ax.plot(original_year.index, original_year[col], label='Original', alpha=0.5)
        ax.plot(imputed_year.index, imputed_year[col], label='Imputado', alpha=0.5)
        ax.set_title(f'{col}')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Valor')
        ax.legend()
        ax.grid(True)
    
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Comparación Original vs Imputado - {prefix.replace("cont_", "").replace("_", "")} - {year}')
    plt.tight_layout()
    
    plt.savefig(os.path.join(PLOTS_FOLDER, f'comparacion_{prefix.replace("cont_", "").replace("_", "")}_{year}.png'))
    plt.close()

def guardar_resumen(resultados: dict, output_folder: str) -> None:
    """Guarda un resumen de los resultados del clustering."""
    resumen = []
    
    for contaminante, datos in resultados.items():
        try:
            cluster_dist = datos['clusters']['Cluster'].value_counts().sort_index()
            
            resumen.append({
                'Contaminante': contaminante,
                'Número de estaciones': len(datos['clusters']),
                'Distribución de clusters': cluster_dist.to_dict()
            })
        except Exception as e:
            continue
    
    resumen_df = pd.DataFrame(resumen)
    resumen_df.to_csv(os.path.join(output_folder, 'resumen_clustering_7fix.csv'), index=False)

# %%
# 1. Cargar datos originales
data_df = read_merged_files(INPUT_FOLDER, START_YEAR, END_YEAR)
data_df.index = pd.to_datetime(data_df.index.to_series().apply(
    lambda x: f"{x} 00:00:00" if len(str(x)) == 10 else str(x)))

# Verificar valores faltantes en datos originales
print("\n=== VERIFICACIÓN DE VALORES FALTANTES EN DATOS ORIGINALES ===")
cont_columns_original = [col for col in data_df.columns if col.startswith('cont_')]
total_nan_original = 0
for col in cont_columns_original:
    nan_count = data_df[col].isna().sum()
    total_nan_original += nan_count
    if nan_count > 0:
        print(f"  {col}: {nan_count} valores NaN")

print(f"\nTotal de valores NaN en datos originales: {total_nan_original}")
if total_nan_original == 0:
    print("⚠️  ADVERTENCIA: No hay valores faltantes en los datos originales.")
    print("   No se realizará ninguna imputación.")
else:
    print(f"✅ Se encontraron {total_nan_original} valores faltantes para imputar.")

# %%
# 1.1. Limpiar datos: eliminar columnas de meteo WRF
wrf_columns = []
for col in data_df.columns:
    for hour in range(24):
        if col.endswith(f'_h{hour}'):
            wrf_columns.append(col)
            break

if len(wrf_columns) > 0:
    data_df = data_df.drop(columns=wrf_columns)

# %%
# 2. Crear índice temporal completo
complete_index = create_complete_time_index(START_YEAR, END_YEAR)

# %%
# 3. Detectar grupos de contaminantes
def detect_contaminant_groups(df: pd.DataFrame) -> list:
    """Detecta automáticamente los grupos de contaminantes."""
    cont_columns = [col for col in df.columns if col.startswith('cont_')]
    groups = set()
    for col in cont_columns:
        parts = col.split('_')
        if len(parts) >= 2:
            group_prefix = f"cont_{parts[1]}_"
            groups.add(group_prefix)
    
    groups_list = sorted(list(groups))
    return groups_list

column_groups = detect_contaminant_groups(data_df)

# %%
# 4. Procesar imputación por grupos
data_imputed = pd.DataFrame(index=complete_index)

# Inicializar todas las columnas de contaminantes
all_cont_columns = [col for col in data_df.columns if col.startswith('cont_')]
for col in all_cont_columns:
    data_imputed[col] = np.nan
    data_imputed[f"i_{col}"] = -1

# Copiar datos observados y establecer banderas correctamente
mask_observed = data_imputed.index.isin(data_df.index)
for col in all_cont_columns:
    # Copiar valores observados
    data_imputed.loc[mask_observed, col] = data_df[col]
    
    # Establecer banderas: 'none' para valores observados, -1 para NaN
    observed_values = data_df[col]
    observed_flags = observed_values.apply(lambda x: 'none' if pd.notna(x) else -1)
    data_imputed.loc[mask_observed, f"i_{col}"] = observed_flags

print(f"\n=== VERIFICACIÓN DESPUÉS DE INICIALIZACIÓN ===")
print(f"Total de columnas de contaminantes: {len(all_cont_columns)}")
print(f"Total de filas: {len(data_imputed)}")

# Verificar banderas iniciales
total_flags_minus1 = 0
total_flags_none = 0
for col in all_cont_columns:
    flag_col = f"i_{col}"
    flags_minus1 = (data_imputed[flag_col] == -1).sum()
    flags_none = (data_imputed[flag_col] == 'none').sum()
    total_flags_minus1 += flags_minus1
    total_flags_none += flags_none

print(f"Total de banderas -1 (para imputar): {total_flags_minus1}")
print(f"Total de banderas 'none' (observados): {total_flags_none}")

if total_flags_minus1 == 0:
    print("⚠️  ADVERTENCIA: No hay valores marcados para imputación.")
    print("   Verificando datos originales...")
    
    # Verificar valores NaN en datos originales
    total_nan_original = 0
    for col in all_cont_columns:
        nan_count = data_df[col].isna().sum()
        total_nan_original += nan_count
    
    print(f"Valores NaN en datos originales: {total_nan_original}")
    
    if total_nan_original > 0:
        print("❌ ERROR: Hay valores NaN en datos originales pero no se marcaron para imputación.")
        print("   Esto indica un problema en la lógica de inicialización.")
    else:
        print("✅ No hay valores faltantes en los datos originales.")
else:
    print(f"✅ Se marcaron {total_flags_minus1} valores para imputación.")

# %%
# 5. Procesar cada grupo
for group in column_groups:
    print(f"\n=== Procesando grupo: {group} ===")
    columns = [col for col in data_df.columns if col.startswith(group)]
    print(f"Columnas del grupo: {columns}")
    
    climatology = generate_climatology(data_df, columns)
    
    # Verificar estado antes de imputación
    print(f"\nEstado antes de imputación para {group}:")
    for col in columns:
        flag_col = f"i_{col}"
        nan_count = data_imputed[col].isna().sum()
        flag_counts = data_imputed[flag_col].value_counts()
        print(f"  {col}: {nan_count} NaN, {flag_counts.to_dict()}")
    
    # Imputar con métodos básicos
    print(f"\nAplicando imputación con promedio de fila...")
    data_imputed = impute_with_row_avg(data_imputed, columns)
    
    # Verificar después de promedio de fila
    print(f"Estado después de promedio de fila:")
    for col in columns:
        flag_col = f"i_{col}"
        flag_counts = data_imputed[flag_col].value_counts()
        print(f"  {col}: {flag_counts.to_dict()}")
    
    # Imputar con persistencia
    print(f"\nAplicando imputación con persistencia...")
    data_imputed = impute_with_persistence(data_imputed, columns)
    
    # Verificar después de persistencia
    print(f"Estado después de persistencia:")
    for col in columns:
        flag_col = f"i_{col}"
        flag_counts = data_imputed[flag_col].value_counts()
        print(f"  {col}: {flag_counts.to_dict()}")
    
    # Imputar con climatología
    print(f"\nAplicando imputación con climatología...")
    data_imputed = impute_with_climatology(data_imputed, climatology, columns)
    
    # Verificar después de climatología
    print(f"Estado después de climatología:")
    for col in columns:
        flag_col = f"i_{col}"
        flag_counts = data_imputed[flag_col].value_counts()
        nan_count = data_imputed[col].isna().sum()
        print(f"  {col}: {flag_counts.to_dict()}, {nan_count} NaN restantes")
    
    # Guardar climatología
    climatology.to_csv(os.path.join(CLIMATOLOGY_FOLDER, f'climatology_{group.replace("cont_", "").replace("_", "")}_7fix.csv'))

# %%
# 6. Actualizar data_df con datos imputados y banderas
cont_columns_imputed = [col for col in data_imputed.columns if col.startswith('cont_') and not col.startswith('i_cont_')]
for col in cont_columns_imputed:
    if col in data_df.columns:
        data_df[col] = data_imputed[col]

flag_columns = [col for col in data_imputed.columns if col.startswith('i_cont_')]
for col in flag_columns:
    data_df[col] = data_imputed[col]

# %%
# 7. Guardar resultados
data_df.to_csv(os.path.join(OUTPUT_FOLDER, 'data_imputed_7_fix_full.csv'))

for year in range(START_YEAR, END_YEAR + 1):
    yearly_data = data_df[data_df.index.year == year]
    yearly_data.to_csv(os.path.join(OUTPUT_FOLDER, f'data_imputed_7_fix_{year}.csv'))

# %%
# 8. Análisis y visualización
resultados = {}
for contaminante in column_groups:
    try:
        climatology, clusters, distancias = analizar_contaminante(
            data_df, contaminante, CLIMATOLOGY_FOLDER
        )
        resultados[contaminante] = {
            'climatology': climatology,
            'clusters': clusters,
            'distancias': distancias
        }
    except Exception as e:
        print(f"Error al analizar {contaminante}: {str(e)}")
        continue

# Visualizar series temporales
for contaminante in column_groups:
    plot_time_series_imputed(data_imputed, contaminante)

# Visualizar comparaciones
for contaminante in column_groups:
    plot_comparison_original_imputed(data_df, data_imputed, contaminante)

guardar_resumen(resultados, CLIMATOLOGY_FOLDER)

# %%
# 9. Limpiar y exportar versión final
data_df_clean = data_df.copy()

# Eliminar columnas WRF
wrf_columns = []
for col in data_df_clean.columns:
    for hour in range(24):
        if col.endswith(f'_h{hour}'):
            wrf_columns.append(col)
            break
data_df_clean = data_df_clean.drop(columns=wrf_columns)

# Eliminar banderas no contaminantes
non_cont_flags = [col for col in data_df_clean.columns if col.startswith('i_') and not col.startswith('i_cont_')]
data_df_clean = data_df_clean.drop(columns=non_cont_flags)

# %%
# 10. Guardar versión limpia
data_df_clean.to_csv(os.path.join(OUTPUT_FOLDER, 'data_imputed_7fix_full.csv'))

for year in range(START_YEAR, END_YEAR + 1):
    yearly_data = data_df_clean[data_df_clean.index.year == year]
    yearly_data.to_csv(os.path.join(OUTPUT_FOLDER, f'data_imputed_7fix_{year}.csv')) 
# %%
