#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_forecast_hourly.py - Ejecutor simple de pronósticos cada hora
"""

import subprocess
import time
import sys
import os
import glob
import sched
from datetime import datetime, timedelta

def clean_nc_files():
    """Limpia archivos .nc del directorio /dev/shm/tem_ram_forecast"""
    nc_path = "/dev/shm/tem_ram_forecast"
    
    try:
        if not os.path.exists(nc_path):
            print(f"📁 El directorio {nc_path} no existe")
            return
            
        # Buscar todos los archivos .nc
        nc_files = glob.glob(os.path.join(nc_path, "*.nc"))
        
        if not nc_files:
            print(f"📄 No se encontraron archivos .nc en {nc_path}")
            return
            
        print(f"🧹 Limpiando {len(nc_files)} archivos .nc de {nc_path}")
        
        # Eliminar cada archivo
        deleted_count = 0
        for file_path in nc_files:
            try:
                os.remove(file_path)
                deleted_count += 1
                print(f"  ✅ Eliminado: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"  ❌ Error eliminando {os.path.basename(file_path)}: {e}")
        
        print(f"🗑️  Limpieza completada: {deleted_count}/{len(nc_files)} archivos eliminados")
        
    except Exception as e:
        print(f"💥 Error en limpieza de archivos .nc: {e}")

def schedule_daily_cleanup(scheduler):
    """Programa la limpieza diaria a las 5:30 AM"""
    now = datetime.now()
    
    # Calcular el próximo 5:30 AM
    next_cleanup = now.replace(hour=5, minute=30, second=0, microsecond=0)
    
    # Si ya pasó el 5:30 AM de hoy, programar para mañana
    if now >= next_cleanup:
        next_cleanup += timedelta(days=1)
    
    # Calcular segundos hasta el próximo 5:30 AM
    delay = (next_cleanup - now).total_seconds()
    
    print(f"📅 Próxima limpieza programada para: {next_cleanup}")
    print(f"⏰ En {delay/3600:.1f} horas")
    
    # Programar la limpieza
    scheduler.enter(delay, 1, run_daily_cleanup, (scheduler,))

def run_daily_cleanup(scheduler):
    """Ejecuta la limpieza diaria y programa la siguiente"""
    print("=" * 50)
    print(f"🧹 HORA DE LIMPIEZA DIARIA - {datetime.now()}")
    clean_nc_files()
    print("=" * 50)
    
    # Programar la siguiente limpieza para mañana
    schedule_daily_cleanup(scheduler)

def run_forecast():
    """Ejecuta operativo_pro_01.py"""
    try:
        print(f"🚀 Ejecutando operativo_pro_01.py...")
        print("=" * 50)
        
        # Ejecutar el proceso directamente en la terminal (sin capturar salida)
        result = subprocess.run([sys.executable, "operativo_pro_01.py"], 
                              timeout=3600)
        
        print("=" * 50)
        if result.returncode == 0:
            print("✅ Pronóstico completado exitosamente")
        else:
            print(f"❌ Error en pronóstico (código {result.returncode})")
            
    except subprocess.TimeoutExpired:
        print("⏰ Timeout: El pronóstico tardó más de 1 hora")
    except Exception as e:
        print(f"💥 Error: {e}")

def main():
    """Función principal"""
    print("🌟 INICIANDO EJECUTOR DE PRONÓSTICOS HORARIOS")
    print(f"🕐 Iniciado: {datetime.now()}")
    
    # Crear scheduler
    scheduler = sched.scheduler(time.time, time.sleep)
    
    # Programar la limpieza diaria
    schedule_daily_cleanup(scheduler)
    
    # Ejecutar inmediatamente la primera vez
    run_forecast()
    
    # Ciclo principal con scheduler
    try:
        while True:
            # Ejecutar eventos programados (incluyendo limpieza diaria)
            scheduler.run(blocking=False)
            
            print("=" * 50)
            print(f"⏳ Esperando 1 hora... ({datetime.now()})")
            time.sleep(3600)  # cada 1 hora = 3600 segundos
            
            print("=" * 50)
            print(f"🔄 NUEVA INFERENCIA - {datetime.now()}")
            run_forecast()
            print("✅ FIN INFERENCIA")
            print("=" * 50)
            
    except KeyboardInterrupt:
        print("\n🛑 Detenido por el usuario")
    except Exception as e:
        print(f"💥 Error en ciclo: {e}")
        print("🔄 Reintentando en 5 minutos...")
        time.sleep(300)

if __name__ == "__main__":
    main() 
