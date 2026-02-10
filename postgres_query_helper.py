#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo específico para consultas a PostgreSQL.
Self-contained y simple para obtener la última fecha de ozono.
"""

import psycopg2
import netrc
from typing import Optional, Tuple
from datetime import datetime, timedelta
import logging

# Configurar logging básico
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PostgresQueryHelper:
    """
    Helper específico para consultas a PostgreSQL.
    Usa configuración de netrc para conexión.
    """
    
    def __init__(self):
        self.connection = None
        self._host = None
        self._user = None
        self._password = None
        self._database = "contingencia"
        
    def _load_credentials(self):
        """Carga credenciales desde archivo netrc."""
        try:
            secrets = netrc.netrc()
            self._user, self._host, self._password = secrets.hosts['AMATE-OPERATIVO']
            logger.info(f"✅ Credenciales cargadas para host: {self._host}")
        except Exception as e:
            logger.error(f"❌ Error cargando credenciales netrc: {e}")
            raise
    
    def connect(self) -> psycopg2.extensions.connection:
        """Establece conexión con PostgreSQL."""
        if not self._host:
            self._load_credentials()
            
        try:
            logger.info("🔌 Conectando a PostgreSQL...")
            self.connection = psycopg2.connect(
                database=self._database,
                user=self._user,
                host=self._host,
                password=self._password
            )
            logger.info(f"✅ Conectado a PostgreSQL en {self._host}")
            return self.connection
        except Exception as e:
            logger.error(f"❌ Error conectando a PostgreSQL: {e}")
            raise
    
    def disconnect(self):
        """Cierra la conexión."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("🔌 Conexión cerrada")
    
    def execute_query(self, query: str) -> Optional[list]:
        """Ejecuta una consulta SQL y retorna resultados."""
        if not self.connection:
            self.connect()
            
        cursor = None
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            return results
        except Exception as e:
            logger.error(f"❌ Error ejecutando consulta: {e}")
            logger.error(f"Query: {query}")
            raise
        finally:
            if cursor:
                cursor.close()
    
    def get_last_ozone_date(self) -> Optional[str]:
        """
        Obtiene la última fecha disponible de ozono en PostgreSQL.
        Busca en cont_otres y aplica delta +1 hora para el target.
        
        Returns:
            str: Fecha en formato 'YYYY-MM-DD HH:MM:SS' o None si no hay datos
        """
        try:
            query = "SELECT MAX(fecha) as ultima_fecha FROM cont_otres"
            logger.info(f"🔍 Ejecutando consulta: {query}")
            
            results = self.execute_query(query)
            
            if results and results[0] and results[0][0] is not None:
                last_date = results[0][0]
                logger.info(f"🗄️ Última fecha en BD (cont_otres): {last_date}")
                                
                # Aplicar delta +1 hora para el target
                if isinstance(last_date, str):
                    from datetime import datetime
                    last_date = datetime.fromisoformat(last_date.replace('Z', '+00:00'))
                elif hasattr(last_date, 'replace'):
                    # Si es un objeto datetime
                    pass
                else:
                    logger.error("❌ Formato de fecha no reconocido")
                    return None
                
                # Aplicar +1 hora
                target_date = last_date + timedelta(hours=1)
                target_str = target_date.strftime('%Y-%m-%d %H:%M:%S')
                logger.info(f"🎯 Target (+1 hora): {target_str}")
                
                return target_str
            else:
                logger.warning("⚠️ No se encontraron fechas en la BD de cont_otres")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error consultando última fecha de ozono: {e}")
            return None


def get_ozone_target_datetime() -> str:
    """
    Función simple y directa para obtener el target de ozono (+1 hora desde última fecha disponible).
    Si falla PostgreSQL, usa fallback del sistema.
    
    Returns:
        str: Fecha y hora en formato 'YYYY-MM-DD HH:MM:SS'
    """
    helper = PostgresQueryHelper()
    
    try:
        # Intentar obtener target desde PostgreSQL (última fecha + 1 hora)
        target_date = helper.get_last_ozone_date()
        if target_date:
            return target_date
            
    except Exception as e:
        logger.error(f"❌ Error con PostgreSQL: {e}")
    
    finally:
        # Siempre cerrar conexión
        helper.disconnect()
    
    # Fallback: usar hora del sistema
    logger.info("🔄 Fallback: usando hora del sistema")
    now = datetime.now()
    last_hour = now.replace(minute=0, second=0, microsecond=0)
    previous_hour = last_hour - timedelta(hours=1)
    
    fallback_time = previous_hour.strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"🕐 Última hora (sistema): {fallback_time}")
    
    return fallback_time


# Función de conveniencia para logging
def log_to_file(message: str):
    """Función simple para logging (compatible con el código existente)."""
    logger.info(message)


if __name__ == "__main__":
    # Test del módulo
    print("🧪 Probando módulo PostgreSQL...")
    try:
        result = get_ozone_target_datetime()
        print(f"✅ Resultado: {result}")
    except Exception as e:
        print(f"❌ Error en test: {e}") 