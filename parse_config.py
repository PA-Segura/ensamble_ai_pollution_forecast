"""
Módulo de compatibilidad para checkpoints antiguos de PyTorch.

Este archivo mantiene la compatibilidad con modelos guardados anteriormente
que referencian 'parse_config' directamente. Re-exporta desde conf.parse_config.
"""

# Re-exportar todo desde conf.parse_config para compatibilidad con checkpoints antiguos
from conf.parse_config import (
    ConfigParser,
    _update_config,
    _get_opt_name,
    _set_by_path,
    _get_by_path
)

# Asegurar que ConfigParser esté disponible directamente
__all__ = ['ConfigParser', '_update_config', '_get_opt_name', '_set_by_path', '_get_by_path']
