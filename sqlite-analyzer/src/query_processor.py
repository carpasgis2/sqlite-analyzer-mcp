import logging
from typing import Dict, Any, Optional, Tuple, List
import json

from .sql_generator import SQLGenerator
from .sql_validator import SQLValidator
from .pipeline import execute_query_with_timeout

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self, db_connector, config_file: str = None):
        """
        Inicializa el procesador de consultas.
        
        Args:
            db_connector: Conector de base de datos
            config_file: Ruta al archivo de configuración (opcional)
        """
        self.db_connector = db_connector
        
        # Cargar configuración
        self.config = self._load_config(config_file)
        
        # Inicializar generador SQL
        self.sql_generator = SQLGenerator(
            self.config.get('allowed_tables', []),
            self.config.get('allowed_columns', {})
        )
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """
        Carga la configuración desde un archivo o usa valores predeterminados.
        
        Args:
            config_file: Ruta al archivo de configuración
            
        Returns:
            Diccionario de configuración
        """
        default_config = {
            'allowed_tables': ['PATI_PATIENTS', 'ONCO_EVENT_INDICATIONS'],
            'allowed_columns': {
                'PATI_PATIENTS': ['PATI_ID', 'PATI_NAME', 'PATI_BIRTH_DATE'],
                'ONCO_EVENT_INDICATIONS': ['EVIN_ID', 'EVIN_DESCRIPTION_ES', 'EVIN_DATE']
            }
        }
        
        if not config_file:
            return default_config
            
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error al cargar configuración: {str(e)}. Usando valores predeterminados.")
            return default_config
    
    def process_llm_query(self, llm_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa una consulta JSON extraída de un LLM.
        
        Args:
            llm_json: Diccionario con la consulta extraída
            
        Returns:
            Resultados de la consulta
        """
        try:
            logger.info(f"Información JSON extraída: {llm_json}")
            
            # Generar SQL
            sql, params = self.sql_generator.generate_sql(llm_json)
            logger.info(f"SQL original: {sql}")
            
            # Si es una consulta parametrizada, registrar los parámetros
            if params:
                logger.info(f"SQL parametrizado: {sql}")
                logger.info(f"Valores: {params}")
            
            # Ejecutar consulta SQL
            logger.info(f"Ejecutando SQL parametrizado: {sql}")
            if params:
                logger.info(f"Valores de parámetros: {params}")
                results, error = execute_query_with_timeout(self.db_connector, sql, params, timeout_seconds=10)
                if error:
                    logger.error(f"Error al ejecutar consulta: {error}")
                    return {
                        'sql': sql,
                        'results': [{'mensaje': f"Error: {error}"}]
                    }
            else:
                results, error = execute_query_with_timeout(self.db_connector, sql, timeout_seconds=10)
                if error:
                    logger.error(f"Error al ejecutar consulta: {error}")
                    return {
                        'sql': sql,
                        'results': [{'mensaje': f"Error: {error}"}]
                    }
            
            return {
                'sql': sql,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error al procesar consulta: {str(e)}")
            return {
                'sql': "SELECT 'Error al procesar consulta' AS mensaje",
                'results': [{'mensaje': f"Error: {str(e)}"}]
            }
