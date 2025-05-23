import re
import time
import sys
import os
from dotenv import load_dotenv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import requests
from functools import lru_cache
from typing import Dict, Any, List, Optional, Tuple, Set
import logging
import json
import sqlite3
import openai  
import difflib
from difflib import get_close_matches  # MOVIDO ARRIBA
from json.decoder import JSONDecodeError
import unicodedata
from pathlib import Path

# Configuración de logging temprana para el módulo pipeline
pipeline_module_logger = logging.getLogger("PipelineModuleInit")
# Asegurarse de que este logger escribe en el archivo de log principal si ya está configurado
# o configurar uno básico si es necesario.
if not logging.getLogger().handlers: # Si el logger raíz no tiene handlers
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")

pipeline_module_logger.info("Iniciando carga del módulo pipeline.py")

# Ajustar el path para ejecutar el script directamente
pipeline_module_logger.info("Comprobando si se ejecuta como script principal...")
if __name__ == "__main__" and __package__ is None:
    pipeline_module_logger.info("Ejecutando como script principal, ajustando sys.path...")
    import sys, os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    pipeline_module_logger.info(f"sys.path ajustado: {sys.path}")

pipeline_module_logger.info("Intentando importar llm_utils...")
# Manejar import cuando se ejecuta como script vs módulo
try:
    from src.llm_utils import call_llm, call_llm_with_fallbacks, extract_info_from_question_llm
    pipeline_module_logger.info("Importado src.llm_utils")
except ImportError:
    from llm_utils import call_llm, call_llm_with_fallbacks, extract_info_from_question_llm
    pipeline_module_logger.info("Importado llm_utils (fallback)")
pipeline_module_logger.info("Intentando importar sql_organizer.parse_join_string...")
from src.sql_organizer import parse_join_string
pipeline_module_logger.info("Importado sql_organizer.parse_join_string")

# logging.getLogger().setLevel(logging.DEBUG) # Esto ya debería estar configurado por el script de prueba o arriba

pipeline_module_logger.info("Configurando variables de archivo...")
# Ajustar el path para incluir el directorio actual
current_dir = os.path.dirname(os.path.abspath(__file__))
DICTIONARY_FILE = os.path.join(current_dir, "data", "dictionary.json")
SCHEMA_ENHANCER_FILE = os.path.join(current_dir, "data", "schema_enhanced.json")
CONNECTION_ENHANCER_CACHE = {}
CONNECTION_RAG_CACHE = {}
pipeline_module_logger.info("Variables de archivo configuradas.")

pipeline_module_logger.info("Intentando importar sql_utils...")
from src.sql_utils import (
    validate_sql_query,
    fix_count_query,
    format_results,
    sanitize_sql_identifier,
    detect_table_from_question,
    validate_where_conditions,
    correct_typos_in_question,
    load_terms_mapping,
    parameterize_query,
    whitelist_validate_query
)
pipeline_module_logger.info("Importado sql_utils.")

pipeline_module_logger.info("Intentando importar table_relationship...")
# Importar módulos de relación de tablas
from src.table_relationship import (
    generate_table_relationships_map, 
    save_table_relationships, 
    load_table_relationships,
    infer_patient_relationship,
    discover_table_relationships
)
pipeline_module_logger.info("Importado table_relationship.")

pipeline_module_logger.info("Intentando importar sql_validator...")
# Importar los nuevos módulos SQL locales
try:
    from src.sql_validator import SQLValidator, whitelist_validate_query as new_whitelist_validate_query
    pipeline_module_logger.info("Importado src.sql_validator.")
except ImportError:
    from sql_validator import SQLValidator, whitelist_validate_query as new_whitelist_validate_query
    pipeline_module_logger.info("Importado sql_validator (fallback).")

pipeline_module_logger.info("Intentando importar más de sql_organizer...")
from src.sql_organizer import (
    normalize_structured_info,
    validate_and_fix_relations,
    enhance_structured_info
)
pipeline_module_logger.info("Importado más de sql_organizer.")

pipeline_module_logger.info("Intentando importar db_relationship_graph...")
from src.db_relationship_graph import (
    build_relationship_graph,
    find_join_path,
    generate_join_path,
    direct_join_exists,
    get_join_definition,
    infer_join_relation_with_llm,
    infer_join_by_convention,
    get_columns_for_table
)
pipeline_module_logger.info("Importado db_relationship_graph.")

pipeline_module_logger.info("Intentando importar rag_enhancements...")
# Importar el sistema RAG mejorado
from src.rag_enhancements import initialize_enhanced_rag, EnhancedSchemaRAG
pipeline_module_logger.info("Importado rag_enhancements.")

pipeline_module_logger.info("Intentando importar SchemaEnhancer...")
from src.schema_enhancer import SchemaEnhancer
pipeline_module_logger.info("Importado SchemaEnhancer.")

pipeline_module_logger.info("Intentando importar DBConnector de db_config...")
# Importar DBConnector para anotaciones de tipo en generate_sql
try:
    from src.db_config import DBConnector
    pipeline_module_logger.info("Importado src.db_config.DBConnector.")
except ImportError:
    from db_config import DBConnector
    pipeline_module_logger.info("Importado db_config.DBConnector (fallback).")

pipeline_module_logger.info("Intentando importar get_db_connector de db_config...")
# Importar helper de conexión y configuración por defecto
try:
    from src.db_config import get_db_connector, DEFAULT_DB_CONFIG
    pipeline_module_logger.info("Importado src.db_config.get_db_connector.")
except ImportError:
    from db_config import get_db_connector, DEFAULT_DB_CONFIG
    pipeline_module_logger.info("Importado db_config.get_db_connector (fallback).")

pipeline_module_logger.info("Intentando importar SQLGenerator...")
from src.sql_generator import SQLGenerator
pipeline_module_logger.info("Importado SQLGenerator.")

pipeline_module_logger.info("Definiendo load_schema_as_string...")
# Definir la función load_schema_as_string localmente
def load_schema_as_string(file_path: str) -> str:
    """Carga el contenido de un archivo de esquema como una cadena."""
    pipeline_module_logger.debug(f"Intentando cargar esquema desde: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            schema_content = f.read()
        pipeline_module_logger.info(f"Esquema cargado exitosamente desde {file_path} ({len(schema_content)} caracteres)")
        return schema_content
    except FileNotFoundError:
        pipeline_module_logger.error(f"Archivo de esquema no encontrado en: {file_path}")
        return "{}"  # Devuelve un JSON vacío como string si no se encuentra
    except Exception as e:
        pipeline_module_logger.error(f"Error al cargar el archivo de esquema {file_path}: {e}")
        return "{}"  # Devuelve un JSON vacío como string en caso de otros errores

pipeline_module_logger.info("Función load_schema_as_string definida localmente.")

pipeline_module_logger.info("Definiendo create_schema_enhancer...")
# Definir la función create_schema_enhancer que faltaba
def create_schema_enhancer(schema_path: str) -> SchemaEnhancer:
    """Crea y devuelve un SchemaEnhancer usando el archivo especificado"""
    return SchemaEnhancer(schema_path)
pipeline_module_logger.info("Función create_schema_enhancer definida.")

pipeline_module_logger.info("Inicializando CONNECTION_ENHANCER_CACHE...")
# Ruta por defecto para la base de datos
# Diccionario global para almacenar instancias de enhancer por ID de conexión
CONNECTION_ENHANCER_CACHE = {} # Ya definido arriba, pero reafirmando.
pipeline_module_logger.info("CONNECTION_ENHANCER_CACHE inicializado.")

pipeline_module_logger.info("Cargando variables de entorno desde .env...")
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    pipeline_module_logger.info(f"Variables de entorno cargadas desde {env_path}")
else:
    pipeline_module_logger.warning(f"Archivo .env no encontrado en {env_path}. Usando variables de entorno del sistema si existen.")

pipeline_module_logger.info("Definiendo rutas de esquema y relaciones...")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # .../sqlite-analyzer/src
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..")) # .../sqlite-analyzer
WORKSPACE_ROOT_DIR = os.path.abspath(os.path.join(PROJECT_ROOT_DIR, "..")) # .../sina_mcp

SCHEMA_DATA_DIR = os.path.join(SCRIPT_DIR, "data") # .../sqlite-analyzer/src/data/
SCHEMA_FULL_PATH = os.path.join(SCHEMA_DATA_DIR, "schema_enhanced.json")
SCHEMA_SIMPLE_PATH = os.path.join(SCHEMA_DATA_DIR, "schema_simple.json") # Asegúrate de que este archivo exista y sea una versión reducida del esquema
RELACIONES_PATH_FOR_LLM = os.path.join(WORKSPACE_ROOT_DIR, "table_relationships_corrected.json") # Usar el archivo corregido
RELACIONES_PATH = os.path.join(WORKSPACE_ROOT_DIR, "table_relationships_corrected.json") # Para build_relationship_graph, etc.

TERMS_DICT_PATH = os.path.join(SCHEMA_DATA_DIR, "dictionary.json") # Ejemplo

pipeline_module_logger.info("Rutas de esquema y relaciones definidas.")

pipeline_module_logger.info("Fin de la carga inicial del módulo pipeline.py")

# Clase de memoria de conversación - Corregida la definición con docstrings adecuados
class ChatMemory:
    """Clase para gestionar el historial de conversación del chatbot"""
    
    def __init__(self, max_history: int = 10):
        """Inicializa el historial de conversación"""
        self.history: List[Dict[str, str]] = []
        self.max_history = max_history

    def add(self, role: str, content: str):
        """Añade un mensaje al historial de la conversación"""
        self.history.append({"role": role, "content": content})
        # Mantener el historial limitado al tamaño máximo
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get(self) -> List[Dict[str, str]]:
        """Obtiene el historial completo de la conversación"""
        return self.history
        
    def clear(self):
        """Limpia el historial de la conversación"""
        self.history = []

# Configuración y logging
def load_config(path: str = "config.ini") -> Dict[str, Any]:
    # ...parsear archivo config.ini con API keys, URLs, modelos...
    return {}

def setup_logging(level: str = "INFO"):
    """Configura el sistema de logging con el nivel especificado"""
    
    # Mapeo de niveles de texto a constantes de logging
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    # Asegurar que el nivel sea válido o usar INFO como fallback
    log_level = levels.get(level.upper(), logging.INFO)
    
    # Configurar el logging con formato detallado y nivel solicitado
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    logging.info(f"Sistema de logging inicializado con nivel: {level}")

# Nueva función para cargar y gestionar el diccionario de términos
def load_terms_dictionary(file_path: str = DICTIONARY_FILE) -> Dict[str, Any]:
    """
    Carga el diccionario de términos que mapea lenguaje natural a elementos SQL.
    
    Args:
        file_path: Ruta al archivo JSON del diccionario
        
    Returns:
        Diccionario con mapeos de términos
    """
    # Utilizamos la función mejorada desde sql_utils
    return load_terms_mapping(file_path)

# Preprocesamiento mejorado con corrección de typos y términos del diccionario
def preprocess_question(question: str, terms_dict: Dict[str, Any] = None) -> Tuple[str, List[str], str]:
    """
    Preprocesa la pregunta para mejorar la extracción de información.
    - Corrige posibles typos utilizando el diccionario de términos
    - Normaliza el texto (lowercase, elimina caracteres especiales)
    - Identifica palabras clave relacionadas con consultas
    - Enriquece con términos del diccionario si está disponible
    - Añade sugerencias y descripciones enriquecidas si están disponibles
    
    Returns:
        Una tupla: (enriched_question_string, identified_tables_list, identified_query_type)
    """
    # Corregir typos si hay términos disponibles
    if terms_dict and 'valid_terms' in terms_dict:
        question, corrections = correct_typos_in_question(question, terms_dict['valid_terms'])
        if corrections:
            logging.info(f"Correcciones aplicadas: {corrections}")
    original_question_for_terms = question  # Guardar la pregunta original para la búsqueda de términos
    question = question.lower().strip()
    question_type = "select"
    if re.search(r'(cuant[oa]s|número|total|contar|count)', question):
        question_type = "count"
    elif re.search(r'(promedio|media|avg)', question):
        question_type = "avg"
    elif re.search(r'(máximo|max|mayor)', question):
        question_type = "max"
    elif re.search(r'(mínimo|min|menor)', question):
        question_type = "min"
    structured_metadata = {}
    filter_patterns = [
        (r'(entre|between)\s+(\d+)\s+y\s+(\d+)', "range"),
        (r'mayor\s+(?:que|a)\s+(\d+)', "greater_than"),
        (r'menor\s+(?:que|a)\s+(\d+)', "less_than"),
        (r'igual\s+a\s+(\d+)', "equals")
    ]
    for pattern, filter_type in filter_patterns:
        match = re.search(pattern, question)
        if match:
            if filter_type == "range":
                structured_metadata["filter"] = {
                    "type": "range",
                    "min": match.group(2),
                    "max": match.group(3)
                }
            elif filter_type == "greater_than":
                structured_metadata["filter"] = {
                    "type": "greater_than",
                    "value": match.group(1)
                }
            elif filter_type == "less_than":
                structured_metadata["filter"] = {
                    "type": "less_than",
                    "value": match.group(1)
                }
            elif filter_type == "equals":
                structured_metadata["filter"] = {
                    "type": "equals",
                    "value": match.group(1)
                }
            break
    enriched_question = f"[QUERY_TYPE: {question_type}]"
    if structured_metadata:
        enriched_question += f" [METADATA: {json.dumps(structured_metadata, ensure_ascii=False)}]"
    enriched_question += f" {original_question_for_terms}"
    
    identified_tables_from_terms = set()  # Usar un set para evitar duplicados
    matched_terms = []             # <--- INICIALIZACIÓN CORREGIDA
    matched_descriptions = []      # <--- INICIALIZACIÓN CORREGIDA
    matched_synonyms = []          # <--- INICIALIZACIÓN CORREGIDA

    # Enriquecer con términos, descripciones y sinónimos del diccionario
    if terms_dict:
        if 'table_mappings' in terms_dict:
            for term, table_data in terms_dict['table_mappings'].items():
                # Verificar que table_data es un diccionario
                if not isinstance(table_data, dict):
                    logging.warning(f"Se esperaba un diccionario para el término '{term}' en 'table_mappings', pero se encontró {type(table_data)}. Omitiendo este término.")
                    continue

                pattern = r'(^|\W)' + re.escape(term) + r'\b'
                try:
                    if re.search(pattern, original_question_for_terms, re.IGNORECASE):
                        table_name = table_data.get('table')
                        if table_name:
                            identified_tables_from_terms.add(table_name)
                            enriched_term = f"- {term}:table:{table_name}"
                            if enriched_term not in matched_terms:
                                matched_terms.append(enriched_term)
                        
                        description = table_data.get('description')
                        if description and description not in matched_descriptions:
                            matched_descriptions.append(f"- {term}:desc:{description}")

                        synonyms = table_data.get('synonyms', [])
                        # Asegurarse de que los sinónimos sean una lista
                        if not isinstance(synonyms, list):
                            logging.warning(f"Se esperaba una lista para los sinónimos del término '{term}', pero se encontró {type(synonyms)}. Omitiendo sinónimos para este término.")
                            synonyms = []
                            
                        for syn in synonyms:
                            syn_entry = f"- {term}:syn:{syn}"
                            if syn_entry not in matched_synonyms:
                                matched_synonyms.append(syn_entry)
                except re.error as e:
                    logging.warning(f"Error de regex con término '{term}' y patrón '{pattern}': {e}")
        else:
            logging.warning("La clave 'table_mappings' no se encontró en terms_dict.")
        
        # Columnas
        if 'column_mappings' in terms_dict:
            for term, col_data in terms_dict['column_mappings'].items():
                pattern = r'(^|\W)' + re.escape(term) + r'\b'
                try:
                    if re.search(pattern, original_question_for_terms, re.IGNORECASE):
                        pass
                except re.error as e:
                    logging.warning(f"Error de regex con término de columna '{term}' y patrón '{pattern}': {e}")
        else:
            logging.warning("La clave 'column_mappings' no se encontró en terms_dict.")

        if matched_terms:
            enriched_question += f" [TERMS: {', '.join(matched_terms)}]"
        if matched_descriptions:
            enriched_question += f" [DESCRIPTIONS: {', '.join(matched_descriptions)}]"
        if matched_synonyms:
            enriched_question += f" [SYNONYMS: {', '.join(matched_synonyms)}]"
        logging.info(f"Términos identificados: {matched_terms}")
        logging.info(f"Descripciones identificadas: {matched_descriptions}")
        logging.info(f"Sinónimos identificados: {matched_synonyms}")
    logging.debug(f"Pregunta preprocesada: {enriched_question}")
    return enriched_question, list(identified_tables_from_terms), question_type

def validate_table_names(sql_query, db_structure):
    """Verifica y corrige nombres de tablas en consulta SQL generada"""
    # --- NUEVO: proteger contra None u otros tipos ---
    if not isinstance(sql_query, str) or not sql_query:
        logging.debug(f"validate_table_names: sql_query inválida ({type(sql_query)}), omitiendo")
        return sql_query

    logging.debug(f"Validando nombres de tablas en la consulta: {sql_query}")
    original_query = sql_query
    table_matches = re.findall(r'\bFROM\s+(\w+)|\bJOIN\s+(\w+)', sql_query, re.IGNORECASE)
    logging.debug(f"Tablas encontradas en la consulta: {table_matches}")
    
    logging.debug(f"Tablas encontradas en la consulta: {table_matches}")
    
    for table_name in table_matches:
        table = next(filter(None, table_name))
        logging.debug(f"Verificando tabla: {table}")
        
        if table not in db_structure:
            logging.warning(f"Tabla '{table}' no encontrada en la estructura de la base de datos")
            match = difflib.get_close_matches(table, list(db_structure.keys()), n=1, cutoff=0.6)
            
            if match:
                logging.info(f"Corrigiendo tabla '{table}' a '{match[0]}'")
                sql_query = sql_query.replace(table, match[0])
            else:
                logging.warning(f"No se encontró una coincidencia adecuada para la tabla '{table}'")
    
    if original_query != sql_query:
        logging.info(f"Consulta corregida: {sql_query}")
    
    return sql_query

def get_schema_enhancer(schema_path: str = SCHEMA_ENHANCER_FILE) -> SchemaEnhancer:
    """
    Obtiene o crea una instancia del SchemaEnhancer.
    
    Args:
        schema_path: Ruta al archivo del esquema mejorado
        
    Returns:
        Instancia de SchemaEnhancer (nueva o existente del caché)
    """
    enhancer_id = schema_path
    
    # Verificar si el path existe antes de intentar cargar
    if not os.path.exists(schema_path):
        logging.warning(f"El archivo de esquema {schema_path} no existe. Se creará al guardar.")
    
    # Recuperar del caché o crear nueva instancia
    if enhancer_id not in CONNECTION_ENHANCER_CACHE:
        logging.info(f"Creando nueva instancia de SchemaEnhancer para {schema_path}")
        try:
            CONNECTION_ENHANCER_CACHE[enhancer_id] = SchemaEnhancer(schema_path)
        except Exception as e:
            logging.error(f"Error al crear SchemaEnhancer: {e}")
            # Crear una instancia vacía como fallback
            CONNECTION_ENHANCER_CACHE[enhancer_id] = SchemaEnhancer(None)
            CONNECTION_ENHANCER_CACHE[enhancer_id].loaded = False
    else:
        logging.debug(f"Usando instancia existente de SchemaEnhancer para {schema_path}")
    
    return CONNECTION_ENHANCER_CACHE[enhancer_id]

def ensure_relationships_map(db_structure: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, str]]:
    """
    Garantiza que exista un mapa de relaciones entre tablas, creándolo si es necesario.
    
    Args:
        db_structure: Diccionario con la estructura de la base de datos
        
    Returns:
        Diccionario con las relaciones entre tablas
    """
    # Crear una clave única para esta estructura de base de datos
    structure_id = hash(frozenset(db_structure.keys()))
    
    # Verificar si ya tenemos el mapa en caché
    if hasattr(ensure_relationships_map, 'cache') and structure_id in ensure_relationships_map.cache:
        logging.debug(f"Usando mapa de relaciones en caché para estructura {structure_id}")
        return ensure_relationships_map.cache[structure_id]
    
    # Intentar cargar relaciones desde archivo
    try:
        relationships = load_table_relationships()
        if relationships:
            logging.info(f"Mapa de relaciones cargado con {len(relationships)} relaciones")
            # Guardar en caché
            if not hasattr(ensure_relationships_map, 'cache'):
                ensure_relationships_map.cache = {}
            ensure_relationships_map.cache[structure_id] = relationships
            return relationships
    except Exception as e:
        logging.warning(f"No se pudo cargar el mapa de relaciones: {e}")
    
    # Si no se pudo cargar, generar nuevas relaciones
    try:
        logging.info("Generando mapa de relaciones entre tablas")
        relationships = generate_table_relationships_map(db_structure)
        
        # Detectar relaciones adicionales mediante análisis de nombres de columnas
        additional_relations = discover_table_relationships(db_structure)
        relationships.update(additional_relations)
        
        # Inferir relaciones con tabla de pacientes si existe
        if "PATI_PATIENTS" in db_structure:
            patient_relations = infer_patient_relationship(db_structure)
            relationships.update(patient_relations)
        
        # Guardar el mapa generado
        try:
            save_table_relationships(relationships)
            logging.info(f"Mapa de relaciones guardado con {len(relationships)} relaciones")
        except Exception as save_error:
            logging.warning(f"No se pudo guardar el mapa de relaciones: {save_error}")
        
        # Guardar en caché
        if not hasattr(ensure_relationships_map, 'cache'):
            ensure_relationships_map.cache = {}
        ensure_relationships_map.cache[structure_id] = relationships
        
        return relationships
    except Exception as e:
        logging.error(f"Error al generar mapa de relaciones: {e}")
        # Devolver un mapa vacío como fallback
        return {}

def load_semantic_mappings():
    """
    Carga los sinónimos y términos comunes de tablas y columnas desde dictionary.json y schema_enhanced.json.
    Devuelve dos diccionarios: {sinonimo/termino: nombre_real_tabla} y {sinonimo/termino: (tabla, columna)}
    """
    table_map = {}
    column_map = {}
    # Cargar dictionary.json
    try:
        with open(DICTIONARY_FILE, encoding="utf-8") as f:
            dct = json.load(f)
            for tname, tinfo in dct.get("tables", {}).items():
                # Agregar nombre real y sinónimos
                table_map[tname.lower()] = tname
                for term in tinfo.get("common_terms", []):
                    clean = term.lower().replace("**términos/sinónimos comunes**:", "").strip("- ")
                    if clean:
                        table_map[clean] = tname
            for cname, cinfo in dct.get("columns", {}).items():
                column_map[cname.lower()] = cinfo.get("table", ""), cname
                for term in cinfo.get("common_terms", []):
                    clean = term.lower().replace("**términos/sinónimos comunes**:", "").strip("- ")
                    if clean:
                        column_map[clean] = (cinfo.get("table", ""), cname)
    except Exception as e:
        logging.warning(f"No se pudo cargar dictionary.json: {e}")
    # Cargar schema_enhanced.json
    try:
        with open(SCHEMA_ENHANCER_FILE, encoding="utf-8") as f:
            schema = json.load(f)
            for tname, tinfo in schema.get("schema_knowledge", {}).get("tables", {}).items():
                table_map[tname.lower()] = tname
                # Extraer sinónimos del campo description (si es JSON)
                desc = tinfo.get("description", "")
                if desc.startswith("```json"):
                    try:
                        desc_json = json.loads(desc.strip("`json \n"))
                        for syn in desc_json.get("sinonimos", []):
                            table_map[syn.lower()] = tname
                    except Exception:
                        pass
            # Columnas (si están presentes)
            for tname, tinfo in schema.get("schema_knowledge", {}).get("tables", {}).items():
                # Si hay columnas
                if "columns" in tinfo:
                    for cname, cinfo in tinfo["columns"].items():
                        column_map[cname.lower()] = (tname, cname)
                        # Sinónimos de columna
                        desc = cinfo.get("description", "")
                        if desc.startswith("```json"):
                            try:
                                desc_json = json.loads(desc.strip("`json \n"))
                                for syn in desc_json.get("sinonimos", []):
                                    column_map[syn.lower()] = (tname, cname)
                            except Exception:
                                pass
    except Exception as e:
        logging.warning(f"No se pudo cargar schema_enhanced.json: {e}")
    return table_map, column_map

def normalize_name(name):
    """Normaliza un nombre quitando acentos y pasando a minúsculas"""
    if not name:
        return ""
    return unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii').lower().strip()

def normalize_table_and_column_names(structured_info):
    """
    Normaliza los nombres de tablas y columnas en structured_info usando los mappings semánticos.
    """
    table_map, column_map = load_semantic_mappings()
    # Tablas
    if "tables" in structured_info:
        new_tables = []
        for t in structured_info["tables"]:
            t_norm = normalize_name(t)
            mapped = table_map.get(t_norm, t)
            new_tables.append(mapped)
        structured_info["tables"] = new_tables
    # Columnas
    if "columns" in structured_info:
        new_columns = []
        for c in structured_info["columns"]:
            c_norm = normalize_name(c)
            mapped = column_map.get(c_norm)
            if mapped:
                new_columns.append(f"{mapped[0]}.{mapped[1]}")
            else:
                new_columns.append(c)
        structured_info["columns"] = new_columns
    # Condiciones
    if "conditions" in structured_info:
        for cond in structured_info["conditions"]:
            if isinstance(cond, dict) and "column" in cond:
                c_norm = normalize_name(cond["column"])
                mapped = column_map.get(c_norm)
                if mapped:
                    cond["column"] = f"{mapped[0]}.{mapped[1]}"
    # Joins
    if "joins" in structured_info:
        for join in structured_info["joins"]:
            if isinstance(join, dict):
                for k in ["table", "foreign_table"]:
                    if k in join:
                        t_norm = normalize_name(join[k])
                        mapped = table_map.get(t_norm, join[k])
                        join[k] = mapped
                for k in ["column", "foreign_column"]:
                    if k in join:
                        c_norm = normalize_name(join[k])
                        mapped = column_map.get(c_norm)
                        if mapped:
                            join[k] = mapped[1]
    return structured_info

def check_table_references(structured_info: dict) -> dict:
    """
    Verifica que todas las tablas referenciadas en condiciones estén en la lista de tablas.
    """
    tables = set(structured_info.get("tables", []))
    tables_in_conditions = set()
    for condition in structured_info.get("conditions", []):
        if isinstance(condition, dict) and "column" in condition:
            if "." in condition["column"]:
                table_name = condition["column"].split(".")[0]
                tables_in_conditions.add(table_name)
        elif isinstance(condition, str):
            matches = re.findall(r'([A-Za-z0-9_]+)\.', condition)
            tables_in_conditions.update(matches)
    missing_tables = tables_in_conditions - tables
    if missing_tables:
        logging.warning(f"Condiciones hacen referencia a tablas no incluidas: {missing_tables}")
        for table in missing_tables:
            if table:
                structured_info.setdefault("tables", []).append(table)
                logging.info(f"Tabla {table} añadida porque está referenciada en condiciones")
    return structured_info

def generate_sql(structured_info: Dict[str, Any], db_structure: Dict[str, Any], 
                 db_connector: 'DBConnector',
                 rag: Optional['EnhancedSchemaRAG'] = None, config: Dict[str, Any] = None) -> str:
    """
    Genera una consulta SQL robusta a partir de la información estructurada, aplicando normalización semántica,
    validación y corrección de nombres, y manejo de errores. Utiliza LLM y RAG si están disponibles.
    """
    try:
        # Importación local de SQLGenerator para evitar ciclos en la carga inicial de módulos
        from src.sql_generator import SQLGenerator

        # 1) Normalizar la estructura y referencias
        structured_info = check_table_references(structured_info)
        structured_info = normalize_table_and_column_names(structured_info)
        
        # 2) Validar que las tablas existan en el esquema
        tablas = structured_info.get("tables", [])
        tablas_validas = [t for t in tablas if t in db_structure]
        if not tablas_validas:
            logging.error(f"Ninguna tabla válida identificada: {tablas}")
            return "SELECT 'No se identificaron tablas válidas' AS mensaje"
        structured_info["tables"] = tablas_validas

        # 3) Generar SQL (usando generador LLM o heurístico)
        sql_query = None
        params = []
        try:
            allowed_tables_list = list(db_structure.keys())
            allowed_columns_map = {table: list(cols.keys()) for table, cols in db_structure.items()}
            sql_gen = SQLGenerator(allowed_tables=allowed_tables_list, allowed_columns=allowed_columns_map)

            sql_query, params = sql_gen.generate_sql(structured_info)

            logging.info("DEBUG: [pipeline.py] JUST RETURNED FROM sql_gen.generate_sql()")
            print("STDOUT DEBUG: [pipeline.py] JUST RETURNED FROM sql_gen.generate_sql()")

            logging.info(f"DEBUG: [pipeline.py] SQL Query Type: {type(sql_query)}, Value: {sql_query}")
            print(f"STDOUT DEBUG: [pipeline.py] SQL Query Type: {type(sql_query)}, Value: {sql_query}")
            logging.info(f"DEBUG: [pipeline.py] Params Type: {type(params)}, Value: {params}")
            print(f"STDOUT DEBUG: [pipeline.py] Params Type: {type(params)}, Value: {params}")

            if not sql_query: # Comprobación si la consulta es None o vacía
                logging.error("[pipeline.py] SQLGenerator devolvió una consulta vacía o None.")
                print("STDOUT ERROR: [pipeline.py] SQLGenerator devolvió una consulta vacía o None.")
                return "SELECT 'Error: No se pudo generar la consulta SQL' AS mensaje"
            
            logging.info(f"DEBUG: [pipeline.py] SQL generado (después de chequeo): {sql_query}")
            print(f"STDOUT DEBUG (pipeline.py): SQL generado (después de chequeo): {sql_query}")
            logging.info(f"DEBUG: [pipeline.py] Parámetros (después de chequeo): {params}")
            print(f"STDOUT DEBUG (pipeline.py): Parámetros (después de chequeo): {params}")

        except Exception as e_gensql:
            logging.error(f"DEBUG: [pipeline.py] EXCEPCIÓN durante o inmediatamente después de generate_sql: {e_gensql}", exc_info=True)
            print(f"STDOUT DEBUG: [pipeline.py] EXCEPCIÓN durante o inmediatamente después de generate_sql: {e_gensql}")
            return "SELECT 'Error durante la generación de SQL' AS mensaje"

        # 4) Corregir nombres de columnas/tablas en el SQL
        sql_query = correct_column_names(sql_query, tablas_validas, db_structure)
        sql_query = validate_table_names(sql_query, db_structure)

        # 5) Validar el SQL final
        if not validate_sql_query(sql_query, db_connector):
            logging.error(f"SQL inválido tras corrección: {sql_query}")
            return "SELECT 'Error: SQL inválido tras corrección' AS mensaje"

        # 6) Devolver la consulta final
        return sql_query

    except Exception as e:
        logging.error(f"Error en generate_sql: {e}")
        return "SELECT 'Error al generar SQL' AS mensaje"

def correct_column_names(sql_query, tables, db_structure):
    # --- NUEVO: Normalizar nombres de columnas y tablas usando mappings semánticos antes de aplicar heurísticas ---
    table_map, column_map = load_semantic_mappings()
    # Corregir nombres de tablas en la consulta (sinónimos y fuzzy)
    for t in tables:
        t_norm = normalize_name(t)
        mapped = table_map.get(t_norm)
        if mapped and mapped != t:
            sql_query = re.sub(rf'\b{re.escape(t)}\b', mapped, sql_query)
        # Fuzzy: si no hay match directo, buscar parecido
        elif not mapped:
            close = difflib.get_close_matches(t_norm, table_map.keys(), n=1, cutoff=0.7)
            if close:
                sql_query = re.sub(rf'\b{re.escape(t)}\b', table_map[close[0]], sql_query)
    # Corregir nombres de columnas en la consulta (sinónimos y fuzzy)
    for (tbl, col) in column_map.values():
        if col and col in sql_query:
            continue  # Ya está bien
        # Buscar sinónimos presentes en la consulta
        for syn, (syn_tbl, syn_col) in column_map.items():
            if syn_col and re.search(rf'\b{re.escape(syn)}\b', sql_query, re.IGNORECASE):
                sql_query = re.sub(rf'\b{re.escape(syn)}\b', syn_col, sql_query, flags=re.IGNORECASE)
            # Fuzzy: si no hay match directo, buscar parecido
            else:
                close = difflib.get_close_matches(syn, [col.lower() for (_, col) in column_map.values() if col], n=1, cutoff=0.7)
                if close and close[0] not in [col]:
                    sql_query = re.sub(rf'\b{re.escape(syn)}\b', close[0], sql_query, flags=re.IGNORECASE)
    # Validación clásica: si aún quedan columnas/tablas no válidas, intentar heurística clásica
    # 1. Crear mapeo de columnas válidas por tabla
    valid_columns = {}
    for table in tables:
        if table in db_structure:
            valid_columns[table] = [col.get("name", "") for col in db_structure[table].get("columns", [])]
    # 2. Buscar referencias a columnas no válidas y corregir
    for table in tables:
        if table in db_structure:
            table_pattern = rf'(?:\b{re.escape(table)}\b|\b[a-zA-Z0-9_]+\b)\.([a-zA-Z0-9_]+)'
            for match in re.finditer(table_pattern, sql_query, re.IGNORECASE):
                col_name = match.group(1)
                if col_name in valid_columns[table]:
                    continue
                best_match = difflib.get_close_matches(col_name, valid_columns[table], n=1, cutoff=0.6)
                if best_match:
                    corrected_name = best_match[0]
                    sql_query = sql_query.replace(f"{col_name}", f"{corrected_name}")
                    logging.info(f"Corregido nombre de columna similar: '{col_name}' → '{corrected_name}' en tabla {table}")
    return sql_query

def chatbot_pipeline(
    question: str, 
    db_connector: Any, 
    config: Optional[Dict[str, Any]] = None, 
    user_id: Optional[str] = None, 
    session_id: Optional[str] = None, 
    tablas_relevantes: Optional[List[str]] = None, 
    condiciones_relevantes: Optional[List[Dict[str, Any]]] = None, # Nuevo parámetro
    max_retries_on_error: int = 1,
    max_retries_on_empty: int = 1,
    use_rag_enhancement: bool = True,
    use_schema_enhancer: bool = True,
    use_dynamic_examples: bool = True,
    use_direct_join_enhancement: bool = True,
    use_relationship_inference: bool = True,
    use_llm_join_inference: bool = True,
    use_convention_join_inference: bool = True,
    force_regenerate_schema: bool = False,
    force_regenerate_relationships: bool = False,
    output_intermediate_steps: bool = False,
    memory: Optional[Any] = None, # Reemplazar Any con ChatMemory si está definida
    db_path_param: Optional[str] = None,
    schema_path_param: Optional[str] = None,
    relationships_path_param: Optional[str] = None,
    log_level: str = "INFO",
    debug_prompts: bool = False,
    debug_responses: bool = False,
    debug_sql: bool = False,
    debug_data: bool = False,
    debug_flow: bool = False,
    debug_cache: bool = False,
    debug_all: bool = False
) -> Dict[str, Any]:
    """Procesa una pregunta en lenguaje natural, la convierte en una consulta SQL y la ejecuta."""
    logger = logging.getLogger(__name__) 
    logger.info(f"DEBUG: [pipeline.py] Inicio de chatbot_pipeline. Pregunta: '{question}', Tablas Relevantes: {tablas_relevantes}, Condiciones: {condiciones_relevantes}")
    print(f"DEBUG: [pipeline.py] Inicio de chatbot_pipeline. Pregunta: '{question}', Tablas Relevantes: {tablas_relevantes}, Condiciones: {condiciones_relevantes} (stdout)")

    # Inicializar structured_info a un diccionario vacío si aún no está definido.
    structured_info: Dict[str, Any] = {}
    initial_query_type = None # Para almacenar el query_type de preprocess_question
    initial_identified_tables = [] # Para almacenar las tablas de preprocess_question

    # Inicializar db_structure y terms_dict al principio para que estén disponibles en ambas ramas
    logger.info("DEBUG: [pipeline.py] Obteniendo estructura de la BD (común)...")
    db_structure = db_connector.get_database_structure() # type: ignore
    if not db_structure:
        logger.error("DEBUG: [pipeline.py] No se pudo obtener la estructura de la base de datos. Saliendo temprano.")
        return {"response": "Error crítico: No se pudo acceder a la estructura de la base de datos.", "sql": None, "tables": [], "columns": [], "data": None, "intermediate_steps": {"error_db_structure": "db_structure is None or empty"}}

    if db_structure and 'ACCI_PATIENT_CONDITIONS' in db_structure:
        logger.info(f"DEBUG: [pipeline.py] Estructura INICIAL de ACCI_PATIENT_CONDITIONS: {db_structure['ACCI_PATIENT_CONDITIONS']}")
        if isinstance(db_structure['ACCI_PATIENT_CONDITIONS'], dict) and 'columns' in db_structure['ACCI_PATIENT_CONDITIONS']:
            columns_list = db_structure['ACCI_PATIENT_CONDITIONS'].get('columns', [])
            column_names = [col.get('name') for col in columns_list if isinstance(col, dict)]
            logger.info(f"DEBUG: [pipeline.py] Columnas INICIALES de ACCI_PATIENT_CONDITIONS: {column_names}")
            if "PATI_ID" in column_names:
                logger.info("DEBUG: [pipeline.py] PATI_ID encontrado en columnas INICIALES de ACCI_PATIENT_CONDITIONS.")
            else:
                logger.info("DEBUG: [pipeline.py] PATI_ID NO encontrado en columnas INICIALES de ACCI_PATIENT_CONDITIONS.")
        else:
            logger.info(f"DEBUG: [pipeline.py] Estructura de columnas INICIAL de ACCI_PATIENT_CONDITIONS es inválida o no encontrada. Tipo de 'columns': {type(db_structure['ACCI_PATIENT_CONDITIONS'].get('columns'))}")
    elif db_structure:
        logger.info("DEBUG: [pipeline.py] ACCI_PATIENT_CONDITIONS no encontrada en la estructura INICIAL de la BD.")
    else:
        logger.info("DEBUG: [pipeline.py] db_structure es None después de la carga inicial.")

    logger.info("DEBUG: [pipeline.py] Cargando diccionario de términos (común)...")
    terms_dict = load_terms_dictionary()

    try:
        if tablas_relevantes:
            logger.info(f"Paso FORZADO: Usando tablas relevantes proporcionadas: {tablas_relevantes}")
            structured_info = {
                "tables": tablas_relevantes,
                "columns": ["*"], 
                "conditions": [],
                "joins": [],
                "query_type": "SELECT",
                "order_by": [],
                "limit": None,
                "group_by": [],
                "having": []
            }
            if condiciones_relevantes:
                logger.info(f"Paso FORZADO: Usando condiciones relevantes proporcionadas: {condiciones_relevantes}")
                structured_info["conditions"] = condiciones_relevantes
            
            logger.info(f"Paso 1 y 2 OMITIDOS (extracción de información por LLM) debido a tablas/condiciones forzadas.")
            logger.info(f"structured_info inicializado con tablas/condiciones forzadas: {structured_info}")
            
            # Normalizar y validar la información estructurada inicial
            structured_info = normalize_structured_info(structured_info)
            logger.info(f"structured_info después de normalización inicial (forzado): {structured_info}")
            
            # Se omite la parte de preprocess_question y enhance_structured_info 
            # ya que la información viene forzada y el LLM no interviene en su formación inicial.
            # También se omite la inicialización de RAG para este flujo simplificado por ahora,
            # a menos que sea estrictamente necesario para SQLGenerator.

            # structured_info_copy es lo que se pasará a SQLGenerator
            structured_info_copy = structured_info.copy() if isinstance(structured_info, dict) else {}

        else: # Flujo original cuando no hay tablas_relevantes forzadas
            logger.info("DEBUG: [pipeline.py] Paso 1: Obteniendo estructura de la BD...")
            # db_structure ya se obtuvo arriba
            logger.info("DEBUG: [pipeline.py] Paso 1 completado. {len(db_structure) if isinstance(db_structure, dict) else 'N/A'} tablas obtenidas.")

            logger.info("DEBUG: [pipeline.py] Paso 2: Cargando diccionario de términos...")
            # terms_dict ya se obtuvo arriba
            logger.info("DEBUG: [pipeline.py] Paso 2 completado. Diccionario de términos cargado.")

            logger.info("DEBUG: [pipeline.py] Paso 3: Preprocesando pregunta...")
            enriched_question, initial_identified_tables, initial_query_type = preprocess_question(question, terms_dict)
            logger.info(f"DEBUG: [pipeline.py] Paso 3 completado. Pregunta enriquecida: {enriched_question}")
            logger.info(f"DEBUG: [pipeline.py] Tablas identificadas en Paso 3: {initial_identified_tables}")
            logger.info(f"DEBUG: [pipeline.py] Tipo de consulta identificado en Paso 3: {initial_query_type}")
            
            logger.info("DEBUG: [pipeline.py] Paso 4: Determinando structured_info (tablas) con LLM...")
            
            # Asegurarse de que structured_info esté inicializado antes de este bloque si no lo estaba ya.
            # Esta es una salvaguarda adicional, aunque ya se inicializa arriba.
            if structured_info is None: # Aunque ya se inicializó arriba, por si acaso.
                structured_info = {}

            # --- INICIO CAMBIO: Poblar structured_info con valores iniciales de preprocess_question ---
            # Esto sirve como fallback si el LLM no identifica tablas o tipo de consulta.
            if not structured_info.get("tables") and initial_identified_tables:
                structured_info["tables"] = initial_identified_tables
                logger.info(f"DEBUG: [pipeline.py] structured_info['tables'] poblado con tablas de Paso 3: {initial_identified_tables}")

            if not structured_info.get("query_type") and initial_query_type:
                structured_info["query_type"] = initial_query_type
                logger.info(f"DEBUG: [pipeline.py] structured_info['query_type'] poblado con tipo de consulta de Paso 3: {initial_query_type}")
            # --- FIN CAMBIO ---

            db_schema_str_simple = load_schema_as_string(SCHEMA_SIMPLE_PATH)
            if not db_schema_str_simple:
                logger.warning(f"ADVERTENCIA: No se pudo cargar el esquema simple desde {SCHEMA_SIMPLE_PATH}. Se usará un JSON vacío.")
                db_schema_str_simple = "{}" 
            
            relaciones_tablas_str = load_schema_as_string(RELACIONES_PATH_FOR_LLM)
            if not relaciones_tablas_str:
                logger.warning(f"ADVERTENCIA: No se pudieron cargar las relaciones desde {RELACIONES_PATH_FOR_LLM}. Se usará un JSON vacío.")
                relaciones_tablas_str = "{}"
            
            db_schema_str_full_details = load_schema_as_string(SCHEMA_FULL_PATH)
            if not db_schema_str_full_details:
                logger.warning(f"ADVERTENCIA: No se pudo cargar el esquema completo desde {SCHEMA_FULL_PATH}. La normalización podría ser menos precisa.")
                # Si el esquema completo no se carga, normalizamos structured_info aquí.
                # structured_info podría no estar completamente poblado por el LLM si este paso falla.
                structured_info = normalize_structured_info(structured_info) # Asegura que sea un dict con claves esperadas
            
            if not isinstance(structured_info, dict):
                logger.error(f"DEBUG: [pipeline.py] Error: structured_info no es un dict después de la interacción con LLM y antes de la normalización final del Paso 4. Valor: {structured_info}")
                logger.info(f"DEBUG: [pipeline.py] Reintentando normalización o inicialización de structured_info.")
                structured_info = normalize_structured_info(structured_info if isinstance(structured_info, dict) else {})
                if not isinstance(structured_info, dict): # Si sigue sin ser un dict, es un error grave
                    return {"response": f"Error crítico: La información estructurada no es válida tras el LLM. Valor: {structured_info}", "sql": None, "tables": [], "columns": []}

            # --- INICIO CAMBIO: Reforzar tablas y query_type después del LLM ---
            # Si el LLM eliminó las tablas o el query_type, y teníamos unos iniciales, los restauramos.
            if not structured_info.get("tables") and initial_identified_tables:
                logger.info(f"DEBUG: [pipeline.py] LLM no devolvió tablas. Restaurando desde Paso 3: {initial_identified_tables}")
                structured_info["tables"] = initial_identified_tables
            
            if not structured_info.get("query_type") and initial_query_type:
                logger.info(f"DEBUG: [pipeline.py] LLM no devolvió query_type. Restaurando desde Paso 3: {initial_query_type}")
                structured_info["query_type"] = initial_query_type
            
            # Asegurar que query_type exista, por defecto a 'SELECT' si aún no está.
            if not structured_info.get("query_type"):
                logger.warning(f"DEBUG: [pipeline.py] query_type sigue ausente. Estableciendo a 'SELECT' por defecto.")
                structured_info["query_type"] = "SELECT"
            # --- FIN CAMBIO ---

            logger.info(f"DEBUG: [pipeline.py] structured_info ANTES de Paso 5 (normalización final del LLM): {structured_info}")
            structured_info = normalize_structured_info(structured_info) # Normaliza de nuevo para asegurar estructura
            logger.info(f"DEBUG: [pipeline.py] structured_info DESPUÉS de Paso 5 (normalización final del LLM): {structured_info}")

            if not isinstance(structured_info, dict) or not structured_info.get("tables"):
                 # Si después de todo esto, no tenemos tablas, es un problema.
                logger.error(f"DEBUG: [pipeline.py] Error CRÍTICO: structured_info no es un dict o no contiene tablas después del Paso 5. Valor: {structured_info}")
                if initial_identified_tables: # Último recurso
                    logger.warning(f"DEBUG: [pipeline.py] REINTENTO FINAL: Usando tablas de Paso 3: {initial_identified_tables}")
                    structured_info["tables"] = initial_identified_tables
                    structured_info["query_type"] = initial_query_type if initial_query_type else "SELECT"
                else:
                    return {"response": "Error: No se pudieron identificar las tablas necesarias para la consulta.", "sql": None, "tables": [], "columns": []}

            logger.info("DEBUG: [pipeline.py] Paso 5 completado.")

            logger.info("DEBUG: [pipeline.py] Paso 6: Mejorando structured_info...")
            # Guardar el query_type antes de enhance_structured_info, ya que a veces lo modifica incorrectamente.
            current_query_type = structured_info.get("query_type", initial_query_type) # Priorizar el del LLM si existe
            
            structured_info = enhance_structured_info(structured_info, enriched_question)
            
            if not isinstance(structured_info, dict):
                logger.error("DEBUG: [pipeline.py] Error: structured_info no es un dict después de enhance_structured_info.")
                # Intentar recuperar usando la copia antes de enhance o los valores iniciales
                structured_info = normalize_structured_info(structured_info_copy_before_enhance if 'structured_info_copy_before_enhance' in locals() else {})
                if not structured_info.get("tables") and initial_identified_tables: structured_info["tables"] = initial_identified_tables
                if not structured_info.get("query_type") and current_query_type: structured_info["query_type"] = current_query_type
                if not structured_info.get("query_type"): structured_info["query_type"] = "SELECT"

                if not isinstance(structured_info, dict) or not structured_info.get("tables"):
                     return {"response": f"Error al estructurar la consulta (mejora): {structured_info}", "sql": None, "tables": [], "columns": []}

            # --- INICIO CAMBIO: Restaurar query_type si enhance_structured_info lo alteró indebidamente ---
            if current_query_type and structured_info.get("query_type") != current_query_type:
                logger.warning(f"DEBUG: [pipeline.py] query_type fue modificado por enhance_structured_info de '{structured_info.get('query_type')}' a '{current_query_type}'. Restaurando.")
                structured_info["query_type"] = current_query_type
            elif not structured_info.get("query_type") and current_query_type: # Si enhance lo eliminó
                logger.warning(f"DEBUG: [pipeline.py] query_type fue eliminado por enhance_structured_info. Restaurando a '{current_query_type}'.")
                structured_info["query_type"] = current_query_type
            elif not structured_info.get("query_type"): # Si sigue sin existir
                 logger.warning(f"DEBUG: [pipeline.py] query_type sigue ausente después de enhance. Estableciendo a 'SELECT'.")
                 structured_info["query_type"] = "SELECT"
            # --- FIN CAMBIO ---
            
            # Asegurar que las tablas iniciales estén presentes si structured_info las perdió
            if not structured_info.get("tables") and initial_identified_tables:
                logger.warning(f"DEBUG: [pipeline.py] Tablas perdidas después de enhance_structured_info. Restaurando desde Paso 3: {initial_identified_tables}")
                structured_info["tables"] = initial_identified_tables

            logger.info(f"DEBUG: [pipeline.py] structured_info DESPUÉS de enhance_structured_info (y corrección de query_type): {structured_info}")
            logger.info("DEBUG: [pipeline.py] Paso 6 completado.")

            # structured_info_copy es lo que se pasará a SQLGenerator
            # Crear la copia DESPUÉS de todas las manipulaciones de structured_info
            structured_info_copy = {}
            if isinstance(structured_info, dict):
                structured_info_copy = structured_info.copy()
            else: # Fallback por si structured_info no es un dict a estas alturas
                logger.error("ERROR CRÍTICO: structured_info no es un diccionario antes de crear structured_info_copy.")
                structured_info_copy["tables"] = initial_identified_tables if initial_identified_tables else []
                structured_info_copy["query_type"] = initial_query_type if initial_query_type else "SELECT"
                structured_info_copy["conditions"] = [] # Añadir otras claves necesarias con valores por defecto

            # Asegurar que structured_info_copy tenga los elementos mínimos
            if "tables" not in structured_info_copy: structured_info_copy["tables"] = initial_identified_tables if initial_identified_tables else []
            if "query_type" not in structured_info_copy: structured_info_copy["query_type"] = initial_query_type if initial_query_type else "SELECT"
            if "columns" not in structured_info_copy: structured_info_copy["columns"] = []
            if "conditions" not in structured_info_copy: structured_info_copy["conditions"] = []
            if "joins" not in structured_info_copy: structured_info_copy["joins"] = []
            # ... añadir otras claves que SQLGenerator espera con valores por defecto

        # --- INICIO: Lógica para Multi-Hop Joins ---
        if structured_info and isinstance(structured_info.get("tables"), list) and len(structured_info["tables"]) > 1:
            logger.info("DEBUG: [pipeline.py] Iniciando lógica de Multi-Hop Joins.")
            
            relationships_from_dict = terms_dict.get("table_relationships") 
            
            if not relationships_from_dict:
                logger.warning("DEBUG: [pipeline.py] 'table_relationships' no encontrado en dictionary.json. No se puede realizar Multi-Hop Join avanzado.")
            else:
                formatted_relationships_for_graph = {}
                if isinstance(relationships_from_dict, list): # Asegurar que es una lista
                    for rel in relationships_from_dict:
                        if isinstance(rel, dict): # Asegurar que cada elemento es un dict
                            from_table = rel.get("from_table")
                            if from_table: # Asegurar que from_table existe
                                if from_table not in formatted_relationships_for_graph:
                                    formatted_relationships_for_graph[from_table] = []
                                
                                # Solo añadir si la relación tiene la información necesaria
                                if rel.get("from_column") and rel.get("to_table") and rel.get("to_column"):
                                    formatted_relationships_for_graph[from_table].append({
                                        "column": rel.get("from_column"),
                                        "foreign_table": rel.get("to_table"),
                                        "foreign_column": rel.get("to_column"),
                                        "type": "FK_DICT_JSON"
                                    })
                                else:
                                    logger.warning(f"DEBUG: [pipeline.py] Relación incompleta en dictionary.json para from_table '{from_table}': {rel}")
                            else:
                                logger.warning(f"DEBUG: [pipeline.py] Relación sin 'from_table' en dictionary.json: {rel}")
                        else:
                            logger.warning(f"DEBUG: [pipeline.py] Elemento no diccionario en 'table_relationships': {rel}")
                else:
                    logger.warning(f"DEBUG: [pipeline.py] 'table_relationships' en dictionary.json no es una lista: {type(relationships_from_dict)}")

                logger.debug(f"DEBUG: [pipeline.py] Relaciones formateadas para grafo: {formatted_relationships_for_graph}")
                
                relationship_graph = build_relationship_graph(db_connector, table_relationships=formatted_relationships_for_graph, db_structure=db_structure)
                logger.debug(f"DEBUG: [pipeline.py] Grafo de relaciones construido para Multi-Hop: {relationship_graph}")

                current_tables_in_query = set(structured_info.get("tables", []))
                existing_joins = structured_info.get("joins", [])
                
                for join_def in existing_joins:
                    if isinstance(join_def, dict):
                        current_tables_in_query.add(join_def.get("table"))
                        current_tables_in_query.add(join_def.get("foreign_table"))
                current_tables_in_query.discard(None)

                if len(current_tables_in_query) > 1:
                    main_query_table = structured_info["tables"][0] if structured_info.get("tables") else (list(current_tables_in_query)[0] if current_tables_in_query else None)

                    if main_query_table:
                        covered_by_join = {main_query_table}
                        for join_def in existing_joins:
                            if isinstance(join_def, dict):
                                covered_by_join.add(join_def.get("table"))
                                covered_by_join.add(join_def.get("foreign_table"))
                        covered_by_join.discard(None)

                        newly_added_joins = []
                        
                        tables_to_connect_iteratively = list(current_tables_in_query - covered_by_join)
                        
                        # Bucle para intentar conectar tablas progresivamente
                        max_connection_passes = len(tables_to_connect_iteratively) + 1 # Como máximo, tantas pasadas como tablas a conectar
                        for _pass_num in range(max_connection_passes):
                            if not tables_to_connect_iteratively: # Todas conectadas
                                break
                            
                            logger.debug(f"DEBUG: [pipeline.py] Pase de conexión Multi-Hop #{_pass_num + 1}. Tablas pendientes: {tables_to_connect_iteratively}, Cubiertas: {covered_by_join}")
                            
                            connected_in_this_pass = set()

                            for target_table_to_connect in list(tables_to_connect_iteratively): # Iterar sobre copia para poder modificar original
                                if target_table_to_connect in covered_by_join: 
                                    connected_in_this_pass.add(target_table_to_connect)
                                    continue

                                best_path_found = None
                                
                                for origin_table_already_covered in list(covered_by_join):
                                    if origin_table_already_covered == target_table_to_connect: continue

                                    logger.debug(f"DEBUG: [pipeline.py] Buscando camino de {origin_table_already_covered} a {target_table_to_connect}")
                                    path = find_join_path(relationship_graph, origin_table_already_covered, target_table_to_connect, max_depth=config.get("max_join_depth", 3))
                                    
                                    if path and (best_path_found is None or len(path) < len(best_path_found)):
                                        best_path_found = path
                                
                                if best_path_found:
                                    logger.info(f"DEBUG: [pipeline.py] Camino Multi-Hop encontrado para {target_table_to_connect}: {best_path_found}")
                                    path_joins = generate_join_path(relationship_graph, best_path_found) # Asumo que generate_join_path usa el grafo para obtener cols
                                    logger.debug(f"DEBUG: [pipeline.py] Joins para el camino: {path_joins}")
                                    
                                    for p_join in path_joins:
                                        is_duplicate = False
                                        for ex_join in existing_joins + newly_added_joins:
                                            if (isinstance(ex_join, dict) and isinstance(p_join, dict) and
                                                {ex_join.get("table"), ex_join.get("foreign_table")} == {p_join.get("table"), p_join.get("foreign_table")} and
                                                ex_join.get("column") == p_join.get("column") and
                                                ex_join.get("foreign_column") == p_join.get("foreign_column")):
                                                is_duplicate = True
                                                break
                                        if not is_duplicate:
                                            newly_added_joins.append(p_join)
                                            if isinstance(p_join, dict): # Añadir tablas del join a cubiertas
                                                covered_by_join.add(p_join.get("table"))
                                                covered_by_join.add(p_join.get("foreign_table"))
                                                covered_by_join.discard(None)
                                    
                                    connected_in_this_pass.add(target_table_to_connect) # Marcar como conectada en este pase
                                else:
                                    logger.warning(f"DEBUG: [pipeline.py] No se encontró camino para conectar {target_table_to_connect} en este pase.")
                            
                            # Actualizar la lista de tablas pendientes
                            tables_to_connect_iteratively = [t for t in tables_to_connect_iteratively if t not in connected_in_this_pass]

                        if newly_added_joins:
                            logger.info(f"DEBUG: [pipeline.py] Añadiendo {len(newly_added_joins)} JOINs inferidos por Multi-Hop: {newly_added_joins}")
                            structured_info.setdefault("joins", []).extend(newly_added_joins)
                            for join_def in newly_added_joins:
                                 if isinstance(join_def, dict):
                                    structured_info.setdefault("tables", []).append(join_def.get("table"))
                                    structured_info.setdefault("tables", []).append(join_def.get("foreign_table"))
                            structured_info["tables"] = sorted(list(set(filter(None, structured_info.get("tables",[])))))
                        
                        if tables_to_connect_iteratively: # Si aún quedan tablas sin conectar
                            logger.warning(f"DEBUG: [pipeline.py] No se pudieron conectar todas las tablas requeridas. Pendientes: {tables_to_connect_iteratively}")


        logger.info(f"DEBUG: [pipeline.py] structured_info ANTES de SQLGenerator (después de Multi-Hop): {structured_info}")
        # --- FIN: Lógica para Multi-Hop Joins ---

        # --- Flujo común para generación de SQL y ejecución --- 
        logger.info(f"DEBUG: [pipeline.py] Paso 8: Generando SQL... (Flujo común)")
        print(f"STDOUT DEBUG: [pipeline.py] Paso 8: Generando SQL... (Flujo común)", flush=True)
        
        logger.info(f"DEBUG: [pipeline.py] structured_info_copy ANTES de SQLGenerator: {structured_info_copy}")
        print(f"STDOUT DEBUG: [pipeline.py] structured_info_copy ANTES de SQLGenerator: {structured_info_copy}", flush=True)

        sql_query = None
        params = []
        
        try: # Línea 875
            if not db_structure: # Comprobación crucial añadida aquí también por si acaso, aunque ya está arriba.
                logger.error("[pipeline.py] db_structure no está disponible en el bloque de generación de SQL. No se puede proceder.")
                return {
                    "response": "Error interno: la estructura de la base de datos no está disponible para la generación de SQL.",
                    "sql": None, "tables": [], "columns": [], "data": None,
                    "intermediate_steps": {"error_db_structure_pre_sqlgen": "db_structure is None or empty"}
                }

            allowed_tables_list = list(db_structure.keys())
            allowed_columns_map = {
                table_name: [
                    col_data['name']
                    for col_data in table_details.get('columns', [])
                    if isinstance(col_data, dict) and 'name' in col_data
                ]
                for table_name, table_details in db_structure.items()
                if isinstance(table_details, dict)
            }
            sql_gen_instance = SQLGenerator(allowed_tables=allowed_tables_list, allowed_columns=allowed_columns_map)

            current_sql_query, current_params = sql_gen_instance.generate_sql(structured_info_copy) # Línea 880

            # Inicio de la corrección del IndentationError y la lógica faltante
            sql_query = current_sql_query
            params = current_params

            if not sql_query:
                logger.error("[pipeline.py] SQLGenerator devolvió una consulta vacía o None en el flujo común.")
                # Considerar devolver un error o una respuesta indicativa aquí si es crítico
                # return {"response": "Error: No se pudo generar la consulta SQL (flujo común).", "sql": None, "tables": [], "columns": [], "data": None, "intermediate_steps": {}}

            logger.info(f"DEBUG: [pipeline.py] SQL generado (flujo común): {sql_query}")
            logger.info(f"DEBUG: [pipeline.py] Parámetros (flujo común): {params}")

        except Exception as e_gensql_common:
            logger.error(f"DEBUG: [pipeline.py] EXCEPCIÓN durante la generación de SQL (flujo común): {e_gensql_common}", exc_info=True)
            return {
                "response": f"Lo siento, ocurrió un error al intentar generar la consulta SQL: {e_gensql_common}",
                "sql": None,
                "tables": structured_info_copy.get("tables", []),
                "columns": structured_info_copy.get("columns", []),
                "data": None,
                "intermediate_steps": {"error_generating_sql_common": str(e_gensql_common)}
            }

        # Continuación del pipeline después de la generación de SQL
        logger.info(f"DEBUG: [pipeline.py] Paso 9: Validando partes de la consulta generada...")

        # Validar las partes de la consulta (structured_info_copy) usando new_whitelist_validate_query
        # allowed_tables_list y allowed_columns_map ya están definidos y en ámbito.
        
        are_parts_valid, parts_validation_msg = new_whitelist_validate_query(
            structured_info_copy, 
            allowed_tables_list, 
            allowed_columns_map
        )
        if not are_parts_valid:
            logger.error(f"Partes de la consulta inválidas según whitelist: {structured_info_copy}. Razón: {parts_validation_msg}")
            return {
                "response": f"Las partes de la consulta generada no son válidas: {parts_validation_msg}",
                "sql": sql_query, # El sql_query actual
                "tables": structured_info_copy.get("tables", []),
                "columns": structured_info_copy.get("columns", []),
                "data": None,
                "intermediate_steps": {"sql_parts_validation_error": parts_validation_msg}
            }
        logger.info(f"DEBUG: [pipeline.py] Partes de la consulta validadas exitosamente (whitelist).")

        logger.info(f"DEBUG: [pipeline.py] Paso 10: Ejecutando SQL: {sql_query} con params: {params}")
        results, column_names_from_db = db_connector.execute_query(sql_query, params if params else None)

        if results is None: # Asumiendo que None indica un error o fallo en la ejecución
            logger.error(f"La ejecución de la consulta SQL ({sql_query}) no devolvió resultados o falló.")
            return {
                "response": "La consulta se ejecutó pero no produjo resultados o hubo un error interno.",
                "sql": sql_query,
                "tables": structured_info_copy.get("tables", []),
                "columns": structured_info_copy.get("columns", []),
                "data": [],
                "intermediate_steps": {"execution_issue": "No results or internal error"}
            }

        logger.info(f"DEBUG: [pipeline.py] Paso 11: Formateando resultados...")
        formatted_results = format_results(results, column_names_from_db)
        logger.info(f"DEBUG: [pipeline.py] Resultados formateados (primeros 200 chars): {str(formatted_results)[:200]}...")

        logger.info("DEBUG: [pipeline.py] Paso 12: Generando respuesta en lenguaje natural desde resultados SQL...")
        natural_language_response_prompt = (
            f"Dada la pregunta del usuario: '{question}'\\n"
            f"Y los siguientes resultados de la base de datos:\\n{formatted_results}\\n\\n"
            f"Por favor, genera una respuesta concisa y clara en lenguaje natural para el usuario. "
            f"Si los resultados están vacíos o no son informativos, indícalo amablemente."
        )
        
        try:
            llm_config_for_nl_response = {
                "api_key": os.environ.get("DEEPSEEK_API_KEY"),
                "base_url": os.environ.get("DEEPSEEK_API_URL"),
                "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
                "provider": "deepseek",
                "temperature": 0.5,
                "max_tokens": 300 # Aumentado ligeramente por si acaso
            }
            messages_for_nl_response = [{"role": "user", "content": natural_language_response_prompt}]

            nl_response_from_llm = call_llm_with_fallbacks(
                messages=messages_for_nl_response,
                config=llm_config_for_nl_response
            )

            if not nl_response_from_llm or not isinstance(nl_response_from_llm, str) or not nl_response_from_llm.strip():
                logger.warning("La respuesta del LLM para el lenguaje natural fue vacía o inválida. Usando resultados formateados directamente.")
                final_response_text = formatted_results if formatted_results else "No se encontraron resultados para tu consulta."
            else:
                final_response_text = nl_response_from_llm.strip()
                logger.info(f"Respuesta en lenguaje natural generada por LLM: {final_response_text}")

        except Exception as e_nl_llm:
            logger.error(f"Error al generar respuesta en lenguaje natural con LLM: {e_nl_llm}", exc_info=True)
            final_response_text = f"Se obtuvieron los siguientes datos: {formatted_results}. (Hubo un problema al generar una explicación en lenguaje natural)."

        # Preparar datos para el log final y la respuesta
        intermediate_steps_dict = {
            "question_preprocessed": enriched_question if 'enriched_question' in locals() else question,
            "structured_info_final": structured_info_copy,
            "generated_sql": sql_query,
            "sql_params": params,
            "raw_results_count": len(results) if results is not None else 0 # Asegurar que results no sea None
        }
        # Añadir más detalles si están disponibles
        if 'error_generating_sql_common' in locals() and e_gensql_common: # type: ignore
            intermediate_steps_dict["error_generating_sql_common"] = str(e_gensql_common) # type: ignore
        if 'sql_validation_error' in locals() and validation_msg: # type: ignore
            intermediate_steps_dict["sql_validation_error"] = validation_msg # type: ignore


        logger.info(f"DEBUG: [pipeline.py] Devolviendo respuesta final: {final_response_text[:100]}...")
        return {
            "response": final_response_text,
            "sql": sql_query,
            "tables": structured_info_copy.get("tables", []),
            "columns": column_names_from_db if column_names_from_db else structured_info_copy.get("columns", []),
            "data": results if results is not None else [], # Devolver lista vacía si results es None
            "column_names": column_names_from_db if column_names_from_db else [],
            "formatted_data": formatted_results if formatted_results else "",
            "intermediate_steps": intermediate_steps_dict
        }

    except JSONDecodeError as json_err:
        logger.error(f"DEBUG: [pipeline.py] Error JSONDecodeError en chatbot_pipeline: {json_err}", exc_info=True)
        return {"response": f"Error al procesar JSON: {json_err}", "sql": None, "tables": [], "columns": []}
    except Exception as e:
        logger.error(f"DEBUG: [pipeline.py] Error catastrófico en chatbot_pipeline: {e}", exc_info=True)
        print(f"ERROR: [pipeline.py] Error catastrófico en chatbot_pipeline: {e} (stdout)")
        return {"response": f"Ocurrió un error interno en el pipeline: {str(e)}", "sql": None, "tables": [], "columns": []}
    finally:
        print("STDOUT DEBUG (pipeline.py): ENTERING MAIN FINALLY BLOCK OF CHATBOT_PIPELINE")
        logger.info("DEBUG: [pipeline.py] Fin de chatbot_pipeline.")
        print("STDOUT DEBUG (pipeline.py): Fin de chatbot_pipeline.")
        print("STDOUT DEBUG (pipeline.py): EXITING MAIN FINALLY BLOCK OF CHATBOT_PIPELINE")
