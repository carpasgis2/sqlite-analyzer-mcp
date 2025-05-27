import logging
import time
import json
import re
import threading
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta # NUEVO

# Importaciones directas/absolutas para módulos en el mismo directorio src
from db_config import DBConnector, DEFAULT_DB_CONFIG as DATABASE_CONFIG # MODIFICADO
from llm_utils import call_llm_with_fallbacks
from sql_generator import SQLGenerator

from sql_organizer import normalize_structured_info # Importación corregida
from table_relationship import ( # Importaciones corregidas
    load_table_relationships, 
    generate_table_relationships_map, 
    save_table_relationships
)

from config import (
    SCHEMA_SIMPLE_PATH, SCHEMA_FULL_PATH, RELACIONES_PATH_FOR_LLM,
    RELACIONES_PATH, TERMS_DICT_PATH, MAX_RESPONSE_DATA_ROWS_CONFIG
)
import os 
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


from dotenv import load_dotenv
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
# Se modifica la importación para que coincida con las definiciones en db_config.py
try:
    from src.db_config import get_db_connector, DEFAULT_DB_CONFIG, DBConnector
    pipeline_module_logger.info("Importado src.db_config.get_db_connector, DEFAULT_DB_CONFIG y DBConnector.")
except ImportError:
    # Este fallback asume que db_config.py está en el mismo directorio o en sys.path
    # Este fallback asume que db_config.py está en el mismo directorio o en sys.path
    from db_config import get_db_connector, DEFAULT_DB_CONFIG, DBConnector
    pipeline_module_logger.info("Importado db_config.get_db_connector, DEFAULT_DB_CONFIG y DBConnector (fallback).")

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
RELACIONES_PATH_FOR_LLM = os.path.join(WORKSPACE_ROOT_DIR, "table_relationships.json") # Usar el archivo corregido
RELACIONES_PATH = os.path.join(WORKSPACE_ROOT_DIR, "table_relationships.json") # Para build_relationship_graph, etc.

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

def normalize_text(text):
    """Normaliza texto a minúsculas y sin acentos para matching robusto."""
    if not isinstance(text, str):
        return text
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join([c for c in text if unicodedata.category(c) != 'Mn'])
    return text

def is_likely_sql_query(text: str) -> bool:
    """Verifica si el texto de entrada es probablemente una consulta SQL."""
    text_upper = text.strip().upper()
    sql_keywords_initial = (
        "SELECT", "INSERT", "UPDATE", "DELETE", "WITH", 
        "CREATE", "ALTER", "DROP", "EXPLAIN", "PRAGMA"
    )
    sql_keywords_contained = [
        "FROM", "WHERE", "JOIN", "GROUP", "ORDER", 
        "VALUES", "SET", "TABLE", "LIMIT", "HAVING", "UNION"
    ]

    if any(text_upper.startswith(keyword) for keyword in sql_keywords_initial):
        return True

    contained_count = sum(1 for keyword in sql_keywords_contained if keyword in text_upper)
    
    # Considerar SQL si tiene FROM y al menos otra palabra clave común,
    # o si tiene múltiples palabras clave comunes y una longitud razonable.
    if "FROM" in text_upper and contained_count > 1:
        return True
    if contained_count > 2 and len(text_upper.split()) > 3: # Evitar falsos positivos en frases cortas
        return True
        
    return False

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
    # Corregir typos si hay términos disponibles y NO es una consulta SQL directa
    # para evitar corromper nombres de tablas/columnas en SQL válidos.
    if not is_likely_sql_query(question) and terms_dict and 'valid_terms' in terms_dict:
        question_before_typo_correction = question # Guardar para logging si es necesario
        question, corrections = correct_typos_in_question(question, terms_dict['valid_terms'])
        if corrections:
            logging.info(f"Correcciones de typos aplicadas (pregunta original: '{question_before_typo_correction}'): {corrections}")
            logging.info(f"Pregunta después de corrección de typos: '{question}'")
    
    original_question_for_terms = question  # Guardar la pregunta (posiblemente corregida) para la búsqueda de términos
    question = question.lower().strip()
    normalized_question = normalize_text(question)
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

    identified_tables_from_terms = set()
    matched_terms = []
    matched_descriptions = []
    matched_synonyms = []

    # --- MEJORA: Matching robusto y exhaustivo ---
    if terms_dict:
        # Recopilar todos los términos y sinónimos posibles para cada tabla
        table_candidates = {} # {normalized_synonym: table_name_upper}
        # Convertir nombres de tabla a MAYÚSCULAS consistentemente
        for table, syns in terms_dict.get('table_synonyms', {}).items():
            for syn in syns:
                table_candidates[normalize_text(syn)] = table.upper()
        for table, syns in terms_dict.get('table_common_terms', {}).items():
            for syn in syns:
                table_candidates[normalize_text(syn)] = table.upper()
        for term, table in terms_dict.get('table_mappings', {}).items():
             table_candidates[normalize_text(term)] = table.upper()
        
        # MODIFICADO: de logging.debug a logging.info y mostrar una muestra de los candidatos
        logging.info(f"[preprocess_question] Candidatos de tabla generados desde terms_dict (norm_syn:TABLE_UPPER, total {len(table_candidates)}): {dict(list(table_candidates.items())[:20])}")

        # Tokenizar la pregunta normalizada para buscar términos compuestos
        # normalized_question ya está disponible y normalizado (lower, no acentos)
        
        sorted_candidates = sorted(table_candidates.items(), key=lambda item: len(item[0]), reverse=True)

        temp_question_for_matching = normalized_question
        
        for norm_syn, table_name_upper in sorted_candidates:
            if norm_syn in temp_question_for_matching:
                identified_tables_from_terms.add(table_name_upper)
                matched_terms.append(norm_syn) 
                # Considerar no reemplazar para permitir que múltiples términos mapeen a la misma tabla o tablas relacionadas
                # temp_question_for_matching = temp_question_for_matching.replace(norm_syn, "", 1) 

        if identified_tables_from_terms:
            logging.info(f"[preprocess_question] Tablas identificadas desde terms_dict: {identified_tables_from_terms} usando términos normalizados: {matched_terms} en pregunta normalizada: '{normalized_question}'")
        else:
            # MEJORADO: Logging más detallado si no se identifican tablas
            logging.info(f"[preprocess_question] No se identificaron tablas directamente desde terms_dict para la pregunta original: '{original_question_for_terms}' (normalizada: '{normalized_question}').")
            if not table_candidates:
                logging.warning("[preprocess_question] `table_candidates` está vacío. Verificar `terms_dict` y su carga (ej. contenido de dictionary.json).")
            else:
                # Mostrar una muestra de los candidatos que se intentaron y no coincidieron.
                # Claves de table_candidates son los 'norm_syn'
                sample_candidate_keys = list(table_candidates.keys())[:20] 
                logging.info(f"[preprocess_question] Muestra de claves candidatas normalizadas que se intentaron (total {len(table_candidates)}): {sample_candidate_keys}")
                # No es necesario loguear normalized_question de nuevo aquí si ya se hizo arriba, pero puede ser útil para contexto directo.
                # logging.info(f"[preprocess_question] La pregunta normalizada contra la que se comparó fue: '{normalized_question}'")
    else:
        logging.warning("[preprocess_question] terms_dict no fue proporcionado o está vacío. No se pueden identificar tablas basadas en términos.")

    # ... (resto de la función)

    if matched_terms:
        enriched_question += f" [TERMINOS_RELEVANTES: {', '.join(list(set(matched_terms)))}]"
    if matched_descriptions:
        enriched_question += f" [DESCRIPCIONES_TERMINOS: {'; '.join(list(set(matched_descriptions)))}]"
    if matched_synonyms: # Podría ser redundante si los sinónimos ya se usaron para el matching
        enriched_question += f" [SINONIMOS_TERMINOS: {'; '.join(list(set(matched_synonyms)))}]"
    
    # Convertir el set a lista para el retorno
    final_identified_tables = list(identified_tables_from_terms)
    
    logging.debug(f"Preprocesamiento: Pregunta original: '{original_question_for_terms}', Enriquecida: '{enriched_question}', Tablas identificadas por términos: {final_identified_tables}, Tipo de consulta: {question_type}")
    return enriched_question, final_identified_tables, question_type

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
    # Cargar dictionary.json y extraer sinónimos de common_terms o de la descripción si está vacío
    try:
        with open(DICTIONARY_FILE, encoding="utf-8") as f:
            dct = json.load(f)
            # Procesar tablas
            for tname, tinfo in dct.get("tables", {}).items():
                desc = tinfo.get("description", "")
                synonyms = tinfo.get("common_terms", []) or []
                if not synonyms or synonyms == [tname.lower()]:
                    m = re.search(r"\[(.*?)\]", desc)
                    if m:
                        items = [s.strip().strip("'\"") for s in m.group(1).split(",")]
                        synonyms = [s.lower() for s in items if s]
                table_map[tname.lower()] = tname
                for term in synonyms:
                    table_map[term] = tname
            # Procesar columnas
            for cname, cinfo in dct.get("columns", {}).items():
                table = cinfo.get("table", "")
                col_key = f"{table}.{cname}" if table else cname
                desc = cinfo.get("description", "")
                synonyms = cinfo.get("common_terms", []) or []
                if not synonyms or synonyms == [col_key.lower()]:
                    m = re.search(r"\[(.*?)\]", desc)
                    if m:
                        items = [s.strip().strip("'\"") for s in m.group(1).split(",")]
                        synonyms = [s.lower() for s in items if s]
                column_map[col_key.lower()] = (table, cname)
                for term in synonyms:
                    column_map[term] = (table, cname)
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

def normalize_table_and_column_names(structured_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normaliza los nombres de tablas y columnas en structured_info usando los mappings semánticos.
    También se asegura de que las listas de tablas/columnas no contengan Nones.
    """
    table_map, column_map = load_semantic_mappings()

    # Tablas
    if "tables" in structured_info:
        original_tables = structured_info.get("tables")
        if isinstance(original_tables, list):
            new_tables = []
            for t in original_tables:
                if t is None:
                    logging.debug("normalize_table_and_column_names: Se encontró un valor de tabla None. Se omitirá.")
                    continue
                if not isinstance(t, str):
                    logging.warning(f"normalize_table_and_column_names: Nombre de tabla no es string: '{t}'. Se omitirá.")
                    continue
                t_norm = normalize_name(t)
                mapped = table_map.get(t_norm, t)
                new_tables.append(mapped)
            structured_info["tables"] = new_tables
        elif original_tables is None:
            structured_info["tables"] = []
        else:
            logging.warning(f"normalize_table_and_column_names: 'tables' en structured_info no es una lista ni None, sino {type(original_tables)}. Se dejará como está.")

    # Columnas
    if "columns" in structured_info:
        original_columns = structured_info.get("columns")
        if isinstance(original_columns, list):
            new_columns = []
            for c in original_columns:
                if c is None:
                    logging.debug("normalize_table_and_column_names: Se encontró un valor de columna None. Se omitirá.")
                    continue
                if not isinstance(c, str):
                    logging.warning(f"normalize_table_and_column_names: Nombre de columna no es string: '{c}' (tipo: {type(c)}). Se omitirá.")
                    continue
                
                c_norm = normalize_name(c)
                mapped_data = column_map.get(c_norm)
                column_to_add = c # Default to original (confirmed string)

                if isinstance(mapped_data, tuple) and len(mapped_data) == 2:
                    table_name_map, column_name_map = mapped_data
                    if table_name_map and column_name_map: # e.g., ("PATI_PATIENTS", "PATI_NAME")
                        column_to_add = f"{table_name_map}.{column_name_map}"
                    elif column_name_map: # e.g., (None, "PATI_NAME") or ("", "PATI_NAME")
                        column_to_add = column_name_map
                elif isinstance(mapped_data, str): # Direct mapping to a column name string
                     column_to_add = mapped_data

                new_columns.append(column_to_add)
            structured_info["columns"] = new_columns
        elif original_columns is None:
            structured_info["columns"] = []
        else:
            logging.warning(f"normalize_table_and_column_names: 'columns' en structured_info no es una lista ni None, sino {type(original_columns)}. Se dejará como está.")

    # Condiciones
    if "conditions" in structured_info:
        original_conditions = structured_info.get("conditions")
        if isinstance(original_conditions, list):
            new_conditions = []
            for cond in original_conditions:
                if isinstance(cond, dict) and "column" in cond:
                    original_column_in_cond = cond["column"]
                    if original_column_in_cond is None:
                        logging.warning("normalize_table_and_column_names: Columna None encontrada en una condición. La condición se mantendrá tal cual.")
                        # Potentially skip this condition or handle more gracefully if it causes issues downstream
                        # new_conditions.append(cond) # or skip: continue
                    elif not isinstance(original_column_in_cond, str):
                        logging.warning(f"normalize_table_and_column_names: Columna no string '{original_column_in_cond}' en condición. Se mantendrá original.")
                    else:
                        c_norm = normalize_name(original_column_in_cond)
                        mapped_data = column_map.get(c_norm)
                        if isinstance(mapped_data, tuple) and len(mapped_data) == 2:
                            table_name_map, column_name_map = mapped_data
                            if table_name_map and column_name_map:
                                cond["column"] = f"{table_name_map}.{column_name_map}"
                            elif column_name_map:
                                cond["column"] = column_name_map
                            # else: keep original cond["column"]
                        elif isinstance(mapped_data, str): # Direct mapping
                            cond["column"] = mapped_data
                new_conditions.append(cond) # Add modified or original condition
            structured_info["conditions"] = new_conditions
        elif original_conditions is None:
            structured_info["conditions"] = []
        else:
            logging.warning(f"normalize_table_and_column_names: 'conditions' en structured_info no es una lista ni None. Se dejará como está.")

    # Joins
    if "joins" in structured_info:
        original_joins = structured_info.get("joins")
        if isinstance(original_joins, list):
            new_joins = []
            for join in original_joins:
                if isinstance(join, dict):
                    for k_table in ["table", "foreign_table"]:
                        if k_table in join:
                            original_table_in_join = join[k_table]
                            if original_table_in_join is None:
                                logging.warning(f"normalize_table_and_column_names: Tabla None en join.{k_table}. Se mantendrá None.")
                            elif not isinstance(original_table_in_join, str):
                                logging.warning(f"normalize_table_and_column_names: Tabla no string '{original_table_in_join}' en join.{k_table}. Se mantendrá original.")
                            else:
                                t_norm = normalize_name(original_table_in_join)
                                join[k_table] = table_map.get(t_norm, original_table_in_join)
                    
                    for k_col in ["column", "foreign_column"]:
                        if k_col in join:
                            original_column_in_join = join[k_col]
                            if original_column_in_join is None:
                                logging.warning(f"normalize_table_and_column_names: Columna None en join.{k_col}. Se mantendrá None.")
                            elif not isinstance(original_column_in_join, str):
                                logging.warning(f"normalize_table_and_column_names: Columna no string '{original_column_in_join}' en join.{k_col}. Se mantendrá original.")
                            else:
                                col_to_normalize = original_column_in_join.split('.')[-1] if '.' in original_column_in_join else original_column_in_join
                                c_norm = normalize_name(col_to_normalize)
                                mapped_data = column_map.get(c_norm)
                                if isinstance(mapped_data, tuple) and len(mapped_data) == 2 and mapped_data[1]:
                                    join[k_col] = mapped_data[1] # Use only column part from mapping
                                elif isinstance(mapped_data, str): # Direct mapping to column name
                                    join[k_col] = mapped_data
                                # else: keep original join[k_col]
                new_joins.append(join)
            structured_info["joins"] = new_joins
        elif original_joins is None:
            structured_info["joins"] = []
        else:
            logging.warning(f"normalize_table_and_column_names: 'joins' en structured_info no es una lista ni None. Se dejará como está.")
            
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

def execute_query_with_timeout(db_connector, sql_query: str, params: Optional[List[Any]], logger, timeout_seconds: int = 10) -> Tuple[Optional[Union[List[Dict[str, Any]], int]], Optional[str]]:
    """
    Ejecuta una consulta SQL con timeout, devolviendo resultados o mensaje de error.
    """
    if not isinstance(sql_query, str) or not sql_query:
        logger.debug(f"execute_query_with_timeout: Consulta inválida: {sql_query}")
        return None, "Consulta SQL inválida o vacía"
    result_data = None
    error_message = None
    completed = threading.Event()
    def run():
        nonlocal result_data, error_message
        try:
            data, err = db_connector.execute_query(sql_query, params)
            result_data, error_message = data, err
        except Exception as e:
            logger.error(f"Error en hilo de ejecución: {e}", exc_info=True)
            error_message = str(e)
        finally:
            completed.set()
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    completed.wait(timeout=timeout_seconds)
    if not completed.is_set():
        logger.error(f"Timeout {timeout_seconds}s en consulta: {sql_query}")
        return None, f"Timeout de {timeout_seconds}s alcanzado"
    return result_data, error_message


def generate_sql(structured_info: Dict[str, Any], db_structure: Dict[str, Any], 
                 db_connector: 'DBConnector', logger: logging.Logger,
                 rag: Optional['EnhancedSchemaRAG'] = None, config: Dict[str, Any] = None) -> Tuple[str, List[Any]]: # MODIFIED return type
    """
    Genera una consulta SQL robusta a partir de la información estructurada, aplicando normalización semántica,
    validación y corrección de nombres, y manejo de errores. Utiliza LLM y RAG si están disponibles.
    """
    # params_list debe ser inicializada aquí para estar en el alcance de todos los returns
    params_list: List[Any] = []
    sql_query_str: Optional[str] = None # Para almacenar el SQL string

    try: # OUTER TRY
        from src.sql_generator import SQLGenerator
        structured_info = check_table_references(structured_info)
        structured_info = normalize_table_and_column_names(structured_info)
        
        tablas = structured_info.get("tables", [])
        tablas_validas = [t for t in tablas if t and t in db_structure]

        if not tablas_validas:
            tablas_faltantes = [t for t in tablas if t and t not in db_structure]
            if tablas_faltantes:
                logger.error(f"No existen en el esquema las siguientes tablas requeridas: {tablas_faltantes}")
                return f"SELECT 'Error: No se puede responder porque faltan tablas en la base de datos: {', '.join(tablas_faltantes)}' AS mensaje", []
            logger.error(f"Ninguna tabla válida identificada después del filtrado: {tablas}")
            return "SELECT 'No se identificaron tablas válidas para la consulta' AS mensaje", []

        structured_info["tables"] = tablas_validas
        
        try: # INNER TRY
            logger.info("--- Iniciando inspección de db_structure para allowed_columns_map ---")
            problematic_tables_for_logging = []

            allowed_tables_list = list(db_structure.keys()) if db_structure else []
            allowed_columns_map = {}
            if db_structure:
                for table_name_for_map, cols_data in db_structure.items():
                    table_name_upper = table_name_for_map.upper() # Consistent casing for map keys
                    
                    if not isinstance(table_name_for_map, str) or not table_name_for_map.strip():
                        logger.warning(f"Se encontró un nombre de tabla inválido o vacío en db_structure: '{table_name_for_map}'. Se omitirá para allowed_columns_map.")
                        continue

                    current_table_cols = []
                    if isinstance(cols_data, dict) and 'columns' in cols_data:
                        columns_list_or_dict = cols_data['columns']
                        if isinstance(columns_list_or_dict, list):
                            # Este es el formato esperado donde cols_data['columns'] es una lista
                            col_names_for_map = []
                            for col_entry in columns_list_or_dict:
                                if isinstance(col_entry, dict) and isinstance(col_entry.get('name'), str) and col_entry.get('name').strip():
                                    col_names_for_map.append(col_entry['name'])
                                elif isinstance(col_entry, str) and col_entry.strip():
                                    col_names_for_map.append(col_entry)
                                else:
                                    logger.debug(f"Entrada de columna inválida o vacía para tabla '{table_name_for_map}': {col_entry}")
                            current_table_cols = [c.upper() for c in col_names_for_map]
                            if not current_table_cols and columns_list_or_dict: # Log if original list was not empty but processed is
                                logger.warning(f"Investigación: Tabla '{table_name_for_map}' tiene una lista 'columns' ({len(columns_list_or_dict)} entradas) pero resultó en una lista vacía de nombres de columna procesados. Original: {columns_list_or_dict[:5]}")
                                problematic_tables_for_logging.append(f"{table_name_for_map} (lista 'columns' procesada a vacía pese a no estar vacía originalmente)")
                        else:
                            # cols_data['columns'] no es una lista
                            logger.warning(f"Investigación: Tabla '{table_name_for_map}' tiene clave 'columns', pero NO es una lista. Tipo: {type(columns_list_or_dict)}. Valor: {str(columns_list_or_dict)[:200]}. Se usará lista vacía de columnas.")
                            problematic_tables_for_logging.append(f"{table_name_for_map} ('columns' no es lista)")
                    elif isinstance(cols_data, list): # Podría ser que cols_data sea directamente la lista de columnas
                        logger.info(f"Investigación: Para tabla '{table_name_for_map}', cols_data es una lista (posiblemente de columnas directamente, {len(cols_data)} entradas): {str(cols_data[:3])[:200]}")
                        col_names_for_map = []
                        for col_entry in cols_data:
                            if isinstance(col_entry, dict) and isinstance(col_entry.get('name'), str) and col_entry.get('name').strip():
                                col_names_for_map.append(col_entry['name'])
                            elif isinstance(col_entry, str) and col_entry.strip():
                                col_names_for_map.append(col_entry)
                        current_table_cols = [c.upper() for c in col_names_for_map]
                        if not current_table_cols and cols_data: # Log if original list was not empty but processed is
                             problematic_tables_for_logging.append(f"{table_name_for_map} (cols_data era lista no vacía, procesada a vacía)")
                    else:
                        logger.warning(f"Investigación: Estructura inesperada para la tabla '{table_name_for_map}' en db_structure o 'columns' falta. Tipo de cols_data: {type(cols_data)}. Valor: {str(cols_data)[:200]}. Se usará lista vacía de columnas.")
                        problematic_tables_for_logging.append(f"{table_name_for_map} (estructura inesperada o sin 'columns')")
                    
                    allowed_columns_map[table_name_upper] = current_table_cols
                    logger.debug(f"Investigación: Tabla '{table_name_upper}' añadida a allowed_columns_map con {len(current_table_cols)} columnas: {current_table_cols[:5]}")

            else: # db_structure es None o vacío
                logger.warning("db_structure está vacío o es None. allowed_columns_map estará vacío.")
            
            if problematic_tables_for_logging:
                logger.warning(f"--- Resumen de tablas problemáticas durante la creación de allowed_columns_map (resultaron en 0 columnas procesadas a partir de datos que podrían no haber sido vacíos): {problematic_tables_for_logging} ---")
            else:
                logger.info("--- Inspección de db_structure para allowed_columns_map completada. Si hubo tablas con 0 columnas, fue porque la entrada original ya estaba vacía o no era procesable. ---")
            
            # sql_generator_instance = SQLGenerator(allowed_tables=allowed_tables_list, allowed_columns=allowed_columns_map) # Duplicado, eliminar


            sql_gen = SQLGenerator(allowed_tables=[t.upper() for t in allowed_tables_list], allowed_columns=allowed_columns_map) # Asegurar mayúsculas
            returned_from_sql_gen = sql_gen.generate_sql(structured_info, db_structure, None)

            if isinstance(returned_from_sql_gen, tuple) and len(returned_from_sql_gen) == 2:
                sql_query_str, params_list = returned_from_sql_gen
                if not isinstance(sql_query_str, str):
                    logger.error(f"SQLGenerator.generate_sql devolvió una tupla, pero el SQL no es string: {type(sql_query_str)}")
                    sql_query_str = None 
            elif isinstance(returned_from_sql_gen, str):
                sql_query_str = returned_from_sql_gen
                # params_list ya es []
            else:
                logger.error(f"SQLGenerator.generate_sql devolvió un tipo inesperado: {type(returned_from_sql_gen)}. Valor: {returned_from_sql_gen}")
                sql_query_str = None

            if sql_query_str:
                logger.debug(f"SQL generado por sql_gen: {sql_query_str}, Params: {params_list}")
                
                # Preparar allowed_tables y allowed_columns para new_whitelist_validate_query
                # Ya tenemos allowed_columns_map y allowed_tables_list (que se convirtió a mayúsculas para SQLGenerator)
                # Solo necesitamos asegurar que allowed_tables_list también esté en mayúsculas para la validación.
                allowed_tables_for_validation_upper = [t.upper() for t in allowed_tables_list if isinstance(t, str) and t.strip()]

                # Log detallado de lo que se pasa al validador
                logger.debug(f"Argumentos para new_whitelist_validate_query:")
                logger.debug(f"  structured_info: {json.dumps(structured_info, indent=2, ensure_ascii=False)}")
                logger.debug(f"  allowed_tables_for_validation_upper: {allowed_tables_for_validation_upper}")
                # Loguear una muestra de allowed_columns_map si es muy grande
                log_cols_map_sample = {k: v[:5] + ['...'] if len(v) > 5 else v for k, v in allowed_columns_map.items()}
                logger.debug(f"  allowed_columns_map (muestra): {log_cols_map_sample}")

                is_safe, validation_msg = new_whitelist_validate_query(
                    structured_info, 
                    allowed_tables_for_validation_upper, # Usar la lista de tablas en mayúsculas
                    allowed_columns_map # Ya está en el formato {TABLE_UPPER: [COL_UPPER, ...]}
                )
                logger.info(f"Paso 6.1 completado. ¿Validación de partes estructuradas segura? {is_safe}, Mensaje: {validation_msg}")

                if not is_safe:
                    logger.error(f"Validación de whitelist de partes estructuradas fallida: {validation_msg} para structured_info: {structured_info}")
                    # Devolver un diccionario como en el original para mantener la interfaz, aunque la función espera tupla
                    return {"response": f"Error: La consulta generada no es segura según la validación de estructura. {validation_msg}", 
                            "sql_query": sql_query_str, "params": params_list, "structured_info": structured_info} # type: ignore

                logger.info(f"Paso 6.2: Ejecutando SQL (para validación en generate_sql): {sql_query_str} con params: {params_list}")
                results_val, error_msg_val = execute_query_with_timeout(
                    db_connector, sql_query_str, params_list, logger, timeout_seconds=5
                )
                # results_val, error_msg_val = (None, "Validación con execute_query_with_timeout deshabilitada temporalmente") # Placeholder
                if error_msg_val:
                    logger.error(f"generate_sql: Error from execute_query_with_timeout (validation): {error_msg_val}")
                    error_sql = f"SELECT 'Error durante la validación SQL en generate_sql: {str(error_msg_val)[:100]}' AS mensaje"
                    return error_sql, []
                logger.info(f"generate_sql: execute_query_with_timeout (for validation) completed. SQL: {sql_query_str}")
            # else: sql_query_str es None (o se invalidó), se manejará después.

        except Exception as e_gensql:
            logger.error(f"DEBUG: [pipeline.py] EXCEPCIÓN durante la generación/validación de SQL interna: {e_gensql}", exc_info=True)
            logger.info("generate_sql: Returning error tuple due to e_gensql exception.")
            return "SELECT 'Error durante la generación interna de SQL' AS mensaje", []

        logger.info(f"generate_sql: After try-except e_gensql. sql_query_str: {sql_query_str}, params_list: {params_list}")

        if sql_query_str is None:
            logger.error("Fallo crítico en la generación de SQL: sql_query_str es None después del bloque de generación.")
            logger.info("generate_sql: Returning error tuple due to sql_query_str is None.")
            return "SELECT 'Error: Fallo interno crítico en la generación de SQL (query_str nula)' AS mensaje", []

        logger.info(f"generate_sql: About to correct column names. SQL: {sql_query_str}")
        
        # Asegurarse de que db_structure se pasa correctamente a correct_column_names
        # y que las tablas en tablas_validas están en el formato esperado por correct_column_names.
        # Si correct_column_names espera nombres de tabla en un caso específico (ej. mayúsculas), ajustar aquí.
        # Por ahora, asumimos que tablas_validas (que vienen de structured_info y luego filtradas por db_structure)
        # ya tienen el casing correcto o que correct_column_names lo maneja.
        corrected_sql_query = correct_column_names(sql_query_str, tablas_validas, db_structure, logger) # Pasar logger
        
        if corrected_sql_query is None: 
             logger.error("correct_column_names devolvió None.")
             logger.info("generate_sql: Returning error tuple due to correct_column_names returning None.")
             return "SELECT 'Error: Fallo en la corrección de nombres de columna (resultado nulo)' AS mensaje", []
        logger.info(f"generate_sql: Column names corrected. SQL: {corrected_sql_query}")

        final_sql_query = validate_table_names(corrected_sql_query, db_structure)
        if final_sql_query is None: 
             logger.error("validate_table_names devolvió None.")
             logger.info("generate_sql: Returning error tuple due to validate_table_names returning None.")
             return "SELECT 'Error: Fallo en la validación de nombres de tabla (resultado nulo)' AS mensaje", []
        logger.info(f"generate_sql: Table names validated. SQL: {final_sql_query}")
        
        logger.info(f"generate_sql: About to validate final SQL query: {final_sql_query}")
        if not validate_sql_query(final_sql_query, db_connector): 
            logger.error(f"SQL inválido tras corrección y validación final: {final_sql_query}")
            logger.info("generate_sql: Returning error tuple due to final SQL validation failure.")
            return "SELECT 'Error: SQL inválido tras corrección final' AS mensaje", []

        logger.info(f"generate_sql: Final SQL query validated. SQL: {final_sql_query}")

        logger.info(f"generate_sql: Returning final SQL query and params. SQL: {final_sql_query}, Params: {params_list}")
        return final_sql_query, params_list
    
    except Exception as e: # OUTER CATCH
        logger.error(f"Error en generate_sql (captura externa): {e}", exc_info=True)
        logger.info("generate_sql: Returning error tuple due to outer exception.")
        return "SELECT 'Error al generar SQL (captura externa)' AS mensaje", []

def correct_column_names(sql_query: str, tables: List[str], db_structure: Dict[str, Any], logger: logging.Logger) -> Optional[str]:
    """
    Corrige nombres de columnas en una consulta SQL basándose en db_structure.
    Intenta hacer coincidir columnas ambiguas o mal escritas con las columnas reales de las tablas involucradas.
    """
    if not sql_query or not tables or not db_structure:
        logger.debug("correct_column_names: Entrada inválida (query, tablas o estructura vacía).")
        return sql_query # Devuelve la consulta original si no hay suficiente info para procesar

    logger.debug(f"correct_column_names: Iniciando corrección para query: {sql_query[:100]}..., tablas: {tables}")

    # Extraer todas las columnas candidatas de la consulta (simplificado)
    # Esta regex es básica y podría mejorarse para manejar casos más complejos (alias, funciones, etc.)
    # Buscamos identificadores que podrían ser columnas.
    potential_columns_in_query = set(re.findall(r'\\b([a-zA-Z_][a-zA-Z0-9_\\.]*)\\b', sql_query))
    logger.debug(f"correct_column_names: Columnas potenciales extraídas de la query: {potential_columns_in_query}")

    # Crear un mapa de columnas permitidas para las tablas dadas, en MAYÚSCULAS
    # Ejemplo: {'TABLE_A': ['COL1', 'COL2'], 'TABLE_B': ['COL_X', 'COL_Y']}
    valid_columns_for_tables: Dict[str, List[str]] = {}
    for table_name in tables:
        table_name_upper = table_name.upper()
        # Buscar la tabla en db_structure, puede que no esté en mayúsculas allí
        actual_table_key_in_db_structure = None
        if table_name_upper in db_structure:
            actual_table_key_in_db_structure = table_name_upper
        else: # Intentar encontrarla con el case original o minúsculas si es necesario
            if table_name in db_structure:
                actual_table_key_in_db_structure = table_name
            elif table_name.lower() in db_structure:
                 actual_table_key_in_db_structure = table_name.lower()
            # Podríamos añadir más lógica de fuzzy matching para el nombre de la tabla si fuera necesario

        if actual_table_key_in_db_structure and isinstance(db_structure.get(actual_table_key_in_db_structure), dict):
            cols_data = db_structure[actual_table_key_in_db_structure].get('columns', [])
            col_names = []
            for col_entry in cols_data:
                if isinstance(col_entry, dict) and isinstance(col_entry.get('name'), str):
                    col_names.append(col_entry['name'].upper())
                elif isinstance(col_entry, str): # Si la columna es solo un string
                    col_names.append(col_entry.upper())
            valid_columns_for_tables[table_name_upper] = col_names
        else:
            logger.warning(f"correct_column_names: No se encontró la tabla '{table_name}' o su estructura de columnas en db_structure.")
            valid_columns_for_tables[table_name_upper] = []
            
    logger.debug(f"correct_column_names: Columnas válidas para las tablas dadas (UPPER): {valid_columns_for_tables}")

    modified_query = sql_query
    corrections_made = False

    for p_col_full in potential_columns_in_query:
        p_col_full_upper = p_col_full.upper()
        
        # Ignorar palabras clave SQL comunes y números
        sql_keywords_to_ignore = {'SELECT', 'FROM', 'WHERE', 'JOIN', 'ON', 'AS', 'AND', 'OR', 'NOT', 'NULL', 'GROUP', 'ORDER', 'BY', 'LIMIT', 'OFFSET', 'INSERT', 'INTO', 'VALUES', 'UPDATE', 'SET', 'DELETE', 'COUNT', 'AVG', 'SUM', 'MIN', 'MAX', 'DISTINCT', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'LIKE', 'IN', 'BETWEEN', 'IS', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'FULL'}
        if p_col_full_upper in sql_keywords_to_ignore or p_col_full.isdigit():
            logger.debug(f"correct_column_names: Ignorando palabra clave SQL o número: {p_col_full}")
            continue

        table_prefix_upper = None
        column_part_upper = p_col_full_upper

        if '.' in p_col_full_upper:
            parts = p_col_full_upper.split('.', 1)
            table_prefix_upper = parts[0]
            column_part_upper = parts[1]
            logger.debug(f"correct_column_names: Procesando columna con prefijo: {p_col_full_upper} -> Prefijo: {table_prefix_upper}, Columna: {column_part_upper}")
            
            # Verificar si el prefijo es una tabla válida de la consulta
            if table_prefix_upper not in valid_columns_for_tables: # valid_columns_for_tables tiene claves de tabla en MAYÚSCULAS
                logger.debug(f"correct_column_names: Prefijo de tabla '{table_prefix_upper}' no está en las tablas de la consulta {list(valid_columns_for_tables.keys())}. Podría ser un alias o una tabla no relevante.")
                # Si el prefijo no es una de las tablas principales, podría ser un alias.
                # La lógica actual no resuelve alias aquí, asume que el SQL es estructuralmente simple
                # o que los alias ya coinciden con nombres de tabla (lo cual es una mala práctica).
                # Por ahora, si el prefijo no es una tabla conocida, no intentamos corregir esta columna.
                continue 

            # El prefijo es una tabla conocida. Verificar la parte de la columna.
            if column_part_upper not in valid_columns_for_tables.get(table_prefix_upper, []):
                logger.debug(f"correct_column_names: Columna '{column_part_upper}' (con prefijo '{table_prefix_upper}') no encontrada directamente. Buscando fuzzy match en '{table_prefix_upper}'.")
                # Intentar fuzzy match para la parte de la columna dentro de esa tabla
                matches = get_close_matches(column_part_upper, valid_columns_for_tables.get(table_prefix_upper, []), n=1, cutoff=0.8)
                if matches:
                    correct_col_name = matches[0]
                    # Reconstruir el nombre completo con el prefijo original (case-insensitive replace)
                    # y la columna corregida (con su case original de db_structure)
                    # Necesitamos el case original de la columna de db_structure
                    original_case_correct_col_name = correct_col_name # Default
                    for actual_col_name_in_db_structure in db_structure.get(table_prefix_upper, {}).get('columns',[]): # Asume que table_prefix_upper es la clave correcta en db_structure
                        if isinstance(actual_col_name_in_db_structure, dict) and actual_col_name_in_db_structure.get('name','').upper() == correct_col_name:
                            original_case_correct_col_name = actual_col_name_in_db_structure['name']
                            break
                        elif isinstance(actual_col_name_in_db_structure, str) and actual_col_name_in_db_structure.upper() == correct_col_name:
                            original_case_correct_col_name = actual_col_name_in_db_structure
                            break
                    
                    # Para el reemplazo, necesitamos el prefijo original (case de la query)
                    original_prefix_in_query = p_col_full.split('.')[0]

                    regex_pattern = r'\\b' + re.escape(original_prefix_in_query) + r'\\s*\\.\\s*' + re.escape(p_col_full.split('.')[1]) + r'\\b'
                                        
                    new_col_full_name = f"{original_prefix_in_query}.{original_case_correct_col_name}"
                    
                    logger.info(f"correct_column_names: Corrección (prefijo): Reemplazando '{p_col_full}' por '{new_col_full_name}' en la query.")
                    
                    # Usar re.sub para reemplazo case-insensitive del patrón original
                    # Esto es complejo porque p_col_full ya es el match, y queremos reemplazarlo exactamente.
                    # La regex debe ser precisa para no reemplazar subcadenas incorrectas.
                    # Usamos word boundaries \\b para mayor precisión.
                    
                    # modified_query = re.sub(r'\\b' + re.escape(p_col_full) + r'\\b', new_col_full_name, modified_query, flags=re.IGNORECASE)
                    # El reemplazo anterior podría ser demasiado agresivo si p_col_full es solo una parte.
                    # Es mejor reemplazar la instancia exacta que se está evaluando.
                    # Sin embargo, p_col_full es solo un string, no tiene contexto de su posición.
                    # Esta es una limitación de este enfoque simple.
                    # Por ahora, un reemplazo directo es lo más sencillo, asumiendo que los nombres son suficientemente únicos.
                    
                    # Intentar un reemplazo más seguro usando el patrón regex que construimos
                    num_replacements = 0
                    temp_query = modified_query
                    
                    # Función para reemplazar preservando el case del prefijo original
                    def replacer(match_obj):
                        nonlocal num_replacements
                        num_replacements +=1
                        # match_obj.group(0) es toda la coincidencia, ej. "p.NMAE"
                        # match_obj.group(1) sería el prefijo si lo capturamos, ej. "p"
                        # match_obj.group(2) sería la columna si la capturamos, ej. "NMAE"
                        # Aquí, el patrón es para toda la expresión original_prefix_in_query.original_column_part
                        # Necesitamos el prefijo original de la query, no el del match necesariamente si el regex es más amplio.
                        
                        # El prefijo original de la query es `original_prefix_in_query`
                        # La columna corregida con case original es `original_case_correct_col_name`
                        return f"{original_prefix_in_query}.{original_case_correct_col_name}"

                    modified_query, count = re.subn(regex_pattern, replacer, temp_query, flags=re.IGNORECASE)
                    if count > 0:
                        logger.info(f"  Reemplazo con regex tuvo {count} ocurrencias.")
                        corrections_made = True
                    else:
                        logger.warning(f"  Reemplazo con regex no encontró ocurrencias para el patrón: {regex_pattern} en la query (esto es inesperado).")


                else:
                    logger.debug(f"correct_column_names: No se encontró fuzzy match para '{column_part_upper}' en tabla '{table_prefix_upper}'.")

        else: # Columna sin prefijo (p_col_full_upper es solo el nombre de la columna)
            logger.debug(f"correct_column_names: Procesando columna sin prefijo: {p_col_full_upper}")
            # Verificar si esta columna existe en ALGUNA de las tablas de la consulta
            found_in_table = None
            for table_key_upper, cols_list_upper in valid_columns_for_tables.items():
                if p_col_full_upper in cols_list_upper:
                    found_in_table = table_key_upper # Encontrada directamente
                    logger.debug(f"correct_column_names: Columna sin prefijo '{p_col_full_upper}' encontrada directamente en tabla '{table_key_upper}'. No se corrige.")
                    break
            
            if not found_in_table: # No se encontró directamente, intentar fuzzy match
                logger.debug(f"correct_column_names: Columna sin prefijo '{p_col_full_upper}' no encontrada directamente. Buscando fuzzy match en todas las tablas de la consulta: {list(valid_columns_for_tables.keys())}")
                best_match_col = None
                best_match_table = None
                highest_similarity = 0.0 # No usado por get_close_matches directamente, pero para lógica custom

                possible_matches_across_tables = []
                for table_key_upper_iter, cols_list_upper_iter in valid_columns_for_tables.items():
                    matches_in_table = get_close_matches(p_col_full_upper, cols_list_upper_iter, n=1, cutoff=0.8) # Cutoff más alto para más precisión
                    if matches_in_table:
                        # Guardar la coincidencia y de qué tabla vino
                        possible_matches_across_tables.append({'col': matches_in_table[0], 'table': table_key_upper_iter})
                
                if possible_matches_across_tables:
                    if len(possible_matches_across_tables) == 1:
                        best_match_col = possible_matches_across_tables[0]['col']
                        best_match_table = possible_matches_across_tables[0]['table'] # La tabla donde se encontró el mejor match
                        logger.debug(f"correct_column_names: Fuzzy match único para '{p_col_full_upper}' -> '{best_match_col}' en tabla '{best_match_table}'.")
                    else:
                        # Múltiples matches en diferentes tablas, o múltiples buenos matches en una tabla (si n > 1)
                        # Esto es ambiguo. Podríamos tener una política (ej. tomar el primero, o no corregir)
                        logger.warning(f"correct_column_names: Múltiples fuzzy matches para '{p_col_full_upper}': {possible_matches_across_tables}. No se corregirá debido a ambigüedad.")
                        best_match_col = None # No corregir si es ambiguo

                if best_match_col and best_match_table: # Si hay un match claro y único
                    # Necesitamos el case original de la columna corregida
                    original_case_correct_col_name = best_match_col # Default
                    # Encontrar el case original de la columna en db_structure para la best_match_table
                    # Primero, encontrar la clave correcta para best_match_table en db_structure (puede no ser UPPER)
                    actual_db_structure_table_key = None
                    if best_match_table in db_structure: actual_db_structure_table_key = best_match_table
                    else:
                        for k_db in db_structure.keys():
                            if k_db.upper() == best_match_table:
                                actual_db_structure_table_key = k_db
                                break
                    
                    if actual_db_structure_table_key:
                        for col_entry_db in db_structure.get(actual_db_structure_table_key, {}).get('columns', []):
                            if isinstance(col_entry_db, dict) and col_entry_db.get('name','').upper() == best_match_col:
                                original_case_correct_col_name = col_entry_db['name']
                                break
                            elif isinstance(col_entry_db, str) and col_entry_db.upper() == best_match_col:
                                original_case_correct_col_name = col_entry_db
                                break
                    
                    logger.info(f"correct_column_names: Corrección (sin prefijo): Reemplazando '{p_col_full}' por '{original_case_correct_col_name}' en la query.")
                    # Reemplazar la columna original (p_col_full, que es el case de la query) por la corregida (original_case_correct_col_name)
                    # Usar word boundaries para evitar reemplazar subcadenas.
                    modified_query, count = re.subn(r'\\b' + re.escape(p_col_full) + r'\\b', original_case_correct_col_name, modified_query, flags=re.IGNORECASE)
                    if count > 0:
                        corrections_made = True
                    else:
                        logger.warning(f"  Reemplazo de columna sin prefijo no encontró ocurrencias para '{p_col_full}' (esto es inesperado).")
                else:
                    logger.debug(f"correct_column_names: No se encontró fuzzy match claro o único para columna sin prefijo '{p_col_full_upper}'.")
            # else: la columna sin prefijo se encontró directamente, no se hace nada.

    if corrections_made:
        logger.info(f"correct_column_names: Query modificada: {modified_query}")
        return modified_query
    else:
        logger.debug("correct_column_names: No se realizaron correcciones en la query.")
        return sql_query # Devuelve la original si no hubo cambios

def chatbot_pipeline(
    question: str,
    db_connector: Optional[DBConnector] = None,
    logger_param: Optional[logging.Logger] = None,
    relevant_tables_override: Optional[List[str]] = None,
    conditions_override: Optional[List[str]] = None,
    disable_direct_sql_execution: bool = False,
    disable_llm_enhancement: bool = False, # NUEVO PARÁMETRO
    timeout_seconds: int = 20 # NUEVO PARÁMETRO
) -> Dict[str, Any]:
    """
    Pipeline principal para procesar la pregunta del usuario y devolver una respuesta.
    """
    # Configurar logger si no se proporciona
    logger = logger_param if logger_param else logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")
        logger = logging.getLogger(__name__) # Re-obtener después de basicConfig

    logger.info(f"DEBUG: [pipeline.py] Inicio de chatbot_pipeline. Pregunta: '{question[:100]}...', DisableDirectSQL: {disable_direct_sql_execution}, DisableLLMEnhancement: {disable_llm_enhancement}, Timeout: {timeout_seconds}s")
    logger.info("PIPELINE_BENCHMARK: chatbot_pipeline START")
    pipeline_start_time = time.time() # Unificado
    prev_step_time_tracker = pipeline_start_time # Inicializar tracker

    # Inicializar db_connector si no se proporciona
    if db_connector is None:
        logger.info("DEBUG: [pipeline.py] db_connector no proporcionado, obteniendo uno por defecto...")
        db_connector = get_db_connector()
        logger.info("DEBUG: [pipeline.py] db_connector obtenido.")

    # Cargar estructuras y configuraciones comunes
    logger.info("DEBUG: [pipeline.py] Obteniendo estructura de la BD (común)...")
    db_schema_dict = db_connector.get_database_structure()
    logger.info(f"DEBUG: [pipeline.py] Estructura de BD obtenida: {type(db_schema_dict)}")

    # --- Definición de allowed_tables_list y allowed_columns_map ---
    # Esto asume que db_schema_dict es un diccionario de tablas a información de columnas.
    # La estructura exacta de db_schema_dict (ej. dict de dicts, dict de listas) impacta cómo se extraen los nombres.
    allowed_tables_list = list(db_schema_dict.keys())
    allowed_columns_map: Dict[str, List[str]] = {}
    for table_name, columns_data in db_schema_dict.items():
        cols_for_table = []
        if isinstance(columns_data, dict): # Ej: {'col1':{details}, 'col2':{details}}
            cols_for_table = list(columns_data.keys())
        elif isinstance(columns_data, list): # Ej: [{'name':'col1'}, {'name':'col2'}] o ['col1', 'col2']
            if columns_data:
                if isinstance(columns_data[0], dict) and 'name' in columns_data[0]:
                    cols_for_table = [col_info['name'] for col_info in columns_data if isinstance(col_info, dict) and 'name' in col_info]
                elif isinstance(columns_data[0], str):
                    cols_for_table = [col_name for col_name in columns_data if isinstance(col_name, str)]
        allowed_columns_map[table_name] = cols_for_table
    logger.info(f"DEBUG: [pipeline.py] Allowed tables ({len(allowed_tables_list)}), Allowed columns map generado.")
    # --- Fin definición ---

    # Cargar esquemas desde archivos
    db_schema_str_simple = load_schema_as_string(SCHEMA_SIMPLE_PATH)
    db_schema_enhanced_content = load_schema_as_string(SCHEMA_FULL_PATH)
    # Usar RELACIONES_PATH_FOR_LLM consistentemente si es diferente de RELACIONES_PATH para el LLM
    # Aquí se asume que relaciones_tablas_str es el que se usa para el LLM.
    relaciones_tablas_str = load_schema_as_string(RELACIONES_PATH_FOR_LLM)

    # Cargar mapa de relaciones (para uso interno, no para el LLM directamente)
    relaciones_tablas_map = load_table_relationships(RELACIONES_PATH)
    if not relaciones_tablas_map:
        logger.warning("No se pudo cargar el mapa de relaciones desde el archivo. Intentando generar uno nuevo.")
        relaciones_tablas_map = generate_table_relationships_map(db_schema_dict) # Pasar db_schema_dict
        save_table_relationships(relaciones_tablas_map, RELACIONES_PATH)
    logger.info(f"Mapa de relaciones cargado con {len(relaciones_tablas_map)} relaciones")

    logger.info("DEBUG: [pipeline.py] Cargando diccionario de términos (común)...")
    start_time_terms = time.time()
    terms_dict = load_terms_dictionary(TERMS_DICT_PATH)
    logger.info(f"PIPELINE_BENCHMARK: terms_dict loaded - {time.time() - start_time_terms:.4f}s")
    prev_step_time_tracker = time.time() # Actualizar tracker

    # Preprocesar la pregunta

   
   
    start_time_preprocess = time.time()
    enriched_question, identified_tables_from_preprocess, identified_query_type = preprocess_question(question, terms_dict)
    logger.info(f"PIPELINE_BENCHMARK: preprocess_question completed - {time.time() - start_time_preprocess:.4f}s")
    logger.info(f"DEBUG: [pipeline.py] Pregunta preprocesada: '{enriched_question}', Tablas por preproc: {identified_tables_from_preprocess}, Tipo por preproc: {identified_query_type}")
    prev_step_time_tracker = time.time() # Actualizar tracker

    # Inicializar structured_info
    structured_info: Dict[str, Any] = {
        "tables": relevant_tables_override if relevant_tables_override is not None else identified_tables_from_preprocess,
        "columns": [],
        "conditions": conditions_override if conditions_override is not None else [],
        "query_type": identified_query_type,
        "joins": [],
        "original_question": question,
        "enriched_question": enriched_question,
        "is_complex_query": False
    }
    logger.info(f"DEBUG: [pipeline.py] structured_info inicializado: {structured_info}")
    llm_extracted_info: Optional[Dict[str, Any]] = None # Inicializar llm_extracted_info aquí

    # Determinar si es una consulta SQL directa
    is_direct_sql = False
    if not disable_direct_sql_execution and is_likely_sql_query(question):
        is_direct_sql = True
        logger.info(f"DEBUG_DIRECT_SQL: Clasificado como SQL directo para la pregunta original: '{question}'")
    else:
        logger.info(f"DEBUG_DIRECT_SQL: No clasificado como SQL directo. disable_direct_sql_execution={disable_direct_sql_execution}, is_likely_sql_query('{question}')={is_likely_sql_query(question)}")

    # Ejecutar SQL directo o continuar con el pipeline LLM
    if is_direct_sql:
        logger.info(f"Procesando como consulta SQL directa: '{question[:100]}...'")
        sql_query_to_execute = question
        params_to_execute: List[Any] = []

        logger.info(f"DEBUG: [pipeline.py] Ejecutando consulta SQL directa: '{sql_query_to_execute[:100]}...'")
        start_time_direct_sql = time.time()

        # --- VALIDACIÓN ROBUSTA DE SQL DIRECTO ---
        from src.sql_utils import validate_sql_entities
        if not validate_sql_entities(sql_query_to_execute, db_schema_dict):
            logger.error(f"Validación de entidades fallida para SQL directo: {sql_query_to_execute}")
            final_response = {
                "response": "Error: La consulta contiene columnas o tablas que no existen en la base de datos. "
                "La validación se realiza usando el esquema real extraído de la base de datos (basado en los archivos JSON de esquema). "
                "Por favor, revisa los nombres y vuelve a intentarlo. "
                "Sugerencia: Puedes pedirme que te muestre las tablas o columnas reales disponibles relacionadas con tu consulta.",
                "query_used": sql_query_to_execute,
                "parameters_used": params_to_execute,
                "structured_info": structured_info,
                "error": "DIRECT_SQL_ENTITY_VALIDATION_ERROR",
                "data": None
            }
            logger.info(f"PIPELINE_BENCHMARK: chatbot_pipeline END (direct SQL, validation error) - {time.time() - pipeline_start_time:.4f}s")
            return final_response
        # --- FIN VALIDACIÓN ROBUSTA ---
        
        results, error_msg = execute_query_with_timeout(
            db_connector, sql_query_to_execute, params_to_execute, logger, timeout_seconds=timeout_seconds
        )
        logger.info(f"PIPELINE_BENCHMARK: Direct SQL execution took {time.time() - start_time_direct_sql:.4f}s")

        final_response: Dict[str, Any]
        if error_msg:
            logger.error(f"Error en consulta SQL directa: {error_msg}. SQL: {sql_query_to_execute}")
            final_response = {
                "response": f"Error al ejecutar la consulta SQL directa: {error_msg}",
                "query_used": sql_query_to_execute,
                "parameters_used": params_to_execute,
                "structured_info": structured_info,
                "error": f"DIRECT_SQL_EXECUTION_ERROR: {error_msg}",
                "data": None
            }
        else:
            logger.info(f"Consulta SQL directa ejecutada con éxito. Resultados (tipo): {type(results)}")
            response_message = "Consulta directa ejecutada con éxito."
            data_for_response = results
            
            if isinstance(results, list):
                original_length = len(results)
                if original_length > MAX_RESPONSE_DATA_ROWS_CONFIG:
                    data_for_response = results[:MAX_RESPONSE_DATA_ROWS_CONFIG]
                    response_message = f"Consulta directa ejecutada. Se devuelven las primeras {MAX_RESPONSE_DATA_ROWS_CONFIG} filas de {original_length} resultados."
                elif original_length == 0:
                    response_message = "Consulta directa ejecutada. La consulta no devolvió filas."
                else:
                    response_message = f"Consulta directa ejecutada. {original_length} fila{'s' if original_length != 1 else ''} devuelta{'s' if original_length != 1 else ''}."
            elif results is None:
                response_message = "La consulta directa no devolvió un conjunto de resultados (resultado None)."
            else: # int (rowcount)
                response_message = f"Consulta directa ejecutada. Filas afectadas: {results}."

            final_response = {
                "query_used": sql_query_to_execute,
                "parameters_used": params_to_execute,
                "response": response_message,
                "structured_info": structured_info,
                "error": None,
                "data": data_for_response,
            }
        logger.info(f"DEBUG: [pipeline.py] Respuesta final directa SQL (resumen): {{'response_message': '{final_response.get('response')}', 'sql_present': {bool(final_response.get('query_used'))}, 'data_rows': len(final_response.get('data')) if isinstance(final_response.get('data'), list) else (1 if final_response.get('data') is not None else 0)}}")
        logger.info(f"PIPELINE_BENCHMARK: chatbot_pipeline END (direct SQL) - {time.time() - pipeline_start_time:.4f}s")
        return final_response
    else:
        # Continuar con el pipeline basado en LLM para preguntas en lenguaje natural
        logger.info("Procesando como pregunta en lenguaje natural (no SQL directo).")
        # 4. Fallback LLM para tablas si es necesario
        if not structured_info.get("tables"):
            logger.info("DEBUG: [pipeline.py] No hay tablas identificadas por preprocesamiento, intentando fallback LLM para tablas.")
            # Asumimos que las siguientes funciones están definidas e importadas correctamente:
            # create_prompt_for_table_identification, get_llm_config, call_llm_with_fallbacks
            # y que existe lógica para procesar llm_response_for_tables y actualizar structured_info.
            from src.llm_utils import create_prompt_for_table_identification, get_llm_config, call_llm_with_fallbacks
            messages_for_tables = create_prompt_for_table_identification(enriched_question, db_schema_str_simple, relaciones_tablas_str)
            llm_config_for_tables = get_llm_config("table_identification")

            if not db_schema_str_simple or not relaciones_tablas_str:
                logger.warning("DEBUG: [pipeline.py] Falta schema simple o relaciones para el LLM de identificación de tablas. Saltando.")
            elif not messages_for_tables or not llm_config_for_tables:
                logger.warning("DEBUG: [pipeline.py] Falta messages_for_tables o llm_config_for_tables. Saltando LLM para tablas.")
            else:
                try:
                    llm_response_for_tables = call_llm_with_fallbacks(messages_for_tables, llm_config_for_tables)
                    # Aquí debería ir la lógica para procesar llm_response_for_tables
                    # y actualizar structured_info["tables"] con identified_tables_from_llm.
                    # Ejemplo (debe ser adaptado):
                    # identified_tables_from_llm = parse_llm_table_response(llm_response_for_tables)
                    # if identified_tables_from_llm:
                    #     structured_info["tables"] = identified_tables_from_llm
                    #     logger.info(f"DEBUG: [pipeline.py] Tablas identificadas por LLM: {identified_tables_from_llm}")
                    # else:
                    #     logger.info("DEBUG: [pipeline.py] LLM no identificó tablas adicionales.")
                    logger.info(f"DEBUG: [pipeline.py] LLM para identificación de tablas invocado. Respuesta: {llm_response_for_tables}") # Ajustar según la estructura de respuesta
                except ImportError:
                    logger.error("DEBUG: [pipeline.py] Error de importación al intentar llamar a LLM para tablas (call_llm_with_fallbacks o dependencia).")
                except Exception as e_llm_tables:
                    logger.error(f"DEBUG: [pipeline.py] Excepción durante fallback LLM para tablas: {e_llm_tables}", exc_info=True)
        # ... (resto de la lógica de identified_tables_from_llm y actualización de structured_info)

    prev_step_time_tracker = time.time() # Actualizar tracker

    # 5. Mejora LLM para consultas complejas
    needs_llm_structured_extraction = False
    if not structured_info.get("conditions") and not structured_info.get("joins"):
        needs_llm_structured_extraction = True

    if needs_llm_structured_extraction and not disable_llm_enhancement:
        logger.info(f"DEBUG: [pipeline.py] Delegando al LLM la detección de complejidad y extracción estructurada para la pregunta: '{question}'")
        if not db_schema_str_simple:
            logger.warning("DEBUG: [pipeline.py] db_schema_str_simple no cargado. Necesario para extracción LLM.")
        if not relaciones_tablas_str: # Usar relaciones_tablas_str consistentemente
            logger.warning("DEBUG: [pipeline.py] relaciones_tablas_str no cargado. Necesario para extracción LLM.")
        if not db_schema_enhanced_content:
            logger.warning("DEBUG: [pipeline.py] db_schema_enhanced_content no cargado. Necesario para extracción LLM.")

        if db_schema_str_simple and relaciones_tablas_str and db_schema_enhanced_content:
            try:
                # Asumimos que las siguientes funciones están definidas e importadas correctamente:
                # create_prompt_for_structured_extraction, get_llm_config, call_llm_with_fallbacks
                # y que existe lógica para procesar llm_extracted_info y fusionar con structured_info.
                from src.llm_utils import create_prompt_for_structured_extraction, get_llm_config, call_llm_with_fallbacks
                messages_for_llm_extraction = create_prompt_for_structured_extraction(
                    enriched_question,
                    db_schema_str_simple, # O db_schema_enhanced_content según la necesidad del prompt
                    relaciones_tablas_str,
                    # Podría necesitar más argumentos como identified_tables
                )
                llm_config_for_extraction = get_llm_config("structured_extraction")
                
                llm_extracted_info = call_llm_with_fallbacks(messages_for_llm_extraction, llm_config_for_extraction)
                
                if llm_extracted_info and not llm_extracted_info.get("error"):
                    # Aquí debería ir la lógica para fusionar llm_extracted_info con structured_info.
                    # structured_info.update(llm_extracted_info) # Ejemplo de fusión
                    logger.info(f"DEBUG: [pipeline.py] LLM para extracción estructurada invocado. Información extraída: {llm_extracted_info}")
                    # Asegúrate de que structured_info se actualice correctamente. Por ejemplo:
                    # structured_info["tables"] = llm_extracted_info.get("tables", structured_info["tables"])
                    # structured_info["columns"] = llm_extracted_info.get("columns", structured_info["columns"])
                    # structured_info["conditions"] = llm_extracted_info.get("conditions", structured_info["conditions"])
                    # structured_info["joins"] = llm_extracted_info.get("joins", structured_info["joins"])
                    # structured_info["is_complex_query"] = llm_extracted_info.get("is_complex_query", structured_info["is_complex_query"])
                elif llm_extracted_info and llm_extracted_info.get("error"):
                    logger.error(f"DEBUG: [pipeline.py] LLM para extracción estructurada devolvió un error: {llm_extracted_info.get('error')}")
                else:
                    logger.warning("DEBUG: [pipeline.py] LLM para extracción estructurada no devolvió información.")

            except KeyError as ke:
                logger.error(f"DEBUG: [pipeline.py] KeyError durante la extracción LLM: {ke}. Esto puede indicar un problema con el formato de respuesta del LLM o el manejo de claves.", exc_info=True)
                llm_extracted_info = {"error": f"KeyError: {str(ke)}"} # Marcar error
            except Exception as e_llm_extract:
                logger.error(f"DEBUG: [pipeline.py] Excepción durante la extracción LLM: {e_llm_extract}", exc_info=True)
                llm_extracted_info = {"error": f"Exception: {str(e_llm_extract)}"} # Marcar error
        else:
            logger.warning("DEBUG: [pipeline.py] No se cargaron esquemas/relaciones necesarios para la extracción LLM completa. Saltando esta mejora.")
    else:
        logger.info(f"DEBUG: [pipeline.py] No se delega la detección de complejidad al LLM o está deshabilitado. Saltando extracción LLM estructurada.")

    logger.info(f"PIPELINE_BENCHMARK: LLM extraction and info merge (if any) - {time.time() - prev_step_time_tracker:.4f}s (Total: {time.time() - pipeline_start_time:.4f}s)")
    prev_step_time_tracker = time.time()

    if not is_direct_sql and needs_llm_structured_extraction and not disable_llm_enhancement and (not llm_extracted_info or llm_extracted_info.get("error")):
        error_detail = llm_extracted_info.get("error") if llm_extracted_info else "Información no extraída"
        logger.error(
            f"Fallo crítico: La extracción de información estructurada mediante LLM era necesaria pero falló o no se ejecutó. Detalle: {error_detail}"
        )
        return {
            "response": f"Error: No se pudo interpretar la pregunta compleja. Detalle: {error_detail}. Por favor, reformula la pregunta o contacta al administrador.",
            "query_used": None,
            "parameters_used": [],
            "structured_info": structured_info,
            "error": f"LLM_STRUCTURED_EXTRACTION_FAILED: {error_detail}",
            "data": None
        }

    # 6. Normalización y heurística de condiciones
    logger.debug(f"DEBUG: [pipeline.py] structured_info ANTES de Paso 5 (normalización): {json.dumps(structured_info, indent=2, ensure_ascii=False)}")
    structured_info = normalize_structured_info(structured_info)
    logger.info(f"DEBUG: [pipeline.py] structured_info DESPUÉS de Paso 5 (normalización): {json.dumps(structured_info, indent=2, ensure_ascii=False)}")
    logger.info(f"PIPELINE_BENCHMARK: structured_info normalized - {(time.time() - prev_step_time_tracker):.4f}s")
    prev_step_time_tracker = time.time()

    # Heurística para ID de paciente
    if structured_info.get("tables") and not structured_info.get("conditions"):
        main_table = structured_info["tables"][0]
        question_text_for_ids = question # o enriched_question
        patient_id_value = None
        id_source = None
        logger.info(f"Heurística ID Paciente: Iniciando. question_text_for_ids='{question_text_for_ids}', main_table='{main_table}'")
        explicit_id_regex = r'\b(ID|PATI_ID)\s*(?:=|ES|ES DE|CON)?\s*(\d+)\b'
        logger.info(f"Heurística: Regex explícito a usar: r'{explicit_id_regex}'")
        original_col_candidate_from_regex = None # Ya estaba inicializada
        try:
            explicit_match = re.search(explicit_id_regex, question_text_for_ids, re.IGNORECASE)
        except Exception as e:
            logger.error(f"Heurística: Intento 1 - ERROR durante re.search: {e}", exc_info=True)
            explicit_match = None
        if explicit_match:
            logger.info(f"Heurística: Intento 1 - Regex explícito ENCONTRADO. Grupos: {explicit_match.groups()}")
            try:
                patient_id_value = int(explicit_match.group(2))
                id_source = f"regex_explicit ('{explicit_id_regex}')"
                original_col_candidate_from_regex = explicit_match.group(1) 
            except ValueError:
                logger.warning(f"Heurística: Intento 1 - No se pudo convertir a int: {explicit_match.group(2)}")
                patient_id_value = None
            except Exception as e_group:
                logger.error(f"Heurística: Intento 1 - Error al procesar grupos del regex: {e_group}", exc_info=True)
                patient_id_value = None
        else:
            logger.info("Heurística: Intento 1 - Regex explícito NO ENCONTRADO.")
        if patient_id_value is None:
            logger.info("Heurística: Intento 1 falló o no aplicó. Procediendo a Intento 2 ('paciente' + número).")
            try:
                # Placeholder para la lógica del segundo intento (estaba comentada en el original)
                # fallback_match = re.search(r'\b(?:paciente|caso)\s+(\d+)\b', question_text_for_ids, re.IGNORECASE)
                # if fallback_match:
                #    patient_id_value = int(fallback_match.group(1))
                #    id_source = "regex_fallback_paciente_numero"
                #    logger.info(f"Heurística: Intento 2 - Regex fallback ENCONTRADO. ID: {patient_id_value}")
                pass # Mantener vacío si la lógica no está presente
            except ValueError:                    
                logger.warning(f"Heurística: Intento 2 - No se pudo convertir a int (si aplica).")
                patient_id_value = None
            except Exception as e_fallback:                    
                logger.error(f"Heurística: Intento 2 - Error durante regex fallback: {e_fallback}", exc_info=True)
                patient_id_value = None
        else:
            logger.info("Heurística: Intento 1 tuvo éxito. Saltando Intento 2.")
        if patient_id_value is not None:
            logger.info(f"Heurística: ID de paciente FINALMENTE encontrado: {patient_id_value} (fuente: {id_source}). Procediendo a determinar columna.")
            id_column_name = None
            if main_table in db_schema_dict and isinstance(db_schema_dict.get(main_table), dict): # Usar db_schema_dict
                logger.info(f"Heurística: Buscando columna ID en {main_table} (esquema: {list(db_schema_dict[main_table].keys())}). (Lógica de determinación de columna placeholder)")
                # Placeholder: Lógica para encontrar la columna ID en main_table
                # Ejemplo:
                # for col_name_candidate in ["PATI_ID", "ID_PACIENTE", "PACIENTE_ID", "ID"]: # Orden de preferencia
                #    if col_name_candidate in db_schema_dict[main_table]:
                #        id_column_name = col_name_candidate
                #        logger.info(f"Heurística: Columna ID encontrada: {id_column_name} en {main_table}")
                #        break
            if not id_column_name:
                logger.warning(f"Heurística: No se pudo determinar id_column_name para {main_table} automáticamente. Usando 'PATI_ID' como fallback o necesita configuración.")
                id_column_name = "PATI_ID" # Fallback o necesita lógica más robusta
            
            if id_column_name: # Solo añadir si tenemos una columna
                condition = {"column": id_column_name, "operator": "=", "value": patient_id_value}
                structured_info.setdefault("conditions", []).append(condition)
                logger.info(f"DEBUG: [pipeline.py] Condición de paciente extraída heurísticamente: {condition}")
            else:
                logger.warning(f"DEBUG: [pipeline.py] No se pudo determinar la columna para el ID de paciente {patient_id_value}, no se añadió condición.")
        else:
            logger.info(f"DEBUG: [pipeline.py] No se aplicó heurística de ID de paciente (ningún método tuvo éxito en encontrar un ID).")
        if not isinstance(structured_info, dict) or not structured_info.get("tables"):
            logger.error(f"DEBUG: [pipeline.py] Error CRÍTICO FINAL: structured_info no es un dict o no contiene tablas DESPUÉS de todos los intentos. Valor: {structured_info}")
            return {"response": "Error: No se pudieron identificar las tablas necesarias para la consulta (todos los intentos fallaron).", "sql": None, "data": None, "error": "Table identification failed"}

    logger.info(f"DEBUG: [pipeline.py] Usando structured_info para Paso 6 (Generación SQL): {json.dumps(structured_info, indent=2, ensure_ascii=False)}")
    logger.info("DEBUG: [pipeline.py] Paso 6: Generando SQL a partir de structured_info...")
    
    sql_generator_instance = SQLGenerator(
        allowed_tables=allowed_tables_list,
        allowed_columns=allowed_columns_map,
        enhanced_schema_path=SCHEMA_FULL_PATH
    )
    generation_result = sql_generator_instance.generate_sql(
        structured_info,
        db_schema_enhanced_content if db_schema_enhanced_content else db_schema_str_simple,
        relaciones_tablas_map
    )
    
    sql_query_str: Optional[str] = None
    params_list: List[Any] = []
    error_message_from_generator: Optional[str] = None

    if isinstance(generation_result, tuple) and len(generation_result) == 2:
        sql_query_str, params_list = generation_result
        logger.info(f"DEBUG: [pipeline.py] SQL y parámetros generados: SQL='{sql_query_str}', Params={params_list}")
    elif isinstance(generation_result, dict) and "response" in generation_result: # Asumiendo que el error viene en 'response'
        error_message_from_generator = generation_result["response"]
        sql_query_str = generation_result.get("sql_query") # Si el generador lo provee incluso con error
        params_list = generation_result.get("params", [])
        logger.error(f"DEBUG: [pipeline.py] generate_sql devolvió un diccionario de error: {error_message_from_generator}")
    else:
        logger.error(f"DEBUG: [pipeline.py] generate_sql devolvió un formato inesperado: {generation_result}")
        error_message_from_generator = "Error interno durante la generación de SQL: formato de respuesta inesperado del generador."
    
    # BENCHMARKING: prev_step_time_tracker se actualizó antes de la heurística de ID.
    # La generación SQL es el paso actual.
    current_time = time.time()
    logger.info(f"PIPELINE_BENCHMARK: SQL generated - {current_time - prev_step_time_tracker:.4f}s (Total: {current_time - pipeline_start_time:.4f}s)")
    prev_step_time_tracker = current_time


    if not sql_query_str and error_message_from_generator:
        logger.error(f"DEBUG: [pipeline.py] Error al generar la consulta SQL: {error_message_from_generator}")
        return {
            "response": error_message_from_generator,
            "query_used": sql_query_str, # Podría ser None o el SQL fallido
            "parameters_used": params_list,
            "structured_info": structured_info,
            "error": "SQL_GENERATION_VALIDATION_FAILED",
            "data": None
        }
    elif not sql_query_str:
        logger.error("DEBUG: [pipeline.py] Error: SQL string es None o vacío y no hay mensaje de error del generador.")
        return {
            "response": "Error: No se pudo generar la consulta SQL (string vacío o nulo).",
            "query_used": None,
            "parameters_used": [],
            "structured_info": structured_info,
            "error": "SQL_GENERATION_FAILED_EMPTY",
            "data": None
        }

    logger.info(f"DEBUG: [pipeline.py] SQL validado (o generado) correctamente: {sql_query_str}")

    # 7. Ejecución de la consulta SQL
    logger.info(f"DEBUG: [pipeline.py] Paso 7: Ejecutando consulta SQL: '{sql_query_str}' con params: {params_list}")
    execute_query_start_time = time.time()

    

    result, error = execute_query_with_timeout(db_connector, sql_query_str, params_list, logger, timeout_seconds)

    logger.info(f"PIPELINE_BENCHMARK: Query executed - {(time.time() - execute_query_start_time):.4f}s (Total: {(time.time() - pipeline_start_time):.4f}s)")

    if error:
        logger.error(f"Error al ejecutar la consulta SQL generada: {error}. SQL: {sql_query_str}, Params: {params_list}")
        return {
            "response": f"Error al ejecutar la consulta SQL: {error}",
            "query_used": sql_query_str,
            "parameters_used": params_list,
            "structured_info": structured_info,
            "error": f"SQL_EXECUTION_ERROR: {error}",
            "data": None
        }

        # Procesar resultado y formar respuesta final
        # ... (código existente para procesar `result` y formar `final_response`)
        # Asegurarse que MAX_RESPONSE_DATA_ROWS se reemplaza por MAX_RESPONSE_DATA_ROWS_CONFIG
        MAX_RESPONSE_DATA_ROWS = MAX_RESPONSE_DATA_ROWS_CONFIG # Usar la constante importada
        response_message = "Consulta ejecutada con éxito."
        data_for_response = result
        if isinstance(result, list):
            original_length = len(result)
            if original_length > MAX_RESPONSE_DATA_ROWS: # Usar la constante
                data_for_response = result[:MAX_RESPONSE_DATA_ROWS] # Usar la constante
                response_message = f"Consulta ejecutada con éxito. Se devuelven las primeras {MAX_RESPONSE_DATA_ROWS} filas de {original_length} resultados."
            elif original_length == 0:
                response_message = "Consulta ejecutada con éxito. La consulta no devolvió filas."
            else:
                response_message = f"Consulta ejecutada con éxito. {original_length} fila{'s' if original_length != 1 else ''} devuelta{'s' if original_length != 1 else ''}."
        elif result is None:
            response_message = "La consulta no devolvió resultados (resultado None)."
        else: # int (rowcount)
            response_message = f"Consulta ejecutada. Filas afectadas: {result}."
        
        final_response = {
            "query_used": sql_query_str,
            "parameters_used": params_list,
            "response": response_message,
            "structured_info": structured_info,
            "error": None,
            "data": data_for_response,
        }
        # ... (código existente para response_summary_for_log y retorno final)
        response_summary_for_log = {
            "response_message": final_response.get("response"),
            "sql_present": bool(final_response.get("query_used")),
            "data_rows": len(data_for_response) if isinstance(data_for_response, list) else (1 if data_for_response is not None else 0),
        }
        if isinstance(data_for_response, list) and data_for_response:
            response_summary_for_log["data_first_row_type"] = type(data_for_response[0]).__name__
        elif data_for_response is not None:
             response_summary_for_log["data_type"] = type(data_for_response).__name__
        logger.info(f"DEBUG: [pipeline.py] Respuesta final preparada (resumen): {response_summary_for_log}")
        logger.info(f"PIPELINE_BENCHMARK: chatbot_pipeline END - {(time.time() - pipeline_start_time):.4f}s")
        return final_response

# --- Sub-pipeline: Médicos para condición y medicación específicas ---

# def handle_condition_medication_doctors_subpipeline(
#     condition_term: str,
#     medication_term: str,
#     db_structure: Dict[str, Any],
#     db_connector: DBConnector,
#     logger: logging.Logger,
#     timeout_seconds: int = 20
# ) -> Dict[str, Any]:
#     \"\"\"
#     Sub-pipeline que usa CTEs para obtener médicos que trataron pacientes con una condición y medicación dadas.
#     \"\"\"
#     # Identificar tablas relevantes
#     diag_table = next((t for t in db_structure if 'DIABETES' in t.upper()), None)
#     med_table = next((t for t in db_structure if 'MEDICATIONS' in t.upper()), None)
#     patient_med_table = next((t for t in db_structure if 'USUAL_MEDICATION' in t.upper()), None)
#     # Seleccionar la tabla/columna de doctor adecuada
#     if 'EPIS_EPISODES' in db_structure:
#         appt_table = 'EPIS_EPISODES'
#         doctor_col = 'EPIS_REFERAL_DOCTOR'
#     else:
#         appt_table = next((t for t in db_structure if 'APPOINTMENTS' in t.upper()), None)
#         # Fallback: usar APPR_ID como profesional si no hay columna DOCTOR explícita
#         doctor_col = next((c for c in db_structure.get(appt_table, []) if 'APPR' in c.upper()), 'APPR_ID')

#     # Columnas clave en cada tabla
#     pati_id = 'PATI_ID'
#     cond_col = next((c for c in db_structure.get(diag_table, []) if 'DESCRIPTION' in c.upper()), 'DESCRIPTION')
#     med_id = next((c for c in db_structure.get(med_table, []) if 'MEDI_ID' in c.upper()), 'MEDI_ID')
#     med_name_col = next((c for c in db_structure.get(med_table, []) if 'NAME' in c.upper()), 'MEDI_NAME')
#     doctor_col = next((c for c in db_structure.get(appt_table, []) if 'DOCTOR' in c.upper()), 'APPO_DOCTOR_ID')

#     # Construir consulta con CTEs
#     sql = f\"\""
# WITH pacientes_condicion AS (
#   SELECT {pati_id}
#   FROM {diag_table}
#   WHERE {cond_col} LIKE ?
# ), pacientes_medicacion AS (
#   SELECT {pati_id}
#   FROM {patient_med_table}
#   JOIN {med_table} ON {patient_med_table}.{med_id} = {med_table}.{med_id}
#   WHERE {med_table}.{med_name_col} LIKE ?
#     AND {pati_id} IN (SELECT {pati_id} FROM pacientes_condicion)
# )
# SELECT DISTINCT {doctor_col} AS doctor_id
# FROM {appt_table}
# WHERE {pati_id} IN (SELECT {pati_id} FROM pacientes_medicacion);
# \"\""
#     params = [f"%{condition_term}%", f"%{medication_term}%"]
    
#     # Ejecutar consulta con timeout
#     result, error = execute_query_with_timeout(db_connector, sql, params, logger, timeout_seconds)
#     if error:
#         return {
#             "response": f"Error en sub-pipeline: {error}",
#             "query_used": sql,
#             "parameters_used": params,
#             "data": None,
#             "error": "SUBPIPELINE_ERROR"
#         }

#     # Extraer lista de doctores
#     doctors = [row.get('doctor_id') for row in result] if isinstance(result, list) else []
#     return {
#         "response": f"Médicos que trataron pacientes con '{condition_term}' tomando '{medication_term}': {doctors}",
#         "query_used": sql,
#         "parameters_used": params,
#         "data": doctors,
#         "error": None
#     }
