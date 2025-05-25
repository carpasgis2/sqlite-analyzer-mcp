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
# Se modifica la importación para que coincida con las definiciones en db_config.py
try:
    from src.db_config import get_db_connector, DEFAULT_DB_CONFIG, DBConnector
    pipeline_module_logger.info("Importado src.db_config.get_db_connector, DEFAULT_DB_CONFIG y DBConnector.")
except ImportError:
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
            logger.error(f"Ninguna tabla válida identificada después del filtrado: {tablas}")
            return "SELECT 'No se identificaron tablas válidas para la consulta' AS mensaje", []

        structured_info["tables"] = tablas_validas
        
        try: # INNER TRY
            allowed_tables_list = list(db_structure.keys()) if db_structure else []
            allowed_columns_map = {}
            if db_structure:
                for table_name_for_map, cols_data in db_structure.items():
                    if isinstance(cols_data, dict) and isinstance(cols_data.get('columns'), list):
                        col_names_for_map = []
                        for col_entry in cols_data['columns']:
                            if isinstance(col_entry, dict) and isinstance(col_entry.get('name'), str):
                                col_names_for_map.append(col_entry['name'])
                            elif isinstance(col_entry, str): # Si la columna es solo un string
                                col_names_for_map.append(col_entry)
                        allowed_columns_map[table_name_for_map] = col_names_for_map
                    else:
                        # Manejar el caso donde cols_data no es un dict o no tiene 'columns'
                        logger.warning(f"Estructura inesperada para la tabla '{table_name_for_map}' en db_structure. No se pudieron extraer columnas.")
                        allowed_columns_map[table_name_for_map] = []
        
            sql_generator_instance = SQLGenerator(allowed_tables=allowed_tables_list, allowed_columns=allowed_columns_map)


            sql_gen = SQLGenerator(allowed_tables=allowed_tables_list, allowed_columns=allowed_columns_map)
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
                allowed_tables_for_validation = list(db_structure.keys()) if db_structure else []
                allowed_columns_for_validation = {}
                if db_structure:
                    for table_name, table_data in db_structure.items():
                        if isinstance(table_data, dict) and 'columns' in table_data and isinstance(table_data['columns'], list):
                            column_names = []
                            for col_info in table_data['columns']:
                                if isinstance(col_info, dict) and isinstance(col_info.get('name'), str):
                                    column_names.append(col_info['name'])
                                elif isinstance(col_info, str):
                                    column_names.append(col_info)
                            allowed_columns_for_validation[table_name.upper()] = [col.upper() for col in column_names if col] # Asegurar mayúsculas y que col no sea None
                        else:
                            logger.warning(f"Estructura inesperada para la tabla '{table_name}' en db_structure al preparar para validación.")
                            allowed_columns_for_validation[table_name.upper()] = []
                
                is_safe, validation_msg = new_whitelist_validate_query(structured_info, allowed_tables_for_validation, allowed_columns_for_validation)
                logger.info(f"Paso 6.1 completado. ¿Validación de partes estructuradas segura? {is_safe}, Mensaje: {validation_msg}")

                if not is_safe:
                    logger.error(f"Validación de whitelist de partes estructuradas fallida: {validation_msg} para structured_info: {structured_info}")
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
        corrected_sql_query = correct_column_names(sql_query_str, tablas_validas, db_structure)
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

def correct_column_names(sql_query, tables, db_structure):
    logger = logging.getLogger(__name__)
    logger.info(f"correct_column_names: Iniciando. SQL: {sql_query}, Tablas: {tables}")

    # --- NUEVO: Normalizar nombres de columnas y tablas usando mappings semánticos antes de aplicar heurísticas ---
    logger.info("correct_column_names: Cargando mapeos semánticos...")
    table_map, column_map = load_semantic_mappings()
    logger.info(f"correct_column_names: Mapeos semánticos cargados. table_map size: {len(table_map)}, column_map size: {len(column_map)}")

    # Corregir nombres de tablas en la consulta (sinónimos y fuzzy)
    logger.info("correct_column_names: Iniciando corrección de nombres de tablas.")
    original_sql_query_for_table_correction = sql_query
    for t_idx, t in enumerate(tables):
        logger.debug(f"correct_column_names: Procesando tabla {t_idx + 1}/{len(tables)}: '{t}'")
        t_norm = normalize_name(t)
        mapped = table_map.get(t_norm)
        if mapped and mapped != t:
            logger.debug(f"correct_column_names: Mapeo directo de tabla encontrado para '{t}' -> '{mapped}'")
            sql_query = re.sub(fr'\b{re.escape(t)}\b', mapped, sql_query, flags=re.IGNORECASE) # Corrected
        # Fuzzy: si no hay match directo, buscar parecido
        elif not mapped:
            logger.debug(f"correct_column_names: No hay mapeo directo para tabla '{t}'. Intentando fuzzy match con {len(table_map.keys())} candidatos.")
            # Ensure list(table_map.keys()) does not contain None or non-string items if difflib requires it
            valid_table_map_keys = [str(k) for k in table_map.keys() if k is not None]
            close = difflib.get_close_matches(t_norm, valid_table_map_keys, n=1, cutoff=0.7)
            if close:
                # Asegúrate de que close[0] es una clave válida en table_map
                if close[0] in table_map:
                    logger.debug(f"correct_column_names: Fuzzy match para tabla '{t}' -> '{table_map[close[0]]}' (original: '{close[0]}')")
                    sql_query = re.sub(fr'\b{re.escape(t)}\b', table_map[close[0]], sql_query, flags=re.IGNORECASE) # Corrected
                else:
                    logger.warning(f"correct_column_names: Clave '{close[0]}' de fuzzy match no encontrada en table_map. Se omite la corrección para '{t}'.")
            else:
                logger.debug(f"correct_column_names: No se encontró fuzzy match para tabla '{t}'") # Corregido
    if original_sql_query_for_table_correction != sql_query:
        logger.info(f"correct_column_names: SQL después de corrección de tablas: {sql_query}")
    else:
        logger.info("correct_column_names: No se realizaron cambios en SQL durante la corrección de tablas.")

    # Corregir nombres de columnas en la consulta (sinónimos y fuzzy)
    logger.info("correct_column_names: Iniciando corrección de nombres de columnas (sinónimos).")
    original_sql_query_for_column_correction = sql_query
    
    column_map_items = list(column_map.items()) # Cache items
    for syn_idx, (syn, (syn_tbl, syn_col)) in enumerate(column_map_items):
        if syn_idx % 500 == 0: 
            logger.debug(f"correct_column_names: Procesando sinónimo de columna {syn_idx + 1}/{len(column_map_items)}: '{syn}' -> ('{syn_tbl}', '{syn_col}')")
        
        if syn_col and syn: # Ensure syn_col and syn are not None or empty
            try:
                # Check if the synonym (syn) is present in the query and needs replacement by syn_col
                if re.search(fr'\b{re.escape(str(syn))}\b', sql_query, re.IGNORECASE):
                    logger.debug(f"correct_column_names: Sinónimo de columna '{syn}' encontrado. Reemplazando con '{syn_col}'")
                    sql_query = re.sub(fr'\b{re.escape(str(syn))}\b', str(syn_col), sql_query, flags=re.IGNORECASE) # Corrected
            except re.error as e:
                logger.error(f"correct_column_names: Error de regex procesando sinónimo '{syn}': {e}")
                continue # Skip to next synonym on regex error

    if original_sql_query_for_column_correction != sql_query:
        logger.info(f"correct_column_names: SQL después de corrección de sinónimos de columnas: {sql_query}")
    else:
        logger.info("correct_column_names: No se realizaron cambios en SQL durante la corrección de sinónimos de columnas.")

    # Validación clásica: si aún quedan columnas/tablas no válidas, intentar heurística clásica
    logger.info("correct_column_names: Iniciando validación clásica y corrección heurística de columnas.")
    valid_columns = {}
    logger.debug("correct_column_names: Creando mapa de columnas válidas por tabla para heurística.")
    for table_idx, table in enumerate(tables):
        if table in db_structure:
            cols_data = db_structure[table].get("columns", [])
            valid_columns[table] = [col.get("name", "") for col in cols_data if isinstance(col, dict) and col.get("name")]
        else:
            logger.warning(f"correct_column_names: Tabla '{table}' no encontrada en db_structure durante la creación de valid_columns para heurística.")
    logger.debug("correct_column_names: Mapa de columnas válidas para heurística creado.")

    original_sql_query_for_heuristic_correction = sql_query
    for table_h_idx, table_name_heuristic in enumerate(tables): # Renombrar variable de bucle
        logger.debug(f"correct_column_names: Heurística para tabla {table_h_idx + 1}/{len(tables)}: '{table_name_heuristic}'")
        if table_name_heuristic in valid_columns and valid_columns[table_name_heuristic]:
            qualified_column_pattern = fr'\\b{re.escape(table_name_heuristic)}\\.([a-zA-Z0-9_]+)\\b'
            logger.debug(f"correct_column_names: Buscando con patrón calificado específico: {qualified_column_pattern} en SQL: {sql_query}")
            current_sql_offset = 0
            new_sql_parts = []
            for match_idx, match in enumerate(re.finditer(qualified_column_pattern, sql_query, re.IGNORECASE)):
                col_name_from_query = match.group(1)
                full_match_str = match.group(0) # e.g., "PATI_PATIENT_ALLERGIES.PATI_ID"
                logger.debug(f"correct_column_names: ... Heurística (calificada): Encontrado '{full_match_str}', columna extraída: '{col_name_from_query}' para tabla '{table_name_heuristic}'")
                if col_name_from_query.upper() in [vc.upper() for vc in valid_columns[table_name_heuristic]]:
                    logger.debug(f"correct_column_names: ... Columna '{col_name_from_query}' ya es válida para tabla '{table_name_heuristic}'.")
                    new_sql_parts.append(sql_query[current_sql_offset:match.end()])
                    current_sql_offset = match.end()
                    continue
                
                logger.debug(f"correct_column_names: ... Columna '{col_name_from_query}' no es directamente válida. Intentando fuzzy match con {valid_columns[table_name_heuristic]}.")
                valid_cols_for_diff = [str(vc) for vc in valid_columns[table_name_heuristic] if vc is not None]
                best_match_list = difflib.get_close_matches(col_name_from_query, valid_cols_for_diff, n=1, cutoff=0.6)
                
                if best_match_list:
                    corrected_col_name = best_match_list[0]
                    logger.info(f"correct_column_names: Corregido (heurística, calificada) nombre de columna: '{col_name_from_query}' -> '{corrected_col_name}' en tabla '{table_name_heuristic}' (match original: '{full_match_str}')")
                    corrected_full_match = f"{table_name_heuristic}.{corrected_col_name}"
                    new_sql_parts.append(sql_query[current_sql_offset:match.start()])
                    new_sql_parts.append(corrected_full_match)
                    current_sql_offset = match.end()
                else:
                    logger.debug(f"correct_column_names: ... No se encontró fuzzy match para columna calificada '{col_name_from_query}' en tabla '{table_name_heuristic}'. Manteniendo original.")
                    new_sql_parts.append(sql_query[current_sql_offset:match.end()])
                    current_sql_offset = match.end()
            
            new_sql_parts.append(sql_query[current_sql_offset:])
            sql_query = "".join(new_sql_parts)
            logger.debug(f"correct_column_names: Heurística para columnas no calificadas (omitida por ahora para '{table_name_heuristic}').")
        elif table_name_heuristic not in db_structure:
            logger.warning(f"correct_column_names: Tabla '{table_name_heuristic}' no en db_structure durante la corrección heurística.")
        elif not valid_columns.get(table_name_heuristic):
            logger.warning(f"correct_column_names: No hay columnas válidas definidas para la tabla '{table_name_heuristic}' en la corrección heurística.")
            
    if original_sql_query_for_heuristic_correction != sql_query:
        logger.info(f"correct_column_names: SQL después de corrección heurística: {sql_query}")
    else:
        logger.info("correct_column_names: No se realizaron cambios en SQL durante la corrección heurística.")

    logger.info(f"correct_column_names: Finalizando. SQL resultante: {sql_query}")
    return sql_query

def chatbot_pipeline(
    question: str, 
    db_connector: DBConnector, 
    logger: logging.Logger, 
    relevant_tables: Optional[List[str]] = None, 
    conditions: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
    timeout_seconds: int = 10,
    max_retries: int = 1,
    use_llm_for_complex_query: bool = False,
    force_tables: Optional[List[str]] = None,
    force_conditions: Optional[List[str]] = None,
    force_query_type: Optional[str] = None, 
    disable_rag: bool = False,
    disable_llm_enhancement: bool = False,
    disable_direct_sql_execution: bool = False
) -> Dict[str, Any]:
    try:
        pipeline_start_time = time.time()
        prev_step_time_tracker = pipeline_start_time # Inicializar prev_step_time_tracker aquí
        logger.info(f"DEBUG: [pipeline.py] Inicio de chatbot_pipeline. Pregunta: '{question[:500]}...', Tablas Relevantes: {relevant_tables}, Condiciones: {conditions}, DisableDirectSQL: {disable_direct_sql_execution}")
        logger.info(f"PIPELINE_BENCHMARK: chatbot_pipeline START")

        # 1. Preprocesamiento y obtención de estructura/diccionario
        logger.info("DEBUG: [pipeline.py] Obteniendo estructura de la BD (común)...")
        db_structure = db_connector.get_database_structure()  
        logger.info(f"DEBUG: [pipeline.py] Estructura de BD obtenida: {type(db_structure)}")
        if not db_structure:
            logger.error("No se pudo obtener la estructura de la base de datos.")
            return {"response": "Error crítico: No se pudo cargar la estructura de la base de datos.", "query": "", "data": None, "error_type": "db_structure_failure"}

        allowed_tables_list = list(db_structure.keys()) if db_structure else []
        allowed_columns_map = {}
        if db_structure:
            for table_name, table_data in db_structure.items():
                # Asumiendo que table_data['columns'] es una lista de diccionarios con una clave 'name'
                column_names = [col.get('name') for col in table_data.get('columns', []) if col.get('name')]
                allowed_columns_map[table_name] = column_names
        # ...

        # Cargar esquemas y relaciones necesarios más arriba
        db_schema_str_simple = ""
        db_schema_enhanced_content = "" # Anteriormente db_schema_str_full_details
        relaciones_tablas_str_for_llm = "" # Para el prompt del LLM
        relaciones_tablas_map = {} # Para SQLGenerator

        try:
            db_schema_str_simple = load_schema_as_string(SCHEMA_SIMPLE_PATH)
            db_schema_enhanced_content = load_schema_as_string(SCHEMA_FULL_PATH)
            relaciones_tablas_str_for_llm = load_schema_as_string(RELACIONES_PATH_FOR_LLM)
            # Cargar el mapa de relaciones para SQLGenerator
            relaciones_tablas_map = ensure_relationships_map(db_structure) 
            if not relaciones_tablas_map: # Asegurar que no sea None si ensure_relationships_map falla
                relaciones_tablas_map = {}
                logger.warning("El mapa de relaciones (relaciones_tablas_map) está vacío después de ensure_relationships_map.")

        except Exception as e_load_schemas_early:
            logger.error(f"Error cargando esquemas/relaciones esenciales al inicio del pipeline: {e_load_schemas_early}", exc_info=True)
            # Podríamos decidir devolver un error aquí si son críticos

        logger.info("DEBUG: [pipeline.py] Cargando diccionario de términos (común)...")
        terms_dict = {}
        try:
            if 'TERMS_DICT_PATH' in globals() and TERMS_DICT_PATH:
                terms_dict = load_terms_dictionary(TERMS_DICT_PATH) 
                logger.info(f"PIPELINE_BENCHMARK: terms_dict loaded - {(time.time() - pipeline_start_time):.4f}s")
            else:
                logger.warning("TERMS_DICT_PATH no está definido o está vacío. No se cargará el diccionario de términos.")
                terms_dict = {}
        except Exception as e_terms:
            logger.error(f"Error al cargar el diccionario de términos: {e_terms}", exc_info=True)
            terms_dict = {}

        enriched_question, identified_tables_from_preprocessing, initial_query_type = preprocess_question(question, terms_dict)
        logger.info(f"PIPELINE_BENCHMARK: preprocess_question completed - {(time.time() - prev_step_time_tracker):.4f}s")
        prev_step_time_tracker = time.time() # Actualizar para el siguiente paso

        logger.info(f"DEBUG: [pipeline.py] Pregunta preprocesada: '{enriched_question[:500]}...', Tablas por preproc: {identified_tables_from_preprocessing}, Tipo por preproc: {initial_query_type}")

        # 2. Inicializar structured_info
        structured_info: Dict[str, Any] = {
            "tables": force_tables if force_tables is not None else identified_tables_from_preprocessing,
            "columns": [], 
            "conditions": force_conditions if force_conditions is not None else (conditions if conditions is not None else []),
            "query_type": force_query_type if force_query_type is not None else initial_query_type,
            "joins": [], 
            "original_question": question,
            "enriched_question": enriched_question,
            "is_complex_query": False
        }
        logger.info(f"DEBUG: [pipeline.py] structured_info inicializado: {structured_info}")

        # 3. Detección y ejecución de SQL directo (temprano)
        logger.info(f"DEBUG_DIRECT_SQL: Initial check. disable_direct_sql_execution={disable_direct_sql_execution}, question='{question[:100]}...'")
        is_direct_sql = False
        if not disable_direct_sql_execution:
            question_upper = question.strip().upper()
            sql_keywords_initial = ("SELECT", "INSERT", "UPDATE", "DELETE", "WITH", "CREATE", "ALTER", "DROP", "EXPLAIN", "PRAGMA")
            sql_keywords_contained = ["FROM", "WHERE", "JOIN", "GROUP", "ORDER", "VALUES", "SET", "TABLE", "LIMIT"]
            logger.info(f"DEBUG_DIRECT_SQL: Evaluating question_upper='{question_upper[:200]}...'")
            starts_with_sql_keyword = any(question_upper.startswith(keyword) for keyword in sql_keywords_initial)
            logger.info(f"DEBUG_DIRECT_SQL: starts_with_sql_keyword={starts_with_sql_keyword}")
            contained_sql_keywords_count = sum(1 for keyword in sql_keywords_contained if keyword in question_upper)
            found_keywords = [keyword for keyword in sql_keywords_contained if keyword in question_upper]
            logger.info(f"DEBUG_DIRECT_SQL: contained_sql_keywords_count={contained_sql_keywords_count} (Found: {found_keywords})")
            is_information_schema_query = "INFORMATION_SCHEMA" in question_upper
            logger.info(f"DEBUG_DIRECT_SQL: is_information_schema_query={is_information_schema_query}")
            if is_information_schema_query:
                is_direct_sql = True
                logger.info("DEBUG_DIRECT_SQL: Classified as direct SQL due to 'INFORMATION_SCHEMA'.")
            elif (starts_with_sql_keyword and contained_sql_keywords_count > 0) or \
                 (contained_sql_keywords_count > 1 and len(question_upper.split()) > 3):
                is_direct_sql = True
                logger.info(f"DEBUG_DIRECT_SQL: Classified as direct SQL based on keyword start/contain criteria. Starts: {starts_with_sql_keyword}, Contained count: {contained_sql_keywords_count}, Word count: {len(question_upper.split())}")
            else:
                logger.info(f"DEBUG_DIRECT_SQL: NOT classified as direct SQL by detection logic. Starts: {starts_with_sql_keyword}, Contained count: {contained_sql_keywords_count}, Word count: {len(question_upper.split())}")
        else:
            logger.info("DEBUG_DIRECT_SQL: Direct SQL execution is disabled by 'disable_direct_sql_execution' parameter being True.")

        if is_direct_sql:
            logger.info(f"Procesando como consulta SQL directa: '{question[:200]}...'")
            sql_query_str = question 
            params_list = [] 
            logger.info(f"DEBUG: [pipeline.py] Paso 7: Ejecutando consulta SQL directa: \\'{sql_query_str[:500]}\\' con params: {params_list}")
            try:
                execute_direct_sql_start_time = time.time()
                result, error = execute_query_with_timeout(db_connector, sql_query_str, params_list, logger, timeout_seconds)
                execute_direct_sql_end_time = time.time()
                logger.info(f"PIPELINE_BENCHMARK: Direct SQL execution took {(execute_direct_sql_end_time - execute_direct_sql_start_time):.4f}s")
                if error:
                    logger.error(f"Error al ejecutar SQL directo: {error}")
                    final_response = {"response": f"Error al ejecutar la consulta SQL directa: {error}", "sql": sql_query_str, "data": None, "error": error}
                else:
                    logger.info(f"SQL directo ejecutado con éxito. Resultado (primeras 200 chars): {str(result)[:200]}")
                    final_response = {
                        "query_used": sql_query_str,
                        "parameters_used": params_list,
                        "response": "Consulta SQL directa ejecutada con éxito.",
                        "data": result, 
                        "error": None
                    }
            except Exception as e_exec:
                logger.error(f"Excepción inesperada durante la ejecución de SQL directo: {e_exec}", exc_info=True)
                final_response = {"response": f"Error inesperado al procesar la consulta SQL directa: {e_exec}", "sql": sql_query_str, "data": None, "error": str(e_exec)}
            end_time = time.time()
            logger.info(f"PIPELINE_BENCHMARK: chatbot_pipeline END (direct SQL) - {(end_time - pipeline_start_time):.4f}s")
            return final_response

        # 4. Fallback LLM para tablas si es necesario
        # db_schema_str_simple y relaciones_tablas_str_for_llm ya deberían estar cargados
        if not structured_info.get("tables"):
            logger.info("DEBUG: [pipeline.py] No hay tablas. Intentando fallback con LLM para identificar TODAS las tablas relevantes...")
            system_message_template = '''Eres un asistente experto en SQL y bases de datos médicas. Tu tarea es analizar la pregunta del usuario y devolver una LISTA CONCISA de las tablas ABSOLUTAMENTE ESENCIALES para responder la pregunta. Prioriza las tablas de datos principales sobre las tablas de parámetros o unidades.

Contexto de la Base de Datos:
Esquema Simplificado:
 {}

Relaciones entre Tablas (ayuda a entender cómo se conectan los datos):
 {}

Pregunta del usuario: {}

Consideraciones Adicionales:
- Enfócate en las tablas que contienen los datos directamente solicitados.
- Evita incluir tablas de parámetros (ej. aquellas que empiezan con 'PARA_') o tablas de unidades a menos que la pregunta trate EXPLÍCITAMENTE sobre esos parámetros o unidades. Por ejemplo, si la pregunta es sobre 'unidades de medida', entonces 'PARA_MEASUREMENT_UNITS' sería relevante. De lo contrario, probablemente no lo sea.
- Devuelve solo las tablas estrictamente necesarias. Si una tabla de parámetros solo describe códigos en otra tabla principal que ya has seleccionado, y la pregunta no pide la descripción de esos códigos, no incluyas la tabla de parámetros.

Responde ÚNICAMENTE con un objeto JSON que contenga una clave "tables" con la lista de nombres de tablas.
Ejemplo: {{"tables": ["TABLE_X", "TABLE_Y"]}}'''
            llm_config_for_tables = {
                "api_key": os.environ.get("DEEPSEEK_API_KEY"),
                "base_url": os.environ.get("DEEPSEEK_API_URL"),
                "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
                "provider": "deepseek", 
                "temperature": 0.0,
                "max_tokens": 150, 
                "response_format": {"type": "json_object"} 
            }
            if not db_schema_str_simple: # Doble chequeo por si falló la carga inicial
                try:
                    db_schema_str_simple = load_schema_as_string(SCHEMA_SIMPLE_PATH)
                except Exception as e_load_simple_schema_fallback:
                    logger.error(f"Error cargando SCHEMA_SIMPLE_PATH para fallback de tablas LLM: {e_load_simple_schema_fallback}", exc_info=True)
                    db_schema_str_simple = "{}" # Asegurar que sea string
            if not relaciones_tablas_str_for_llm: # Doble chequeo
                try:
                    relaciones_tablas_str_for_llm = load_schema_as_string(RELACIONES_PATH_FOR_LLM)
                except Exception as e_load_rel_fallback:
                    logger.error(f"Error cargando RELACIONES_PATH_FOR_LLM para fallback de tablas LLM: {e_load_rel_fallback}", exc_info=True)
                    relaciones_tablas_str_for_llm = "{}" # Asegurar que sea string
            
            system_content_for_llm = system_message_template.format(db_schema_str_simple, relaciones_tablas_str_for_llm, question) 
            messages_for_tables = [
                {"role": "system", "content": system_content_for_llm},
                {"role": "user", "content": f"Pregunta del usuario: {question}"}
            ]
            try:
                from src.llm_utils import call_llm_with_fallbacks, extract_json_from_llm_response 
            except ImportError:
                from llm_utils import call_llm_with_fallbacks, extract_json_from_llm_response
            llm_response_for_tables = call_llm_with_fallbacks(messages_for_tables, llm_config_for_tables)
            identified_tables_from_llm = []
            if llm_response_for_tables and not str(llm_response_for_tables).startswith("ERROR:"):
                logger.info(f"DEBUG: [pipeline.py] Respuesta LLM para fallback de tablas (robust): {llm_response_for_tables}")
                extracted_data = extract_json_from_llm_response(str(llm_response_for_tables))
                if extracted_data: 
                    raw_tables = extracted_data.get("tables", [])
                    if isinstance(raw_tables, list) and all(isinstance(t, str) for t in raw_tables):
                        known_tables_set = set(db_structure.keys()) if db_structure else set()
                        valid_tables_from_llm_initial = [table for table in raw_tables if table in known_tables_set]
                        
                        if len(valid_tables_from_llm_initial) < len(raw_tables):
                            logger.warning(f"LLM fallback (robust) sugirió tablas no existentes: {set(raw_tables) - set(valid_tables_from_llm_initial)}. Se usarán solo las válidas.")

                        # --- INICIO DE NUEVO FILTRADO ---
                        valid_tables_from_llm = list(valid_tables_from_llm_initial) # Copiar para modificar
                        
                        if len(valid_tables_from_llm) > 0: # Aplicar filtro solo si hay tablas válidas
                            AUXILIARY_TABLES_TO_FILTER = {"PARA_MEASUREMENT_UNITS", "PARA_UNITS"}
                            justification_keywords = {"unidad", "unidades", "medida", "medidas", "parámetro", "parámetros", "código", "códigos", "terminología", "término"}
                            
                            question_lower_normalized = normalize_text(question) 

                            is_justified_by_question = any(keyword in question_lower_normalized for keyword in justification_keywords)
                            
                            candidates_for_removal = AUXILIARY_TABLES_TO_FILTER.intersection(set(valid_tables_from_llm))

                            if candidates_for_removal and not is_justified_by_question:
                                non_auxiliary_tables_present = set(valid_tables_from_llm) - AUXILIARY_TABLES_TO_FILTER
                                if non_auxiliary_tables_present:
                                    tables_actually_removed = set()
                                    temp_filtered_tables = []
                                    for t_filter in valid_tables_from_llm: # Renombrar variable de bucle
                                        if t_filter in AUXILIARY_TABLES_TO_FILTER:
                                            logger.info(f"Filtrado post-LLM: Tabla auxiliar '{t_filter}' candidata a eliminación y no justificada por la pregunta. Será eliminada porque hay otras tablas principales.")
                                            tables_actually_removed.add(t_filter)
                                        else:
                                            temp_filtered_tables.append(t_filter)
                                    if tables_actually_removed:
                                         valid_tables_from_llm = temp_filtered_tables
                                         logger.info(f"Filtrado post-LLM: Tablas auxiliares eliminadas: {tables_actually_removed}. Tablas restantes: {valid_tables_from_llm}")
                                else:
                                    logger.info(f"Filtrado post-LLM: Tablas candidatas a eliminación ({candidates_for_removal}) no eliminadas porque no hay otras tablas 'principales' y la pregunta podría referirse a ellas, o no están justificadas pero son la única opción.")
                            elif candidates_for_removal and is_justified_by_question:
                                logger.info(f"Filtrado post-LLM: Tablas auxiliares ({candidates_for_removal}) mantenidas porque la pregunta parece justificarlas con palabras clave.")
                        # --- FIN DE NUEVO FILTRADO ---

                        if valid_tables_from_llm: # Usar la lista potencialmente filtrada
                            identified_tables_from_llm = valid_tables_from_llm
                            structured_info["tables"] = identified_tables_from_llm
                            logger.info(f"DEBUG: [pipeline.py] structured_info['tables'] poblado por LLM fallback (robust, post-filtrado): {identified_tables_from_llm}")
                        else:
                            logger.warning("LLM fallback (robust) no identificó tablas válidas o existentes (o fueron filtradas) a pesar de una respuesta JSON válida.")
                    else:
                        logger.warning(f"LLM fallback (robust) devolvió un JSON, pero la clave 'tables' no es una lista de strings: {raw_tables}")
                else:
                    logger.error(f"LLM fallback (robust) no pudo extraer JSON de la respuesta: {llm_response_for_tables}")
            else:
                logger.error(f"LLM fallback (robust) no devolvió una respuesta válida o devolvió un error: {llm_response_for_tables}")
            if not identified_tables_from_llm:
                logger.info("DEBUG: [pipeline.py] Fallback robusto de LLM para tablas falló.")

        # 5. Mejora LLM para consultas complejas
        # --- NUEVO: Delega la detección de complejidad al LLM ---
        # Si no hay condiciones ni joins, o la pregunta parece ambigua, pide al LLM que decida si es compleja
        needs_llm_structured_extraction = False
        if not structured_info.get("conditions") and not structured_info.get("joins"):
            # Si el usuario fuerza, o si no hay condiciones ni joins, pide al LLM decidir
            needs_llm_structured_extraction = True

        if needs_llm_structured_extraction and not disable_llm_enhancement:
            logger.info(f"DEBUG: [pipeline.py] Delegando al LLM la detección de complejidad y extracción estructurada para la pregunta: '{question}'")
            # db_schema_str_simple, relaciones_tablas_str_for_llm, db_schema_enhanced_content ya deberían estar cargados
            if not db_schema_str_simple: # Doble chequeo
                db_schema_str_simple = load_schema_as_string(SCHEMA_SIMPLE_PATH)
            if not relaciones_tablas_str_for_llm: # Doble chequeo
                relaciones_tablas_str_for_llm = load_schema_as_string(RELACIONES_PATH_FOR_LLM)
            if not db_schema_enhanced_content: # Doble chequeo
                db_schema_enhanced_content = load_schema_as_string(SCHEMA_FULL_PATH)

            if db_schema_str_simple and relaciones_tablas_str_for_llm:
                try:
                    # El prompt debe pedir al LLM que decida si la pregunta es compleja y extraiga la estructura adecuada
                    llm_extracted_info = extract_info_from_question_llm(
                        question=question,
                        db_schema_str_full_details=db_schema_enhanced_content, 
                        db_schema_str_simple=db_schema_str_simple,
                        relaciones_tablas_str=relaciones_tablas_str_for_llm,
                        config=config or {}
                    )
                    logger.info(f"PIPELINE_BENCHMARK: extract_info_from_question_llm completed - {(time.time() - prev_step_time_tracker):.4f}s")
                    prev_step_time_tracker = time.time()
                    logger.info(f"DEBUG: [pipeline.py] Información extraída por LLM para pregunta (detección complejidad delegada): {llm_extracted_info}")
                    if llm_extracted_info:
                        if llm_extracted_info.get("tables"):
                            structured_info["tables"] = list(set(structured_info.get("tables", []) + llm_extracted_info.get("tables", [])))
                        if llm_extracted_info.get("columns"):
                            structured_info["columns"] = llm_extracted_info["columns"]
                        if llm_extracted_info.get("conditions"):
                            structured_info["conditions"] = llm_extracted_info["conditions"]
                        if llm_extracted_info.get("joins"):
                            structured_info["joins"] = llm_extracted_info["joins"]
                        for key in ["group_by", "order_by", "limit", "distinct", "query_type"]:
                            if llm_extracted_info.get(key):
                                structured_info[key] = llm_extracted_info[key]
                        # El LLM puede devolver un campo "is_complex_query"
                        if "is_complex_query" in llm_extracted_info:
                            structured_info["is_complex_query"] = llm_extracted_info["is_complex_query"]
                        logger.info(f"DEBUG: [pipeline.py] structured_info DESPUÉS de fusión LLM (detección complejidad delegada): {json.dumps(structured_info, indent=2, ensure_ascii=False)}")
                    else:
                        logger.warning("DEBUG: [pipeline.py] extract_info_from_question_llm no devolvió información válida o devolvió None/string.")
                except KeyError as ke:
                    logger.error(
                        f"KeyError durante la extracción LLM (llamada a extract_info_from_question_llm en llm_utils.py): {ke}. "
                        f"Esto usualmente indica un problema con la plantilla de prompt interna en llm_utils.py, "
                        f"que espera una clave (placeholder) '{ke}' que no fue satisfecha por los argumentos posicionales "
                        f"utilizados en su formateo (db_schema_str_simple, relaciones_tablas_str). "
                        f"Es necesario revisar la definición de 'system_message_template' y su uso en 'extract_info_from_question_llm' dentro de 'llm_utils.py'.",
                        exc_info=True
                    )
                    llm_extracted_info = None # Asegurar que llm_extracted_info sea None en este caso.
                except Exception as e_llm_extract:
                    logger.error(f"Error durante la extracción LLM para pregunta (detección complejidad delegada): {e_llm_extract}", exc_info=True)
                    llm_extracted_info = None # Asegurar que llm_extracted_info sea None en caso de otros errores.
            else:
                logger.warning("DEBUG: [pipeline.py] No se cargaron esquemas/relaciones necesarios para la extracción LLM completa. Saltando esta mejora.")
        else:
            logger.info(f"DEBUG: [pipeline.py] No se delega la detección de complejidad al LLM. Saltando extracción LLM estructurada.")

        # NUEVA COMPROBACIÓN: Si la extracción LLM era necesaria y falló, devolver error.
        if not is_direct_sql and needs_llm_structured_extraction and not disable_llm_enhancement and not llm_extracted_info:
            logger.error(
                "Fallo crítico: La extracción de información estructurada mediante LLM era necesaria pero falló "
                "(llm_extracted_info está vacío o es None). La pregunta original no es SQL directo. "
                "No se puede generar una consulta SQL significativa."
            )
            return {
                "response": "Error: No se pudo interpretar la pregunta compleja debido a un fallo en el componente de extracción de información. Por favor, reformula la pregunta o contacta al administrador.",
                "query_used": None,
                "parameters_used": [],
                "structured_info": structured_info, # Devolver el structured_info actual para depuración
                "error": "LLM_STRUCTURED_EXTRACTION_FAILED",
                "data": None
            }

        # 6. Normalización y heurística de condiciones
        # prev_step_time_tracker ya fue actualizado después de la extracción LLM o antes si se saltó.
        logger.info(f"PIPELINE_BENCHMARK: LLM extraction and info merge (if any) - {time.time() - prev_step_time_tracker:.4f}s (Total: {time.time() - pipeline_start_time:.4f}s)")
        prev_step_time_tracker = time.time() # Actualizar para el siguiente paso de normalización
        logger.debug(f"DEBUG: [pipeline.py] structured_info ANTES de Paso 5 (normalización): {json.dumps(structured_info, indent=2, ensure_ascii=False)}")
        structured_info = normalize_structured_info(structured_info) 
        logger.info(f"DEBUG: [pipeline.py] structured_info DESPUÉS de Paso 5 (normalización): {json.dumps(structured_info, indent=2, ensure_ascii=False)}")
        logger.info(f"PIPELINE_BENCHMARK: structured_info normalized - {(time.time() - prev_step_time_tracker):.4f}s")
        prev_step_time_tracker = time.time() # Actualizar para el siguiente paso

        # Heurística para ID de paciente
        if structured_info.get("tables") and not structured_info.get("conditions"):
            main_table = structured_info["tables"][0]
            question_text_for_ids = question
            patient_id_value = None
            id_source = None
            logger.info(f"Heurística ID Paciente: Iniciando. question_text_for_ids='{question_text_for_ids}', main_table='{main_table}'")
            explicit_id_regex = r'\b(ID|PATI_ID)\s*(?:=|ES|ES DE|CON)?\s*(\d+)\b'
            logger.info(f"Heurística: Regex explícito a usar: r'{explicit_id_regex}'")
            try:
                explicit_match = re.search(explicit_id_regex, question_text_for_ids, re.IGNORECASE)
            except Exception as e:
                logger.error(f"Heurística: Intento 1 - ERROR durante re.search: {e}", exc_info=True)
                explicit_match = None
            if explicit_match:
                logger.info(f"Heurística: Intento 1 - Regex explícito ENCONTRADO. Grupos: {explicit_match.groups()}")
                try:
                    col_name_candidate = explicit_match.group(1)
                    id_val_str = explicit_match.group(2)
                    patient_id_value = int(id_val_str)
                    id_source = f"regex explícito ({col_name_candidate})"
                    logger.info(f"Heurística: ID de paciente {patient_id_value} (columna candidata '{col_name_candidate}') encontrado vía {id_source}.")
                except ValueError:
                    logger.warning(f"Heurística: Intento 1 - Valor '{id_val_str}' (de columna '{col_name_candidate}') no es un entero.")
                    patient_id_value = None
                except Exception as e_group:
                    logger.error(f"Heurística: Intento 1 - ERROR extrayendo grupos del match: {e_group}", exc_info=True)
                    patient_id_value = None
            else:
                logger.info("Heurística: Intento 1 - Regex explícito NO ENCONTRADO.")
            if patient_id_value is None:
                logger.info("Heurística: Intento 1 falló o no aplicó. Procediendo a Intento 2 ('paciente' + número).")
                try:
                    numbers_in_question = re.findall(r'\b\d+\b', question_text_for_ids) 
                    logger.info(f"Heurística: Intento 2 - Números encontrados en pregunta: {numbers_in_question}")
                    if "paciente" in question_text_for_ids.lower() and numbers_in_question:
                        patient_id_value = int(numbers_in_question[0])
                        id_source = "palabra 'paciente' y número"
                        logger.info(f"Heurística: Intento 2 - ID de paciente {patient_id_value} encontrado vía {id_source}.")
                    else:
                        logger.info("Heurística: Intento 2 - Condiciones NO cumplidas (sin números o sin 'paciente').")
                except ValueError: # pragma: no cover
                    logger.warning(f"Heurística: Intento 2 - Valor '{numbers_in_question[0] if numbers_in_question else 'desconocido'}' no es un entero.")
                    patient_id_value = None
                except Exception as e_fallback: # pragma: no cover
                    logger.error(f"Heurística: Intento 2 - ERROR durante fallback regex: {e_fallback}", exc_info=True)
                    patient_id_value = None # Asegurar que no se use un valor inválido
            else:
                logger.info("Heurística: Intento 1 tuvo éxito. Saltando Intento 2.")
            if patient_id_value is not None:
                logger.info(f"Heurística: ID de paciente FINALMENTE encontrado: {patient_id_value} (fuente: {id_source}). Procediendo a determinar columna.")
                id_column_name = None
                # Lógica para determinar id_column_name (basada en la implementación anterior)
                if main_table in db_structure and db_structure[main_table]:
                    logger.info(f"Heurística: Buscando columna ID en tabla principal '{main_table}'.")
                    # Prioridad 1: FK a PATI_PATIENTS
                    for fk in db_structure[main_table].get('foreign_keys', []):
                        if fk.get('referenced_table', '').upper() == 'PATI_PATIENTS':
                            id_column_name = fk.get('foreign_key_column')
                            logger.info(f"Heurística: Columna ID para '{main_table}' (por FK a PATI_PATIENTS): {id_column_name}")
                            break
                    # Prioridad 2: Nombres comunes
                    if not id_column_name:
                        common_id_column_names = ["PATI_ID", "PACIENTE_ID", "PPA_PATIENT_ID", "PATIENT_ID", "ID_PACIENTE"]
                        logger.info(f"Heurística: Buscando nombres comunes de columna ID: {common_id_column_names}")
                        for col_info in db_structure[main_table].get('columns', []):
                            col_name_upper = col_info.get('name','').upper()
                            if col_name_upper in common_id_column_names:
                                id_column_name = col_info.get('name')
                                logger.info(f"Heurística: Columna ID para '{main_table}' (por nombre común '{col_name_upper}'): {id_column_name}")
                                break
                # --- CORRECCIÓN: Inicializar la variable antes de usarla ---
                original_col_candidate_from_regex = None
                if not id_column_name:
                    if id_source and "regex explícito" in id_source and explicit_match:
                        try:
                            original_col_candidate_from_regex = explicit_match.group(1)
                        except Exception as e:
                            logger.warning(f"Error extrayendo group(1) de explicit_match: {e}")
                            original_col_candidate_from_regex = None
                    if original_col_candidate_from_regex:
                        id_column_name = original_col_candidate_from_regex
                        logger.info(f"Heurística: Usando columna '{id_column_name}' (del regex explícito) como ID para '{main_table}'.")
                    else:
                        id_column_name = "PATI_ID" # Fallback general
                        logger.warning(f"Heurística: No se encontró una columna de ID de paciente clara para '{main_table}'. Asumiendo '{id_column_name}'.")
                condition = {"column": id_column_name, "operator": "=", "value": patient_id_value}
                structured_info.setdefault("conditions", []).append(condition)
                logger.info(f"DEBUG: [pipeline.py] Condición de paciente extraída heurísticamente: {condition}")
            else:
                logger.info(f"DEBUG: [pipeline.py] No se aplicó heurística de ID de paciente (ningún método tuvo éxito en encontrar un ID).")
            if not isinstance(structured_info, dict) or not structured_info.get("tables"):
                logger.error(f"DEBUG: [pipeline.py] Error CRÍTICO FINAL: structured_info no es un dict o no contiene tablas DESPUÉS de todos los intentos. Valor: {structured_info}")
                return {"response": "Error: No se pudieron identificar las tablas necesarias para la consulta (todos los intentos fallaron).", "sql": None, "data": None, "error": "Table identification failed"}

        # 7. Generación y validación SQL
        logger.info(f"DEBUG: [pipeline.py] Usando structured_info para Paso 6 (Generación SQL): {json.dumps(structured_info, indent=2, ensure_ascii=False)}")
        logger.info("DEBUG: [pipeline.py] Paso 6: Generando SQL a partir de structured_info...")
        
        # --- MODIFICACIÓN: Manejar mejor el resultado de generate_sql ---
        # sql_generator_instance ya está creada.
        # Usar db_schema_enhanced_content y relaciones_tablas_map
        sql_generator_instance = SQLGenerator(allowed_tables=allowed_tables_list, allowed_columns=allowed_columns_map)
        generation_result = sql_generator_instance.generate_sql(
            structured_info, 
            db_schema_enhanced_content if db_schema_enhanced_content else db_schema_str_simple, 
            relaciones_tablas_map
        )
        
        sql_query_str = None
        params_list = []
        error_message_from_generator = None

        if isinstance(generation_result, tuple) and len(generation_result) == 2:
            sql_query_str, params_list = generation_result
            logger.info(f"DEBUG: [pipeline.py] Returned from generate_sql. Value: {generation_result}, Type: {type(generation_result)}")
        elif isinstance(generation_result, dict) and "response" in generation_result:
            error_message_from_generator = generation_result["response"]
            # Opcional: también podrías querer registrar el SQL y params propuestos si están en el dict
            # sql_query_str = generation_result.get("sql_query") 
            # params_list = generation_result.get("params", [])
            logger.error(f"DEBUG: [pipeline.py] generate_sql devolvió un diccionario de error: {generation_result}")
        else:
            logger.error(f"DEBUG: [pipeline.py] generate_sql devolvió un formato inesperado: {generation_result}")
            error_message_from_generator = "Error interno durante la generación de SQL: formato de respuesta inesperado del generador."

        step_time_tracker = time.time() # Reset para el siguiente paso
        logger.info(f"PIPELINE_BENCHMARK: SQL generated - {step_time_tracker - prev_step_time_tracker:.4f}s (Total: {step_time_tracker - pipeline_start_time:.4f}s)")
        prev_step_time_tracker = step_time_tracker

        logger.info(f"DEBUG: [pipeline.py] SQL generado (str): {sql_query_str}")
        logger.info(f"DEBUG: [pipeline.py] Parámetros generados: {params_list}")

        if not sql_query_str and error_message_from_generator:
            logger.error(f"DEBUG: [pipeline.py] Error al generar la consulta SQL: {error_message_from_generator}")
            return {
                "response": error_message_from_generator,
                "query_used": None, # O el SQL propuesto si se extrajo del dict de error
                "parameters_used": params_list, # O los params propuestos
                "structured_info": structured_info,
                "error": "SQL_GENERATION_VALIDATION_FAILED",
                "data": None
            }
        elif not sql_query_str:
            logger.error("DEBUG: [pipeline.py] Error al generar la consulta SQL. SQL string es None o vacío después de la asignación y no hay mensaje de error específico del generador.")
            return {
                "response": "Error: No se pudo generar la consulta SQL (string vacío o nulo).",
                "query_used": None,
                "parameters_used": [],
                "structured_info": structured_info,
                "error": "SQL_GENERATION_FAILED_EMPTY",
                "data": None
            }
        
        logger.info(f"DEBUG: [pipeline.py] SQL validado correctamente: {sql_query_str}")

        # 7. Ejecución de la consulta SQL
        logger.info(f"DEBUG: [pipeline.py] Paso 7: Ejecutando consulta SQL: '{sql_query_str}' con params: {params_list}")
        execute_query_start_time = time.time()
        
        result, error = execute_query_with_timeout(db_connector, sql_query_str, params_list, logger, timeout_seconds)
        
        if error:
            logger.error(f"ERROR durante la ejecución de la consulta: {error}")
        else:
            logger.info("DEBUG: execute_query_with_timeout returned without error.")
            if result is None:
                logger.info("DEBUG: Result from execute_query_with_timeout is None.")
            else:
                logger.info(f"DEBUG: Result type: {type(result).__name__}")
                if isinstance(result, list):
                    logger.info(f"DEBUG: Result is a list. Length: {len(result)}")
                    if result:
                        first_element = result[0]
                        if isinstance(first_element, dict):
                            logger.info(f"DEBUG: First element type: {type(first_element).__name__}")
                            for key_val, value_val in first_element.items():
                                value_type = type(value_val).__name__
                                value_repr = repr(value_val)
                                truncated_value_repr = value_repr[:100] + ('...' if len(value_repr) > 100 else '')
                                logger.info(f"DEBUG: Key='{key_val}', Value_Type='{value_type}', Truncated_Value_Repr='{truncated_value_repr}'")
                        else:
                            logger.info("DEBUG: First element is not a dict.")
                    else:
                        logger.info("DEBUG: Result list is empty.")
                else:
                    logger.info("DEBUG: Result is not a list.")
            output_str_res = str(result)
            truncated_output_res = output_str_res[:500] + ('...' if len(output_str_res) > 500 else '')
            logger.info(f"DEBUG: [pipeline.py] Consulta SQL ejecutada. Full result (controlled): {'(list too long to log full)' if isinstance(result, list) and len(result) > 5 else truncated_output_res}")
        logger.info(f"PIPELINE_BENCHMARK: Query executed - {(time.time() - execute_query_start_time):.4f}s (Total: {(time.time() - pipeline_start_time):.4f}s)")
        if error:
            return {"response": f"Error al ejecutar la consulta SQL: {error}", "sql": sql_query_str, "data": None, "error": error}
        MAX_RESPONSE_DATA_ROWS = 100
        response_message = "Consulta ejecutada con éxito."
        data_for_response = result
        if isinstance(result, list):
            original_length = len(result)
            if original_length > MAX_RESPONSE_DATA_ROWS:
                data_for_response = result[:MAX_RESPONSE_DATA_ROWS]
                response_message = f"Consulta ejecutada con éxito. Se devuelven las primeras {MAX_RESPONSE_DATA_ROWS} filas de {original_length} resultados."
            elif original_length == 0:
                response_message = "Consulta ejecutada con éxito. La consulta no devolvió filas."
            else:
                response_message = f"Consulta ejecutada con éxito. {original_length} fila{'s' if original_length != 1 else ''} devuelta{'s' if original_length != 1 else ''}."
        elif result is None:
            response_message = "La consulta no devolvió resultados (resultado None)."
        else:
            response_message = "Consulta ejecutada. El resultado no es una lista de filas."
        final_response = {
            "query_used": sql_query_str,
            "parameters_used": params_list,
            "response": response_message,
            "structured_info": structured_info,
            "error": None, 
            "data": data_for_response,
        }
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

    except Exception as e:
        current_logger = logger if 'logger' in locals() and isinstance(logger, logging.Logger) else logging.getLogger("chatbot_pipeline_fallback_logger")
        current_logger.error(f"Excepción inesperada en chatbot_pipeline: {e}", exc_info=True)
        return {"response": f"Error inesperado en el pipeline: {e}", "sql": None, "data": None, "error": str(e)}

def execute_query_with_timeout(db_connector, sql_query, params, logger, timeout_seconds=5):
    """Ejecuta una consulta SQL con un timeout estricto"""
    import threading
    import time

    result = None
    error = None
    completed_flag_obj = {'value': False} # Usar un objeto mutable para 'completed'

    def execute_query_thread():
        thread_logger = logger 
        thread_logger.info("execute_query_thread: Starting.")
        try:
            thread_logger.info(f"execute_query_thread: About to call db_connector.execute_query for SQL: {sql_query}")
            temp_result = db_connector.execute_query(sql_query, params)
            
            nonlocal result
            result = temp_result 
            
            if isinstance(result, list):
                thread_logger.info(f"execute_query_thread: db_connector.execute_query returned. Result is list: True. Length if list: {len(result)}.")
                if result: 
                    first_element = result[0]
                    if isinstance(first_element, dict):
                        thread_logger.info(f"execute_query_thread: First element type: {type(first_element).__name__}")
                        for key_val, value_val in first_element.items(): # Renombrar variables para evitar conflicto
                            value_type = type(value_val).__name__
                            value_repr = repr(value_val)
                            truncated_value_repr = value_repr[:50] + ('...' if len(value_repr) > 50 else '')
                            thread_logger.info(f"execute_query_thread: Key='{key_val}', TruncatedValue='{truncated_value_repr}'")
                    else:
                        thread_logger.info("execute_query_thread: First element is not a dict.")
                else: 
                    thread_logger.info("execute_query_thread: Result list is empty.")
            elif result is None:
                 thread_logger.info("execute_query_thread: Result is None.")
            else:
                 thread_logger.info(f"execute_query_thread: Result is not a list. Type: {type(result).__name__}")

        except Exception as e:
            thread_logger.error(f"execute_query_thread: Exception during db_connector.execute_query: {e}", exc_info=True)
            nonlocal error
            error = f"Error en la ejecución de la consulta: {e}"
        finally:
            thread_logger.info("execute_query_thread: In finally block. Setting completed flag to True.")
            completed_flag_obj['value'] = True 
            thread_logger.info("execute_query_thread: Completed flag set to True. Exiting thread function.")

    thread = threading.Thread(target=execute_query_thread)
    logger.info(f"execute_query_with_timeout: Starting thread for query: '{sql_query}' with timeout: {timeout_seconds}s.")
    thread.start()
    
    logger.info(f"execute_query_with_timeout: Thread started ({thread.name}). About to join with timeout: {timeout_seconds}s.") # LOG ANTES DEL JOIN
    
    thread.join(timeout_seconds) 
    
    logger.info(f"execute_query_with_timeout: Thread join finished or timed out. Thread alive: {thread.is_alive()}. Completed flag: {completed_flag_obj['value']}.") # LOG DESPUÉS DEL JOIN

    if thread.is_alive(): 
        logger.warning(f"execute_query_with_timeout: Query execution timed out after {timeout_seconds} seconds. SQL: {sql_query}")
        error = f"Timeout: La consulta excedió los {timeout_seconds} segundos."
    elif not completed_flag_obj['value']: 
        logger.error("execute_query_with_timeout: Thread finished but 'completed' flag is false. This might indicate an issue if no error was set by the thread.")
        if not error: 
            error = "Error desconocido: el hilo de la consulta terminó inesperadamente sin marcar como completado y sin error explícito."
    elif error:
        logger.info(f"execute_query_with_timeout: Thread finished with an error: {error}")
    else: 
        logger.info("execute_query_with_timeout: Thread finished successfully and 'completed' flag is true.")

    logger.info(f"execute_query_with_timeout: Returning. Result is present: {result is not None}. Error is present: {error is not None}.")
    return result, error
