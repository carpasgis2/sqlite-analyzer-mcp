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
        table_candidates = {}
        for table, syns in terms_dict.get('table_synonyms', {}).items():
            for syn in syns:
                table_candidates[normalize_text(syn)] = table
        for table, syns in terms_dict.get('table_common_terms', {}).items():
            for syn in syns:
                table_candidates[normalize_text(syn)] = table
        for term, table in terms_dict.get('table_mappings', {}).items():
            table_candidates[normalize_text(term)] = table
        # Matching robusto: buscar coincidencias parciales y exactas
        for syn_norm, table in table_candidates.items():
            if not syn_norm:
                continue
            # Coincidencia exacta o parcial (palabra dentro de la pregunta)
            if syn_norm in normalized_question or syn_norm in question or any(syn_norm in w for w in normalized_question.split()):
                identified_tables_from_terms.add(table)
                if syn_norm not in matched_synonyms:
                    matched_synonyms.append(syn_norm)
        # Fuzzy matching adicional si no hay match directo
        if not identified_tables_from_terms and table_candidates:
            from difflib import get_close_matches
            words = normalized_question.split()
            for word in words:
                close = get_close_matches(word, list(table_candidates.keys()), n=1, cutoff=0.7)
                if close:
                    identified_tables_from_terms.add(table_candidates[close[0]])
                    if close[0] not in matched_synonyms:
                        matched_synonyms.append(close[0])
        # Añadir descripciones si están disponibles
        for table in identified_tables_from_terms:
            desc = terms_dict.get('table_descriptions', {}).get(table)
            if desc and desc not in matched_descriptions:
                matched_descriptions.append(desc)
        if identified_tables_from_terms:
            for table in identified_tables_from_terms:
                matched_terms.append(f"- {table}")
        if matched_terms:
            enriched_question += f" [TERMS: {', '.join(matched_terms)}]"
        if matched_descriptions:
            enriched_question += f" [DESCRIPTIONS: {', '.join(matched_descriptions)}]"
        if matched_synonyms:
            enriched_question += f" [SYNONYMS: {', '.join(matched_synonyms)}]"
        logging.info(f"Términos identificados: {matched_terms}")
        logging.info(f"Descripciones identificadas: {matched_descriptions}")
        logging.info(f"Sinónimos identificados: {matched_synonyms}")
        if not identified_tables_from_terms:
            logging.warning(f"No se identificaron tablas en la pregunta '{question}'. Términos candidatos: {list(table_candidates.keys())}")
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
                 db_connector: 'DBConnector',
                 rag: Optional['EnhancedSchemaRAG'] = None, config: Dict[str, Any] = None) -> str:
    """
    Genera una consulta SQL robusta a partir de la información estructurada, aplicando normalización semántica,
    validación y corrección de nombres, y manejo de errores. Utiliza LLM y RAG si están disponibles.
    """
    try: # OUTER TRY
        # Importación local de SQLGenerator para evitar ciclos en la carga inicial de módulos
        from src.sql_generator import SQLGenerator

        # 1) Normalizar la estructura y referencias
        structured_info = check_table_references(structured_info)
        structured_info = normalize_table_and_column_names(structured_info)
        
        # 2) Validar que las tablas existan en el esquema
        tablas = structured_info.get("tables", [])
        # Filtrar Nones potenciales que podrían venir de una normalización defectuosa o datos de entrada
        tablas_validas = [t for t in tablas if t and t in db_structure]

        if not tablas_validas:
            logging.error(f"Ninguna tabla válida identificada después del filtrado: {tablas}")
            return "SELECT 'No se identificaron tablas válidas para la consulta' AS mensaje"
        structured_info["tables"] = tablas_validas

        # 3) Generar SQL (usando generador LLM o heurístico)
        sql_query_str: Optional[str] = None
        params_list: List[Any] = []
        
        try: # INNER TRY (catches e_gensql for generation/validation/execution issues)
            # Asegurarse de que db_structure y sus elementos son accedidos de forma segura
            allowed_tables_list = list(db_structure.keys()) if db_structure else []
            
            allowed_columns_map = {}
            if db_structure:
                for table_name_for_map, cols_data in db_structure.items():
                    if isinstance(cols_data, dict) and isinstance(cols_data.get('columns'), list):
                        # Extraer nombres de columnas, asegurándose de que sean strings
                        col_names_for_map = []
                        for col_entry in cols_data['columns']:
                            if isinstance(col_entry, dict) and isinstance(col_entry.get('name'), str):
                                col_names_for_map.append(col_entry['name'])
                            elif isinstance(col_entry, str): # Si la columna es solo un string
                                col_names_for_map.append(col_entry)
                        allowed_columns_map[table_name_for_map] = col_names_for_map
                    else:
                        allowed_columns_map[table_name_for_map] = [] # Fallback a lista vacía
            
            sql_gen = SQLGenerator(allowed_tables=allowed_tables_list, allowed_columns=allowed_columns_map)

            returned_from_generate_sql = sql_gen.generate_sql(structured_info, db_structure, None)
            logging.debug("JUST RETURNED FROM sql_gen.generate_sql()")

            if isinstance(returned_from_generate_sql, tuple) and len(returned_from_generate_sql) == 2:
                sql_query_str, params_list = returned_from_generate_sql
                if not isinstance(sql_query_str, str): # Asegurar que la query es string
                    logging.error(f"SQLGenerator.generate_sql devolvió una tupla, pero el primer elemento no es string: {type(sql_query_str)}")
                    sql_query_str = None # Invalidar para que se maneje como error abajo
                logging.debug(f"SQL Query Type: {type(sql_query_str)}, Value: {sql_query_str}")
                logging.debug(f"Params Type: {type(params_list)}, Value: {params_list}")
            elif isinstance(returned_from_generate_sql, str):
                sql_query_str = returned_from_generate_sql
                logging.debug(f"SQL Query (str) Type: {type(sql_query_str)}, Value: {sql_query_str}")
            else:
                logging.error(f"SQLGenerator.generate_sql devolvió un tipo inesperado: {type(returned_from_generate_sql)}. Valor: {returned_from_generate_sql}")
                # sql_query_str permanece None, será manejado más adelante

            if sql_query_str: # Solo proceder si tenemos una cadena SQL
                logging.debug(f"SQL generado (antes de validación): {sql_query_str}")
                logging.debug(f"Parámetros (antes de validación): {params_list}")

                logging.info("Paso 6.1: Validando consulta SQL (estructurada) con whitelist...")
                
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
                            logging.warning(f"Estructura inesperada para la tabla '{table_name}' en db_structure al preparar para validación.")
                            allowed_columns_for_validation[table_name.upper()] = []
                
                is_safe, validation_msg = new_whitelist_validate_query(structured_info, allowed_tables_for_validation, allowed_columns_for_validation)
                logging.info(f"Paso 6.1 completado. ¿Validación de partes estructuradas segura? {is_safe}, Mensaje: {validation_msg}")

                if not is_safe:
                    logging.error(f"Validación de whitelist de partes estructuradas fallida: {validation_msg} para structured_info: {structured_info}")
                    return {"response": f"Error: La consulta generada no es segura según la validación de estructura. {validation_msg}", 
                            "sql_query": sql_query_str, "params": params_list, "structured_info": structured_info} # type: ignore

                logging.info(f"Paso 6.2: Ejecutando SQL: {sql_query_str} con params: {params_list}")
                # result = None # No se usa 'result' más adelante en este bloque
                try:
                    if db_connector:
                        logging.debug("Llamando a db_connector.execute_query...")
                        # La ejecución real se hace en chatbot_pipeline, aquí solo generamos y validamos.
                        # result = db_connector.execute_query(sql_query_str, params_list) 
                        logging.debug(f"Llamada a db_connector.execute_query (simulada/omitida en generate_sql) completada.")
                    else:
                        logging.error("db_connector no está inicializado. No se puede ejecutar la consulta (simulación).")
                        return {"response": "Error crítico: El conector de base de datos no está disponible.", 
                                "sql_query": sql_query_str, "params": params_list, "structured_info": structured_info} # type: ignore
                except Exception as e_exec: # Captura de error de ejecución (si se hiciera aquí)
                    logging.error(f"Error al ejecutar la consulta SQL (simulación): {e_exec}", exc_info=True)
                    return {"response": "Error al ejecutar la consulta SQL (simulación).", "sql_query": sql_query_str, "params": params_list} # type: ignore
            # else: sql_query_str es None o no es string. Se manejará después del bloque try-except interno.

        except Exception as e_gensql:
            logging.error(f"DEBUG: [pipeline.py] EXCEPCIÓN durante la generación/validación de SQL interna: {e_gensql}", exc_info=True)
            # Devolver un string SQL de error es lo esperado por el resto del pipeline
            return "SELECT 'Error durante la generación interna de SQL' AS mensaje"

        # ---- FIN DEL BLOQUE try...except e_gensql ----

        # Verificar si sql_query_str se generó correctamente.
        # Si sql_query_str es None en este punto, significa que SQLGenerator.generate_sql no produjo una cadena válida
        # y no se manejó como un error que retornara antes.
        if sql_query_str is None:
            logging.error("Fallo crítico en la generación de SQL: sql_query_str es None después del bloque de generación.")
            return "SELECT 'Error: Fallo interno crítico en la generación de SQL (query_str nula)' AS mensaje"

        # A partir de aquí, sql_query_str DEBE ser una cadena SQL.
        # Las funciones de corrección y validación operarán sobre esta cadena.
        # La variable `sql_query` que se usaba antes se reemplaza por el uso directo de `sql_query_str`
        # o una variable renombrada para claridad.

        # 4) Corregir nombres de columnas/tablas en el SQL
        # Se asume que correct_column_names y validate_table_names esperan y devuelven string.
        # Y que `tablas_validas` es una lista de strings.
        corrected_sql_query = correct_column_names(sql_query_str, tablas_validas, db_structure)
        if corrected_sql_query is None: # Chequeo de seguridad
             logging.error("correct_column_names devolvió None.")
             return "SELECT 'Error: Fallo en la corrección de nombres de columna (resultado nulo)' AS mensaje"

        final_sql_query = validate_table_names(corrected_sql_query, db_structure)
        if final_sql_query is None: # Chequeo de seguridad
             logging.error("validate_table_names devolvió None.")
             return "SELECT 'Error: Fallo en la validación de nombres de tabla (resultado nulo)' AS mensaje"


        # 5) Validar el SQL final (sintaxis básica, etc., si validate_sql_query lo hace)
        # Asumimos que db_connector es necesario para esta validación.
        if not validate_sql_query(final_sql_query, db_connector): # type: ignore
            logging.error(f"SQL inválido tras corrección y validación final: {final_sql_query}")
            return "SELECT 'Error: SQL inválido tras corrección final' AS mensaje"

        # 6) Devolver la consulta final y sus parámetros
        # La función está tipada para devolver str, pero el pipeline maneja (str, list) o dict.
        # Por ahora, devolvemos solo el string SQL como antes, y los params se manejan en chatbot_pipeline.
        # Si la función debe devolver también params_list, la signatura y el manejo deben cambiar.
        # Por coherencia con el error de "TypeError: argument of type 'NoneType' is not iterable"
        # el problema principal es que se pasaba None a funciones que esperaban string.
        # El retorno de esta función es usado por chatbot_pipeline, que espera un string o un dict.
        # Si todo va bien, devolvemos el string SQL. params_list se usa en chatbot_pipeline.
        return final_sql_query 
    
    except Exception as e: # OUTER CATCH
        logging.error(f"Error en generate_sql (captura externa): {e}", exc_info=True)
        return "SELECT 'Error al generar SQL (captura externa)' AS mensaje"

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
            
            # Asegurarse de que structured_info esté inicializado
            if structured_info is None: 
                structured_info = {}

            if initial_identified_tables:
                structured_info["tables"] = initial_identified_tables
                logger.info(f"DEBUG: [pipeline.py] structured_info['tables'] poblado con tablas de Paso 3: {initial_identified_tables}")

            if initial_query_type:
                structured_info["query_type"] = initial_query_type
                logger.info(f"DEBUG: [pipeline.py] structured_info['query_type'] poblado con tipo de consulta de Paso 3: {initial_query_type}")
            
            # Cargar esquemas necesarios
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
                logger.warning(f"ADVERTENCIA: No se pudo cargar el esquema completo desde {SCHEMA_FULL_PATH}.")
            
            # Si no se identificaron tablas en el Paso 3, intentar fallback con LLM
            if not structured_info.get("tables"):
                logger.info("DEBUG: [pipeline.py] No hay tablas de Paso 3. Intentando fallback con LLM para identificar tablas...")
                table_synonyms = terms_dict.get('table_synonyms', {})
                table_common_terms = terms_dict.get('table_common_terms', {})
                # table_mappings = terms_dict.get('table_mappings', {}) # No usado en el prompt actual
                prompt = (
                    "No se identificaron tablas relevantes de forma directa. "
                    "A continuación tienes la lista de tablas y sus sinónimos más comunes extraídos del diccionario:\n"
                )
                for table, syns in table_synonyms.items():
                    prompt += f"- {table}: {', '.join(syns)}\n"
                for table, syns in table_common_terms.items():
                    prompt += f"- {table}: {', '.join(syns)}\n"
                prompt += "\nPregunta del usuario: " + question + "\n"
                prompt += "\nResponde SOLO con el nombre exacto de la tabla más relevante para la consulta."
                
                llm_config = {
                    "api_key": os.environ.get("DEEPSEEK_API_KEY"),
                    "base_url": os.environ.get("DEEPSEEK_API_URL"),
                    "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
                    "provider": "deepseek",
                    "temperature": 0.0,
                    "max_tokens": 50
                }
                messages = [
                    {"role": "system", "content": "Eres un asistente experto en SQL y bases de datos médicas."},
                    {"role": "user", "content": prompt}
                ]
                try:
                    from src.llm_utils import call_llm_with_fallbacks
                except ImportError:
                    from llm_utils import call_llm_with_fallbacks
                
                llm_response = call_llm_with_fallbacks(messages, llm_config)
                
                if llm_response and isinstance(llm_response, str):
                    cleaned_llm_response = llm_response.strip()
                    identified_table_from_llm = None
                    known_tables = list(db_structure.keys()) if db_structure else []
                    
                    if not known_tables:
                        logger.warning("La lista de tablas conocidas (db_structure.keys()) está vacía. La extracción de tablas del LLM puede no ser fiable.")
                        identified_table_from_llm = cleaned_llm_response.split()[0] if cleaned_llm_response.split() else None
                    elif cleaned_llm_response in known_tables:
                        identified_table_from_llm = cleaned_llm_response
                        logger.info(f"Tabla identificada por LLM fallback (coincidencia exacta): {identified_table_from_llm}")
                    else:
                        words_in_response = cleaned_llm_response.replace(",", " ").replace(";", " ").split()
                        for word in words_in_response:
                            if word in known_tables:
                                identified_table_from_llm = word
                                logger.info(f"Tabla encontrada en respuesta LLM fallback '{cleaned_llm_response}' buscando palabras conocidas: {identified_table_from_llm}")
                                break
                        if not identified_table_from_llm:
                            potential_table_name = cleaned_llm_response.split()[0] if cleaned_llm_response.split() else None
                            if potential_table_name:
                                if potential_table_name not in known_tables and known_tables:
                                    logger.warning(f"Fallback: La tabla '{potential_table_name}' (primera palabra de '{cleaned_llm_response}') no es una tabla conocida. Se usará igualmente.")
                                elif not known_tables:
                                     logger.info(f"Fallback: Usando primera palabra '{potential_table_name}' de '{cleaned_llm_response}' (lista de tablas conocidas vacía).")
                                elif potential_table_name in known_tables:
                                     logger.info(f"Fallback: Usando primera palabra '{potential_table_name}' de '{cleaned_llm_response}' (es una tabla conocida).")
                                else:
                                     logger.info(f"Fallback: Usando primera palabra '{potential_table_name}' de '{cleaned_llm_response}' (no es un sinónimo conocido, pero es la primera palabra).")
                                identified_table_from_llm = potential_table_name
                            else:
                                logger.warning(f"LLM fallback no pudo extraer un nombre de tabla de la respuesta: '{cleaned_llm_response}'")

                    if identified_table_from_llm:
                        structured_info["tables"] = [identified_table_from_llm]
                        logger.info(f"DEBUG: [pipeline.py] structured_info['tables'] poblado por LLM fallback: {[identified_table_from_llm]}")
                        # Asegurar que query_type esté presente
                        if not structured_info.get("query_type") and initial_query_type:
                            structured_info["query_type"] = initial_query_type
                        elif not structured_info.get("query_type"):
                            structured_info["query_type"] = "SELECT" # Default
                    else:
                        logger.error(f"LLM fallback no pudo identificar una tabla válida de la respuesta: '{llm_response}'")
                else:
                    logger.error(f"LLM fallback no devolvió una respuesta de texto válida. Respuesta: {llm_response}")
            
            # Asegurar que query_type exista, por defecto a 'SELECT' si aún no está.
            if not structured_info.get("query_type"):
                if initial_query_type:
                    structured_info["query_type"] = initial_query_type
                else:
                    logger.warning(f"DEBUG: [pipeline.py] query_type sigue ausente. Estableciendo a 'SELECT' por defecto.")
                    structured_info["query_type"] = "SELECT"

            logger.info(f"DEBUG: [pipeline.py] structured_info ANTES de Paso 5 (normalización): {structured_info}")
            structured_info = normalize_structured_info(structured_info) 
            logger.info(f"DEBUG: [pipeline.py] structured_info DESPUÉS de Paso 5 (normalización): {structured_info}")

            # --- INICIO: Heurística para extraer condiciones de ID de paciente si están vacías ---
            if structured_info.get("tables") and not structured_info.get("conditions"):
                main_table = structured_info["tables"][0]
                # Usar enriched_question o question original. enriched_question puede tener [QUERY_TYPE]
                question_text_for_ids = question # Usar la pregunta original sin procesar para buscar IDs
                numbers_in_question = re.findall(r'\b\d+\b', question_text_for_ids)
                
                if numbers_in_question and "paciente" in question_text_for_ids.lower():
                    patient_id_value = None
                    try:
                        patient_id_value = int(numbers_in_question[0])
                    except ValueError:
                        logger.warning(f"No se pudo convertir el ID extraído '{numbers_in_question[0]}' a entero.")

                    if patient_id_value is not None:
                        id_column_name = None
                        # Lógica mejorada para encontrar la columna de ID de paciente
                        if main_table in db_structure:
                            # Prioridad 1: FK a una tabla de pacientes conocida (ej. PATI_PATIENTS)
                            for fk in db_structure[main_table].get('foreign_keys', []):
                                if fk.get('referenced_table', '').upper() == 'PATI_PATIENTS':
                                    id_column_name = fk.get('foreign_key_column')
                                    logger.info(f"Columna ID paciente encontrada por FK a PATI_PATIENTS: {id_column_name} en tabla {main_table}")
                                    break
                            
                            # Prioridad 2: Nombres de columna comunes para ID de paciente en la tabla actual
                            if not id_column_name:
                                for col_info in db_structure[main_table].get('columns', []):
                                    col_name_upper = col_info.get('name','').upper()
                                    # Lista más exhaustiva de posibles nombres de columnas de ID de paciente
                                    if col_name_upper in ["PATI_ID", "PACIENTE_ID", "PPA_PATIENT_ID", "PATIENT_ID", "ID_PACIENTE"]:
                                        id_column_name = col_info.get('name')
                                        logger.info(f"Columna ID paciente encontrada por nombre común: {id_column_name} en tabla {main_table}")
                                        break
                            
                            # Prioridad 3: Si la tabla es PATI_PATIENTS, usar su PK
                            if not id_column_name and main_table.upper() == "PATI_PATIENTS":
                                for col_info in db_structure[main_table].get('columns', []):
                                    if col_info.get('primary_key'):
                                        id_column_name = col_info.get('name')
                                        logger.info(f"Columna ID paciente encontrada por PK en PATI_PATIENTS: {id_column_name}")
                                        break
                        
                        if not id_column_name: # Fallback si no se encuentra nada mejor
                             id_column_name = "PATI_ID" # Asunción común
                             logger.warning(f"No se encontró una columna de ID de paciente clara para {main_table}. Asumiendo '{id_column_name}'.")

                        condition = {"column": id_column_name, "operator": "=", "value": patient_id_value}
                        structured_info["conditions"] = [condition]
                        logger.info(f"DEBUG: [pipeline.py] Condición de paciente extraída heurísticamente: {condition}")
                else:
                    logger.info(f"DEBUG: [pipeline.py] No se aplicó heurística de ID de paciente (no 'paciente' en pregunta, no números, o tabla no relevante).")
            # --- FIN: Heurística para extraer condiciones de ID de paciente ---

            # Verificación final: si después de todos los intentos (Paso 3 y fallback LLM y heurística) no hay tablas
            if not isinstance(structured_info, dict) or not structured_info.get("tables"):
                logger.error(f"DEBUG: [pipeline.py] Error CRÍTICO FINAL: structured_info no es un dict o no contiene tablas DESPUÉS de todos los intentos. Valor: {structured_info}")
                return {"response": "Error: No se pudieron identificar las tablas necesarias para la consulta (todos los intentos fallaron).", "sql": None, "tables": [], "columns": []}
            
            # structured_info_copy se asigna aquí, DESPUÉS de que structured_info haya sido modificado.
            structured_info_copy = structured_info.copy() if isinstance(structured_info, dict) else {}
            logger.info(f"DEBUG: [pipeline.py] structured_info_copy preparada para normalización (Paso 6): {structured_info_copy}") # Este log puede ser confuso, structured_info_copy es para el siguiente paso.

        # --- INICIO CAMBIO: Normalizar structured_info antes de pasar a SQLGenerator ---
        # La variable structured_info ya está normalizada y verificada.
        logger.info(f"DEBUG: [pipeline.py] Usando structured_info para Paso 6: {structured_info}")
        if not isinstance(structured_info, dict) or not structured_info.get("tables"): # Doble check, debería ser redundante si la lógica anterior es correcta
            logger.error(f"DEBUG: [pipeline.py] Error CRÍTICO (inesperado): structured_info no es un dict o no contiene tablas antes de generar SQL. Valor: {structured_info}")
            return {"response": "Error crítico: La información estructurada no es válida antes de la generación de SQL.", "sql": None, "tables": [], "columns": []}
        # --- FIN CAMBIO: Normalizar structured_info antes de pasar a SQLGenerator ---
        logger.info(f"DEBUG: [pipeline.py] Paso 6: Generando SQL a partir de structured_info...")
        
        # --- INICIO CAMBIO: Manejo explícito de la salida de generate_sql ---
        returned_from_generate_sql = generate_sql(
            structured_info=structured_info, 
            db_structure=db_structure, 
            db_connector=db_connector,
            rag=None, 
            config=config
        )
        
        sql_query_str: Optional[str] = None
        params_list: List[Any] = []

        if isinstance(returned_from_generate_sql, tuple) and len(returned_from_generate_sql) == 2:
            sql_query_str, params_list = returned_from_generate_sql
            if not isinstance(sql_query_str, str) or not isinstance(params_list, list):
                logger.error(f"DEBUG: [pipeline.py] generate_sql devolvió una tupla, pero los tipos son incorrectos: q_type={type(sql_query_str)}, p_type={type(params_list)}")
                return {"response": "Error: Tipos inesperados en la consulta SQL generada.", "sql": None, "tables": [], "columns": []}
        elif isinstance(returned_from_generate_sql, str):
            sql_query_str = returned_from_generate_sql
            # params_list ya es []
        else:
            logger.error(f"DEBUG: [pipeline.py] Salida inesperada de generate_sql: {returned_from_generate_sql}")
            return {"response": "Error: Formato inesperado de la consulta SQL generada.", "sql": None, "tables": [], "columns": []}

        logger.info(f"DEBUG: [pipeline.py] SQL generado (str): {sql_query_str}")
        logger.info(f"DEBUG: [pipeline.py] Parámetros generados: {params_list}")

        if not sql_query_str:
            logger.error("DEBUG: [pipeline.py] Error al generar la consulta SQL. SQL string es None o vacío.")
            return {"response": "Error: No se pudo generar la consulta SQL.", "sql": None, "tables": [], "columns": []}
        # --- FIN CAMBIO: Manejo explícito de la salida de generate_sql ---

        # Los logs de depuración del usuario que siguen deberían ahora reflejar sql_query_str y params_list
        # Ejemplo: logger.info(f"STDOUT DEBUG (pipeline.py): SQL generado (después de chequeo): {sql_query_str}")
        # Ejemplo: logger.info(f"STDOUT DEBUG (pipeline.py): Parámetros (después de chequeo): {params_list}")

        # --- INICIO CAMBIO: Validación del SQL generado ---
        if not validate_sql_query(sql_query_str, db_connector):
            logger.error(f"DEBUG: [pipeline.py] SQL inválido tras la generación: {sql_query_str}")
            return {"response": "Error: SQL generado es inválido.", "sql": sql_query_str, "tables": structured_info.get("tables", []), "columns": structured_info.get("columns", [])} # Devolver SQL para depuración
        # --- FIN CAMBIO: Validación del SQL generado ---   
        logger.info(f"DEBUG: [pipeline.py] SQL validado correctamente: {sql_query_str}")
        
        # --- INICIO CAMBIO: Corrección de nombres de columnas y tablas en el SQL ---
        sql_query_str = correct_column_names(sql_query_str, structured_info.get("tables", []), db_structure)
        sql_query_str = validate_table_names(sql_query_str, db_structure)
        logger.info(f"DEBUG: [pipeline.py] SQL corregido: {sql_query_str}")
        # --- FIN CAMBIO: Corrección de nombres de columnas y tablas en el SQL ---
        
        # --- INICIO CAMBIO: Ejecución de la consulta SQL ---
        logger.info(f"DEBUG: [pipeline.py] Paso 7: Ejecutando consulta SQL: '{sql_query_str}' con params: {params_list}")
        # --- EJECUCIÓN REAL DE LA CONSULTA SQL ---
        logger.info(f"DEBUG: A punto de ejecutar db_connector.execute_query con query: '{sql_query_str}' y params: {params_list}")
        try:
            result = db_connector.execute_query(sql_query_str, params_list)
            logger.info(f"DEBUG: Resultado de db_connector.execute_query: {result}")
        except Exception as exec_err:
            logger.error(f"ERROR: Excepción al ejecutar db_connector.execute_query: {exec_err}", exc_info=True)
            result = [{"error": str(exec_err)}]
        logger.info(f"DEBUG: [pipeline.py] Consulta SQL ejecutada. Resultado: {'(demasiado largo para loguear)' if isinstance(result, list) and len(result) > 5 else result}")
        
        # El log original del usuario para "result" puede ser muy verboso.
        # Considerar loguear solo una parte o un resumen del resultado.
        # logger.info(f"DEBUG: [pipeline.py] Consulta SQL ejecutada. Resultado: {result}") 

        if result is None or (isinstance(result, list) and not result): # Manejar tanto None como lista vacía
            logger.warning(f"DEBUG: [pipeline.py] La consulta SQL ('{sql_query_str}' con params {params_list}) no devolvió resultados o el resultado fue None.")
            return {"response": "No se encontraron resultados para la consulta.", "sql": sql_query_str, "tables": structured_info.get("tables", []), "columns": structured_info.get("columns", []), "data": [] if result is None else result} # Devolver lista vacía si es None
        # --- FIN CAMBIO: Ejecución de la consulta SQL ---
        
        # --- INICIO CAMBIO: Preparación de la respuesta final ---
        response = {
            "response": "Consulta ejecutada con éxito.",
            "sql": sql_query_str,
            "tables": structured_info.get("tables", []),
            "columns": structured_info.get("columns", []),
            "data": result,
            "intermediate_steps": {
                "structured_info": structured_info,
                "sql_query": sql_query_str
            }
        }
        if output_intermediate_steps:
            response["intermediate_steps"]["raw_question"] = question
            response["intermediate_steps"]["enriched_question"] = enriched_question
            response["intermediate_steps"]["db_structure"] = db_structure
            response["intermediate_steps"]["terms_dict"] = terms_dict
        logger.info(f"DEBUG: [pipeline.py] Respuesta final preparada: {response}")
        # --- FIN CAMBIO: Preparación de la respuesta final ---
        return response
    except Exception as e:
        logger.error(f"DEBUG: [pipeline.py] Error al ejecutar el flujo de pipeline: {e}")
        raise e
