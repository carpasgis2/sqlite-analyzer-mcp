# flexible_search_config.py
from typing import Dict, List, Any, Optional, Callable, Union
import json
import logging
from pathlib import Path
import sqlite3

logger = logging.getLogger(__name__)

# --- Variables de Caché ---
_SCHEMA_INFO_CACHE: Optional[Dict[str, Any]] = None
_NORMALIZED_SYNONYMS_CACHE: Optional[Dict[str, List[str]]] = None
_TEXT_COLUMNS_CACHE: Optional[List[str]] = None

# --- Funciones de Validación y Carga de Configuración ---

def validate_react_format(response_text: str) -> bool:
    """
    Valida si la respuesta es un JSON válido.
    """
    if not response_text:
        logger.error("Respuesta vacía o nula. No se puede procesar.")
        return False
    try:
        json.loads(response_text)
        return True
    except json.JSONDecodeError:
        logger.error(f"La respuesta no es un JSON válido: {response_text}")
        return False

# Intenta importar la función de invocación del LLM desde llm_utils.py
LLM_CALLER_FUNCTION: Optional[Callable[[Dict[str, Any], List[Dict[str, str]]], str]] = None # Updated type hint
try:
    from .llm_utils import call_llm # Cambiado de invoke_llm a call_llm
    LLM_CALLER_FUNCTION = call_llm
    logger.info("Función call_llm cargada exitosamente desde llm_utils.")
except ImportError:
    logger.warning("No se pudo importar call_llm desde llm_utils. La funcionalidad LLM no estará disponible.")
except Exception as e:
    logger.error(f"Error al importar call_llm: {e}. La funcionalidad LLM no estará disponible.")

# Opción para habilitar/deshabilitar la carga desde LLM (podría venir de config.py o variables de entorno)
USE_LLM_FOR_CONFIG = True # Mover aquí para que esté disponible globalmente antes de los getters

# --- Configuraciones Estáticas (Fallback) ---
DEFAULT_DIAGNOSIS_SYNONYMS_CONFIG: Dict[str, List[str]] = {
    "insuficiencia cardiaca": ["falla cardiaca", "fallo cardiaco", "insuficiencia del corazón", "cardiac insufficiency", "ic"],
    "hipertension": ["presion arterial alta", "hta", "hypertension", "hipertensión arterial", "hta"],
    "diabetes": ["diabetes mellitus", "azucar alta", "dm", "dbt"],
    "neumonia": ["infeccion pulmonar", "pulmonia", "pneumonia"],
    "infarto de miocardio": ["ataque al corazon", "iam", "miocardial infarction", "infarto"],
    "accidente cerebrovascular": ["acv", "derrame cerebral", "stroke", "ictus"],
    "bronquitis cronica": ["epoc", "enfermedad pulmonar obstructiva cronica", "bronquitis"]
    # Puedes añadir más diagnósticos y abreviaturas aquí
}

# --- Mapeo de grupos terapéuticos antihipertensivos y principios activos ---
# Esto puede venir de la BD, pero aquí se define un fallback manual
ANTIHIPERTENSIVE_GROUPS = [
    "antihipertensivo", "antihipertensivos", "ieca", "ara ii", "ara-ii", "betabloqueante", "calcioantagonista", "diurético", "diuréticos", "inhibidor de la eca", "inhibidores de la eca"
]
# Principios activos comunes (puedes ampliar esta lista)
ANTIHIPERTENSIVE_ACTIVE_INGREDIENTS = [
    "enalapril", "lisinopril", "captopril", "ramipril", "losartan", "valsartan", "olmesartan", "amlodipino", "nifedipino", "diltiazem", "verapamilo", "atenolol", "metoprolol", "bisoprolol", "nebivolol", "clortalidona", "hidroclorotiazida"
]

# -- Generar dinámicamente el mapa de diagnósticos a códigos (se poblará desde la BD)
DEFAULT_DIAGNOSIS_CODE_MAP: Dict[str, List[str]] = { term: [] for term in DEFAULT_DIAGNOSIS_SYNONYMS_CONFIG.keys() }
# --- Auto-carga dinámica de códigos de diagnóstico si DEFAULT_DIAGNOSIS_CODE_MAP está vacío ---
_db_path = Path(__file__).parent / 'db' / 'database_new.sqlite3.db'

# >>> INICIO CAMBIO: Comentar la carga dinámica de códigos para evitar error CODG_ID
# if _db_path.exists():
#     try:
#         conn = sqlite3.connect(str(_db_path))
#         cur = conn.cursor()
#         for term, codes in DEFAULT_DIAGNOSIS_CODE_MAP.items():
#             if not codes:
#                 like_pat = f"%{term}%"
#                 # La tabla CODR_DIAGNOSTIC_GROUPS usa CODG_ID y CODG_DESCRIPTION_ES
#                 # ¡¡¡REVISAR ESTA CONSULTA Y EL ESQUEMA DE LA BD!!!
#                 cur.execute(
#                     "SELECT CODG_ID FROM CODR_DIAGNOSTIC_GROUPS WHERE lower(CODG_DESCRIPTION_ES) LIKE ?", (like_pat,)
#                 )
#                 fetched = cur.fetchall()
#                 DEFAULT_DIAGNOSIS_CODE_MAP[term] = [str(r[0]) for r in fetched]
#         conn.close()
#     except Exception as e:
#         logger.warning(f"No se pudieron cargar dinámicamente códigos de diagnóstico: {e}")
# <<< FIN CAMBIO

# --- Utilidad para extraer variantes de diagnóstico desde un hint en la query (para el pipeline) ---
def extract_diagnosis_variants_from_hint(query: str) -> list:
    """
    Extrae variantes de diagnóstico del hint /*DIAGNOSIS_VARIANTS:...*/ en la query.
    Devuelve una lista de variantes (strings) o [] si no hay hint.
    """
    import re
    m = re.search(r"/\*DIAGNOSIS_VARIANTS:([^*]+)\*/", query)
    if m:
        variants = m.group(1).split("|")
        return [v.strip().lower() for v in variants if v.strip()]
    return []

DEFAULT_TEXT_COLUMNS_FOR_FLEXIBLE_SEARCH: List[str] = [
    "OBSERVACIONES", "DIAGNOSTICO", "MOTIVO_CONSULTA", "DESCRIPCION", "ANTECEDENTES", "EVOLUCION",
    "APPO_OBSERVATIONS", "ALLE_ALLERGY_DESCRIPTION", "PROC_DESCRIPTION", "NOTE_TEXT", "CLINICAL_NOTES",
    "FREE_TEXT_DIAGNOSIS", "PATIENT_HISTORY_NOTES", "SYMPTOM_DESCRIPTION",
    # Nombres genéricos que suelen indicar texto libre
    "TEXTO_LIBRE", "NOTAS", "COMENTARIOS", "DETALLES", "RESUMEN_CLINICO"
]
# Ya no se añaden explícitamente "MEDI_DESCRIPTION_ES" y "ACIN_DESCRIPTION_ES" aquí.
# Se confía en que el LLM las identifique o que estén en el esquema que el LLM analiza.

# --- Funciones de Obtención de Configuración (con carga perezosa y caché) ---

def get_schema_info_for_llm() -> Optional[Dict[str, Any]]:
    """
    Carga y devuelve la información del esquema simplificado desde 'schema_simple.json'.
    Utiliza una caché para evitar recargas.
    """
    global _SCHEMA_INFO_CACHE
    if _SCHEMA_INFO_CACHE is not None:
        return _SCHEMA_INFO_CACHE

    try:
        simple_schema_path = Path(__file__).parent / 'data' / 'schema_simple.json'
        with open(simple_schema_path, 'r', encoding='utf-8') as f:
            schema_data = json.load(f)
        _SCHEMA_INFO_CACHE = {tbl: cols for tbl, cols in schema_data.items()}
        logger.info("Esquema simple cargado y cacheado para prompts LLM.")
        return _SCHEMA_INFO_CACHE
    except Exception as e:
        logger.warning(f"No se pudo cargar esquema simple para prompts LLM: {e}")
        _SCHEMA_INFO_CACHE = {} # Cachear un diccionario vacío en caso de error para evitar reintentos
        return _SCHEMA_INFO_CACHE

# --- Funciones de Interacción con LLM ---

def get_llm_generated_synonyms(
    base_diagnoses: Optional[List[str]] = None, 
    llm_caller: Optional[Callable[[Dict[str, Any], List[Dict[str, str]]], Any]] = None # Puede devolver str u objeto
) -> Dict[str, List[str]]:
    """
    Obtiene sinónimos de diagnóstico generados por un LLM.
    Soporta respuestas tipo string o tipo objeto (con .content).
    """
    if not llm_caller:
        logger.warning("LLM caller (call_llm) no proporcionado para generar sinónimos.")
        return {}

    if not base_diagnoses:
        base_diagnoses = list(DEFAULT_DIAGNOSIS_SYNONYMS_CONFIG.keys())
        if not base_diagnoses:
            logger.warning("No hay diagnósticos base para generar sinónimos.")
            return {}

    prompt_string = f"""
    Eres un asistente experto en terminología médica. 
    Para cada uno de los siguientes términos de diagnóstico médico, proporciona una lista de sinónimos comunes, 
    abreviaturas populares y jerga médica relacionada.
    Los términos son: {json.dumps(base_diagnoses)}.
    Devuelve la respuesta exclusivamente en formato JSON, como un diccionario donde cada clave es el término original 
    y el valor es una lista de sus sinónimos. Ejemplo:
    {{
      "termino1": ["sinonimo1_1", "sinonimo1_2"],
      "termino2": ["sinonimo2_1", "sinonimo2_2"]
    }}
    """
    messages = [{"role": "user", "content": prompt_string}]
    llm_config = {"temperature": 0.1, "max_tokens": 500, "max_retries": 2} 
    step_name = "GenerateSynonymsLLM"

    try:
        logger.info(f"[{step_name}] Enviando prompt al LLM para generar sinónimos para: {base_diagnoses}")
        # Permitir que el llm_caller sea un modelo Langchain o una función tradicional
        response = None
        if hasattr(llm_caller, "invoke"):
            response = llm_caller.invoke(messages)
        else:
            response = llm_caller(llm_config, messages)

        # Extraer el texto de la respuesta
        if hasattr(response, "content"):
            response_text = response.content
        elif isinstance(response, str):
            response_text = response
        else:
            logger.error(f"[{step_name}] Respuesta del LLM no es un string ni tiene atributo 'content': {response}")
            return {}

        if not isinstance(response_text, str):
            logger.error(f"[{step_name}] El texto extraído de la respuesta del LLM no es un string: {response_text}")
            return {}

        # Limpiar posible Markdown de la respuesta del LLM (```json ... ```
        cleaned_response_text = response_text.strip()
        if cleaned_response_text.startswith("```json"):
            cleaned_response_text = cleaned_response_text[7:]
        if cleaned_response_text.endswith("```"):
            cleaned_response_text = cleaned_response_text[:-3]
        cleaned_response_text = cleaned_response_text.strip()

        if not validate_react_format(cleaned_response_text):
            logger.error(f"[{step_name}] Respuesta del LLM no es un JSON válido: {cleaned_response_text[:200]}...")
            return {}

        generated_synonyms = json.loads(cleaned_response_text)
        logger.info(f"[{step_name}] Sinónimos generados por LLM: {generated_synonyms}")
        return generated_synonyms
    except json.JSONDecodeError as e:
        logger.error(f"[{step_name}] Error al decodificar la respuesta JSON del LLM: {e}. Respuesta: {response_text}")
        return {}
    except Exception as e:
        logger.error(f"[{step_name}] Error al interactuar con el LLM para generar sinónimos: {e}")
        return {}

def get_llm_identified_text_columns(
    schema_info: Optional[Dict[str, Any]] = None, 
    llm_caller: Optional[Callable[[Dict[str, Any], List[Dict[str, str]]], str]] = None # Updated type hint
) -> List[str]:
    """
    Identifica columnas de texto relevantes para búsqueda flexible usando un LLM.
    Si falla o no hay llm_client, devuelve una lista vacía.
    """
    if not llm_caller:
        logger.warning("LLM caller (call_llm) no proporcionado para identificar columnas de texto.")
        return []

    # Obtener schema_info usando la nueva función cacheada
    current_schema_info = schema_info
    if current_schema_info is None:
        current_schema_info = get_schema_info_for_llm()

    prompt_string = ""
    if not current_schema_info: # schema_info puede ser un diccionario vacío si la carga falló
        logger.warning("Información del esquema no disponible o no proporcionada al LLM para identificar columnas de texto. Se intentará sin ella, pero los resultados pueden ser limitados.")
        prompt_string = f"""
        Eres un asistente experto en análisis de esquemas de bases de datos médicas.
        Por favor, identifica nombres de columnas genéricos que probablemente contengan texto libre relevante para:
        - Diagnósticos médicos
        - Observaciones clínicas
        - Motivos de consulta
        - Antecedentes del paciente
        - Evolución de la enfermedad o tratamiento
        - Notas médicas generales
        - Descripciones de procedimientos
        - Descripciones de medicamentos
        - Descripciones de principios activos
        - Comentarios adicionales

        Devuelve la respuesta exclusivamente en formato JSON, como una lista de strings, donde cada string es un nombre de columna.
        Ejemplo: ["OBSERVACIONES", "DIAGNOSTICO_PRINCIPAL_TEXTO", "NOTAS_EVOLUCION"]
        """
    else:
        prompt_string = f"""
        Eres un asistente experto en análisis de esquemas de bases de datos médicas.
        Dado el siguiente esquema de base de datos (simplificado como un diccionario de nombres de tabla a listas de nombres de columna):
        {json.dumps(current_schema_info, indent=2)}

        Por favor, identifica todas las columnas en cualquier tabla que probablemente contengan texto libre relevante para:
        - Diagnósticos médicos
        - Observaciones clínicas
        - Motivos de consulta
        - Antecedentes del paciente
        - Evolución de la enfermedad o tratamiento
        - Notas médicas generales
        - Descripciones de procedimientos
        - Descripciones de medicamentos
        - Descripciones de principios activos
        - Comentarios adicionales

        Devuelve la respuesta exclusivamente en formato JSON, como una lista de strings, donde cada string es un nombre de columna (sin el nombre de la tabla).
        Ejemplo: ["OBSERVACIONES", "DIAGNOSTICO_PRINCIPAL_TEXTO", "NOTAS_EVOLUCION"]
        """

    messages = [{"role": "user", "content": prompt_string}]
    llm_config = {"temperature": 0.1, "max_tokens": 500, "max_retries": 2}
    step_name = "IdentifyTextColumnsLLM"

    try:
        logger.info(f"[{step_name}] Enviando prompt al LLM para identificar columnas de texto relevantes.")
        response_text = llm_caller(llm_config, messages) # Corrected argument order

        if not response_text or response_text.startswith("ERROR:"): # Asumiendo que call_llm puede devolver "ERROR:"
            logger.error(f"[{step_name}] LLM call devolvió un error o respuesta vacía: {response_text}")
            return []

        # >>> INICIO CAMBIO: Limpiar posible Markdown de la respuesta del LLM
        cleaned_response_text = response_text.strip()
        if cleaned_response_text.startswith("```json"):
            cleaned_response_text = cleaned_response_text[7:]
        if cleaned_response_text.endswith("```"):
            cleaned_response_text = cleaned_response_text[:-3]
        cleaned_response_text = cleaned_response_text.strip()
        # <<< FIN CAMBIO

        # Validar que la respuesta sea un JSON válido
        if not validate_react_format(cleaned_response_text): # validate_react_format ahora valida JSON
            logger.error(f"[{step_name}] Respuesta del LLM no es un JSON válido: {cleaned_response_text}")
            return []

        identified_columns = json.loads(cleaned_response_text)
        if not isinstance(identified_columns, list) or not all(isinstance(col, str) for col in identified_columns):
            logger.error(f"[{step_name}] El JSON de columnas identificadas por LLM no es una lista de strings: {identified_columns}")
            return []
            
        logger.info(f"[{step_name}] Columnas de texto identificadas por LLM: {identified_columns}")
        return identified_columns
    except json.JSONDecodeError as e:
        # Esto no debería ocurrir si validate_react_format funciona correctamente, pero se mantiene por seguridad.
        logger.error(f"[{step_name}] Error al decodificar la respuesta JSON del LLM para columnas de texto: {e}. Respuesta: {response_text}")
        return []
    except Exception as e:
        logger.error(f"[{step_name}] Error al interactuar con el LLM para identificar columnas de texto: {e}")
        return []

# --- Lógica de Carga de Configuración ---
# Estas variables contendrán la configuración final a ser usada por sql_generator.py

# --- Configuraciones Estáticas (Fallback) ---
DEFAULT_DIAGNOSIS_SYNONYMS_CONFIG: Dict[str, List[str]] = {
    "insuficiencia cardiaca": ["falla cardiaca", "fallo cardiaco", "insuficiencia del corazón", "cardiac insufficiency", "ic"],
    "hipertension": ["presion arterial alta", "hta", "hypertension", "hipertensión arterial", "hta"],
    "diabetes": ["diabetes mellitus", "azucar alta", "dm", "dbt"],
    "neumonia": ["infeccion pulmonar", "pulmonia", "pneumonia"],
    "infarto de miocardio": ["ataque al corazon", "iam", "miocardial infarction", "infarto"],
    "accidente cerebrovascular": ["acv", "derrame cerebral", "stroke", "ictus"],
    "bronquitis cronica": ["epoc", "enfermedad pulmonar obstructiva cronica", "bronquitis"]
    # Puedes añadir más diagnósticos y abreviaturas aquí
}

# --- Mapeo de grupos terapéuticos antihipertensivos y principios activos ---
# Esto puede venir de la BD, pero aquí se define un fallback manual
ANTIHIPERTENSIVE_GROUPS = [
    "antihipertensivo", "antihipertensivos", "ieca", "ara ii", "ara-ii", "betabloqueante", "calcioantagonista", "diurético", "diuréticos", "inhibidor de la eca", "inhibidores de la eca"
]
# Principios activos comunes (puedes ampliar esta lista)
ANTIHIPERTENSIVE_ACTIVE_INGREDIENTS = [
    "enalapril", "lisinopril", "captopril", "ramipril", "losartan", "valsartan", "olmesartan", "amlodipino", "nifedipino", "diltiazem", "verapamilo", "atenolol", "metoprolol", "bisoprolol", "nebivolol", "clortalidona", "hidroclorotiazida"
]

# -- Generar dinámicamente el mapa de diagnósticos a códigos (se poblará desde la BD)
DEFAULT_DIAGNOSIS_CODE_MAP: Dict[str, List[str]] = { term: [] for term in DEFAULT_DIAGNOSIS_SYNONYMS_CONFIG.keys() }
# --- Auto-carga dinámica de códigos de diagnóstico si DEFAULT_DIAGNOSIS_CODE_MAP está vacío ---
_db_path = Path(__file__).parent / 'db' / 'database_new.sqlite3.db'

# >>> INICIO CAMBIO: Comentar la carga dinámica de códigos para evitar error CODG_ID
# if _db_path.exists():
#     try:
#         conn = sqlite3.connect(str(_db_path))
#         cur = conn.cursor()
#         for term, codes in DEFAULT_DIAGNOSIS_CODE_MAP.items():
#             if not codes:
#                 like_pat = f"%{term}%"
#                 # La tabla CODR_DIAGNOSTIC_GROUPS usa CODG_ID y CODG_DESCRIPTION_ES
#                 # ¡¡¡REVISAR ESTA CONSULTA Y EL ESQUEMA DE LA BD!!!
#                 cur.execute(
#                     "SELECT CODG_ID FROM CODR_DIAGNOSTIC_GROUPS WHERE lower(CODG_DESCRIPTION_ES) LIKE ?", (like_pat,)
#                 )
#                 fetched = cur.fetchall()
#                 DEFAULT_DIAGNOSIS_CODE_MAP[term] = [str(r[0]) for r in fetched]
#         conn.close()
#     except Exception as e:
#         logger.warning(f"No se pudieron cargar dinámicamente códigos de diagnóstico: {e}")
# <<< FIN CAMBIO

# --- Utilidad para extraer variantes de diagnóstico desde un hint en la query (para el pipeline) ---
def extract_diagnosis_variants_from_hint(query: str) -> list:
    """
    Extrae variantes de diagnóstico del hint /*DIAGNOSIS_VARIANTS:...*/ en la query.
    Devuelve una lista de variantes (strings) o [] si no hay hint.
    """
    import re
    m = re.search(r"/\*DIAGNOSIS_VARIANTS:([^*]+)\*/", query)
    if m:
        variants = m.group(1).split("|")
        return [v.strip().lower() for v in variants if v.strip()]
    return []

DEFAULT_TEXT_COLUMNS_FOR_FLEXIBLE_SEARCH: List[str] = [
    "OBSERVACIONES", "DIAGNOSTICO", "MOTIVO_CONSULTA", "DESCRIPCION", "ANTECEDENTES", "EVOLUCION",
    "APPO_OBSERVATIONS", "ALLE_ALLERGY_DESCRIPTION", "PROC_DESCRIPTION", "NOTE_TEXT", "CLINICAL_NOTES",
    "FREE_TEXT_DIAGNOSIS", "PATIENT_HISTORY_NOTES", "SYMPTOM_DESCRIPTION",
    # Nombres genéricos que suelen indicar texto libre
    "TEXTO_LIBRE", "NOTAS", "COMENTARIOS", "DETALLES", "RESUMEN_CLINICO"
]
# Ya no se añaden explícitamente "MEDI_DESCRIPTION_ES" y "ACIN_DESCRIPTION_ES" aquí.
# Se confía en que el LLM las identifique o que estén en el esquema que el LLM analiza.

# --- Funciones de Obtención de Configuración (con carga perezosa y caché) ---

def get_schema_info_for_llm() -> Optional[Dict[str, Any]]:
    """
    Carga y devuelve la información del esquema simplificado desde 'schema_simple.json'.
    Utiliza una caché para evitar recargas.
    """
    global _SCHEMA_INFO_CACHE
    if _SCHEMA_INFO_CACHE is not None:
        return _SCHEMA_INFO_CACHE

    try:
        simple_schema_path = Path(__file__).parent / 'data' / 'schema_simple.json'
        with open(simple_schema_path, 'r', encoding='utf-8') as f:
            schema_data = json.load(f)
        _SCHEMA_INFO_CACHE = {tbl: cols for tbl, cols in schema_data.items()}
        logger.info("Esquema simple cargado y cacheado para prompts LLM.")
        return _SCHEMA_INFO_CACHE
    except Exception as e:
        logger.warning(f"No se pudo cargar esquema simple para prompts LLM: {e}")
        _SCHEMA_INFO_CACHE = {} # Cachear un diccionario vacío en caso de error para evitar reintentos
        return _SCHEMA_INFO_CACHE

# --- Funciones de Interacción con LLM ---

def get_llm_generated_synonyms(
    base_diagnoses: Optional[List[str]] = None, 
    llm_caller: Optional[Callable[[Dict[str, Any], List[Dict[str, str]]], Any]] = None # Puede devolver str u objeto
) -> Dict[str, List[str]]:
    """
    Obtiene sinónimos de diagnóstico generados por un LLM.
    Soporta respuestas tipo string o tipo objeto (con .content).
    """
    if not llm_caller:
        logger.warning("LLM caller (call_llm) no proporcionado para generar sinónimos.")
        return {}

    if not base_diagnoses:
        base_diagnoses = list(DEFAULT_DIAGNOSIS_SYNONYMS_CONFIG.keys())
        if not base_diagnoses:
            logger.warning("No hay diagnósticos base para generar sinónimos.")
            return {}

    prompt_string = f"""
    Eres un asistente experto en terminología médica. 
    Para cada uno de los siguientes términos de diagnóstico médico, proporciona una lista de sinónimos comunes, 
    abreviaturas populares y jerga médica relacionada.
    Los términos son: {json.dumps(base_diagnoses)}.
    Devuelve la respuesta exclusivamente en formato JSON, como un diccionario donde cada clave es el término original 
    y el valor es una lista de sus sinónimos. Ejemplo:
    {{
      "termino1": ["sinonimo1_1", "sinonimo1_2"],
      "termino2": ["sinonimo2_1", "sinonimo2_2"]
    }}
    """
    messages = [{"role": "user", "content": prompt_string}]
    llm_config = {"temperature": 0.1, "max_tokens": 500, "max_retries": 2} 
    step_name = "GenerateSynonymsLLM"

    try:
        logger.info(f"[{step_name}] Enviando prompt al LLM para generar sinónimos para: {base_diagnoses}")
        # Permitir que el llm_caller sea un modelo Langchain o una función tradicional
        response = None
        if hasattr(llm_caller, "invoke"):
            response = llm_caller.invoke(messages)
        else:
            response = llm_caller(llm_config, messages)

        # Extraer el texto de la respuesta
        if hasattr(response, "content"):
            response_text = response.content
        elif isinstance(response, str):
            response_text = response
        else:
            logger.error(f"[{step_name}] Respuesta del LLM no es un string ni tiene atributo 'content': {response}")
            return {}

        if not isinstance(response_text, str):
            logger.error(f"[{step_name}] El texto extraído de la respuesta del LLM no es un string: {response_text}")
            return {}

        # Limpiar posible Markdown de la respuesta del LLM (```json ... ```
        cleaned_response_text = response_text.strip()
        if cleaned_response_text.startswith("```json"):
            cleaned_response_text = cleaned_response_text[7:]
        if cleaned_response_text.endswith("```"):
            cleaned_response_text = cleaned_response_text[:-3]
        cleaned_response_text = cleaned_response_text.strip()

        if not validate_react_format(cleaned_response_text):
            logger.error(f"[{step_name}] Respuesta del LLM no es un JSON válido: {cleaned_response_text[:200]}...")
            return {}

        generated_synonyms = json.loads(cleaned_response_text)
        logger.info(f"[{step_name}] Sinónimos generados por LLM: {generated_synonyms}")
        return generated_synonyms
    except json.JSONDecodeError as e:
        logger.error(f"[{step_name}] Error al decodificar la respuesta JSON del LLM: {e}. Respuesta: {response_text}")
        return {}
    except Exception as e:
        logger.error(f"[{step_name}] Error al interactuar con el LLM para generar sinónimos: {e}")
        return {}

def get_llm_identified_text_columns(
    schema_info: Optional[Dict[str, Any]] = None, 
    llm_caller: Optional[Callable[[Dict[str, Any], List[Dict[str, str]]], str]] = None # Updated type hint
) -> List[str]:
    """
    Identifica columnas de texto relevantes para búsqueda flexible usando un LLM.
    Si falla o no hay llm_client, devuelve una lista vacía.
    """
    if not llm_caller:
        logger.warning("LLM caller (call_llm) no proporcionado para identificar columnas de texto.")
        return []

    # Obtener schema_info usando la nueva función cacheada
    current_schema_info = schema_info
    if current_schema_info is None:
        current_schema_info = get_schema_info_for_llm()

    prompt_string = ""
    if not current_schema_info: # schema_info puede ser un diccionario vacío si la carga falló
        logger.warning("Información del esquema no disponible o no proporcionada al LLM para identificar columnas de texto. Se intentará sin ella, pero los resultados pueden ser limitados.")
        prompt_string = f"""
        Eres un asistente experto en análisis de esquemas de bases de datos médicas.
        Por favor, identifica nombres de columnas genéricos que probablemente contengan texto libre relevante para:
        - Diagnósticos médicos
        - Observaciones clínicas
        - Motivos de consulta
        - Antecedentes del paciente
        - Evolución de la enfermedad o tratamiento
        - Notas médicas generales
        - Descripciones de procedimientos
        - Descripciones de medicamentos
        - Descripciones de principios activos
        - Comentarios adicionales

        Devuelve la respuesta exclusivamente en formato JSON, como una lista de strings, donde cada string es un nombre de columna.
        Ejemplo: ["OBSERVACIONES", "DIAGNOSTICO_PRINCIPAL_TEXTO", "NOTAS_EVOLUCION"]
        """
    else:
        prompt_string = f"""
        Eres un asistente experto en análisis de esquemas de bases de datos médicas.
        Dado el siguiente esquema de base de datos (simplificado como un diccionario de nombres de tabla a listas de nombres de columna):
        {json.dumps(current_schema_info, indent=2)}

        Por favor, identifica todas las columnas en cualquier tabla que probablemente contengan texto libre relevante para:
        - Diagnósticos médicos
        - Observaciones clínicas
        - Motivos de consulta
        - Antecedentes del paciente
        - Evolución de la enfermedad o tratamiento
        - Notas médicas generales
        - Descripciones de procedimientos
        - Descripciones de medicamentos
        - Descripciones de principios activos
        - Comentarios adicionales

        Devuelve la respuesta exclusivamente en formato JSON, como una lista de strings, donde cada string es un nombre de columna (sin el nombre de la tabla).
        Ejemplo: ["OBSERVACIONES", "DIAGNOSTICO_PRINCIPAL_TEXTO", "NOTAS_EVOLUCION"]
        """

    messages = [{"role": "user", "content": prompt_string}]
    llm_config = {"temperature": 0.1, "max_tokens": 500, "max_retries": 2}
    step_name = "IdentifyTextColumnsLLM"

    try:
        logger.info(f"[{step_name}] Enviando prompt al LLM para identificar columnas de texto relevantes.")
        response_text = llm_caller(llm_config, messages) # Corrected argument order

        if not response_text or response_text.startswith("ERROR:"): # Asumiendo que call_llm puede devolver "ERROR:"
            logger.error(f"[{step_name}] LLM call devolvió un error o respuesta vacía: {response_text}")
            return []

        # >>> INICIO CAMBIO: Limpiar posible Markdown de la respuesta del LLM
        cleaned_response_text = response_text.strip()
        if cleaned_response_text.startswith("```json"):
            cleaned_response_text = cleaned_response_text[7:]
        if cleaned_response_text.endswith("```"):
            cleaned_response_text = cleaned_response_text[:-3]
        cleaned_response_text = cleaned_response_text.strip()
        # <<< FIN CAMBIO

        # Validar que la respuesta sea un JSON válido
        if not validate_react_format(cleaned_response_text): # validate_react_format ahora valida JSON
            logger.error(f"[{step_name}] Respuesta del LLM no es un JSON válido: {cleaned_response_text}")
            return []

        identified_columns = json.loads(cleaned_response_text)
        if not isinstance(identified_columns, list) or not all(isinstance(col, str) for col in identified_columns):
            logger.error(f"[{step_name}] El JSON de columnas identificadas por LLM no es una lista de strings: {identified_columns}")
            return []
            
        logger.info(f"[{step_name}] Columnas de texto identificadas por LLM: {identified_columns}")
        return identified_columns
    except json.JSONDecodeError as e:
        # Esto no debería ocurrir si validate_react_format funciona correctamente, pero se mantiene por seguridad.
        logger.error(f"[{step_name}] Error al decodificar la respuesta JSON del LLM para columnas de texto: {e}. Respuesta: {response_text}")
        return []
    except Exception as e:
        logger.error(f"[{step_name}] Error al interactuar con el LLM para identificar columnas de texto: {e}")
        return []
