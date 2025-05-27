# flexible_search_config.py
from typing import Dict, List, Any, Optional, Callable, Union
import json
import logging
from pathlib import Path
import sqlite3

logger = logging.getLogger(__name__)

# Intenta importar la función de invocación del LLM desde llm_utils.py
LLM_CALLER_FUNCTION: Optional[Callable[[List[Dict[str, str]], Dict[str, Any], str], str]] = None
try:
    from .llm_utils import call_llm # Cambiado de invoke_llm a call_llm
    LLM_CALLER_FUNCTION = call_llm
    logger.info("Función call_llm cargada exitosamente desde llm_utils.")
except ImportError:
    logger.warning("No se pudo importar call_llm desde llm_utils. La funcionalidad LLM no estará disponible.")
except Exception as e:
    logger.error(f"Error al importar call_llm: {e}. La funcionalidad LLM no estará disponible.")

# --- Configuraciones Estáticas (Fallback) ---
DEFAULT_DIAGNOSIS_SYNONYMS_CONFIG: Dict[str, List[str]] = {
    "insuficiencia cardiaca": ["falla cardiaca", "fallo cardiaco", "insuficiencia del corazón", "cardiac insufficiency", "ic"],
    "hipertension": ["presion arterial alta", "hta", "hypertension"],
    "diabetes": ["diabetes mellitus", "azucar alta", "dm"],
    "neumonia": ["infeccion pulmonar", "pulmonia", "pneumonia"],
    "infarto de miocardio": ["ataque al corazon", "iam", "miocardial infarction", "infarto"],
    "accidente cerebrovascular": ["acv", "derrame cerebral", "stroke", "ictus"],
    "bronquitis cronica": ["epoc", "enfermedad pulmonar obstructiva cronica", "bronquitis"]
}

# -- Generar dinámicamente el mapa de diagnósticos a códigos (se poblará desde la BD)
DEFAULT_DIAGNOSIS_CODE_MAP: Dict[str, List[str]] = { term: [] for term in DEFAULT_DIAGNOSIS_SYNONYMS_CONFIG.keys() }
# --- Auto-carga dinámica de códigos de diagnóstico si DEFAULT_DIAGNOSIS_CODE_MAP está vacío ---
_db_path = Path(__file__).parent / 'db' / 'database_new.sqlite3.db'
if _db_path.exists():
    try:
        conn = sqlite3.connect(str(_db_path))
        cur = conn.cursor()
        for term, codes in DEFAULT_DIAGNOSIS_CODE_MAP.items():
            if not codes:
                like_pat = f"%{term}%"
                # La tabla CODR_DIAGNOSTIC_GROUPS usa CODG_ID y CODG_DESCRIPTION_ES
                cur.execute(
                    "SELECT CODG_ID FROM CODR_DIAGNOSTIC_GROUPS WHERE lower(CODG_DESCRIPTION_ES) LIKE ?", (like_pat,)
                )
                fetched = cur.fetchall()
                DEFAULT_DIAGNOSIS_CODE_MAP[term] = [str(r[0]) for r in fetched]
        conn.close()
    except Exception as e:
        logger.warning(f"No se pudieron cargar dinámicamente códigos de diagnóstico: {e}")
# --- Fin auto-carga dinámica ---

DEFAULT_TEXT_COLUMNS_FOR_FLEXIBLE_SEARCH: List[str] = [
    "OBSERVACIONES", "DIAGNOSTICO", "MOTIVO_CONSULTA", "DESCRIPCION", "ANTECEDENTES", "EVOLUCION",
    "APPO_OBSERVATIONS", "ALLE_ALLERGY_DESCRIPTION", "PROC_DESCRIPTION", "NOTE_TEXT", "CLINICAL_NOTES",
    "FREE_TEXT_DIAGNOSIS", "PATIENT_HISTORY_NOTES", "SYMPTOM_DESCRIPTION",
    # Nombres genéricos que suelen indicar texto libre
    "TEXTO_LIBRE", "NOTAS", "COMENTARIOS", "DETALLES", "RESUMEN_CLINICO"
]
# Ya no se añaden explícitamente "MEDI_DESCRIPTION_ES" y "ACIN_DESCRIPTION_ES" aquí.
# Se confía en que el LLM las identifique o que estén en el esquema que el LLM analiza.

# --- Funciones de Interacción con LLM ---

def get_llm_generated_synonyms(
    base_diagnoses: Optional[List[str]] = None, 
    llm_caller: Optional[Callable[[List[Dict[str, str]], Dict[str, Any], str], str]] = None
) -> Dict[str, List[str]]:
    """
    Obtiene sinónimos de diagnóstico generados por un LLM.
    Si falla o no hay llm_client, devuelve un diccionario vacío o podría lanzar una excepción.
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
    # Configuración mínima para call_llm. Ajusta según sea necesario.
    # Puedes tomar estos valores de una configuración central si lo prefieres.
    llm_config = {"temperature": 0.1, "max_tokens": 500, "max_retries": 2} 
    step_name = "GenerateSynonymsLLM"

    try:
        logger.info(f"[{step_name}] Enviando prompt al LLM para generar sinónimos para: {base_diagnoses}")
        response_text = llm_caller(messages, llm_config, step_name)
        
        if not response_text or response_text.startswith("ERROR:"):
            logger.error(f"[{step_name}] LLM call devolvió un error o respuesta vacía: {response_text}")
            return {}

        generated_synonyms = json.loads(response_text)
        logger.info(f"[{step_name}] Sinónimos generados por LLM: {generated_synonyms}")
        return generated_synonyms
    except json.JSONDecodeError as e:
        logger.error(f"[{step_name}] Error al decodificar la respuesta JSON del LLM para sinónimos: {e}. Respuesta: {response_text}")
        return {}
    except Exception as e:
        logger.error(f"[{step_name}] Error al interactuar con el LLM para generar sinónimos: {e}")
        return {}

def get_llm_identified_text_columns(
    schema_info: Optional[Dict[str, Any]] = None, 
    llm_caller: Optional[Callable[[List[Dict[str, str]], Dict[str, Any], str], str]] = None
) -> List[str]:
    """
    Identifica columnas de texto relevantes para búsqueda flexible usando un LLM.
    Si falla o no hay llm_client, devuelve una lista vacía.
    """
    if not llm_caller:
        logger.warning("LLM caller (call_llm) no proporcionado para identificar columnas de texto.")
        return []

    prompt_string = ""
    if not schema_info:
        logger.warning("Información del esquema no proporcionada al LLM para identificar columnas de texto. Se intentará sin ella, pero los resultados pueden ser limitados.")
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
        {json.dumps(schema_info, indent=2)}

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
        response_text = llm_caller(messages, llm_config, step_name)

        if not response_text or response_text.startswith("ERROR:"):
            logger.error(f"[{step_name}] LLM call devolvió un error o respuesta vacía: {response_text}")
            return []

        identified_columns = json.loads(response_text)
        logger.info(f"[{step_name}] Columnas de texto identificadas por LLM: {identified_columns}")
        return identified_columns
    except json.JSONDecodeError as e:
        logger.error(f"[{step_name}] Error al decodificar la respuesta JSON del LLM para columnas de texto: {e}. Respuesta: {response_text}")
        return []
    except Exception as e:
        logger.error(f"[{step_name}] Error al interactuar con el LLM para identificar columnas de texto: {e}")
        return []

# --- Lógica de Carga de Configuración ---
# Estas variables contendrán la configuración final a ser usada por sql_generator.py

# Opción para habilitar/deshabilitar la carga desde LLM (podría venir de config.py o variables de entorno)
USE_LLM_FOR_CONFIG = True # Cambiar a True para activar la carga desde LLM
# El LLM_CALLER_FUNCTION se carga al inicio del archivo.

# Cargar Sinónimos
_current_diagnosis_synonyms = DEFAULT_DIAGNOSIS_SYNONYMS_CONFIG.copy()
if USE_LLM_FOR_CONFIG and LLM_CALLER_FUNCTION:
    logger.info("Intentando cargar sinónimos de diagnóstico desde LLM...")
    llm_synonyms = get_llm_generated_synonyms(llm_caller=LLM_CALLER_FUNCTION)
    if llm_synonyms:
        # Aquí se podría implementar una lógica de fusión si se desea,
        # por ejemplo, actualizar los sinónimos existentes o añadir nuevos.
        # Por simplicidad, vamos a reemplazar si el LLM devuelve algo.
        _current_diagnosis_synonyms = llm_synonyms
        logger.info("Sinónimos de diagnóstico cargados/actualizados desde LLM.")
    else:
        logger.warning("No se pudieron cargar sinónimos desde LLM, se usarán los valores por defecto.")

# Normalizar los sinónimos (siempre se hace, ya sea con los por defecto o los del LLM)
NORMALIZED_DIAGNOSIS_SYNONYMS_CONFIG: Dict[str, List[str]] = {
    k.lower(): [s.lower() for s in v_list] 
    for k, v_list in _current_diagnosis_synonyms.items() if isinstance(v_list, list)
}

# Extender sinónimos con los códigos para que el LLM los use en condiciones
for diag_term, codes in DEFAULT_DIAGNOSIS_CODE_MAP.items():
    term_lower = diag_term.lower()
    code_strs = [str(c) for c in codes]
    if code_strs:
        if term_lower in NORMALIZED_DIAGNOSIS_SYNONYMS_CONFIG:
            NORMALIZED_DIAGNOSIS_SYNONYMS_CONFIG[term_lower].extend(code_strs)
        else:
            NORMALIZED_DIAGNOSIS_SYNONYMS_CONFIG[term_lower] = code_strs

# -- Fallback usando texto libre para diagnósticos cuando no hay códigos disponibles
DEFAULT_DIAGNOSIS_TEXT_MAP: Dict[str, List[str]] = {
    "insuficiencia cardiaca": ["insuficiencia cardíaca", "falla cardíaca", "insuficiencia cardiaca", "fallo cardíaco"],
    "neumonia": ["neumonía", "infección pulmonar", "neumonitis", "infección respiratoria"],
    "hipertension": ["hipertensión", "tensión alta", "hipertensión arterial"],
    "insuficiencia renal": ["insuficiencia renal", "fallo renal", "falla renal", "enfermedad renal"]
}

# Extender sinónimos con texto libre fallback
for term, text_variants in DEFAULT_DIAGNOSIS_TEXT_MAP.items():
    term_lower = term.lower()
    variants_lower = [t.lower() for t in text_variants]
    if term_lower in NORMALIZED_DIAGNOSIS_SYNONYMS_CONFIG:
        NORMALIZED_DIAGNOSIS_SYNONYMS_CONFIG[term_lower].extend(variants_lower)
    else:
        NORMALIZED_DIAGNOSIS_SYNONYMS_CONFIG[term_lower] = variants_lower

# Cargar esquema simple para prompts LLM
try:
    simple_schema_path = Path(__file__).parent / 'data' / 'schema_simple.json'
    with open(simple_schema_path, 'r', encoding='utf-8') as f:
        _schema_simple = json.load(f)
    SCHEMA_INFO_FOR_LLM = {tbl: cols for tbl, cols in _schema_simple.items()}
    logger.info("Esquema simple cargado para prompts LLM.")
except Exception as e:
    SCHEMA_INFO_FOR_LLM = None
    logger.warning(f"No se pudo cargar esquema simple para prompts LLM: {e}")

# Cargar Columnas de Búsqueda Flexible
_current_text_columns = DEFAULT_TEXT_COLUMNS_FOR_FLEXIBLE_SEARCH[:] # Copia para no modificar la original
if USE_LLM_FOR_CONFIG and LLM_CALLER_FUNCTION:
    logger.info("Intentando cargar columnas de texto para búsqueda flexible desde LLM...")
    # Usar esquema simple cargado para mejorar la identificación de columnas de texto por el LLM
    llm_columns = get_llm_identified_text_columns(schema_info=SCHEMA_INFO_FOR_LLM, llm_caller=LLM_CALLER_FUNCTION)
    
    if llm_columns:
        # Reemplazar lista de columnas si el LLM devolvió resultados
        _current_text_columns = [col.upper() for col in llm_columns if isinstance(col, str)]
        logger.info(f"Columnas para búsqueda flexible cargadas desde LLM: {_current_text_columns}")
    else:
        logger.warning("No se pudieron cargar columnas de texto desde LLM, se usarán los valores por defecto.")

TEXT_COLUMNS_FOR_FLEXIBLE_SEARCH: List[str] = [col.upper() for col in _current_text_columns]

# Log final de las configuraciones cargadas
logger.debug(f"Configuración final de sinónimos normalizados: {NORMALIZED_DIAGNOSIS_SYNONYMS_CONFIG}")
logger.debug(f"Configuración final de columnas de texto para búsqueda flexible: {TEXT_COLUMNS_FOR_FLEXIBLE_SEARCH}")

# Ejemplo de cómo se podría obtener el cliente LLM (esto debería estar en un módulo de configuración o utils)
# def get_llm_client_instance():
#     # from ..llm_utils import MiClienteLLM # Suponiendo que tienes una clase cliente
#     # client = MiClienteLLM(api_key="tu_api_key")
#     # return client
#     return None # Placeholder
#
# if USE_LLM_FOR_CONFIG:
#     LLM_CLIENT_INSTANCE = get_llm_client_instance()
#     if not LLM_CLIENT_INSTANCE:
#         logger.warning("USE_LLM_FOR_CONFIG está activado pero no se pudo obtener una instancia del cliente LLM.")
