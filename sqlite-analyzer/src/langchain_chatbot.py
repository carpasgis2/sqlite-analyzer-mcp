"""
LangChain Tool para usar el pipeline de chatbot existente.
Permite usar el pipeline como herramienta en un agente conversacional LangChain.
"""
import logging # Asegurar que logging se importa al principio
import os
import sys
import re
import json # Asegurarse de que json está importado
import concurrent.futures
from typing import Type, Any # Añadido Any
from pydantic import PrivateAttr  # Añadir esta importación para atributos privados

from langchain.tools import Tool # Descomentado
from langchain_core.tools import BaseTool # Usar BaseTool
from pydantic.v1 import BaseModel, Field # MODIFICADO: Para args_schema si es necesario

from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI # Asegurarse que esta es la importación
from langchain_core.language_models.base import BaseLanguageModel # Para tipado más genérico si se prefiere

from langchain_core.exceptions import OutputParserException
import sqlite3 # Nueva importación

# Gestionar la importación de LLM_MODEL_NAME
try:
    from src.llm_utils import LLM_MODEL_NAME # MODIFICADO: Añadido prefijo src.
except ImportError:
    # El logger puede no estar completamente configurado aquí si esto está en la parte superior.
    # Se podría registrar una advertencia más tarde o usar print.
    print("Advertencia: llm_utils.py no encontrado o LLM_MODEL_NAME no definido. Usando un valor por defecto para LLM_MODEL_NAME.")
    LLM_MODEL_NAME = "deepseek-coder" # Valor por defecto

# Importación para la nueva herramienta BioChat
# Asumiendo que biochat.py está en el directorio raíz del proyecto 'sina_mcp'
# y este archivo (langchain_chatbot.py) está en 'sina_mcp/sqlite-analyzer/src/'
import sys
import os
# Añadir el directorio raíz del proyecto (sina_mcp) al sys.path
# Esto permite importaciones absolutas desde la raíz del proyecto
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from biochat import full_pipeline # Importación absoluta
except ImportError as e:
    print(f"Error crítico: No se pudo importar full_pipeline desde biochat.py. Detalles: {e}. La herramienta BioChat no estará disponible.")
    full_pipeline = None
except Exception as e:
    print(f"Error inesperado al importar full_pipeline: {e}")
    full_pipeline = None

# Importar el pipeline reducido y la función desde biochat
try:
    from biochat import reduced_pipeline as biochat_reduced_pipeline, pubmed_metadata_only_pipeline
except ImportError:
    biochat_reduced_pipeline = None
    pubmed_metadata_only_pipeline = None

from langchain_core.runnables.history import RunnableWithMessageHistory

# MODIFICADO: Usar importación relativa para pipeline
from .pipeline import chatbot_pipeline_entrypoint as chatbot_pipeline
from .sql_utils import list_tables, list_columns, search_schema # Importaciones relativas
from .db_connector import DBConnector # <--- AÑADIDA ESTA LÍNEA

# Configuración del logger
# --- INICIO CONFIGURACIÓN DE LOGGING ---
# Configurar el logger raíz para capturar logs de este script y otros módulos (ej. pipeline)
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Nivel base, puedes cambiar a logging.DEBUG para más detalle

# Definir log_formatter y log_file_path aquí para que estén disponibles globalmente en este módulo
log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] %(message)s"
)
log_file_path = os.path.join(os.path.dirname(__file__), "chatbot_agent.log")

# Evitar añadir múltiples handlers si el script se recarga (común en algunos entornos)
if not logger.handlers:
    # Handler para guardar en archivo
    # Cambiado a mode='w' para que los logs se creen en un archivo nuevo (sobrescribiendo si existe)
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8') 
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    # Handler para mostrar en consola
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
    
    logger.info(f"Logging configurado. Los logs se guardarán en: {log_file_path} (modo overwrite)")
else:
    logger.info(f"El logger ya tiene handlers. Los logs continuarán en: {log_file_path} (modo overwrite)")
# --- FIN CONFIGURACIÓN DE LOGGING ---\

# Configuración de la API de DeepSeek (preferiblemente desde variables de entorno)
_DEEPSEEK_API_URL_FROM_ENV = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")

# Asegurarse de que la URL base no contenga /chat/completions
if _DEEPSEEK_API_URL_FROM_ENV.endswith("/chat/completions"):
    LLM_API_BASE_URL = _DEEPSEEK_API_URL_FROM_ENV[:-len("/chat/completions")]
elif _DEEPSEEK_API_URL_FROM_ENV.endswith("/chat/completions/"):
    LLM_API_BASE_URL = _DEEPSEEK_API_URL_FROM_ENV[:-len("/chat/completions/")]
else:
    LLM_API_BASE_URL = _DEEPSEEK_API_URL_FROM_ENV

# Eliminar barras inclinadas al final de la URL base si las hubiera, para evitar dobles barras
LLM_API_BASE_URL = LLM_API_BASE_URL.rstrip('/')

LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-aedf531ee17447aa95c9102e595f29ae") # Clave API de DeepSeek
LLM_MODEL = LLM_MODEL_NAME # Usar la variable importada o el valor por defecto
LLM_PROVIDER = "deepseek" # Proveedor

# Crear la instancia de LLM que se usará en las herramientas
# Esta instancia se crea aquí porque sql_medical_chatbot_tool y query_planner_tool se instancian en este mismo archivo.
try:
    llm_instance = ChatOpenAI(
        model_name=LLM_MODEL,
        openai_api_base=LLM_API_BASE_URL,
        openai_api_key=LLM_API_KEY,
        temperature=0.0,
    )
    logger.info(f"Instancia de ChatOpenAI ({LLM_MODEL}) creada exitosamente para herramientas.")
except Exception as e:
    logger.error(f"Error al crear la instancia de ChatOpenAI para herramientas: {e}", exc_info=True)
    llm_instance = None

# Instancia el conector de base de datos
# Ruta a la base de datos (relativa a src)
# Asumiendo que db_connector.py está en src y la BD en src/db/database_new.sqlite3.db
# y los esquemas en src/data/
_SCRIPT_DIR_LC = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DB_PATH_LC = os.path.join(_SCRIPT_DIR_LC, "db", "database_new.sqlite3.db")
_DEFAULT_SCHEMA_PATH_LC = os.path.join(_SCRIPT_DIR_LC, "data", "schema_simple.json") # o schema_enhanced.json según necesidad

# Crear la instancia del DBConnector correcto
# Esta instancia será usada por SQLMedicalChatbot
db_connector_instance = DBConnector(
    db_path=_DEFAULT_DB_PATH_LC
)
logger.info(f"Instancia de src.db_connector.DBConnector creada con db_path: {_DEFAULT_DB_PATH_LC}")


class SQLMedicalChatbot(BaseTool):
    name: str = "SQLMedicalChatbot"
    description: str = (
        "Útil para responder preguntas o ejecutar sub-tareas que requieren consultar información médica específica de pacientes "
        "(alergias, medicamentos, diagnósticos, etc.) almacenada en una base de datos SQLite. "
        "Esta herramienta genera y ejecuta consultas SQL. Proporciona la pregunta completa y clara si buscas datos concretos. "
        "Es más efectiva cuando la pregunta es una sub-tarea bien definida de un plan mayor o una consulta directa que no requiere múltiples pasos lógicos complejos. "
        "NO uses esta herramienta para preguntas médicas generales (usa BioChatMedicalInquiry) ni para saludos (usa SinaSuiteAndGeneralInformation). "
        "Si la pregunta original del usuario es muy compleja, considera usar QueryPlannerTool primero para descomponerla."
    )
    
    db_connector: DBConnector
    logger: logging.Logger
    llm: BaseLanguageModel # Usar BaseLanguageModel o ChatOpenAI o Any

    _mcp_context: dict = PrivateAttr(default_factory=dict)  # Contexto MCP: entidades relevantes

    # Si la herramienta toma argumentos específicos, puedes definir un args_schema:
    # class SQLMedicalChatbotArgs(BaseModel):
    #     query: str = Field(description="La pregunta o consulta a procesar")
    # args_schema: Type[BaseModel] = SQLMedicalChatbotArgs

    # Pydantic v2 (usado por Langchain con BaseTool) se encargará de la inicialización
    # de los campos anotados si se pasan como argumentos de palabra clave al crear la instancia.
    # No se necesita un __init__ explícito a menos que haya lógica adicional.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # El contexto MCP ahora es un atributo privado
        # self._mcp_context ya se inicializa por PrivateAttr
        pass

    def _update_mcp_context(self, response_message: str, original_query: str = None, executed_sql_query: str = None, data_results: list = None):
        # Busca tablas y columnas mencionadas en la respuesta y actualiza el contexto
        tablas = re.findall(r"tabla[s]? ([A-Z0-9_]+)", response_message, re.IGNORECASE)
        if tablas:
            self._mcp_context['last_tables'] = tablas
            self.logger.info(f"Contexto MCP actualizado con last_tables: {tablas}")
        columnas = re.findall(r"columna[s]? ([A-Z0-9_]+)", response_message, re.IGNORECASE)
        if columnas:
            self._mcp_context['last_columns'] = columnas
            self.logger.info(f"Contexto MCP actualizado con last_columns: {columnas}")

        # Extraer PATI_ID
        found_patient_id = None
        # 1. De la query original del usuario
        if original_query:
            match_user_query = re.search(
                r"paciente\s+(?:con\s+ID\s+|ID\s+)?(\d+)|(?:patient|patient_id)\s*(?:=|is\s+)?\s*(\d+)|PATI_ID\s*=\s*(\d+)",
                original_query,
                re.IGNORECASE
            )
            if match_user_query:
                found_patient_id = next((g for g in match_user_query.groups() if g is not None), None)
                if found_patient_id:
                     self.logger.info(f"PATI_ID '{found_patient_id}' extraído de la query original del usuario: '{original_query}'")

        # 2. De la query SQL ejecutada (si no se encontró antes)
        if not found_patient_id and executed_sql_query:
            match_sql = re.search(r"PATI_ID\s*=\s*'?(\d+)'?", executed_sql_query, re.IGNORECASE)
            if match_sql:
                found_patient_id = match_sql.group(1)
                if found_patient_id:
                    self.logger.info(f"PATI_ID '{found_patient_id}' extraído de la query SQL ejecutada: '{executed_sql_query}'")
        
        if found_patient_id:
            self._mcp_context['last_patient_id'] = found_patient_id
            self.logger.info(f"Contexto MCP actualizado con last_patient_id: {found_patient_id}")
        # Si no se encuentra un nuevo ID, el anterior persiste, lo que es generalmente el comportamiento deseado
        # para preguntas de seguimiento que no re-especifican un ID.

    def _resolve_ambiguous_reference(self, query: str) -> tuple[str, str | None]:
        """
        Busca referencias ambiguas y resuelve usando el contexto.
        Devuelve un tuple: (tipo_resolucion, valor)
        tipo_resolucion puede ser:
            - "no_ambiguity": no se encontró ambigüedad o no se pudo resolver. La query original se usa.
            - "resolved_query": la query fue modificada. 'valor' es la nueva query.
            - "direct_response": se debe devolver una respuesta directamente. 'valor' es la respuesta.
        """
        # Resolver referencias a tablas
        if re.search(r"esa tabla|dicha tabla|la tabla mencionada", query, re.IGNORECASE):
            tablas = self._mcp_context.get('last_tables')
            if tablas:
                last_table = tablas[-1]
                self.logger.info(f"Referencia ambigua a 'esa tabla' resuelta a: {last_table}.")
                return "direct_response", f"La consulta se refiere a la tabla '{last_table}' (mencionada anteriormente). Para ver sus columnas, puedes usar la herramienta correspondiente para listar columnas de '{last_table}'."
            else:
                return "direct_response", "No se ha mencionado ninguna tabla previamente en la conversación."

        # Resolver referencias a columnas
        if re.search(r"esa columna|dicha columna|el campo mencionado", query, re.IGNORECASE):
            columnas = self._mcp_context.get('last_columns')
            if columnas:
                last_column = columnas[-1]
                self.logger.info(f"Referencia ambigua a 'esa columna' resuelta a: {last_column}")
                return "direct_response", f"Última columna mencionada: {last_column}"
            else:
                return "direct_response", "No se ha mencionado ninguna columna previamente en la conversación."

        # Resolver referencias a pacientes
        patient_ref_pattern = r"\b(este|ese|dicho|el|del|al|para el|para la)\s+paciente\b|\b(sus)\s+(pruebas|datos|informes|alergias|medicamentos|historia|diagnósticos|tratamientos)\b"
        if re.search(patient_ref_pattern, query, re.IGNORECASE):
            patient_id = self._mcp_context.get('last_patient_id')
            if patient_id:
                resolved_query = f"{query} (para el paciente con ID {patient_id})"
                # Casos especiales para queries muy cortas que son solo posesivos
                match_short_query = re.match(r"^\s*(sus)\s+(pruebas|datos|informes|alergias|medicamentos|historia|diagnósticos|tratamientos)\s*$", query, re.IGNORECASE)
                if match_short_query:
                    term = match_short_query.group(2) # El sustantivo (pruebas, datos, etc.)
                    resolved_query = f"{term} para el paciente con ID {patient_id}"

                self.logger.info(f"Referencia ambigua a paciente resuelta. Query original: '{query}'. Query modificada: '{resolved_query}'")
                return "resolved_query", resolved_query
            else:
                return "direct_response", "Se hizo referencia a 'este paciente' (o similar), pero no hay un paciente identificado en el contexto de la conversación. Por favor, especifica el ID del paciente."

        return "no_ambiguity", query

    def _run(self, query: str) -> str:
        """Procesa la consulta de manera segura. Este es el método que Langchain llama."""
        original_user_query = query 

        # --- INICIO: Expansión dinámica de diagnósticos ---
        # Si la consulta contiene palabras clave de diagnóstico, intentamos expandir variantes y sinónimos
        import re
        from src.flexible_search_config import extract_diagnosis_variants_from_hint, get_llm_generated_synonyms
        diagnosis_patterns = [
            r"diagn[oó]stico[s]? de ([\w\sáéíóúüñ\-]+)",
            r"con ([\w\sáéíóúüñ\-]+) ?\(?HTA\)?",  # ejemplo: con hipertensión o con HTA
            r"hipertensi[oó]n|hta|diabetes|epoc|asma|c[aá]ncer|neoplasia|tumor|insuficiencia renal|ictus|infarto|alzheimer|parkinson|artritis|lupus|fibromialgia|sida|vih|covid|enfermedad pulmonar|enfermedad cardiovascular|enfermedad renal|enfermedad hep[aá]tica|enfermedad autoinmune"
        ]
        diagnosis_term = None
        for pat in diagnosis_patterns:
            m = re.search(pat, query, re.IGNORECASE)
            if m:
                diagnosis_term = m.group(1) if m.lastindex else m.group(0)
                diagnosis_term = diagnosis_term.strip()
                break
        # Si se detecta término diagnóstico, expandimos variantes y sinónimos
        if diagnosis_term:
            try:
                # Obtener variantes desde hint (si ya existen en la query)
                variants = extract_diagnosis_variants_from_hint(query)
                if not variants:
                    # Si no hay hint, usar el término detectado y sus sinónimos
                    synonyms_dict = get_llm_generated_synonyms([diagnosis_term], llm_caller=self.llm)
                    variants = [diagnosis_term]
                    if synonyms_dict and diagnosis_term in synonyms_dict:
                        variants += [v for v in synonyms_dict[diagnosis_term] if v.lower() != diagnosis_term.lower()]
                # Añadir un hint a la query para el pipeline (como comentario SQL especial)
                variants_hint = f"/*DIAGNOSIS_VARIANTS:{'|'.join(variants)}*/ "
                query = variants_hint + query
                self.logger.info(f"Expansión dinámica de diagnóstico detectada: '{diagnosis_term}'. Variantes: {variants}")
            except Exception as e:
                self.logger.error(f"Error en expansión dinámica de diagnóstico para '{diagnosis_term}': {e}")
        # --- FIN: Expansión dinámica de diagnósticos ---

        resolution_type, resolved_value = self._resolve_ambiguous_reference(original_user_query)

        if resolution_type == "direct_response":
            self.logger.info(f"Resolución ambigua resultó en respuesta directa: {resolved_value}")
            return resolved_value
        elif resolution_type == "resolved_query":
            self.logger.info(f"Query original '{original_user_query}' resuelta a '{resolved_value}' por _resolve_ambiguous_reference.")
            query = resolved_value 
        
        # query ahora es la original o la resuelta. Proceder con la limpieza.
        self.logger.info(f"Query a procesar después de resolución de ambigüedades: '{query}'")
        
        if not hasattr(self.db_connector, 'get_db_structure_dict'):
            self.logger.error(f"CRITICAL: self.db_connector (tipo: {type(self.db_connector)}) en SQLMedicalChatbot NO tiene get_db_structure_dict. Esto causará un error en el pipeline.")
            return "Error de configuración interna: el conector de base de datos es incorrecto."
        else:
            self.logger.info(f"SQLMedicalChatbot._run usando db_connector de tipo: {type(self.db_connector)} que SÍ tiene get_db_structure_dict.")

        try:
            # --- INICIO DEBUG INFO EN RESPUESTA ---
            debug_info = []
            debug_info.append(f"DEBUG: Attempting to locate DB at: {_DEFAULT_DB_PATH_LC}")
            if os.path.exists(_DEFAULT_DB_PATH_LC):
                debug_info.append(f"DEBUG: Database file FOUND at {_DEFAULT_DB_PATH_LC}")
            else:
                debug_info.append(f"DEBUG: Database file NOT FOUND at {_DEFAULT_DB_PATH_LC}")
                db_dir = os.path.dirname(_DEFAULT_DB_PATH_LC)
                if os.path.exists(db_dir):
                    debug_info.append(f"DEBUG: Contents of {db_dir}: {os.listdir(db_dir)}")
                else:
                    debug_info.append(f"DEBUG: Directory {db_dir} does not exist.")
            debug_text = "\\n".join(debug_info)
            # --- FIN DEBUG INFO EN RESPUESTA ---\
            # original_query_for_log ahora se refiere a la query después de la posible resolución de ambigüedad
            original_query_for_log_after_resolution = query 
            stripped_input = query.strip()
            cleaned_query = stripped_input
            
            markdown_pattern = r"^```(?:sql)?\\s*(.*?)\\s*```$"
            match = re.search(markdown_pattern, stripped_input, re.DOTALL | re.IGNORECASE)
            
            if match:
                extracted_sql = match.group(1).strip()
                if original_query_for_log_after_resolution != extracted_sql:
                    self.logger.info(f"Markdown detectado y extraído. Query (post-resolución) antes: '{original_query_for_log_after_resolution}'")
                    self.logger.info(f"Query después de extraer de Markdown y strip(): '{extracted_sql}'")
                else:
                    self.logger.info(f"Markdown detectado. Query extraída (sin cambios internos, post-resolución): '{extracted_sql}'")
                cleaned_query = extracted_sql
            else:
                if original_query_for_log_after_resolution != stripped_input:
                    self.logger.info(f"No se detectó patrón de Markdown. Query (post-resolución) antes: '{original_query_for_log_after_resolution}'")
                    self.logger.info(f"Query después de strip() inicial (usada como está, post-resolución): '{cleaned_query}'")
                else:
                    self.logger.info(f"No se detectó patrón de Markdown. Query (post-resolución) procesada como está: '{cleaned_query}'")
            
            sql_incompat_warnings = []
            if re.search(r"DATEDIFF\s*\(", cleaned_query, re.IGNORECASE):
                sql_incompat_warnings.append("ADVERTENCIA: 'DATEDIFF' no es compatible con SQLite. Usa (julianday(fecha1) - julianday(fecha2)) para diferencias de fechas en días.")
            if re.search(r"CURRENT_DATE", cleaned_query, re.IGNORECASE):
                sql_incompat_warnings.append("ADVERTENCIA: 'CURRENT_DATE' no es compatible con SQLite. Usa date('now') o datetime('now') según el caso.")
            if re.search(r"GETDATE\s*\(", cleaned_query, re.IGNORECASE):
                sql_incompat_warnings.append("ADVERTENCIA: 'GETDATE()' no es compatible con SQLite. Usa date('now') o datetime('now').")
            
            # Verificar que self.llm está disponible
            if self.llm is None:
                self.logger.error("CRITICAL: La instancia de LLM (self.llm) no está disponible en SQLMedicalChatbot._run.")
                return "Error de configuración interna: el modelo de lenguaje no está disponible."
                
            self.logger.info(f"Procesando consulta: '{cleaned_query}' (con timeout de 90s)")
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # Pasar self.llm como tercer argumento para llm_param, y self.logger como sexto para logger_param
                # Se elimina self.schema_path y self.terms_dict_path de la llamada a chatbot_pipeline
                future = executor.submit(chatbot_pipeline, cleaned_query, self.db_connector, self.llm, self.logger)
                try:
                    pipeline_result = future.result(timeout=90)
                except concurrent.futures.TimeoutError:
                    self.logger.error("chatbot_pipeline excedió el tiempo máximo de 90s y fue cancelado")
                    return "Error: El procesamiento de la consulta tardó demasiado y fue cancelado. Por favor, intenta con una consulta más simple."
                except Exception as e_pipeline:
                    self.logger.error(f"Error inesperado ejecutando chatbot_pipeline: {e_pipeline}", exc_info=True)
                    return f"Error inesperado al procesar la consulta: {str(e_pipeline)}"

            self.logger.info(f"Pipeline ejecutado. Tipo de resultado: {type(pipeline_result)}")
            if pipeline_result is None:
                self.logger.error("Pipeline devolvió None")
                return "No pude obtener resultados para esta consulta."
            if not isinstance(pipeline_result, dict):
                self.logger.error(f"Pipeline devolvió un tipo inesperado: {type(pipeline_result)}")
                return f"Error: Resultado con formato inválido. Por favor reporta este error."
            
            response_message = pipeline_result.get("response_message", "La consulta se ejecutó pero no pude formatear la respuesta correctamente.")
            data_results = pipeline_result.get("data_results")
            # executed_sql_query_info ahora se obtiene de sql_query_generated y params_used
            # para simplificar, ya que el pipeline devuelve el SQL final y sus parámetros.
            executed_sql_query_info = {
                "final_executed_query": pipeline_result.get("sql_query_generated"),
                "params_used": pipeline_result.get("params_used")
            }

            self.logger.info(f"Pipeline returned: response_message='{response_message}', data_results_type='{type(data_results)}', data_results_empty={not data_results}, executed_sql_query_info_type='{type(executed_sql_query_info)}'")

            sql_to_check_for_count_raw = query # Default to original user query for this check
            if isinstance(executed_sql_query_info, dict):
                final_executed_query_from_info = executed_sql_query_info.get("final_executed_query")
                if isinstance(final_executed_query_from_info, str) and final_executed_query_from_info.strip():
                    sql_to_check_for_count_raw = final_executed_query_from_info
                    self.logger.info(f"Usando final_executed_query para la comprobación de COUNT: {sql_to_check_for_count_raw[:100]}...")
            elif isinstance(executed_sql_query_info, str) and executed_sql_query_info.strip():
                sql_to_check_for_count_raw = executed_sql_query_info
                self.logger.info(f"Usando executed_sql_query_info (string) para la comprobación de COUNT: {sql_to_check_for_count_raw[:100]}...")
            
            sql_to_check_for_count_upper = sql_to_check_for_count_raw.upper()
            is_count_query = False
            if re.search(r"COUNT\s*\(", sql_to_check_for_count_upper):
                is_count_query = True
                self.logger.info(f"Consulta identificada como posible consulta COUNT basada en la subcadena 'COUNT('. SQL: {sql_to_check_for_count_raw[:200]}...")

            if is_count_query:
                is_complex_count_query = False
                if "GROUP BY" in cleaned_query.upper():
                    is_complex_count_query = True
                    self.logger.info("La consulta original contiene 'GROUP BY', tratada como consulta de conteo compleja.")

                if data_results and len(data_results) == 1:
                    row = data_results[0]
                    if not is_complex_count_query: # CORREGIDO AQUÍ
                        if isinstance(row, (tuple, list)) and len(row) > 1:
                            is_complex_count_query = True
                            self.logger.info("La fila de resultados tiene múltiples columnas, tratada como consulta de conteo compleja.")
                        elif isinstance(row, dict) and len(row.keys()) > 1:
                            is_complex_count_query = True
                            self.logger.info("La fila de resultados (dict) tiene múltiples claves, tratada como consulta de conteo compleja.")
                    
                    actual_count = None
                    if isinstance(row, (tuple, list)) and len(row) >= 1:
                        actual_count = row[0]
                    elif isinstance(row, dict):
                        if len(row.keys()) == 1:
                            actual_count = list(row.values())[0]
                        else:
                            for key_in_row in row.keys():
                                if isinstance(key_in_row, str) and key_in_row.upper().startswith("COUNT"):
                                    actual_count = row[key_in_row]
                                    break
                            if actual_count is None and row.values():
                                actual_count = list(row.values())[0]
                    elif hasattr(row, '__getitem__') and hasattr(row, '__len__') and len(row) >= 1 and not isinstance(row, str):
                        try:
                            actual_count = row[0]
                        except (KeyError, TypeError, IndexError):
                            try: 
                                if hasattr(row, 'keys') and callable(row.keys) and len(list(row.keys())) > 0:
                                    first_key = list(row.keys())[0]
                                    actual_count = row[first_key]
                            except Exception as e_row_access:
                                self.logger.warning(f"No se pudo acceder al elemento de la fila para la consulta COUNT usando métodos estándar: {e_row_access}")
                    
                    if actual_count is not None and isinstance(actual_count, (int, float)):
                        if not is_complex_count_query:
                            response_message = f"El resultado del conteo es: {actual_count}."
                            self.logger.info(f"Resultado de la consulta COUNT (simple) procesado. Conteo real: {actual_count}. Mensaje de respuesta establecido en: '{response_message}'")
                            data_results = None
                        else:
                            self.logger.info(f"Consulta con COUNT() detectada pero es compleja. Conteo extraído: {actual_count}. Se mantendrán los datos y el mensaje originales.")
                    elif actual_count is not None:
                        self.logger.warning(f"La consulta COUNT devolvió un valor, pero no es un número: {actual_count} (tipo: {type(actual_count)}). Mensaje original: '{response_message}'")
                    else:
                        self.logger.warning(f"La consulta COUNT devolvió una fila, pero el formato es inesperado o el valor no se pudo extraer: {row}. Mensaje original: '{response_message}'")
                elif data_results is None or not data_results:
                     self.logger.warning(f"La consulta COUNT no devolvió datos o data_results está vacío. Mensaje original: '{response_message}'")
                else:
                    self.logger.warning(f"La consulta COUNT devolvió {len(data_results)} filas, se esperaba 1. Datos: {str(data_results)[:200]}... Mensaje original: '{response_message}'")

            if response_message and "La consulta contiene columnas o tablas que no existen" in response_message:
                tablas_mencionadas = re.findall(r"FROM\\s+([A-Z0-9_]+)|JOIN\\s+([A-Z0-9_]+)", cleaned_query, re.IGNORECASE)
                tablas_planas = [t for pair in tablas_mencionadas for t in pair if t]
                # Se actualiza la sugerencia para usar DatabaseSchemaTool
                sugerencia = ("\\nSugerencia: Antes de volver a intentar, usa la herramienta 'DatabaseSchemaTool' "
                             "para verificar las tablas y columnas. Ejemplo: 'DatabaseSchemaTool: describe table NOMBRE_TABLA'.\\n"
                             "Tablas detectadas en tu consulta: " + ", ".join(set(tablas_planas))) if tablas_planas else (
                             "\\nSugerencia: Usa la herramienta 'DatabaseSchemaTool' para explorar el esquema (ej. 'DatabaseSchemaTool: list tables').")
                response_message += sugerencia

                if (re.search(r"NOT IN|NOT EXISTS|LEFT JOIN", cleaned_query, re.IGNORECASE) and
                    re.search(r"\\b(VACC_VACCINES|PATI_PATIENTS|_DATE|_ID)\\b", cleaned_query, re.IGNORECASE) and
                    (re.search(r"'now'", cleaned_query) or re.search(r"\\d{4}", cleaned_query))):
                    response_message += (
                        "\\nADVERTENCIA: Cuando utilices subconsultas (NOT IN, NOT EXISTS) o LEFT JOIN para filtrar entidades..." # Mensaje abreviado
                    )
                    self.logger.warning("Advertencia genérica para subconsultas/fechas activada.")

                is_complex_cte = False
                if "WITH " in cleaned_query.upper():
                    is_complex_cte = re.search(r"\bWITH\s+\w+\s+AS\s*\(", cleaned_query, re.IGNORECASE) is not None
                is_other_complex = False
                if any(kwd in cleaned_query.upper() for kwd in ["OVER", "PARTITION", "CASE"]):
                    is_other_complex = re.search(r"\bOVER\s*\(|PARTITION\s+BY|CASE\s+WHEN.*THEN.*END", cleaned_query, re.IGNORECASE | re.DOTALL) is not None
                
                if is_complex_cte:
                    response_message += "\\n\\nADVERTENCIA CRÍTICA SOBRE CTEs..." # Mensaje abreviado
                    self.logger.warning(f"Consulta compleja con WITH detectada... Query: {cleaned_query[:300]}...")
                elif is_other_complex:
                    response_message += "\\n\\nADVERTENCIA ADICIONAL DE COMPLEJIDAD..." # Mensaje abreviado
                    self.logger.warning(f"Consulta compleja (no CTE) detectada... Query: {cleaned_query[:300]}...")

                # --- INICIO SECCIÓN COMENTADA TEMPORALMENTE ---
                # La siguiente sección dependía de `search_schema` que no está definida y `self.schema_path` que ha sido eliminado.
                # Se elimina esta sección completa.
                # --- FIN SECCIÓN COMENTADA TEMPORALMENTE ---

            self.logger.info(f"Resultado del pipeline procesado: Mensaje='{response_message}', Datos presentes: {data_results is not None}")

            formatted_data = ""
            MAX_ROWS_FOR_AGENT_OBSERVATION = 10
            if data_results:
                if isinstance(data_results, list):
                    num_rows = len(data_results)
                    if num_rows > 0:
                        try:
                            data_to_format = data_results
                            if num_rows > MAX_ROWS_FOR_AGENT_OBSERVATION:
                                self.logger.info(f"Truncando datos para la observación del agente de {num_rows} a {MAX_ROWS_FOR_AGENT_OBSERVATION} filas.")
                                data_to_format = data_results[:MAX_ROWS_FOR_AGENT_OBSERVATION]
                            formatted_data = json.dumps(data_to_format, indent=2, ensure_ascii=False)
                        except Exception as e_format:
                            self.logger.error(f"Error al formatear los datos a JSON: {e_format}")
                            formatted_data = str(data_results[:MAX_ROWS_FOR_AGENT_OBSERVATION])
                elif not isinstance(data_results, list):
                     formatted_data = str(data_results)

            final_response_str = response_message
            if sql_incompat_warnings:
                if not all(warn in final_response_str for warn in sql_incompat_warnings):
                     final_response_str = "\\n".join(sql_incompat_warnings) + "\\n" + final_response_str

            if formatted_data:
                if isinstance(data_results, list) and len(data_results) > MAX_ROWS_FOR_AGENT_OBSERVATION:
                    final_response_str += f"\\\\n(Mostrando primeros {MAX_ROWS_FOR_AGENT_OBSERVATION} registros en esta observación detallada)"
                final_response_str += f"\\nDatos:\\n{formatted_data}"
            else:
                is_entity_validation_error = "La consulta contiene columnas o tablas que no existen" in response_message
                if not is_entity_validation_error:
                    if "no devolvió filas" in response_message or "no se encontraron resultados" in response_message:
                        final_response_str += "\\nSugerencia: Si no obtuviste resultados, intenta usar términos más generales..." # Mensaje abreviado
                    elif response_message:
                        final_response_str += "\\nNo se encontraron datos para esta consulta."
            
            if not isinstance(final_response_str, (str, int, float)):
                self.logger.warning(f"La respuesta combinada no es un tipo básico serializable: {type(final_response_str)}")
                try:
                    final_response_str = str(final_response_str)
                except Exception as e_str_conv:
                    self.logger.error(f"Error al convertir final_response_str a string: {e_str_conv}")
                    return "Error al procesar la respuesta final."
            
            # MAX_OBSERVATION_LENGTH = 10000
            # if len(final_response_str) > MAX_OBSERVATION_LENGTH:
            #     self.logger.warning(f"La observación final excede los {MAX_OBSERVATION_LENGTH} caracteres ({len(final_response_str)}). Truncando...")
            #     final_response_str = final_response_str[:MAX_OBSERVATION_LENGTH] + "... (Observación truncada)"

            # Actualizar contexto MCP al final, antes de devolver la respuesta.
            # Usar original_user_query (la que entró a _run antes de cualquier modificación)
            # y la SQL final ejecutada.
            final_executed_sql = None
            if isinstance(executed_sql_query_info, dict):
                final_executed_sql = executed_sql_query_info.get("final_executed_query")
            elif isinstance(executed_sql_query_info, str):
                final_executed_sql = executed_sql_query_info

            self._update_mcp_context(
                response_message=response_message,
                original_query=original_user_query, 
                executed_sql_query=final_executed_sql,
                data_results=data_results
            )
            
            # El bloque que manejaba "esa tabla" aquí ha sido movido a _resolve_ambiguous_reference
            # y ya no es necesario aquí.
            
            return debug_text + "\n\n" + str(final_response_str)
        except Exception as e_outer:
            self.logger.error(f"Error general en _run de SQLMedicalChatbot: {e_outer}", exc_info=True)
            # Devuelve el error SQL exacto para que el agente lo procese y active el STEP 3
            return f"Error: {str(e_outer)}"

# Importar la función dinámica de expansión de diagnósticos
# from flexible_search_config import build_diagnosis_conditions_dynamic



sql_medical_chatbot_tool = SQLMedicalChatbot(
    db_connector=db_connector_instance,
    logger=logger,
    llm=llm_instance # Pasar la instancia de LLM creada
)

# --- NUEVA HERRAMIENTA: QueryPlannerTool ---
class QueryPlannerTool(BaseTool):
    name: str = "QueryPlannerTool"
    description: str = (
        "Analiza una pregunta compleja del usuario y, si es necesario, la descompone en un plan de múltiples pasos o sub-preguntas. "
        "Debe usarse como primer paso si la pregunta parece requerir información de múltiples fuentes, múltiples consultas a la base de datos, "
        "o un razonamiento secuencial que no puede ser manejado por una sola llamada a SQLMedicalChatbot o BioChatMedicalInquiry. "
        "La salida es un plan estructurado en formato JSON (o una indicación de que la pregunta es simple)."
    )
    llm: BaseLanguageModel
    logger: logging.Logger

    def _run(self, user_question: str) -> str:
        self.logger.info(f"[QueryPlannerTool] Recibida pregunta para planificación: {user_question}")

        planning_prompt_template = """
Dada la siguiente pregunta de un usuario:
</user_question>

Tu tarea es analizar esta pregunta y determinar si es compleja y requiere un plan de múltiples pasos.
Una pregunta es compleja si requiere combinar información, múltiples operaciones lógicas, o inferencias.

**PRINCIPIOS CLAVE PARA LA PLANIFICACIÓN:**

1.  **ESQUEMA Y ENTIDADES:**
    *   Usa SIEMPRE los nombres exactos de tablas y columnas del esquema de la base de datos. NO INVENTES nombres.
    *   Si un paso implica SQL, `SQLMedicalChatbot` necesitará los nombres correctos.

2.  **MAPEO DE CONCEPTOS A CÓDIGOS (ESPECIALMENTE ALERGIAS):**
    *   Si la pregunta involucra conceptos médicos que se mapean a códigos o categorías:
        a.  **Paso 1 (Identificar Código/Categoría):** Tu plan DEBE PRIMERO proponer un paso para obtener el código numérico (ej. `ALCA_ID`, `NMAL_ID`, `CDTE_ID`) usando `SQLMedicalChatbot` para consultar la tabla de diccionario/categoría apropiada.
        b.  **Paso 2 (Usar Código en Datos del Paciente):** Los pasos siguientes deben usar el código obtenido en el Paso 1 para filtrar en las tablas de datos del paciente (ej. `PATI_PATIENT_ALLERGIES`, `EPIS_DIAGNOSTICS`).

    *   **GUÍA ESPECÍFICA PARA ALERGIAS:**
        *   **Tabla de Categorías Principal:** `ALLE_ALLERGY_CATEGORIES` (columnas `ALCA_ID`, `ALCA_DESCRIPTION_ES`).
            *   `ALCA_DESCRIPTION_ES = 'Medicamentosa'` (corresponde a `ALCA_ID = 1`)
            *   `ALCA_DESCRIPTION_ES = 'No medicamentosa'` (corresponde a `ALCA_ID = 2`)
        *   **Tabla de Alérgenos No Medicamentosos Específicos:** `ALLE_NOT_MEDICINAL_ALLERGENS` (columnas `NMAL_ID`, `NMAL_DESCRIPTION_ES`). Esta tabla detalla los alérgenos que caen bajo `ALCA_ID = 2` ('No medicamentosa').

        *   **Si la pregunta es sobre "alergias alimentarias" o "alergias no alimentarias" (en general, como categoría):**
            1.  **Paso 1 (Obtener ALCA_ID):** La sub-pregunta para `SQLMedicalChatbot` debe ser:
                `"Obtén el ALCA_ID de ALLE_ALLERGY_CATEGORIES donde ALCA_DESCRIPTION_ES = 'No medicamentosa'."`
                (Esto te dará `ALCA_ID = 2`. Las alergias alimentarias son un tipo de alergia 'No medicamentosa').
            2.  **Paso 2 (Filtrar pacientes):** Usa el `ALCA_ID` obtenido para filtrar en `PATI_PATIENT_ALLERGIES`. Ejemplo: `WHERE ALCA_ID = {{output_paso_1}}`.

        *   **Si la pregunta es sobre "alergias medicamentosas":**
            1.  **Paso 1 (Obtener ALCA_ID):** Sub-pregunta:
                `"Obtén el ALCA_ID de ALLE_ALLERGY_CATEGORIES donde ALCA_DESCRIPTION_ES = 'Medicamentosa'."` (Esto dará `ALCA_ID = 1`).
            2.  **Paso 2 (Filtrar pacientes):** Usa el `ALCA_ID` obtenido en `PATI_PATIENT_ALLERGIES`.

        *   **Si la pregunta es sobre un ALÉRGENO ESPECÍFICO NO MEDICAMENTOSO (ej. "alergia al polen", "alergia al cacahuete"):**
            1.  **Paso 1 (Obtener ALCA_ID para 'No medicamentosa'):** Sub-pregunta:
                `"Obtén el ALCA_ID de ALLE_ALLERGY_CATEGORIES donde ALCA_DESCRIPTION_ES = 'No medicamentosa'."` (Obtendrás `ALCA_ID = 2`).
            2.  **Paso 2 (Obtener NMAL_ID del alérgeno específico):** Sub-pregunta:
                `"Obtén el NMAL_ID de ALLE_NOT_MEDICINAL_ALLERGENS donde NMAL_DESCRIPTION_ES LIKE '%termino_del_alergeno_especifico%'."` (Ej: `'%polen%'`, `'%cacahuete%'`).
            3.  **Paso 3 (Filtrar pacientes con datos estructurados):** Usa AMBOS, el `ALCA_ID` del Paso 1 Y los `NMAL_ID` del Paso 2, para filtrar en `PATI_PATIENT_ALLERGIES`. Ejemplo: `WHERE ALCA_ID = {{output_paso_1}} AND NMAL_ID IN ({{output_paso_2}})`.
            4.  **Paso 4 (Fallback a Texto Libre SI el Paso 2 o 3 no devuelven resultados concluyentes):** Si el Paso 2 no encontró un `NMAL_ID` o el Paso 3 no encontró pacientes, planifica un paso adicional para buscar el `termino_del_alergeno_especifico` en los campos de texto libre de `PATI_PATIENT_ALLERGIES` (ej. `PALL_OBSERVATIONS`, `PALL_ALLERGY_OBSERVATION`). Describe este paso claramente.
            *   Ejemplo de sub-pregunta para este fallback: `"Busca pacientes con 'polen' en las observaciones de alergias: SELECT PATI_ID, PALL_OBSERVATIONS FROM PATI_PATIENT_ALLERGIES WHERE LOWER(PALL_OBSERVATIONS) LIKE '%polen%' OR LOWER(PALL_ALLERGY_OBSERVATION) LIKE '%polen%'."`

    *   **Para OTROS CONCEPTOS (ej. diagnósticos como 'diabetes tipo 2' mapeado a `CDTE_ID`):**
        1.  **Paso 1 (Identificar Código):** La sub-pregunta a `SQLMedicalChatbot` debe consultar la tabla de diccionario relevante (ej. `DIAG_DIAGNOSES` para `CDTE_ID`) usando `LIKE` en la columna descriptiva para encontrar el código. Ejemplo: `"Obtén el CDTE_ID de DIAG_DIAGNOSES donde DIAG_DESCRIPTION LIKE '%diabetes tipo 2%'."`
        2.  **Paso 2 (Filtrar pacientes con código):** Usa el código obtenido en la tabla de datos del paciente (ej. `EPIS_DIAGNOSTICS`).
        3.  **Paso 3 (Fallback a Texto Libre SI el Paso 1 o 2 no devuelven resultados concluyentes):** Si el Paso 1 no encontró un código o el Paso 2 no encontró pacientes, planifica un paso adicional para buscar el término original en los campos de texto libre relevantes (ej. `DIAG_OTHER_DIAGNOSTIC` en `EPIS_DIAGNOSTICS`). Describe este paso claramente.

3.  **BÚSQUEDA EN TEXTO LIBRE COMO COMPLEMENTO:**
    *   Si buscas un término médico específico (por ejemplo, un alérgeno, diagnóstico, medicamento, etc.) y el plan inicial para buscarlo en la tabla estructurada correspondiente podría no encontrarlo (o si la búsqueda estructurada inicial falla o devuelve 0 resultados):
        a.  **Planifica un paso adicional o alternativo** para buscar ese término en los campos de texto libre relevantes de las tablas de pacientes (ej. `PALL_OBSERVATIONS`, `DIAG_OTHER_DIAGNOSTIC`, `MEDI_DOSAGE_OBSERVATIONS`, etc.), utilizando una consulta con `LIKE '%término%'`.
        b.  El objetivo es complementar la búsqueda estructurada para no omitir registros relevantes por falta de codificación.
    *   **Ejemplo de sub-pregunta para SQLMedicalChatbot para texto libre (si la búsqueda estructurada de 'polen' falló):**
        `"Busca pacientes con 'polen' en las observaciones de alergias: SELECT PATI_ID FROM PATI_PATIENT_ALLERGIES WHERE LOWER(PALL_OBSERVATIONS) LIKE '%polen%' OR LOWER(PALL_ALLERGY_OBSERVATION) LIKE '%polen%'."`

4.  **DERIVACIÓN DE INFO Y MULTI-CRITERIO:**
    *   Si se necesita información no directa (ej. 'edad' de `PATI_BIRTH_DATE`) o hay múltiples filtros, el plan debe incluir pasos o consideraciones para ello, instruyendo a `SQLMedicalChatbot` para que use funciones SQL (ej. `strftime`, `julianday`, `CASE WHEN`).
    *   Asegúrate de que el plan aborde TODOS los criterios de la pregunta original.

5.  **USO DE HERRAMIENTAS Y AMBIGÜEDAD:**
    *   Usa `BioChatMedicalInquiry` SOLO para investigación médica general o conocimiento que NO está en la base de datos.
    *   Si la pregunta es ambigua, el plan debe indicar que se necesita aclaración (ver formato de salida).

**FORMATO DE SALIDA (JSON):**

Si la pregunta es simple (puede ser respondida directamente por `SQLMedicalChatbot`, `BioChatMedicalInquiry`, o `SinaSuiteAndGeneralInformation`):
{{"plan_required": false, "reasoning": "La pregunta es simple y puede ser manejada directamente.", "suggested_tool": "SQLMedicalChatbot_o_BioChatMedicalInquiry_o_SinaSuiteAndGeneralInformation", "original_question": "{user_question}"}}

Si la pregunta es compleja, genera un plan JSON con una lista "plan". Cada paso debe tener: "step_number", "step_description", "tool_to_use", "inputs" (diccionario, usualmente {{"query": "sub-pregunta"}}, puede referenciar salidas previas con {{{{step_N_output}}}}).

Si es ambigua e irresoluble:
{{"plan_required": false, "reasoning": "Pregunta ambigua. Aclaración necesaria: [qué falta].", "suggested_tool": "SinaSuiteAndGeneralInformation", "original_question": "{user_question}"}}

Si hay un error interno al planificar:
{{"error": "No se pudo generar plan.", "reasoning": "Error interno o pregunta incomprensible."}}
"""
        
        formatted_prompt = planning_prompt_template.format(user_question=user_question)
        
        try:
            self.logger.info(f"[QueryPlannerTool] Enviando prompt simplificado al LLM para planificación.")
            llm_response = self.llm.invoke(formatted_prompt)
            
            if hasattr(llm_response, 'content'):
                plan_json_str = llm_response.content
            elif isinstance(llm_response, str):
                plan_json_str = llm_response
            else:
                self.logger.error(f"[QueryPlannerTool] Respuesta del LLM inesperada: {type(llm_response)}. Contenido: {str(llm_response)}")
                return json.dumps({"error": "Respuesta inesperada del LLM durante la planificación.", "details": str(llm_response)})

            self.logger.info(f"[QueryPlannerTool] Respuesta JSON cruda del LLM (prompt simplificado): {plan_json_str}")

            match = re.search(r"\{.*\}", plan_json_str, re.DOTALL)
            if match:
                cleaned_json_str = match.group(0)
                try:
                    json.loads(cleaned_json_str) # Validar JSON
                    self.logger.info(f"[QueryPlannerTool] Plan (prompt simplificado) generado y parseado exitosamente.")
                    return cleaned_json_str 
                except json.JSONDecodeError as e:
                    self.logger.error(f"[QueryPlannerTool] Error al decodificar JSON del plan (prompt simplificado): {e}. JSON intentado: {cleaned_json_str}")
                    return json.dumps({"error": f"Error al decodificar JSON del plan: {e}", "cleaned_json_response": cleaned_json_str, "original_llm_response": plan_json_str})
            else:
                self.logger.warning(f"[QueryPlannerTool] No se pudo extraer un JSON válido de la respuesta del LLM (prompt simplificado). Respuesta: {plan_json_str}")
                return json.dumps({"error": "No se pudo extraer JSON de la respuesta del LLM.", "llm_response": plan_json_str})

        except Exception as e:
            self.logger.error(f"[QueryPlannerTool] Excepción durante la planificación con LLM (prompt simplificado): {e}", exc_info=True)
            return json.dumps({"error": f"Excepción durante la planificación: {str(e)}"})

    async def _arun(self, user_question: str) -> str:
        return self._run(user_question)

query_planner_tool = QueryPlannerTool(
    llm=llm_instance, # Asegurarse de que usa la instancia LLM correcta
    logger=logger     # Asegurarse de que usa el logger correcto
)

# --- FILTRO para preguntas de entidad biomédica (lista de principios activos, grupos, etc.) ---
def is_biomedical_entity_question(question: str) -> bool:
    """
    Solo retorna True para preguntas atómicas de definición o identificación directa,
    NO para listados, agrupaciones o relaciones complejas.
    """
    patterns = [
        r"^¿?qué es\b",
        r"^define\b",
        r"^¿?para qué sirve\b",
        r"^nombre del principio activo\b",
        r"^sinónimos de\b",
        r"^denominación común internacional\b",
        r"^grupo terap(é|e)utico\b",
        r"^explica\b",
        r"^describe\b",
        r"^información sobre\b",
        r"^efectos secundarios de\b",
        r"^indicaciones de\b",
        r"^contraindicaciones de\b",
        r"^mecanismo de acción de\b",
    ]
    import re
    for pat in patterns:
        if re.match(pat, question.strip().lower()):
            return True
    return False

# --- LLM directo para preguntas de entidad biomédica ---
def direct_llm_biomedical_entity_answer(question: str) -> str:
    """Responde usando solo el LLM principal, sin llamar a PubMed ni pipeline."""
    if llm_instance is None:
        return "Error: LLM no disponible para respuesta directa."
    try:
        # Prompt directo, sin contexto adicional
        response = llm_instance.invoke(question)
        if hasattr(response, 'content'):
            return response.content.strip()
        elif isinstance(response, str):
            return response.strip()
        else:
            return str(response)
    except Exception as e:
        logger.error(f"Error en respuesta directa LLM entidad biomédica: {e}", exc_info=True)
        return f"Error al procesar la respuesta directa: {e}"

# --- Herramienta BioChat para consultas médicas generales ---
def invoke_biochat_pipeline(question: str) -> str:
    """
    Invoca el pipeline de BioChat para responder preguntas médicas generales o de investigación.
    Si la pregunta es de entidad biomédica (lista de principios activos, etc.), responde SOLO con el LLM.
    Si la pregunta es de investigación biomédica, usa SOLO el pipeline reducido (solo metadatos PubMed).
    """
    if is_biomedical_entity_question(question):
        logger.info(f"Pregunta detectada como entidad biomédica: '{question}'. Respondiendo solo con LLM.")
        return direct_llm_biomedical_entity_answer(question)
    # Si la pregunta es de investigación biomédica, usar solo el pipeline reducido
    if biochat_reduced_pipeline is not None:
        logger.info(f"Pregunta biomédica compleja: '{question}'. Usando pipeline reducido solo metadatos PubMed.")
        try:
            pipeline_outputs = biochat_reduced_pipeline({"objective": question})
            raw_pubmed_results = pipeline_outputs.get("raw_pubmed_results", "No se pudo obtener resultados de PubMed.")
            return raw_pubmed_results
        except Exception as e:
            logger.error(f"Error en pipeline reducido para pregunta '{question}': {e}", exc_info=True)
            return f"Se produjo un error al procesar tu consulta con el sistema BioChat (solo metadatos PubMed): {str(e)}. Por favor, intenta reformular tu pregunta o inténtalo más tarde."
    # Fallback: pipeline completo si el reducido no está disponible
    if full_pipeline is None:
        logger.error("full_pipeline no está disponible. No se puede procesar la pregunta con BioChat.")
        return "Error: El componente BioChat para consultas médicas generales no está disponible en este momento."
    logger.info(f"Invocando BioChat pipeline completo con la pregunta: '{question}'")
    try:
        pipeline_outputs = full_pipeline({"objective": question})
        final_report_text = pipeline_outputs.get("final_report", "No se pudo generar un informe final desde BioChat.")
        final_report_text += "\n\nEste informe resume la información encontrada para su consulta. Si necesita más detalles o una búsqueda diferente, por favor especifíquela."
        logger.info("Informe final de BioChat generado exitosamente.")
        return final_report_text
    except Exception as e:
        logger.error(f"Error durante la ejecución del BioChat pipeline para la pregunta '{question}': {e}", exc_info=True)
        return f"Se produjo un error al procesar tu consulta con el sistema BioChat: {str(e)}. Por favor, intenta reformular tu pregunta o inténtalo más tarde."

biochat_medical_tool = Tool(
    name="BioChatMedicalInquiry",
    func=invoke_biochat_pipeline,
    description=(
        "Responde a preguntas médicas generales, biomédicas o de investigación que NO requieren consultar directamente la base de datos de pacientes con SQL. "
        "Útil para obtener explicaciones detalladas, resúmenes de investigaciones, información sobre enfermedades, tratamientos, o cuando la pregunta es compleja y requiere un análisis por múltiples agentes especializados. "
        "Ejemplos: '¿Cuáles son los últimos avances en el tratamiento del cáncer de pulmón?', 'Explícame la CRISPR-Cas9', 'Busca información sobre la metformina y sus efectos secundarios'. "
        "NO uses esta herramienta si la pregunta puede responderse con una consulta SQL a la base de datos de pacientes (para eso, usa SQLMedicalChatbot)."
    )
)

# --- Herramienta para información general y SinaSuite ---
def fetch_sinasuite_info(question: str) -> str:
    """
    Busca información sobre SinaSuite, la función del chatbot o responde a saludos y preguntas generales.
    Utiliza esta herramienta para preguntas que no parezcan estar relacionadas con la extracción de datos
    específicos de la base de datos médica.
    """
    question_lower = question.lower()
    
    saludos_keywords = ["hola", "buenos días", "buenas tardes", "buenas noches", "qué tal", "hey"]
    sinasuite_keywords = [
        "sinasuite", "qué es sinasuite", "que es sinasuite",
        "cuál es tu función", "cual es tu funcion", "qué haces", "que haces",
        "quién eres", "quien eres", "para qué sirves", "para que sirves",
        "ayuda", "info", "informacion"
    ]

    if any(saludo in question_lower for saludo in saludos_keywords):
        return "¡Hola! Soy un asistente virtual. Puedo ayudarte con consultas sobre la base de datos médica o proporcionarte información general sobre SinaSuite. ¿En qué puedo ayudarte hoy?"

    if any(keyword in question_lower for keyword in sinasuite_keywords):
        return "Soy un asistente virtual con dos funciones principales: 1) Ayudarte a consultar información específica de la base de datos médica. 2) Proporcionarte información general sobre SinaSuite, que es una plataforma integral para la gestión de datos médicos. Para más detalles sobre SinaSuite, puedes visitar https://www.sinasuite.com/."

    # Si el agente eligió esta herramienta pero no es un saludo ni sobre SinaSuite,
    # podría ser un error del agente o una pregunta muy general.
    return "Puedo ayudarte a consultar la base de datos médica o darte información sobre SinaSuite. ¿Tienes alguna pregunta específica sobre estos temas?"

sinasuite_tool = Tool(
    name="SinaSuiteAndGeneralInformation",  # Nombre sin espacios
    func=fetch_sinasuite_info,
    description="""Útil para responder a saludos, preguntas generales sobre la función de este chatbot, o consultas sobre 'SinaSuite'.
No uses esta herramienta para consultas que requieran acceder a datos médicos (ni de la base de datos ni información médica general).
Ejemplos de cuándo usarla: 'Hola', '¿Qué es SinaSuite?', '¿Quién eres?', 'Ayuda'."""
)

# --- TOOLS DE ESQUEMA (OBSOLETO - REEMPLAZADO POR DatabaseSchemaTool) ---
# from sql_utils import list_tables, list_columns, search_schema # Ya no se usan directamente aquí
# SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "data", "schema_simple.json") # Ya no se usa aquí

# Las funciones tool_list_tables, tool_list_columns, tool_search_schema, tool_list_all_columns
# y la lista TOOLS_SCHEMA han sido eliminadas ya que su funcionalidad
# ahora está cubierta por DatabaseSchemaTool.

def custom_handle_parsing_errors(error: OutputParserException) -> str:
    """
    Genera un mensaje de error personalizado y prescriptivo cuando el LLM no sigue el formato ReAct.
    Intenta extraer la salida problemática del LLM para incluirla en el mensaje de corrección.
    """
    response_str = str(error) # Mensaje completo de la excepción

    # Intentar extraer la salida real del LLM que causó el problema
    problematic_output = getattr(error, 'llm_output', None)
    if problematic_output is None:
        # Intentar parsear desde el string de la excepción
        prefixes_to_check = [
            "Parsing LLM output produced both a final answer and a parse-able action:: ",
            "Could not parse LLM output: ",
            "Invalid Format: ",
            "Invalid tool `",
        ]
        parsed_from_str = False
        for prefix in prefixes_to_check:
            if response_str.startswith(prefix):
                if prefix == "Invalid tool `":
                    end_of_tool_name = response_str.find("`", len(prefix))
                    if end_of_tool_name != -1:
                        problematic_output = response_str[len(prefix):end_of_tool_name]
                        parsed_from_str = True
                        break
                else:
                    problematic_output = response_str[len(prefix):]
                    parsed_from_str = True
                    break
        if not parsed_from_str:
            problematic_output = response_str
    else:
        problematic_output = str(problematic_output)

    # Loguear la salida problemática específica que se enviará al LLM para corrección
    logging.error(f"Salida problemática del LLM (para corrección):\\n---\\n{problematic_output}\\n---")

    # Mensaje prescriptivo para el LLM
    # (El resto de la función que construye el mensaje de error para el LLM permanece igual)
    # ...
    # Asegurarse de que el logger de root también capture la salida problemática real que se envía al LLM
    # (el logger actual en el manejador de errores del agente ya lo hace si error.llm_output está poblado)

    # El mensaje que se devuelve al LLM para que lo corrija:
    # (Este es el formato que ya tenías y es bueno, solo nos aseguramos que `problematic_output` sea más preciso)
    # (El código original para construir el mensaje prescriptivo sigue aquí)
    # ... (resto del código de la función) ...
    # Por ejemplo:
    error_message_template = (
        "CRITICAL ERROR: Your response was not in the correct ReAct format. "
        "You MUST respond with either a valid 'Action:' line followed by an 'Action Input:' line, "
        "OR a 'Final Answer:' line. "
        "DO NOT provide explanations or conversational text outside of the 'Thought:' field. "
        "The available tools are: {tool_names}. "
        "Ensure your Action is one of these tools if you are using an action. "
        "Your problematic output was:\\n'''{problematic_llm_output}'''\\n"
        "Correct your response to strictly follow the ReAct format (Thought, Action, Action Input, or Final Answer)."
    )
    # Obtener nombres de herramientas (asumiendo que están disponibles en algún contexto o globalmente)
    # Esto es solo un ejemplo, necesitarías acceso a `self.tools` o similar si esto está en una clase.
    # Si es una función global, los nombres de las herramientas tendrían que pasarse o ser accesibles.
    # Por ahora, lo omito para mantener el cambio enfocado en la extracción de `problematic_output`.
    
    # tool_names_str = ", ".join([tool.name for tool in self.tools]) if hasattr(self, 'tools') else "Not available here"
    
    # Para este ejemplo, usaré un placeholder para tool_names
    tool_names_str = "[SQLMedicalChatbot, SinaSuiteAndGeneralInformation]" # Placeholder

    # Re-loguear la salida problemática que se usará en el prompt de corrección
    # logging.error(f"Salida problemática del LLM (para corrección del LLM):\\n---\\n{problematic_output}\\n---")
    # Este logging ya se hizo arriba.

    return error_message_template.format(
        tool_names=tool_names_str,
        problematic_llm_output=problematic_output
    )

def get_langchain_agent():
    """
    Inicializa y devuelve un agente LangChain con las herramientas configuradas.
    """
    try:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            base_url=LLM_API_BASE_URL,
            temperature=0.0,
        )
    except Exception as e:
        logger.error("No se pudo inicializar el LLM: %s", e)
        return None

    # Instanciar herramientas con el LLM recién creado
    sql_medical_chatbot_tool = SQLMedicalChatbot(
        db_connector=db_connector_instance,
        logger=logger,
        llm=llm
    )
    query_planner_tool = QueryPlannerTool(
        llm=llm,
        logger=logger
    )
    # Las siguientes herramientas deben estar definidas en el módulo, si no, defínelas antes de esta función
    # biochat_medical_tool y sinasuite_tool ya deberían estar disponibles

    all_tools = [query_planner_tool, sql_medical_chatbot_tool, biochat_medical_tool, sinasuite_tool]

    # Construir el prompt de agente (AGENT_PREFIX) y tool_descriptions_str
    tool_descriptions_str = "\n".join([f"- {tool.name}: {getattr(tool, 'description', '')}" for tool in all_tools])
    tool_names_str = ", ".join([tool.name for tool in all_tools])

    AGENT_PREFIX = f"""Answer the following questions as best you can. Strictly follow the ReAct format.

**ReAct Format (MANDATORY for EVERY response):**
Thought: [Your reasoning. Include plan status if active (e.g., "Plan active, next step X"). Brief and direct.]
Action: [Tool name. One of: {tool_names_str}]
Action Input: [Input for the tool.]
Observation: [Result of the action.]
... (Thought/Action/Action Input/Observation can repeat)
Thought: [Final reasoning.]
Final Answer: [Direct final answer.]

**SQL Query Rules (for SQLMedicalChatbot):**
1.  **Strict Schema Adherence**: ALWAYS use exact table/column names from the provided schema. Verify with `DatabaseSchemaTool` if unsure (e.g., `DatabaseSchemaTool: list tables` or `DatabaseSchemaTool: describe table TABLE_NAME`). NEVER invent names.
2.  **Critical Alerts**:
    *   `PRES_PRESCRIPTIONS` table DOES NOT EXIST. For prescriptions, carefully examine the schema for alternatives (e.g., `PATI_USUAL_MEDICATION`).
    *   Patient gender is NOT in `PATI_GENDER`. Use `PATI_PATIENTS.PATY_ID` and JOIN with the gender types table (likely `PARA_GENDERS`).
3.  **Common Tables (Examples)**:
    *   Diagnoses: `EPIS_DIAGNOSTICS`, `HIST_HISTORY_DIAG_PROC` (Codes: `CDTE_ID`; Text: `HIDI_OTHER_DIAGNOSTIC`, `DIAG_OTHER_DIAGNOSTIC`).
    *   Medications: `MEDI_MEDICATIONS` (Use `MEDI_ID`, `MEDI_DESCRIPTION_ES`).
    *   Usual Medication: `PATI_USUAL_MEDICATION`.

**Internal State Variables (Managed in Thoughts):**
- `active_plan`: (Optional) Full JSON plan from QueryPlannerTool if one is active.
- `current_step_index`: (Optional) 0-based index of the *next* step to execute from `active_plan`.

**Agent Workflow:**

**STEP 1: Initial Analysis & Plan Continuation**
Thought:
    1. Evaluate user input.
    2. If `active_plan` exists and `current_step_index` is valid:
        If user input is a simple continuation (e.g., "yes", "continue") for `active_plan[current_step_index]`:
            Proceed with `active_plan[current_step_index]`. (Go to Action directly, skip QueryPlannerTool).
        Else (new question or plan ended): Discard `active_plan` and `current_step_index`. Treat as new question.
    3. Else (no active plan): Treat as new question.
    4. If new question: "Need to analyze with QueryPlannerTool."
Action: [If continuing plan: `active_plan[current_step_index].tool_to_use`. If new question: `QueryPlannerTool`]
Action Input: [If continuing plan: `active_plan[current_step_index].inputs`. If new question: Original user question.]

**STEP 2: Process QueryPlannerTool Response (If called in STEP 1)**
(Observation will contain QueryPlannerTool JSON)
Thought: Received QueryPlannerTool output.
    - If `"plan_required": true` (Rule A):
        `active_plan` = plan from JSON. `current_step_index` = 0.
        "My plan is: `active_plan`. Executing first step (`active_plan[0]`): '[description]' using '[tool]'.",
    - If `"plan_required": false` and no error (Rule B):
        Discard any prior `active_plan`/`current_step_index`.
        "Question is simple. Suggested tool: '[suggested_tool from JSON]'. Using it with original question."
    - If JSON has "error" (Rule C):
        Discard any prior `active_plan`/`current_step_index`.
        "QueryPlannerTool failed: [error message from JSON]. Informing user."
Action: [Tool from Rule A or B. If Rule C, go to Final Answer.]
Action Input: [Input for the tool, per Rule A or B.]

**STEP 2.1: Process Observation from an Active Plan Step**
Thought:
    Completed step `active_plan[current_step_index -1]` (or current if STEP 1 continued). Observation: `[Result of action]`.
    Increment `current_step_index`.
    If `current_step_index < len(active_plan)` (more steps remain):
        "Plan has more steps. Next is `active_plan[current_step_index]`: '[description]' using '[tool]'. Ready to proceed if user confirms or input is simple continuation."
        (Agent will wait for next user input, which will be processed by STEP 1. If plan dictates auto-continuation, state it and proceed to Action for the next step.)
    Else (last step completed or plan interrupted):
        "All plan steps completed or plan ended. Formulating final answer."
        Discard `active_plan`, `current_step_index`. (Go to Final Answer).

**STEP 3: Handle SQLMedicalChatbot Errors (e.g., `no such table`)**
(Observation contains SQLMedicalChatbot error)
Thought: SQLMedicalChatbot failed. Error: `[error message]`. Original failed query: `[SQL query]`.
    The error indicates a schema issue. I must verify the schema.
    If error is about `PRES_PRESCRIPTIONS`, I remember NOT to use it and will look for alternatives.
Action: DatabaseSchemaTool
Action Input: [`"list tables"` or `"describe table SUSPECTED_TABLE_NAME"`]
---
(After DatabaseSchemaTool Observation)
Thought: Schema obtained: `[DatabaseSchemaTool Observation]`.
    Analyzing schema and original error.
    If `PRES_PRESCRIPTIONS` was the issue, I now have the real table list to find an alternative.
    Retrying SQLMedicalChatbot with a corrected query, strictly adhering to the actual schema.
    (If part of `active_plan`, plan and index remain, but sub-query for this step is corrected).
Action: SQLMedicalChatbot
Action Input: "Original failed query: `[failed query]`. Original error: `[SQL error]`. Relevant schema: `[schema info]`. CORRECT the SQL query to use ONLY VALID tables/columns from the schema and retry."

**Available Tools:**
{tool_descriptions_str}

**Key Reminders & Finalization:**
-   Always include `active_plan` and `current_step_index` in `Thought` if a plan is active.
-   `Final Answer` is ONLY the direct answer. Be concise.
-   **CRITICAL FINALIZATION**: If an action's `Observation` (e.g., from SQLMedicalChatbot) already contains the answer to the user's question (including requested data), YOU MUST end your turn with `Final Answer: [response]`. DO NOT iterate further or re-run queries if you have the data.
"""

    # Crear el agente con LangChain
    agent = initialize_agent(
        tools=all_tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=ConversationBufferMemory(memory_key="chat_history"),
        handle_parsing_errors=custom_handle_parsing_errors,
        agent_kwargs={"prefix": AGENT_PREFIX}
    )
    return agent
