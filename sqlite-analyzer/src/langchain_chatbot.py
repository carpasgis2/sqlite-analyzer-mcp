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
# from pydantic import PrivateAttr  # Añadir esta importación para atributos privados

from langchain.tools import Tool # Descomentado
from langchain_core.tools import BaseTool # Usar BaseTool
from pydantic.v1 import BaseModel, Field # MODIFICADO: Para args_schema si es necesario

from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI # Asegurarse que esta es la importación
from langchain_core.language_models.base import BaseLanguageModel # Para tipado más genérico si se prefiere

from langchain_core.exceptions import OutputParserException
import sqlite3 # Nueva importación
import openai
from openai import RateLimitError

# Gestionar la importación de LLM_MODEL_NAME
try:
    from llm_utils import LLM_MODEL_NAME
except ImportError:
    # El logger puede no estar completamente configurado aquí si esto está en la parte superior.
    # Se podría registrar una advertencia más tarde o usar print.
    print("Advertencia: llm_utils.py no encontrado o LLM_MODEL_NAME no definido. Usando un valor por defecto para LLM_MODEL_NAME.")
    LLM_MODEL_NAME = "deepseek-coder" # Valor por defecto

# Importación para la nueva herramienta BioChat
# Asegura que la raíz del proyecto está en sys.path para importar biochat.py
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
try:
    from biochat import full_pipeline
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

# Cambiar imports relativos a absolutos para ejecución directa
try:
    from pipeline import chatbot_pipeline_entrypoint as chatbot_pipeline
    from sql_utils import list_tables, list_columns, search_schema
    from db_connector import DBConnector
except ImportError:
    # Fallback: intentar imports relativos si falla el absoluto (por compatibilidad)
    from .pipeline import chatbot_pipeline_entrypoint as chatbot_pipeline
    from .sql_utils import list_tables, list_columns, search_schema
    from .db_connector import DBConnector

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

# Configuración de la API de OpenAI
# Elimina cualquier clave hardcodeada para evitar confusión y usa solo la variable de entorno
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("\033[91m[ADVERTENCIA] No se ha definido la variable de entorno OPENAI_API_KEY.\033[0m")
else:
    print("[DEBUG] Clave API usada (primeros 8):", OPENAI_API_KEY[:8], "... (oculta por seguridad)")

LLM_MODEL_NAME = "gpt-3.5-turbo"
LLM_API_BASE_URL = "https://api.openai.com/v1"
LLM_API_KEY = OPENAI_API_KEY
LLM_PROVIDER = "openai"

# Crear la instancia de LLM que se usará en las herramientas
# Esta instancia se crea aquí porque sql_medical_chatbot_tool y query_planner_tool se instancian en este mismo archivo.
try:
    llm_instance = ChatOpenAI(
        model_name=LLM_MODEL_NAME,
        openai_api_base=LLM_API_BASE_URL,
        openai_api_key=LLM_API_KEY,
        temperature=0.0,
    )
    logger.info(f"Instancia de ChatOpenAI ({LLM_MODEL_NAME}) creada exitosamente para herramientas.")
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
        from flexible_search_config import extract_diagnosis_variants_from_hint, get_llm_generated_synonyms
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
You are a robust medical query planner. Route each user question to the correct tool:

- If the question mentions guidelines, PubMed, reviews, literature, articles, studies, or biomedical research, ALWAYS use the BioChatMedicalInquiry tool.
- If the question is about patient data, hospitalizations, diagnoses, procedures, medications, or anything that can be answered with SQL from the patient database, use the SQLMedicalChatbot tool.
- NEVER invent tables, columns, or relationships. If the question cannot be answered with the available schema, respond: 'This cannot be answered with SQL. Please use BioChatMedicalInquiry for biomedical or literature questions.'
- When showing tables or schema, ONLY mention the relevant table(s) by name (maximum 3), NEVER print the full schema or all tables.
- If the question is a greeting or about the assistant, use ChatMedGeneralInfo.
- IMPORTANT: Before stating that information is not available, always check for any related columns or tables that could provide a partial or indirect answer. Never say data is missing if any related field exists in the schema.

Return a JSON object with:
- plan_required: true/false
- reasoning: brief justification
- suggested_tool: tool name (if simple)
- plan: (if complex) list of steps, each with step_number, step_description, tool_to_use, and inputs
- original_question: the user question
- If ambiguous or not answerable, set plan_required: false and reasoning explaining why.
"""
        formatted_prompt = planning_prompt_template + f"\nUser question: {user_question}\n"
        
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

# --- Herramienta para información general y ChatMed ---
def fetch_general_info(question: str) -> str:
    """
    Responde a saludos, preguntas generales sobre el chatbot o su función, o dudas generales sobre el sistema.
    Este asistente es un buscador médico avanzado (ChatMed), capaz de responder consultas clínicas, buscar artículos científicos (PubMed) y ayudarte con información médica relevante.
    """
    question_lower = question.lower()
    
    saludos_keywords = ["hola", "buenos días", "buenas tardes", "buenas noches", "qué tal", "hey"]
    general_keywords = [
        "quién eres", "quien eres", "qué eres", "que eres",
        "qué haces", "que haces", "cuál es tu función", "cual es tu funcion",
        "ayuda", "info", "informacion", "buscador", "chatbot", "chatmed"
    ]

    if any(saludo in question_lower for saludo in saludos_keywords):
        return "¡Hola! Soy ChatMed, tu asistente médico virtual. Puedo ayudarte a buscar información clínica, responder dudas médicas y encontrar artículos científicos. ¿En qué puedo ayudarte hoy?"

    if any(keyword in question_lower for keyword in general_keywords):
        return ("Soy ChatMed, un chatbot médico avanzado. Puedo responder preguntas clínicas, ayudarte a consultar bases de datos médicas, y buscar artículos científicos en PubMed. "
                "No soy un sustituto de un profesional sanitario, pero puedo orientarte y proporcionarte información relevante y actualizada.")

    # Si el agente eligió esta herramienta pero no es un saludo ni una pregunta general,
    # podría ser un error del agente o una pregunta muy general.
    return "Soy ChatMed, tu asistente médico. ¿Tienes alguna consulta clínica o necesitas buscar información científica?"

chatmed_tool = Tool(
    name="ChatMedGeneralInfo",
    func=fetch_general_info,
    description="""Responde a saludos, preguntas generales sobre el chatbot médico, su función, o dudas generales sobre el sistema. Ejemplos: 'Hola', '¿Quién eres?', '¿Qué puedes hacer?', 'Ayuda', '¿Qué es ChatMed?'."""
)

# --- TOOLS DE ESQUEMA (OBSOLETO - REEMPLAZADO POR DatabaseSchemaTool) ---
# from sql_utils import list_tables, list_columns, search_schema # Ya no se usan directamente aquí
# SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "data", "schema_simple.json") # Ya no se usa aquí

# Mover la creación de all_tools y AGENT_PREFIX al ámbito global
# Asegúrate de que todas las herramientas individuales (query_planner_tool, sql_medical_chatbot_tool, etc.)
# ya han sido instanciadas globalmente ANTES de este punto.

all_tools = [query_planner_tool, sql_medical_chatbot_tool, biochat_medical_tool, chatmed_tool]

tool_descriptions_str = "\n".join([f"- {tool.name}: {getattr(tool, 'description', '')}" for tool in all_tools])
tool_names_str = ", ".join([tool.name for tool in all_tools])

# --- UTILIDAD PARA TRUNCAR OBSERVACIONES LARGAS DE ESQUEMA ---
def truncate_schema_observation(observation: str, max_lines: int = 10) -> str:
    """
    Trunca la observación de esquema/tablas si es demasiado larga para evitar flood en la terminal.
    """
    lines = observation.splitlines()
    if len(lines) > max_lines:
        return '\n'.join(lines[:max_lines]) + f"\n... (truncado, {len(lines)-max_lines} líneas más) ..."
    return observation

# --- REFORZAR EL PROMPT DEL AGENTE PARA DISCIPLINA DE SQL ---
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
2.  **Do NOT invent entities, tables, columns, or relationships.** If the user's question does not correspond to any table/column in the schema, respond that it is not possible to answer with SQL and suggest using another tool if appropriate.
3.  **Stay on topic**: Only generate SQL relevant to the user's question. Do NOT introduce unrelated concepts or entities (e.g., do not mention 'polen' if the user asks about hospitalizations).
4.  **Critical Alerts**:
    *   `PRES_PRESCRIPTIONS` table DOES NOT EXIST. For prescriptions, carefully examine the schema for alternatives (e.g., `PATI_USUAL_MEDICATION`).
    *   Patient gender is NOT in `PATI_GENDER`. Use `PATI_PATIENTS.PATY_ID` and JOIN with the gender types table (likely `PARA_GENDERS`).
5.  **Common Tables (Examples)**:
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

# Crear el agente con LangChain (Este es el agente global)
# Definir custom_handle_parsing_errors ANTES de usarla en initialize_agent
def custom_handle_parsing_errors(error: OutputParserException) -> str:
    """Maneja errores de parseo de la salida del agente."""
    logger.error(f"Error de parseo en la salida del agente: {error}", exc_info=True)
    # Devolver el texto problemático o un mensaje de error genérico
    # Podrías intentar extraer 'error.llm_output' o 'error.observation' si están disponibles
    # y son relevantes para que el agente reintente.
    # Por ahora, un mensaje genérico que incluye el error.
    return f"Error al procesar la respuesta del LLM. Por favor, reformula tu pregunta o intenta de nuevo. Detalle del error: {str(error)}"

agent = initialize_agent(
    tools=all_tools,
    llm=llm_instance, # Asegúrate que llm_instance es el LLM global
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=ConversationBufferMemory(memory_key="chat_history"),
    handle_parsing_errors=custom_handle_parsing_errors, # Asegúrate que custom_handle_parsing_errors está definido globalmente o es accesible
    agent_kwargs={"prefix": AGENT_PREFIX}
)

# La función get_langchain_agent() podría ser eliminada o refactorizada si ya no es necesaria,
# ya que ahora tenemos un agente global. Por ahora, la dejamos.
def get_langchain_agent():
    """
    Devuelve la instancia global del agente LangChain.
    """
    global agent # Asegurarse de que estamos usando el agente global
    if agent is None:
        logger.error("Se intentó obtener el agente antes de su inicialización.")
        # Podrías inicializarlo aquí como fallback si es necesario,
        # pero idealmente ya debería estar inicializado.
        # Por ahora, simplemente devolvemos None o lanzamos un error.
        raise RuntimeError("El agente LangChain no ha sido inicializado.")
    return agent

if __name__ == "__main__":
    logger.info("Iniciando chatbot en modo interactivo...")
    # Verificar si biochat.py está disponible y mostrar advertencia si no
    if full_pipeline is None:
        print("\\033[93m⚠️ Advertencia: biochat.py no encontrado o full_pipeline no pudo ser importado.\\033[0m")
        print("\\033[93m   Las funcionalidades de BioChat (investigación médica general) no estarán disponibles.\\033[0m")
        print("\\033[93m   Para habilitarlas, asegúrate de que biochat.py está en la raíz del proyecto (\\033[1m%s\\033[0m\\033[93m).\\033[0m" % _PROJECT_ROOT)
        print("-" * 50)

    print("\\033[92m🤖 Asistente Médico Virtual (SinaMCP - LangChain) listo.\\033[0m")
    print("Escribe tu consulta o usa ':salir' para terminar, ':limpiar' para reiniciar la conversación.")
    
    current_agent = get_langchain_agent() # Obtener el agente global

    while True:
        try:
            user_input = input("\033[94m👤 Tú: \033[0m")
            if user_input.lower() == ':salir':
                print("\033[92m🤖 ¡Hasta luego!\033[0m")
                break
            if user_input.lower() == ':limpiar':
                current_agent.memory.clear()
                print("\033[92m✨ Conversación reiniciada.\033[0m")
                continue

            if not user_input.strip():
                continue

            print("\033[92m🤖 Agente: \033[0m", end="", flush=True)
            max_retries = 5
            wait_time = 3
            for attempt in range(max_retries):
                try:
                    response = current_agent.invoke({"input": user_input})
                    if isinstance(response, dict) and "output" in response:
                        output = response["output"]
                        # Si la respuesta contiene un bloque de esquema/tablas muy largo, trúncalo
                        if "DatabaseSchemaTool Observation" in output or "list tables" in output or "describe table" in output:
                            output = truncate_schema_observation(output, max_lines=10)
                        print(output)
                    elif isinstance(response, str):
                        print(response)
                    else:
                        logger.error(f"Respuesta inesperada del agente: {response}")
                        print("Hubo un problema al obtener la respuesta del agente.")
                    break  # Salir del bucle de reintentos si fue exitoso
                except RateLimitError as rle:
                    logger.warning(f"Rate limit alcanzado (429): {rle}. Reintentando en {wait_time} segundos...")
                    print(f"\033[93m[AVISO] Límite de uso alcanzado. Esperando {wait_time} segundos para reintentar...\033[0m")
                    time.sleep(wait_time)
                    wait_time *= 2  # Exponencial
                except Exception as e:
                    logger.error(f"Error inesperado en el bucle interactivo: {e}", exc_info=True)
                    print(f"\033[91mOcurrió un error inesperado: {e}\033[0m")
                    break
            else:
                print("\033[91mNo se pudo completar la consulta tras varios intentos por límite de uso. Intenta de nuevo en unos minutos.\033[0m")

        except KeyboardInterrupt:
            print("\n\033[92m🤖 ¡Hasta luego! (Interrupción de teclado)\033[0m")
            break
        except Exception as e:
            logger.error(f"Error inesperado fuera del bucle principal: {e}", exc_info=True)
            print(f"\033[91mOcurrió un error inesperado: {e}\033[0m")



