"""
LangChain Tool para usar el pipeline de chatbot existente.
Permite usar el pipeline como herramienta en un agente conversacional LangChain.
"""
import logging # Asegurar que logging se importa al principio
import os
import sys
import re
import json # Añadir importación de json
# import difflib # No parece usarse directamente, se puede quitar si no es necesario
import concurrent.futures
from typing import Type, Any # Añadido Any

from langchain.tools import Tool # Descomentado
from langchain_core.tools import BaseTool # Usar BaseTool
from pydantic.v1 import BaseModel, Field # MODIFICADO: Para args_schema si es necesario

from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI # Asegurarse que esta es la importación
from langchain_core.language_models.base import BaseLanguageModel # Para tipado más genérico si se prefiere

from langchain_core.exceptions import OutputParserException

# Gestionar la importación de LLM_MODEL_NAME
try:
    from src.llm_utils import LLM_MODEL_NAME # MODIFICADO: Añadido prefijo src.
except ImportError:
    # El logger puede no estar completamente configurado aquí si esto está en la parte superior.
    # Se podría registrar una advertencia más tarde o usar print.
    print("Advertencia: llm_utils.py no encontrado o LLM_MODEL_NAME no definido. Usando un valor por defecto para LLM_MODEL_NAME.")
    LLM_MODEL_NAME = "deepseek-coder" # Valor por defecto


# sys.path.append(os.path.dirname(os.path.abspath(__file__))) # ELIMINADO: Esta línea se elimina
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import chatbot_pipeline_entrypoint as chatbot_pipeline # MODIFICADO: Importar con el nombre correcto y alias si es necesario
from src.db_connector import DBConnector # MODIFICADO: Añadido prefijo src.

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

# Crear la instancia de LLM que se usará en la herramienta
# Esta instancia se crea aquí porque sql_medical_chatbot_tool se instancia en este mismo archivo.
try:
    llm_instance = ChatOpenAI(
        model_name=LLM_MODEL,
        openai_api_base=LLM_API_BASE_URL,
        openai_api_key=LLM_API_KEY,
        temperature=0.0,  # Ajustar según sea necesario
        # max_tokens=2000 # Ajustar según sea necesario
    )
    logger.info(f"Instancia de ChatOpenAI ({LLM_MODEL}) creada exitosamente para SQLMedicalChatbot.")
except Exception as e:
    logger.error(f"Error al crear la instancia de ChatOpenAI para SQLMedicalChatbot: {e}", exc_info=True)
    llm_instance = None # Fallback a None si la creación falla

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
    db_path=_DEFAULT_DB_PATH_LC,
    schema_path=_DEFAULT_SCHEMA_PATH_LC
)
logger.info(f"Instancia de src.db_connector.DBConnector creada con db_path: {_DEFAULT_DB_PATH_LC} y schema_path: {_DEFAULT_SCHEMA_PATH_LC}")


class SQLMedicalChatbot(BaseTool):
    name: str = "SQLMedicalChatbot"
    description: str = (
        "Útil para responder preguntas sobre información médica de pacientes, alergias, medicamentos, diagnósticos, etc., "
        "contenida en una base de datos SQLite. La herramienta intentará primero generar y ejecutar una consulta SQL. "
        "Si la pregunta es ambigua, podría pedir clarificación. Si la generación SQL falla o no produce resultados, "
        "intentará una búsqueda semántica (vectorial) sobre metadatos y descripciones de la base de datos como fallback. "
        "Proporciona la pregunta completa y clara. La herramienta puede manejar preguntas complejas que requieran joins o subconsultas. "
        "NO uses esta herramienta para preguntas generales no relacionadas con la base de datos médica."
    )
    
    db_connector: DBConnector
    logger: logging.Logger
    terms_dict_path: str
    schema_path: str
    llm: BaseLanguageModel # Usar BaseLanguageModel o ChatOpenAI o Any

    # Si la herramienta toma argumentos específicos, puedes definir un args_schema:
    # class SQLMedicalChatbotArgs(BaseModel):
    #     query: str = Field(description="La pregunta o consulta a procesar")
    # args_schema: Type[BaseModel] = SQLMedicalChatbotArgs

    # Pydantic v2 (usado por Langchain con BaseTool) se encargará de la inicialización
    # de los campos anotados si se pasan como argumentos de palabra clave al crear la instancia.
    # No se necesita un __init__ explícito a menos que haya lógica adicional.

    def _run(self, query: str) -> str:
        """Procesa la consulta de manera segura. Este es el método que Langchain llama."""
        # La lógica de la anterior función/método safe_process_results va aquí.
        # Asegúrate de usar self.logger, self.db_connector, self.terms_dict_path, self.schema_path
        
        # Verificar que self.db_connector es del tipo correcto (src.db_connector.DBConnector)
        # Esto es más para depuración, se puede quitar después.
        if not hasattr(self.db_connector, 'get_db_structure_dict'):
            self.logger.error(f"CRITICAL: self.db_connector (tipo: {type(self.db_connector)}) en SQLMedicalChatbot NO tiene get_db_structure_dict. Esto causará un error en el pipeline.")
            return "Error de configuración interna: el conector de base de datos es incorrecto."
        else:
            self.logger.info(f"SQLMedicalChatbot._run usando db_connector de tipo: {type(self.db_connector)} que SÍ tiene get_db_structure_dict.")

        try:
            original_query_for_log = query 
            stripped_input = query.strip()
            cleaned_query = stripped_input
            
            markdown_pattern = r"^```(?:sql)?\\s*(.*?)\\s*```$"
            match = re.search(markdown_pattern, stripped_input, re.DOTALL | re.IGNORECASE)
            
            if match:
                extracted_sql = match.group(1).strip()
                if original_query_for_log != extracted_sql:
                    self.logger.info(f"Markdown detectado y extraído. Query original: '{original_query_for_log}'")
                    self.logger.info(f"Query después de extraer de Markdown y strip(): '{extracted_sql}'")
                else:
                    self.logger.info(f"Markdown detectado. Query extraída (sin cambios internos): '{extracted_sql}'")
                cleaned_query = extracted_sql
            else:
                if original_query_for_log != stripped_input:
                    self.logger.info(f"No se detectó patrón de Markdown. Query original: '{original_query_for_log}'")
                    self.logger.info(f"Query después de strip() inicial (usada como está): '{cleaned_query}'")
                else:
                    self.logger.info(f"No se detectó patrón de Markdown. Query procesada como está (sin cambios por strip inicial): '{cleaned_query}'")
            
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
                future = executor.submit(chatbot_pipeline, cleaned_query, self.db_connector, self.llm, self.terms_dict_path, self.schema_path, self.logger)
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
                sugerencia = ("\\nSugerencia: Antes de volver a intentar, usa la herramienta 'Listar columnas de una tabla' "
                             "para cada tabla implicada. Ejemplo: 'Listar columnas de una tabla: NOMBRE_TABLA'.\\n"
                             "Tablas detectadas en tu consulta: " + ", ".join(set(tablas_planas))) if tablas_planas else (
                             "\\nSugerencia: Usa la herramienta de exploración de esquema para ver las columnas reales.")
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
                # La siguiente sección depende de `search_schema` que no está definida.
                # sql_keywords = {'select','from','where','join','on','and','or','distinct','inner','left','right','case','when','then','else','end','as','group','by','order','like','count', 'limit', 'offset', 'having', 'is', 'null', 'not', 'in', 'exists', 'between', 'avg', 'sum', 'min', 'max', 'abs', 'upper', 'lower', 'length', 'date', 'datetime', 'julianday', 'strftime', 'char', 'cast', 'round'}
                # # Usar una expresión regular que capture palabras que puedan ser identificadores SQL (incluyendo '_')
                # palabras = [w for w in re.findall(r"\\b[a-zA-Z_][a-zA-Z0-9_]*\\b", cleaned_query) if w.lower() not in sql_keywords and len(w) >= 3]
                # sugerencias_esquema = []
                # # Tomar hasta 3 palabras clave únicas y relevantes (no demasiado cortas)
                # palabras_unicas_filtradas = sorted(list(set(palabras)), key=len, reverse=True) # Priorizar más largas

                # for kw in palabras_unicas_filtradas[:3]:
                #     self.logger.info(f"Buscando en esquema la palabra clave: {kw} (derivada de la consulta: '{cleaned_query}')")
                #     # La función search_schema se importa de sql_utils
                #     result = search_schema(self.schema_path, kw)
                #     if result.get('tables') or result.get('columns'):
                #         sug_parts = []
                #         if result.get('tables'):
                #             sug_parts.append(f"tablas: {', '.join(result['tables'])}")
                #         if result.get('columns'):
                #             sug_parts.append(f"columnas: {', '.join(result['columns'])}")
                #         if sug_parts:
                #             sugerencias_esquema.append(f"- Para el término '{kw}', se encontró en: {'; '.join(sug_parts)}.")
                
                # if sugerencias_esquema:
                #     response_message += "\\n\\nSugerencias adicionales basadas en tu consulta y el esquema:\\n" + "\\n".join(sugerencias_esquema)
                #     self.logger.info(f"Añadidas sugerencias de esquema al mensaje de respuesta: {' '.join(sugerencias_esquema)}")
                # Fin de la sección de búsqueda en esquema

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
            
            MAX_OBSERVATION_LENGTH = 10000
            if len(final_response_str) > MAX_OBSERVATION_LENGTH:
                self.logger.warning(f"La observación final excede los {MAX_OBSERVATION_LENGTH} caracteres ({len(final_response_str)}). Truncando...")
                final_response_str = final_response_str[:MAX_OBSERVATION_LENGTH] + "... (Observación truncada)"

            return final_response_str
        except Exception as e_outer:
            self.logger.error(f"Error general en _run de SQLMedicalChatbot: {e_outer}", exc_info=True)
            return f"Error crítico al procesar la consulta: {str(e_outer)}"

sql_medical_chatbot_tool = SQLMedicalChatbot(
    db_connector=db_connector_instance,
    logger=logger,
    terms_dict_path=os.path.join(os.path.dirname(__file__), "data", "dictionary.json"),
    schema_path=os.path.join(os.path.dirname(__file__), "data", "schema_simple.json"), # O schema_enhanced.json
    llm=llm_instance # Pasar la instancia de LLM creada
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
    description="""Útil para responder a saludos, preguntas generales sobre la función del chatbot, o consultas sobre SinaSuite.
No uses esta herramienta para consultas que requieran acceder o buscar datos en la base de datos médica.
Ejemplos de cuándo usarla: 'Hola', '¿Qué es SinaSuite?', '¿Quién eres?', 'Ayuda'."""
)

# --- TOOLS DE ESQUEMA ---
from src.sql_utils import list_tables, list_columns, search_schema
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "data", "schema_simple.json")  # Usar el esquema real para las herramientas

def tool_list_tables(_:str=None) -> str:
    """Devuelve la lista de tablas reales del esquema. NO inventar nombres: usar siempre esta herramienta antes de generar SQL."""
    tablas = list_tables(SCHEMA_PATH)
    return "Tablas disponibles:\n" + "\n".join(tablas)

def tool_list_columns(table:str) -> str:
    """Devuelve las columnas reales de una tabla. NO inventar columnas: usar siempre esta herramienta antes de generar SQL."""
    cols = list_columns(SCHEMA_PATH, table)
    # Log temporal para depuración
    logger.info(f"[DEBUG] tool_list_columns: tabla solicitada='{table}', columnas encontradas={cols}")
    if cols:
        return f"Columnas de {table}:\n" + "\n".join(cols)
    # Si no hay columnas, comprobar si la tabla existe realmente
    tablas = list_tables(SCHEMA_PATH)
    tablas_lower = [t.lower() for t in tablas]
    if table.lower() in tablas_lower:
        return (f"La tabla '{table}' existe pero no se encontraron columnas en el esquema. "
                f"Puede ser un problema de formato o de nombres. Por favor, revisa el esquema o pide ayuda.\n"
                f"ADVERTENCIA: Si tu consulta SQL falla, revisa que los nombres de columnas y tablas sean exactos y que las funciones SQL sean compatibles con SQLite. "
                f"Por ejemplo, usa 'julianday' en vez de 'DATEDIFF', y 'date('now')' en vez de 'CURRENT_DATE'.")
    # Sugerir nombres similares si la tabla no existe
    import difflib
    sugerencias = difflib.get_close_matches(table, tablas, n=3, cutoff=0.6)
    if sugerencias:
        return f"No se encontraron columnas para la tabla '{table}'. ¿Quizás quisiste decir: {', '.join(sugerencias)}?"
    return (f"No se encontraron columnas para la tabla '{table}'. "
            f"ADVERTENCIA: Si tu consulta SQL falla, revisa que los nombres de columnas y tablas sean exactos y que las funciones SQL sean compatibles con SQLite. "
            f"Por ejemplo, usa 'julianday' en vez de 'DATEDIFF', y 'date('now')' en vez de 'CURRENT_DATE'.")

def tool_search_schema(keyword:str) -> str:
    """Busca tablas y columnas por palabra clave en el esquema."""
    result = search_schema(SCHEMA_PATH, keyword)
    out = []
    if result.get("tables"):
        out.append("Tablas que coinciden con la palabra clave:" + ", ".join(result["tables"]))
    if result.get("columns"):
        column_details = [f"{col['table']}.{col['column']}" for col in result["columns"]]
        out.append("Columnas que coinciden con la palabra clave (tabla.columna):" + ", ".join(column_details))
    
    if not out:
        return f"No se encontraron tablas ni columnas que coincidan con la palabra clave '{keyword}' en el esquema."
    return "\\n".join(out)

def tool_list_all_columns(_:str=None) -> str:
    """Devuelve todas las columnas de todas las tablas del esquema."""
    tablas = list_tables(SCHEMA_PATH)
    out = []
    if not tablas:
        return "No se encontraron tablas en el esquema."
    for t in tablas:
        cols = list_columns(SCHEMA_PATH, t)
        if cols:
            out.append(f"Tabla: {t}\\n  Columnas: {', '.join(cols)}")
        else:
            out.append(f"Tabla: {t}\\n  Columnas: (No se encontraron columnas o la tabla está vacía en el esquema)")
    return "\\n".join(out)

# Registrar tools en LangChain si se usa agente
TOOLS_SCHEMA = [
    Tool(
        name="Listar tablas del esquema",
        func=tool_list_tables,
        description="Devuelve la lista de tablas reales del esquema de la base de datos médica. Útil para evitar inventar tablas.\n**INSTRUCCIÓN:** Antes de generar SQL, consulta aquí las tablas reales."
    ),
    Tool(
        name="Listar columnas de una tabla",
        func=tool_list_columns,
        description="Devuelve la lista de columnas reales de una tabla. Argumento: nombre de la tabla.\n**INSTRUCCIÓN:** SIEMPRE usa esta herramienta antes de generar SQL para evitar inventar columnas."
    ),
    Tool(
        name="Buscar en el esquema por palabra clave",
        func=tool_search_schema,
        description="Busca tablas y columnas que contengan una palabra clave en su nombre o descripción. Argumento: palabra clave.\n**INSTRUCCIÓN:** Úsala para explorar el esquema antes de escribir SQL."
    ),
    Tool(
        name="Listar todas las columnas del esquema",
        func=tool_list_all_columns,
        description="Devuelve todas las columnas de todas las tablas del esquema. Útil para depuración y para explorar el esquema completo.\n**INSTRUCCIÓN:** Úsala si tienes dudas sobre el esquema."
    
    ),
]

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
    """Inicializa y devuelve el agente LangChain configurado."""
    # Memoria conversacional
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # LLM configurado (DeepSeek o el proveedor definido por las variables globales)
    logger.info(f"Inicializando LLM para el agente con: Provider={LLM_PROVIDER}, Model={LLM_MODEL}, BaseURL={LLM_API_BASE_URL}")
    
    # Usar ChatOpenAI como cliente, ya que la API de DeepSeek es compatible
    llm = ChatOpenAI(
        model=LLM_MODEL, 
        api_key=LLM_API_KEY,
        base_url=LLM_API_BASE_URL, # Usar la URL base saneada
        temperature=0.1,
        # max_tokens=4096 # Opcional, Langchain puede manejarlo o puedes definirlo
    )

    # Inicializa el agente con todas las herramientas, incluyendo las de esquema
    agent = initialize_agent(
        tools=[sql_medical_chatbot_tool, sinasuite_tool] + TOOLS_SCHEMA,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        memory=memory,
        verbose=False,
        handle_parsing_errors=custom_handle_parsing_errors,
        max_iterations=50,
        max_execution_time=300          # <-- Aumentado de 120 a 300 segundos
    )
    return agent

def main():
    # Definir códigos de color ANSI
    GREEN = "\033[92m"
    RESET = "\033[0m"

    logger.info("Iniciando Chatbot médico SQL (LangChain) - Modo Agente")
    logger.info("Escribe 'salir' para terminar.")
    # Se mantienen los prints para la consola con colores, pero la info principal va al log.
    print(f"{GREEN}Chatbot médico SQL (LangChain) - Modo Agente{RESET}")
    print(f"{GREEN}Escribe 'salir' para terminar.{RESET}")

    cli_agent = get_langchain_agent()  # Inicializar el agente para la CLI

    while True:
        question = ""  # Inicializar question
        try:
            question = input(f"{GREEN}Usuario: {RESET}").strip()
            if question.lower() in ("salir", "exit", "quit"):
                logger.info("Bot: ¡Hasta luego!")
                print(f"{GREEN}Bot: {RESET}¡Hasta luego!")
                break
            if not question:
                continue
            
            logger.info(f"Usuario pregunta: {question}")

            # El agente decidirá qué herramienta usar.
            agent_response = cli_agent.invoke({"input": question})
            
            # La estructura de agent_response puede variar, pero comúnmente es un diccionario
            # con una clave 'output' para la respuesta final.
            bot_response = agent_response.get("output", "No pude obtener una respuesta clara del agente.")
            logger.info(f"Respuesta del agente: {bot_response}")

        except OutputParserException as ope:
            logger.error(f"Error de parseo NO RECUPERADO por el agente tras el intento de corrección: {ope}", exc_info=True)
            bot_response = "Lo siento, tuve problemas para entender la estructura de la respuesta interna después de intentar corregirla. Por favor, intenta reformular tu pregunta."
        except Exception as e:
            logger.error(f"Error al invocar el agente para la pregunta '{question}': {e}", exc_info=True)
            bot_response = "Lo siento, ocurrió un error inesperado al procesar tu pregunta. Por favor, intenta reformularla."
            
        print(f"{GREEN}Bot: {RESET}{bot_response}")

if __name__ == "__main__":
    main()
