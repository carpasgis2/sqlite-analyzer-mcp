"""
LangChain Tool para usar el pipeline de chatbot existente.
Permite usar el pipeline como herramienta en un agente conversacional LangChain.
"""
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_deepseek import ChatDeepSeek
from langchain_core.exceptions import OutputParserException

import logging  # Añadir importación de logging
import sys
import os
import re
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importa tu pipeline (ajusta el import si es necesario)
from pipeline import chatbot_pipeline
# Modificación: Importar get_db_connector en lugar de SQLiteConnector
from db_config import get_db_connector, DBConnector # Añadido DBConnector por si se usa para type hints en otro lado

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
# --- FIN CONFIGURACIÓN DE LOGGING ---

# Configuración de la API de Deepseek (preferiblemente desde variables de entorno)
LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-aedf531ee17447aa95c9102e595f29ae")
LLM_API_URL = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
LLM_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
LLM_PROVIDER = "deepseek"

# Instancia el conector de base de datos
# Modificación: Usar get_db_connector()
# La ruta de la base de datos ya está configurada en db_config.py (DEFAULT_DB_CONNECTION_STRING)
db_connector: DBConnector = get_db_connector()

# Herramienta para interactuar con la base de datos médica
def safe_process_results(query: str) -> str:
    """Procesa la consulta de manera segura con mejor manejo de errores y conversión de resultados. Nunca bloquea."""
    import concurrent.futures
    try:
        logger.info(f"Procesando consulta: '{query}' (con timeout de 90s)")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(chatbot_pipeline, query, db_connector, logger) # Añadido logger
            try:
                pipeline_result = future.result(timeout=90)
            except concurrent.futures.TimeoutError:
                logger.error("chatbot_pipeline excedió el tiempo máximo de 90s y fue cancelado")
                return "Error: El procesamiento de la consulta tardó demasiado y fue cancelado. Por favor, intenta con una consulta más simple."
            except Exception as e:
                logger.error(f"Error inesperado ejecutando chatbot_pipeline: {e}", exc_info=True)
                return f"Error inesperado al procesar la consulta: {str(e)}"

        logger.info(f"Pipeline ejecutado. Tipo de resultado: {type(pipeline_result)}")
        if pipeline_result is None:
            logger.error("Pipeline devolvió None")
            return "No pude obtener resultados para esta consulta."
        if not isinstance(pipeline_result, dict):
            logger.error(f"Pipeline devolvió un tipo inesperado: {type(pipeline_result)}")
            return f"Error: Resultado con formato inválido. Por favor reporta este error."
        
        response_message = pipeline_result.get("response", "La consulta se ejecutó pero no pude formatear la respuesta correctamente.")
        data_results = pipeline_result.get("data")

        logger.info(f"Resultado del pipeline procesado: Mensaje='{response_message}', Datos presentes: {data_results is not None}")

        # Formatear los datos si existen
        formatted_data = ""
        MAX_ROWS_FOR_AGENT_OBSERVATION = 10 # Limitar la cantidad de datos crudos en la observación
        if data_results:
            if isinstance(data_results, list):
                num_rows = len(data_results)
                if num_rows > 0:
                    try:
                        # Convertir lista de dicts a una cadena JSON bonita para la respuesta
                        import json
                        # Truncar datos para la observación del agente si son demasiados
                        data_to_format = data_results
                        if num_rows > MAX_ROWS_FOR_AGENT_OBSERVATION:
                            logger.info(f"Truncando datos para la observación del agente de {num_rows} a {MAX_ROWS_FOR_AGENT_OBSERVATION} filas.")
                            data_to_format = data_results[:MAX_ROWS_FOR_AGENT_OBSERVATION]
                        
                        formatted_data = json.dumps(data_to_format, indent=2, ensure_ascii=False)
                        
                    except Exception as e:
                        logger.error(f"Error al formatear los datos a JSON: {e}")
                        formatted_data = str(data_results[:MAX_ROWS_FOR_AGENT_OBSERVATION]) # Fallback a string simple del subconjunto
            elif not isinstance(data_results, list): # Dato único, no lista
                 formatted_data = str(data_results)
            # Si data_results es una lista vacía, formatted_data permanecerá vacío.

        # Combinar el mensaje de respuesta con los datos formateados
        final_response_str = response_message
        if formatted_data:
            # Añadir nota si los datos fueron truncados en la observación
            if isinstance(data_results, list) and len(data_results) > MAX_ROWS_FOR_AGENT_OBSERVATION:
                final_response_str += f"\\n(Mostrando primeros {MAX_ROWS_FOR_AGENT_OBSERVATION} registros en esta observación detallada)"
            final_response_str += f"\\nDatos:\\n{formatted_data}"
        elif "no devolvió filas" in response_message or "no se encontraron resultados" in response_message: # Si no hay datos y el mensaje ya lo indica
            pass # No añadir "Datos:" si el mensaje ya es claro sobre la ausencia de resultados
        elif not data_results and response_message: # No hay datos, pero el mensaje no lo dice explícitamente
             final_response_str += "\\nNo se encontraron datos para esta consulta."
        
        # Asegurar que la respuesta final sea un string simple si es necesario
        if not isinstance(final_response_str, (str, int, float)):
            logger.warning(f"La respuesta combinada no es un tipo básico serializable: {type(final_response_str)}")
            try:
                final_response_str = str(final_response_str)
            except Exception as e:
                logger.error(f"Error al convertir final_response_str a string: {e}")
                return "Error al procesar la respuesta final."
                
        # Limitar la longitud total de la observación para evitar errores de contexto del LLM
        MAX_OBSERVATION_LENGTH = 10000  # Ajustar según sea necesario (en caracteres)
        if len(final_response_str) > MAX_OBSERVATION_LENGTH:
            logger.warning(f"La observación final excede los {MAX_OBSERVATION_LENGTH} caracteres ({len(final_response_str)}). Truncando...")
            final_response_str = final_response_str[:MAX_OBSERVATION_LENGTH] + "... (Observación truncada)"

        return final_response_str
    except Exception as e:
        logger.error(f"Error en safe_process_results: {e}", exc_info=True)
        return f"Error al procesar la consulta: {str(e)}"

chatbot_tool = Tool(
    name="SQLMedicalChatbot",
    func=safe_process_results,  # Usar nuestra función robusta
    description="""Responde preguntas específicas sobre la base de datos médica. Útil para:
- Buscar información de pacientes, citas, diagnósticos, alergias, etc. (ej: 'Busca al paciente con ID 123', '¿Cuáles son las alergias del paciente X?').
- Obtener descripciones para códigos o IDs específicos de tablas de catálogo (ej: si obtienes un ALTY_ID = 1, puedes preguntar 'Cuál es la descripción para ALTY_ID 1 en ALLE_ALLERGY_TYPES' o generar directamente la consulta SQL 'SELECT ALTY_DESCRIPTION_ES FROM ALLE_ALLERGY_TYPES WHERE ALTY_ID = 1').
Usa esta herramienta para cualquier pregunta que implique obtener datos directamente de la base de datos.
Evita preguntas ambiguas; si necesitas traducir un ID a una descripción, formula una pregunta clara para obtener esa descripción de su tabla respectiva."""
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

def custom_handle_parsing_errors(error: OutputParserException) -> str:
    """
    Genera un mensaje de error personalizado y prescriptivo cuando el LLM no sigue el formato ReAct.
    Intenta extraer la salida problemática del LLM para incluirla en el mensaje de corrección.
    """
    response_str = str(error) # Mensaje completo de la excepción

    # Intentar extraer la salida real del LLM que causó el problema
    problematic_output = getattr(error, 'llm_output', None)
    if problematic_output is None: # Si error.llm_output no está disponible o es None
        # Intentar parsear desde el string de la excepción
        # Formatos comunes de OutputParserException:
        # "Parsing LLM output produced both a final answer and a parse-able action:: [ACTUAL_LLM_OUTPUT]"
        # "Could not parse LLM output: [ACTUAL_LLM_OUTPUT]"
        # "Invalid Format: [ACTUAL_LLM_OUTPUT]" (y otros)
        prefixes_to_check = [
            "Parsing LLM output produced both a final answer and a parse-able action:: ",
            "Could not parse LLM output: ",
            "Invalid Format: ",
            "Invalid tool `", # Para errores como "Invalid tool `XYZ`. Did you mean one of [valid_tools]?"
        ]
        
        parsed_from_str = False
        for prefix in prefixes_to_check:
            if response_str.startswith(prefix):
                # Tomar el resto del string después del prefijo
                # Para "Invalid tool", el output problemático es el nombre de la herramienta incorrecta.
                if prefix == "Invalid tool `":
                    end_of_tool_name = response_str.find("`")
                    if end_of_tool_name != -1:
                         problematic_output = response_str[len(prefix):end_of_tool_name]
                         parsed_from_str = True
                         break
                else:
                    problematic_output = response_str[len(prefix):]
                    parsed_from_str = True
                    break
        
        if not parsed_from_str:
            # Si no se pudo parsear con los prefijos conocidos, usar el mensaje de error completo como fallback
            # (aunque esto podría ser menos útil para el LLM)
            problematic_output = response_str 
    else: # Si error.llm_output está disponible
        problematic_output = str(problematic_output)

    # Loguear la salida problemática específica que se enviará al LLM para corrección
    logging.error(f"Salida problemática del LLM (para corrección):\\n---\\n{problematic_output}\\n---")

    # Mensaje prescriptivo para el LLM
    # (El resto de la función que construye el mensaje de error para el LLM permanece igual)
    # ... (código existente para construir el mensaje de error prescriptivo)
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

    # LLM de Deepseek
    llm = ChatDeepSeek(
        api_key=LLM_API_KEY,
        base_url=LLM_API_URL,
        model=LLM_MODEL
    )

    # Inicializa el agente con ambas herramientas
    agent = initialize_agent(
        tools=[chatbot_tool, sinasuite_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        memory=memory,
        verbose=True,
        handle_parsing_errors=custom_handle_parsing_errors  # Usar la función personalizada
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
