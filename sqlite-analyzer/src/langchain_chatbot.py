"""
LangChain Tool para usar el pipeline de chatbot existente.
Permite usar el pipeline como herramienta en un agente conversacional LangChain.
"""
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_deepseek import ChatDeepSeek
from langchain_core.exceptions import OutputParserException

import sys
import os
import re
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importa tu pipeline (ajusta el import si es necesario)
from pipeline import chatbot_pipeline
from db_config import SQLiteConnector

# Configuración de la API de Deepseek (preferiblemente desde variables de entorno)
LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-aedf531ee17447aa95c9102e595f29ae")
LLM_API_URL = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
LLM_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
LLM_PROVIDER = "deepseek"

# Instancia el conector de base de datos (ajusta según tu config)
# Usa SQLiteConnector para conexión real
db_connector = SQLiteConnector(os.path.join(os.path.dirname(__file__), "db/database_new.sqlite3.db"))

# Herramienta para interactuar con la base de datos médica
chatbot_tool = Tool(
    name="SQLMedicalChatbot", # Nombre sin espacios para evitar problemas con algunos LLMs
    func=lambda q: chatbot_pipeline(q, db_connector)["response"],
    description="""Responde preguntas específicas sobre la base de datos médica, como buscar información de pacientes, citas, diagnósticos, etc.
Usa esta herramienta para cualquier pregunta que implique obtener datos de la base de datos.
Ejemplos de cuándo usarla: 'Busca al paciente con ID 123', '¿Cuáles son las alergias del paciente X?', 'Muéstrame las citas para mañana'."""
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
    name="SinaSuiteAndGeneralInformation", # Nombre sin espacios
    func=fetch_sinasuite_info,
    description="""Útil para responder a saludos, preguntas generales sobre la función del chatbot, o consultas sobre SinaSuite.
No uses esta herramienta para consultas que requieran acceder o buscar datos en la base de datos médica.
Ejemplos de cuándo usarla: 'Hola', '¿Qué es SinaSuite?', '¿Quién eres?', 'Ayuda'."""
)

def custom_handle_parsing_errors(error: OutputParserException) -> str:
    """
    Función personalizada para manejar errores de parseo del agente.
    Intenta dar una guía más específica al LLM para que se autocorrija.
    """
    problematic_output = getattr(error, 'llm_output', getattr(error, 'observation', 'No se pudo obtener la salida problemática.'))
    
    # Registrar el error para depuración
    print(f"ERROR DE PARSEO DEL AGENTE DETECTADO: {error}")
    print(f"SALIDA PROBLEMÁTICA DEL LLM QUE CAUSÓ EL ERROR:\n---\n{problematic_output}\n---")

    # Mensaje más detallado para el LLM
    return (
        "ERROR: Tu respuesta no siguió el formato ReAct esperado. "
        "Recuerda, debes generar una línea 'Action: [nombre de la herramienta]' "
        "seguida de una línea 'Action Input: [entrada para la herramienta]'.\n"
        f"La salida que generaste y causó el error fue:\n'''{problematic_output}'''\n"
        "Por favor, revisa el formato ReAct, asegúrate de que la acción sea una de las herramientas disponibles "
        "([SQLMedicalChatbot, SinaSuiteAndGeneralInformation]), y vuelve a intentarlo."
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
        handle_parsing_errors=custom_handle_parsing_errors # Usar la función personalizada
    )
    return agent

def main():
    # Definir códigos de color ANSI
    GREEN = "\033[92m"
    RESET = "\033[0m"

    print(f"{GREEN}Chatbot médico SQL (LangChain) - Modo Agente{RESET}")
    print(f"{GREEN}Escribe 'salir' para terminar.{RESET}")

    cli_agent = get_langchain_agent()  # Inicializar el agente para la CLI

    while True:
        question = input(f"{GREEN}Usuario: {RESET}").strip()
        if question.lower() in ("salir", "exit", "quit"):
            print(f"{GREEN}Bot: {RESET}¡Hasta luego!")
            break
        if not question:
            continue

        try:
            # El agente decidirá qué herramienta usar.
            agent_response = cli_agent.invoke({"input": question})
            
            # La estructura de agent_response puede variar, pero comúnmente es un diccionario
            # con una clave 'output' para la respuesta final.
            bot_response = agent_response.get("output", "No pude obtener una respuesta clara del agente.")

        except OutputParserException as ope:
            # Este error ya debería ser manejado por handle_parsing_errors si el agente lo reintenta.
            # Si llega aquí, es que el agente no pudo recuperarse tras el intento de handle_parsing_errors.
            print(f"Error de parseo NO RECUPERADO por el agente: {ope}")
            bot_response = "Lo siento, tuve problemas para entender la estructura de la respuesta interna después de intentar corregirla. Por favor, intenta reformular tu pregunta."
        except Exception as e:
            # Capturar otros errores durante la invocación del agente
            print(f"Error al invocar el agente: {e}")
            bot_response = "Lo siento, ocurrió un error inesperado al procesar tu pregunta. Por favor, intenta reformularla."
            
        print(f"{GREEN}Bot: {RESET}{bot_response}")

if __name__ == "__main__":
    main()
