import streamlit as st
import sys
import os

# Se asume que este archivo (streamlit_interface.py) está en el mismo directorio (src)
# que langchain_chatbot.py. Si se ejecuta streamlit desde 'sqlite-analyzer/src/',
# langchain_chatbot será directamente importable.
try:
    from langchain_chatbot import get_langchain_agent, logger
except ImportError as e:
    # Si la importación directa falla, podría ser porque Streamlit se ejecuta desde un directorio superior.
    # Intentamos añadir 'src' al path si no está ya.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # También, el directorio padre de 'src' (que sería 'sqlite-analyzer') podría necesitar estar en el path
    # para que las importaciones internas de langchain_chatbot como 'from src.module' funcionen
    # si langchain_chatbot.py usa 'from src. ...' y 'src' es un subdirectorio del path esperado.
    # langchain_chatbot.py ya maneja esto añadiendo su directorio padre ('sqlite-analyzer') al path.
    try:
        from langchain_chatbot import get_langchain_agent, logger
    except ImportError as e_inner:
        st.error(f"Error al importar módulos necesarios: {e_inner}. Asegúrate de que la estructura de directorios es correcta. CWD: {os.getcwd()}, sys.path: {sys.path}")
        st.stop()

# --- Configuración de la página de Streamlit ---
st.set_page_config(page_title="SinaSuite LangChain Chatbot", layout="wide")

st.title("⚕️ SinaSuite LangChain Chatbot 💬")
st.caption("Interactúa con el sistema de consulta médica inteligente.")

# --- Disclaimer ---
st.warning("**AVISO:** Este es un sistema prototipo con fines de demostración. Las respuestas pueden tardar en generarse y la información proporcionada no debe usarse para tomar decisiones médicas reales.", icon="⚠️")

# --- Barra lateral con información y ejemplos ---
st.sidebar.title("ℹ️ Información y Ejemplos")
st.sidebar.markdown("""
Este chatbot puede ayudarte con diversas tareas relacionadas con información médica. Aquí tienes algunos ejemplos de lo que puedes preguntar:

**Saludos y General:**
- "Hola"
- "¿Quién eres?"
- "¿Qué es SinaSuite?"
- "Ayuda"

**Consultas Médicas Generales (BioChat Engine):**
- "¿Cuáles son los últimos avances en el tratamiento del infarto de miocardio?"
- "Explícame la CRISPR-Cas9."
- "Busca información sobre la metformina y sus efectos secundarios."

**Consultas a la Base de Datos Médica (SQL Engine):**
*   "¿Cuál es el número total de pacientes que hay en la base de datos?"
*   "¿Cuál es el nombre completo del paciente con ID 1001?"
*   "¿Cuál es la fecha de nacimiento del paciente con ID 1005?"
*   "¿Qué vacunas ha recibido el paciente con ID 1445 y en qué fechas?"
*   "¿Qué pacientes tienen alergias registradas y cuáles son esas alergias?"
*   "¿Qué tipo de alergias hay? ¿Hay pacientes con alergia al polen?"
*   "¿Cuántos pacientes han sido fusionados y en qué fechas se realizaron las fusiones?"
*   "¿Qué procedimientos médicos están autorizados y a qué tipo pertenecen?"
*   "¿Cuántos diagnósticos principales y secundarios se registraron en cada tipo de episodio durante el año 2024?"
*   "¿Cuál es la prevalencia de alergias alimentarias frente a no alimentarias en hombres y mujeres, por rangos de edad?"
*   "¿Cuál es la distribución de severidad de alergias por tipo de alergeno en pacientes activos?"
*   "¿Qué pacientes han sido fusionados como registros duplicados, cuántas fusiones se realizaron en 2023 y en qué fechas específicas ocurrieron?"
*   "De los 13074 procedimientos autorizados, ¿cuántos quedaron asociados a un episodio cerrado en ‘realizado’ versus ‘cancelado’ o ‘no realizado’, desglosado por tipo de procedimiento?"
*   "¿Cuáles son los pacientes registrados con más alergias y medicaciones?"
*   "Hazme un resumen de los datos clínicos del paciente MARIA DOLORES PRUEBAS101904 SINA101904."
*   "¿Hay algún paciente con pruebas de imagen?"
*   "¿Cuál es la edad del paciente 1010?"
*   "¿Qué pacientes hay hospitalizados?"
*   "¿Qué pruebas médicas o informes tiene Mariana (ID 3163)? ¿Qué información hay sobre ella?"

**Exploración del Esquema de la Base de Datos:**
- "Listar tablas del esquema"
- "Listar columnas de una tabla: PATI_PATIENTS"
- "Buscar en el esquema por palabra clave: diagnóstico"
- "Listar todas las columnas del esquema"

**Preguntas de seguimiento (usando contexto):**
- *(Después de preguntar por un paciente)* "¿Y cuáles son sus medicamentos?"
- *(Después de listar tablas)* "Muéstrame las columnas de la primera tabla."

**Nota:** El motor SQL es más efectivo si las preguntas son claras y se refieren a conceptos presentes en la base de datos. Para investigación general o temas médicos amplios, el motor BioChat es más adecuado.
""")

# --- Inicialización del Agente y Memoria en st.session_state ---
if 'agent' not in st.session_state:
    try:
        with st.spinner("Inicializando agente... Por favor, espera."):
            st.session_state.agent = get_langchain_agent()
        if st.session_state.agent is None:
            st.error("No se pudo inicializar el agente LangChain. Revisa los logs del terminal para más detalles.")
            logger.error("streamlit_interface.py: get_langchain_agent() devolvió None.")
            st.stop()
        logger.info("streamlit_interface.py: Agente LangChain inicializado y guardado en session_state.")
    except Exception as e:
        st.error(f"Excepción al inicializar el agente LangChain: {e}")
        logger.error(f"streamlit_interface.py: Excepción al inicializar el agente: {e}", exc_info=True)
        st.stop()

if 'memory' not in st.session_state:
    # Configurar la memoria aquí si es necesario o asegurarse de que el agente la maneja internamente.
    # Por ahora, asumimos que la memoria ConversationBufferMemory ya está en el agente.
    # Si necesitas pasarla explícitamente o configurarla de forma diferente, este es el lugar.
    # Ejemplo: st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # Y luego pasarla al agente si su inicialización lo permite.
    # Por simplicidad, y dado que el agente ya tiene memoria, no se inicializa explícitamente aquí de nuevo.
    pass # La memoria ya está configurada dentro de get_langchain_agent

if 'history' not in st.session_state:
    st.session_state.history = [] # Formato: [{"role": "user"/"assistant", "content": "..."}]

# --- Mostrar historial del chat ---
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Entrada del usuario ---
user_input = st.chat_input("Escribe tu pregunta aquí...")

if user_input:
    if st.session_state.agent is None:
        st.error("El agente no está inicializado. No se puede procesar la pregunta.")
    else:
        # Añadir pregunta del usuario al historial y mostrarla
        st.session_state.history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Procesar la pregunta con el agente y mostrar la respuesta en streaming
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                # Usar st.spinner para el mensaje "Pensando..."
                # El spinner se mostrará hasta que el primer chunk llegue o termine el stream.
                with st.spinner("Pensando... 🤔"):
                    # El agente ReAct espera un diccionario con la clave "input"
                    # y el método stream devuelve un generador de eventos/chunks.
                    # La memoria se gestiona internamente por el agente si se configuró con ConversationBufferMemory.
                    response_stream = st.session_state.agent.stream({
                        "input": user_input,
                        # "chat_history": st.session_state.history # Descomentar si el agente espera el historial explícitamente
                        })
                    
                    for chunk in response_stream:
                        # Los chunks pueden tener diferentes estructuras dependiendo del agente y LLM.
                        # Para un AgentExecutor, los tokens de salida suelen estar en chunk.get('output')
                        # o necesitarás inspeccionar el chunk para la clave correcta (ej. 'messages', 'content')
                        # Si el chunk es un string directamente (menos común para agentes complejos):
                        # if isinstance(chunk, str):
                        #    full_response += chunk
                        # elif isinstance(chunk, dict) and "output" in chunk:
                        #    full_response += chunk["output"]
                        
                        # Para AgentExecutor.stream, los eventos son diccionarios.
                        # El contenido de la respuesta final del LLM suele estar en eventos "on_chat_model_stream"
                        # bajo data -> chunk -> content
                        if "output" in chunk: # Para la salida final del agente ReAct
                            full_response += chunk["output"]
                            message_placeholder.markdown(full_response + "▌") # Añadir un cursor parpadeante
                        elif isinstance(chunk, dict) and chunk.get("event") == "on_chat_model_stream":
                            content = chunk.get("data", {}).get("chunk", {}).content
                            if content:
                                full_response += content
                                message_placeholder.markdown(full_response + "▌")
                        # Puedes añadir logs aquí para inspeccionar la estructura de los chunks si es necesario
                        # logger.debug(f"Stream chunk: {chunk}")
                
                message_placeholder.markdown(full_response) # Respuesta final sin cursor
                logger.info(f"streamlit_interface.py: User: '{user_input}', Agent: '{full_response}'")

            except Exception as e:
                full_response = f"Error al procesar la pregunta: {e}"
                message_placeholder.error(full_response)
                logger.error(f"streamlit_interface.py: Error al invocar el agente para input '{user_input}': {e}", exc_info=True)

        # Añadir respuesta del asistente al historial
        st.session_state.history.append({"role": "assistant", "content": full_response})

# --- Instrucciones para ejecutar ---
# st.sidebar.info("""
# **Para ejecutar esta aplicación:**
# 1. Asegúrate de tener Streamlit instalado (`pip install streamlit`).
# 2. Abre una terminal.
# 3. Navega al directorio `sqlite-analyzer/src/`.
#    (Ej: `cd c:\Users\cpascual\PycharmProjects\pythonProject\cursos_actividades\sina_mcp\sqlite-analyzer\src\`)
# 4. Ejecuta: `streamlit run streamlit_interface.py`
# """)
