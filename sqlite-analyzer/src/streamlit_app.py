\
import streamlit as st
import os
import base64
import sys
import logging # Importar logging
import time # Python built-in time module

# Añadir el directorio src al path para poder importar langchain_chatbot y sus dependencias
# Esto asume que streamlit_app.py está en sqlite-analyzer/src/
_SCRIPT_DIR_STREAMLIT = os.path.dirname(os.path.abspath(__file__))
# Añadir el directorio 'sqlite-analyzer' al path si es necesario para las importaciones de langchain_chatbot
sys.path.append(os.path.abspath(os.path.join(_SCRIPT_DIR_STREAMLIT, '..')))
# Añadir el directorio actual (src) al path
sys.path.append(_SCRIPT_DIR_STREAMLIT)

# Importar componentes necesarios de langchain_chatbot
# Es crucial que las variables de entorno para el LLM (DEEPSEEK_API_KEY, etc.) estén configuradas
# en el entorno donde se ejecuta Streamlit.
try:
    from langchain_chatbot import SQLMedicalChatbot, db_connector_instance, llm_instance, logger as langchain_logger
    # Paths para terms y schema, deben ser consistentes con langchain_chatbot.py
    _DEFAULT_TERMS_DICT_PATH_STREAMLIT = os.path.join(_SCRIPT_DIR_STREAMLIT, "data", "dictionary.json")
    _DEFAULT_SCHEMA_PATH_STREAMLIT = os.path.join(_SCRIPT_DIR_STREAMLIT, "data", "schema_simple.json")

except ImportError as e:
    # Intentar una importación alternativa si la estructura del proyecto es diferente
    try:
        # Si streamlit_app.py está en la raíz del proyecto y langchain_chatbot en src/
        # Esto es menos probable dado el contexto, pero como fallback.
        from src.langchain_chatbot import SQLMedicalChatbot, db_connector_instance, llm_instance, logger as langchain_logger
        _DEFAULT_TERMS_DICT_PATH_STREAMLIT = os.path.join(_SCRIPT_DIR_STREAMLIT, "src", "data", "dictionary.json")
        _DEFAULT_SCHEMA_PATH_STREAMLIT = os.path.join(_SCRIPT_DIR_STREAMLIT, "src", "data", "schema_simple.json")
        logging.warning("Usando paths de importación alternativos para Streamlit app.")
    except ImportError:
        st.error(f"Error importando módulos necesarios: {e}. Asegúrate de que el PYTHONPATH es correcto, las dependencias están instaladas y el script se ejecuta desde la ubicación esperada.")
        st.stop() # Detener la ejecución si las importaciones fallan


# Configurar un logger para la app Streamlit si es necesario, o usar el de langchain_chatbot
logger = langchain_logger # Usar el logger ya configurado en langchain_chatbot

# --- Función para establecer el fondo ---
def set_bg_hack(main_bg_path):
    if not os.path.exists(main_bg_path):
        logger.error(f"La imagen de fondo no se encontró en la ruta: {main_bg_path}")
        st.warning(f"No se pudo cargar la imagen de fondo desde {main_bg_path}")
        return
    try:
        with open(main_bg_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        
        st.markdown(
             f"""
             <style>
             .stApp {{
                 background-image: url(data:image/png;base64,{encoded_string});
                 background-size: cover;
                 background-repeat: no-repeat;
                 background-attachment: fixed;
             }}

             /* Contenedor general de cada mensaje (avatar + burbuja) */
             .st-chat-message {{
                 /* background-color: transparent !important; */ /* Eliminado para que no haya doble fondo */
                 border-radius: 15px; /* Redondeo general si aplica al contenedor */
                 margin-bottom: 1rem !important;
                 /* border: none !important; */ /* Eliminado */
                 /* padding: 0px !important; */ /* El padding debe estar en la burbuja */
                 display: flex !important; /* Para alinear avatar y burbuja */
                 align-items: flex-end !important; /* Alinea items en la parte inferior */
             }}
             
             /* Ajuste para que el mensaje del usuario flote a la derecha */
             .st-chat-message:has([data-testid="chatAvatarIcon-user"]) {{
                flex-direction: row-reverse !important; /* Invierte el orden para el usuario (burbuja, luego avatar) */
             }}

             /* Burbuja de mensaje del USUARIO */
             [data-testid="chatAvatarIcon-user"] + div {{
                 background-color: rgba(220, 248, 198, 0.95) !important; /* Verde claro */
                 color: #1E1E1E !important;
                 border-radius: 15px !important;
                 padding: 10px 15px !important;
                 box-shadow: 0 3px 6px rgba(0,0,0,0.18) !important;
                 border: 1px solid rgba(0,0,0,0.07) !important;
                 max-width: 75% !important;
                 margin-right: 8px !important; /* Espacio entre avatar y burbuja para el usuario */
                 /* order: 1; */ /* No es necesario con flex-direction: row-reverse */
             }}

             /* Burbuja de mensaje del ASISTENTE */
             [data-testid="chatAvatarIcon-assistant"] + div {{
                 background-color: rgba(255, 255, 255, 0.96) !important; /* Blanco */
                 color: #1E1E1E !important;
                 border-radius: 15px !important;
                 padding: 10px 15px !important;
                 box-shadow: 0 3px 6px rgba(0,0,0,0.18) !important;
                 border: 1px solid rgba(0,0,0,0.07) !important;
                 max-width: 75% !important;
                 margin-left: 8px !important; /* Espacio entre avatar y burbuja para el asistente */
                 /* order: 1; */ /* No es necesario si el avatar es order:0 por defecto */
             }}
             
             /* Estilo para los avatares si es necesario (generalmente Streamlit los maneja bien) */
             /* [data-testid="chatAvatarIcon-user"], [data-testid="chatAvatarIcon-assistant"] {{
                 order: 0; 
             }} */


             /* Campo de entrada de texto */
             .stTextInput > div > div > input {{
                background-color: rgba(248, 249, 250, 0.96) !important; /* Un gris muy claro, casi blanco */
                color: #181818 !important;
                border: 1.5px solid #BCC0C4 !important;
                border-radius: 10px !important;
                padding: 12px 14px !important;
                box-shadow: inset 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.05) !important;
             }}
             /* Placeholder color */
            .stTextInput > div > div > input::placeholder {{
                color: #555c63 !important;
                opacity: 0.8 !important;
            }}

             /* Para asegurar que el input de chat no se solape con mensajes flotantes */
            div[data-testid="stChatInput"] {{
                clear: both;
            }}
             </style>
             """,
             unsafe_allow_html=True
         )
        logger.info(f"Imagen de fondo establecida desde {main_bg_path}")
    except Exception as e:
        logger.error(f"Error al establecer la imagen de fondo: {e}")
        st.warning("Error al cargar la imagen de fondo.")


# --- Configuración de la página de Streamlit ---
st.set_page_config(page_title="Chatbot Médico SINA", page_icon="🩺", layout="wide")
st.title("Chatbot Médico SINA 🩺")

# Establecer el fondo
BACKGROUND_IMAGE_PATH = os.path.join(_SCRIPT_DIR_STREAMLIT, "data", "fondo_sina_es.png")
set_bg_hack(BACKGROUND_IMAGE_PATH)

# --- Inicialización del Chatbot ---
if llm_instance is None:
    st.error("Error crítico: La instancia del LLM no pudo ser creada. Revisa la configuración y logs.")
    logger.error("llm_instance es None en streamlit_app.py. Deteniendo.")
    st.stop()

if db_connector_instance is None:
    st.error("Error crítico: La instancia del DB Connector no pudo ser creada. Revisa la configuración y logs.")
    logger.error("db_connector_instance es None en streamlit_app.py. Deteniendo.")
    st.stop()

try:
    medical_chatbot_tool = SQLMedicalChatbot(
        db_connector=db_connector_instance,
        logger=logger,
        terms_dict_path=_DEFAULT_TERMS_DICT_PATH_STREAMLIT,
        schema_path=_DEFAULT_SCHEMA_PATH_STREAMLIT,
        llm=llm_instance
    )
    logger.info("SQLMedicalChatbot instanciado correctamente en Streamlit.")
except Exception as e:
    st.error(f"Error al inicializar el chatbot: {e}")
    logger.error(f"Error al instanciar SQLMedicalChatbot en Streamlit: {e}", exc_info=True)
    st.stop()

# --- Estado de la sesión para el historial del chat ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hola 👋 ¿En qué puedo ayudarte hoy con la información médica?"}]

# Mostrar mensajes del historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Interfaz de Usuario y Lógica del Chat ---
if prompt := st.chat_input("Escribe tu pregunta sobre datos médicos..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        thinking_messages = [
            "Consultando mis notas... 📝",
            "Analizando los datos... 📊",
            "Buscando la información más precisa... 🔎",
            "Preparando tu respuesta... 💡"
        ]
        
        for msg in thinking_messages:
            message_placeholder.markdown(f"{msg}")
            time.sleep(0.7) # Pequeña pausa para simular progreso

        try:
            with st.spinner("Procesando tu solicitud... Por favor, espera un momento. ⏳"):
                logger.info(f"Streamlit app llamando a medical_chatbot_tool._run con query: '{prompt}'")
                response = medical_chatbot_tool._run(query=prompt)
                logger.info(f"Streamlit app recibió respuesta de _run: '{response[:300].replace('\\n', ' ')}...'") # Log más limpio

            message_placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            logger.error(f"Error durante la ejecución del chatbot desde Streamlit: {e}", exc_info=True)
            error_message = f"Lo siento, ocurrió un error al procesar tu pregunta. Por favor, intenta reformularla o consulta los logs para más detalles. Error: {str(e)}"
            message_placeholder.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

st.sidebar.info("Esta es una aplicación de demostración de un chatbot médico conectado a una base de datos SINA.")
st.sidebar.markdown("---")
if st.sidebar.button("Limpiar historial de chat"):
    st.session_state.messages = [{"role": "assistant", "content": "Hola 👋 ¿En qué puedo ayudarte hoy con la información médica?"}]
    st.rerun()

# Para ejecutar esta app:
# 1. Asegúrate de tener las variables de entorno necesarias configuradas (ej. DEEPSEEK_API_KEY).
# 2. Desde el directorio raíz del proyecto (ej. 'sina_mcp'), ejecuta:
#    streamlit run sqlite-analyzer/src/streamlit_app.py
