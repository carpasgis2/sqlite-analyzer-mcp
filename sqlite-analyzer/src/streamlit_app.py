import streamlit as st
from langchain_community.callbacks import StreamlitCallbackHandler # Descomentado
import os
import sys
import base64

# Añadir la ruta 'src' al sys.path para asegurar que se puedan importar módulos locales
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from langchain_chatbot import get_langchain_agent # Descomentado
except ImportError:
    st.error("Error: No se pudo importar 'get_langchain_agent' desde 'langchain_chatbot.py'. ")

st.set_page_config(page_title="SinaSuite Chatbot Test", layout="wide")

# --- Funciones de fondo (pueden quedar para ver si la UI básica funciona) ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .stTextInput > div > div > input, 
    .stTextArea > div > div > textarea,
    [data-testid="stChatMessageContent"] {{
        background-color: rgba(255, 255, 255, 0.8); 
        color: black;
    }}
    [data-testid="stMarkdownContainer"] p {{
         background-color: rgba(255, 255, 255, 0.7);
         padding: 10px;
         border-radius: 5px;
    }}
    h1 {{
        color: #FFFFFF; 
        background-color: rgba(0, 0, 0, 0.3);
        padding: 10px;
        border-radius: 5px;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

image_path = os.path.join(os.path.dirname(__file__), "data", "fondo_sina_es.png") 
if os.path.exists(image_path):
    set_png_as_page_bg(image_path)
else:
    st.warning(f"No se encontró la imagen de fondo en: {os.path.abspath(image_path)}")
# --- Fin funciones de fondo ---

st.title("SinaSuite Chatbot - Prueba Mínima")

st.info("Prueba de Streamlit con importaciones de LangChain comentadas.")

# Comentamos toda la lógica del agente y del chat por ahora
# if 'agent' not in st.session_state:
#     st.info("Inicialización del agente comentada para prueba.")

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# prompt = st.chat_input("Haz tu pregunta...")

# if prompt:
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     # ... resto de la lógica del chat comentada ...
