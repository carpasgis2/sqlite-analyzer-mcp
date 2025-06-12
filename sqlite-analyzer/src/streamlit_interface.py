# cd C:\Users\cpascual\PycharmProjects\pythonProject\cursos_actividades\sina_mcp\sqlite-analyzer\src

# streamlit run streamlit_interface.py


import streamlit as st
import os
import sys
import re
import time
import json
from datetime import datetime

# --- Configuración de sys.path ---
# Establecer el sys.path correctamente desde el principio.
# Asumimos que este script (streamlit_interface.py) está en .../sina_mcp/sqlite-analyzer/src/
# Y biochat.py está en .../sina_mcp/

_current_file_dir = os.path.dirname(os.path.abspath(__file__))
_sqlite_analyzer_dir = os.path.abspath(os.path.join(_current_file_dir, '..'))
_sina_mcp_dir = os.path.abspath(os.path.join(_sqlite_analyzer_dir, '..'))

# Asegurar que el directorio raíz del proyecto (sina_mcp) esté en sys.path
# para que 'biochat.py' (y otros módulos en sina_mcp) pueda ser importado directamente.
if _sina_mcp_dir not in sys.path:
    sys.path.insert(0, _sina_mcp_dir)

# Asegurar que el directorio 'sqlite-analyzer' (que contiene 'src') esté en sys.path
# para poder hacer importaciones como 'from src.module'.
if _sqlite_analyzer_dir not in sys.path:
    sys.path.insert(0, _sqlite_analyzer_dir)

# Opcionalmente, añadir 'src' directamente si prefieres 'from module_in_src import ...'
# if _current_file_dir not in sys.path:
#     sys.path.insert(0, _current_file_dir)

try:
    # Con _sqlite_analyzer_dir en sys.path, podemos hacer 'from src.langchain_chatbot ...'
    # Con _sina_mcp_dir en sys.path, langchain_chatbot.py debería poder hacer 'from biochat import ...'
    from src.langchain_chatbot import get_langchain_agent, logger
except ImportError as e:
    st.error(f"Error al importar módulos necesarios: {e}. \nCWD: {os.getcwd()}\nsys.path: {sys.path}")
    st.stop()
# --- Fin Configuración de sys.path ---

st.set_page_config(page_title="SinaSuite LangChain Chatbot", layout="wide")

st.write("🛠️ App cargando correctamente")
print("✅ Log visible en consola")
print(f"Versión de Python: {sys.version}")

# --- DISCLAIMER PARA USUARIOS MÓVILES ---
st.info("ℹ️ Si usas la versión móvil, expande la barra lateral (menú >) para ver todas las opciones y resultados correctamente.")

import sys
import os
import time # Añadido para el contador de tiempo
import re # Añadido para análisis de logs
import json # Para analizar la salida JSON de las herramientas
from datetime import datetime # Para marcas de tiempo

def limpiar_respuesta_final(texto):
    if not texto:
        return ""

    # Patrón para buscar "Final Answer:" (insensible a mayúsculas/minúsculas)
    # y capturar todo lo que sigue.
    final_answer_pattern = re.compile(r"Final Answer:(.*)", re.IGNORECASE | re.DOTALL)
    match = final_answer_pattern.search(texto)

    if match:
        # Si se encuentra "Final Answer:", tomar todo lo que sigue
        respuesta_final = match.group(1).strip()
        
        # Aplicar una limpieza mínima a esta respuesta final (espacios, saltos de línea múltiples)
        respuesta_final = re.sub(r"\n{2,}", "\n\n", respuesta_final).strip()
        
        if not respuesta_final:
            # Si después de "Final Answer:" no hay contenido o solo espacios.
            return "Se encontró 'Final Answer:' pero no había contenido posterior."
        return respuesta_final
    else:
        # Si "Final Answer:" no se encuentra, aplicar el conjunto completo de reglas de limpieza previas
        # Eliminar encabezados de Thought
        texto = re.sub(r"^Thought:\s*", "", texto, flags=re.IGNORECASE | re.MULTILINE)

        # Eliminar prefijos conversacionales (ya no se busca "Final Answer:" aquí explícitamente)
        conversational_prefixes_pattern = re.compile(
            r"^(I now know the final answer\.?\s*|The final answer is:?\s*|Here is the final answer:?\s*|Okay, I have the final answer:?\s*)+",
            re.IGNORECASE
        )
        texto = conversational_prefixes_pattern.sub("", texto).strip()

        # Eliminar bloques de plan de pasos multi-step
        texto = re.sub(
            r"🗂️\s*Plan de pasos sugerido por el agente:[\s\S]*?(?=\n\n|\Z)",
            "",
            texto,
            flags=re.IGNORECASE
        )
        texto = re.sub( # También eliminar la justificación si aparece separada
            r"Justificación del plan:[\s\S]*?(?=\n\n|\Z)",
            "",
            texto,
            flags=re.IGNORECASE
        )


        # Eliminar bloques de pensamiento, acción, logs, etc.
        patron_bloques_intermedios = re.compile(
            r"(^|\n)[\s\u2022\-\*]*((🧠\s*)?Pensamiento:.*|Thought:? ?.*|Action:? ?[\w\s]*:? ?.*|Action Input:? ?.*|Razonamiento:? ?.*|\(Mostrando.*registros.*\)|Datos:.*|\[.*?\]:.*|\(y \d+ m[aá]s\)\s*|\(Sin salida o salida vacía\))($|\n)",
            re.IGNORECASE | re.MULTILINE
        )
        texto = patron_bloques_intermedios.sub("\n", texto)

        # Eliminar logs y listados de tablas/columnas, incluyendo logs internos tipo pipeline.py
        patron_logs = re.compile(
            r"(^|\n)[\s\u2022\-\*]*(Tablas disponibles:.*|Columnas de [A-Z_]+:.*|He encontrado.*resultado.*|Datos:.*|\[\{.*?\}\].*|\(Mostrando.*?\)|No se encontraron.*|Tablas que coinciden.*|Columnas que coinciden.*|El resultado del conteo es:.*|No se encontraron datos para esta consulta\.|DEBUG: \[pipeline.py\].*)($|\n)",
            re.IGNORECASE | re.MULTILINE
        )
        texto = patron_logs.sub("\n", texto)

        # Eliminar bloques de SQL, advertencias técnicas y listados de columnas/tuplas
        texto = re.sub(r"Consulta SQL utilizada:.*?;\s*", "", texto, flags=re.DOTALL | re.IGNORECASE)
        texto = re.sub(r"Datos:\s*\(\[.*?\], \[.*?\]\)", "", texto, flags=re.DOTALL | re.IGNORECASE) # Para tuplas de datos
        texto = re.sub(r"Datos:\s*\[.*?\]", "", texto, flags=re.DOTALL | re.IGNORECASE) # Para listas de datos
        texto = re.sub(r"\n*Si necesitas más detalles, por favor aclara tu pregunta\.\n*", "\n", texto, flags=re.IGNORECASE)
        texto = re.sub(r"sql\nCopiar\n", "", texto, flags=re.IGNORECASE)
        texto = re.sub(r"O bien, si quieres usar la descripción del diccionario en lugar del código:[\s\S]*?;\s*", "", texto, flags=re.IGNORECASE)
        texto = re.sub(r"1\. Usando el texto libre.*?;\s*", "", texto, flags=re.IGNORECASE)
        texto = re.sub(r"2\. Usando el código de diagnóstico.*?;\s*", "", texto, flags=re.IGNORECASE)
        texto = re.sub(r"/\*.*?\*/", "", texto, flags=re.DOTALL)  # Elimina comentarios SQL multilinea
        texto = re.sub(r"\n*ADEMAS HA HECHO ESTA COSNULTA.*", "", texto, flags=re.IGNORECASE | re.MULTILINE)
        texto = re.sub(r"No, la consulta tal cual está escrita no es correcta, porque la columna.*", "", texto, flags=re.IGNORECASE | re.MULTILINE)
        texto = re.sub(r"En su lugar, esa tabla tiene:.*", "", texto, flags=re.IGNORECASE | re.MULTILINE)
        texto = re.sub(r"Esto podría indicar que no hay pacientes con ese diagnóstico en la base de datos o que los términos de búsqueda no coinciden exactamente con los registros.*", "", texto, flags=re.IGNORECASE | re.MULTILINE)
        texto = re.sub(r"La consulta no devolvió resultados, aunque la estructura de la consulta parece correcta.*", "", texto, flags=re.IGNORECASE | re.MULTILINE)
        texto = re.sub(r"Parece que hubo un problema al ejecutar la consulta SQL.*", "", texto, flags=re.IGNORECASE | re.MULTILINE)

        # Eliminar bloques de código (JSON, SQL, etc.)
        texto = re.sub(r"```[\w]*\n[\s\S]*?```", "\n", texto)

        # Eliminar duplicados de saltos de línea y espacios
        texto = re.sub(r"\n{2,}", "\n\n", texto).strip()

        # Si la respuesta sigue vacía, mostrar mensaje claro
        if not texto:
            return "No se encontraron resultados para tu consulta."
        return texto

# Se asume que este archivo (streamlit_interface.py) está en el mismo directorio (src)
# que langchain_chatbot.py. Si se ejecuta streamlit desde 'sqlite-analyzer/src/',
# langchain_chatbot será directamente importable.
try:
    from .langchain_chatbot import get_langchain_agent, logger
except ImportError as e:
    # --- CORRECCIÓN DE PATH PARA IMPORTS ROBUSTOS ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # Ahora intenta importar de nuevo
    try:
        from src.langchain_chatbot import get_langchain_agent, logger
    except ImportError as e_inner:
        st.error(f"Error al importar módulos necesarios: {e_inner}. Asegúrate de que la estructura de directorios es correcta. CWD: {os.getcwd()}, sys.path: {sys.path}")
        st.stop()

# --- Configuración de la página de Streamlit ---
# st.set_page_config(page_title="SinaSuite LangChain Chatbot", layout="wide")

st.title("⚕️ SinaSuite LangChain Chatbot 💬")
st.caption("Interactúa con el sistema de consulta médica inteligente.")

# --- Disclaimer ---
st.warning("**AVISO:** Este es un sistema prototipo con fines de demostración. Las respuestas pueden tardar en generarse y la información proporcionada no debe usarse para tomar decisiones médicas reales.", icon="⚠️")

# --- Barra lateral con información y ejemplos ---
st.sidebar.title("🩺 Ejemplos Profesionales de Consulta Médica")
st.sidebar.markdown("""
<div style="background-color:#23272f; color:#fff; padding: 1em; border-radius: 10px; border: 1px solid #444;">

### <span style="color:#ffe066;">👋 Saludos y General</span>
<ul style="color:#fff;">
  <li><b>Saludo inicial</b></li>
  <li><b>¿Quién es el asistente?</b></li>
  <li><b>¿Qué es SinaSuite?</b></li>
  <li><b>Ayuda sobre funcionalidades</b></li>
</ul>

---

### <span style="color:#ffe066;">🧑‍⚕️ Consultas Clínicas Generales (BioChat)</span>
<ul style="color:#fff;">
  <li><i>¿Cuáles son los avances recientes en el manejo del infarto agudo de miocardio?</i></li>
  <li><i>¿Podría explicar el mecanismo de acción y aplicaciones clínicas de la tecnología CRISPR-Cas9?</i></li>
  <li><i>¿Cuáles son los avances recientes en el manejo del infarto agudo de miocardio?</i></li>
  <li><i>¿Qué terapias génicas emergentes existen para enfermedades cardiovasculares?</i></li>
</ul>

---

### <span style="color:#ffe066;">🗄️ Consultas a la Base de Datos Médica (SQL Engine)</span>
<ul style="color:#fff;">
  <li><i>¿Cuál es el número total de pacientes registrados actualmente?</i></li>
  <li><i>¿Cuál es la prevalencia de alergias alimentarias frente a no alimentarias en hombres y mujeres, por rangos de edad?</i></li>
  <li><i>¿Cuántos pacientes tienen prescrita metformina junto con otro antidiabético oral?</i></li>

  <li><i>¿Qué patologías crónicas son más frecuentes por sexo y grupo de edad?</i></li>
  <li><i>¿Qué pacientes mayores de 65 años tienen prescritos medicamentos con alto riesgo de efectos adversos en geriatría según su grupo terapéutico, y han sido diagnosticados de hipertensión o insuficiencia renal?</i></li>
  <li><i>¿Existen pacientes que han sido diagnosticados con diabetes mellitus tipo 2, y además presentan prescripción simultánea de un antidiabético oral y un medicamento con riesgo de hipoglucemia no controlada?</i></li>
</ul>

---

### <span style="color:#ffe066;">🗂️ Exploración del Esquema de la Base de Datos</span>
<ul style="color:#fff;">
  <li><i>Listar todas las tablas del esquema.</i></li>
  <li><i>Mostrar columnas de la tabla: PATI_PATIENTS.</i></li>
  <li><i>Buscar en el esquema por palabra clave: diagnóstico.</i></li>
  <li><i>Listar todas las columnas del esquema.</i></li>
</ul>

---

### <span style="color:#ffe066;">🔄 Preguntas de Seguimiento</span>
<ul style="color:#fff;">
  <li><i>Después de consultar un paciente: "¿Cuáles son sus tratamientos farmacológicos actuales del paicente 2994?"</i></li>
  <li><i>Después de listar tablas: "Muéstrame las columnas de la primera tabla."</i></li>
</ul>

</div>

<style>
    .sidebar-title { font-size: 1.3em; font-weight: bold; margin-bottom: 0.5em; }
    .sidebar-section { margin-bottom: 1em; }
    .sidebar-section h4 { margin-bottom: 0.3em; }
    .sidebar-section ul { margin-top: 0; }
</style>
""", unsafe_allow_html=True)

st.sidebar.info("ℹ️ **Nota sobre el tiempo de respuesta:**\nPara consultas médicas complejas (BioChat Engine), el tiempo promedio de generación de respuesta es de aproximadamente 218.43 segundos.")

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

        # DEBUG: Inspeccionar configuración de la memoria del agente y el prompt
        print("\\n--- DEBUG AGENT INITIALIZATION ---")
        if hasattr(st.session_state.agent, 'memory') and st.session_state.agent.memory is not None:
            memory_obj = st.session_state.agent.memory
            print(f"DEBUG streamlit_interface: Agent memory type: {type(memory_obj)}")
            for attr in ['memory_key', 'input_key', 'output_key', 'return_messages']:
                if hasattr(memory_obj, attr):
                    print(f"DEBUG streamlit_interface: Agent memory.{attr}: {getattr(memory_obj, attr)}")
            
            chat_memory_component = getattr(memory_obj, 'chat_memory', None) # Común en ConversationBufferMemory
            if chat_memory_component and hasattr(chat_memory_component, 'messages'):
                print(f"DEBUG streamlit_interface: Initial messages in agent's ChatMessageHistory: {chat_memory_component.messages}")
            else:
                print(f"DEBUG streamlit_interface: Could not access initial messages in agent's ChatMessageHistory.")
        else:
            print("DEBUG streamlit_interface: Agent does not have a 'memory' attribute or it is None.")

        # Intentar acceder al prompt del agente (esto puede variar mucho según la estructura del agente)
        try:
            if hasattr(st.session_state.agent, 'agent') and hasattr(st.session_state.agent.agent, 'prompt'): # Común para AgentExecutor -> RunnableWithMessageHistory -> BaseSingleActionAgent
                 prompt_to_inspect = st.session_state.agent.agent.prompt
            elif hasattr(st.session_state.agent, 'prompt'): # Si el agente es una cadena más simple
                 prompt_to_inspect = st.session_state.agent.prompt
            else: # Intenta acceder a través de llm_chain si existe
                prompt_to_inspect = st.session_state.agent.agent.llm_chain.prompt

            if prompt_to_inspect:
                print(f"DEBUG streamlit_interface: Agent prompt type: {type(prompt_to_inspect)}")
                if hasattr(prompt_to_inspect, 'input_variables'):
                    print(f"DEBUG streamlit_interface: Agent prompt input_variables: {prompt_to_inspect.input_variables}")
                if hasattr(prompt_to_inspect, 'template'):
                    if isinstance(prompt_to_inspect.template, str):
                         print(f"DEBUG streamlit_interface: Agent prompt template (string): {prompt_to_inspect.template[:500]}...")
                    elif hasattr(prompt_to_inspect, 'messages'): # Para ChatPromptTemplate
                         print(f"DEBUG streamlit_interface: Agent prompt messages: {[str(m.prompt.template if hasattr(m, 'prompt') else m) for m in prompt_to_inspect.messages]}")
            else:
                print("DEBUG streamlit_interface: Could not determine agent's prompt structure for detailed logging.")
        except AttributeError:
            print("DEBUG streamlit_interface: Could not access agent's prompt using common patterns.")
        print("--- END DEBUG AGENT INITIALIZATION ---\\n")

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

import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

# Configuración de credenciales y acceso a Google Sheets (compatible con Streamlit Cloud)
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
try:
    secrets = st.secrets["gspread"]
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(dict(secrets), scope)
    gc = gspread.authorize(credentials)
    sheet = gc.open("RegistroPreguntas").sheet1
except Exception as e:
    sheet = None
    print(f"[Google Sheets] Error al inicializar conexión: {e}")

def log_pregunta_google(pregunta):
    print(f"[Google Sheets] Intentando registrar: {pregunta}")
    if sheet is not None:
        try:
            sheet.append_row([datetime.now().isoformat(), pregunta])
            print("[Google Sheets] ✅ Registro exitoso.")
        except Exception as e:
            print(f"[Google Sheets] ❌ Error al registrar: {e}")
            st.warning("No se pudo registrar la pregunta en Google Sheets.")
    else:
        print("[Google Sheets] ❌ Hoja no inicializada.")

# --- Entrada del usuario ---
user_input = st.chat_input("Escribe tu pregunta aquí...")
if user_input:
    try:
        log_pregunta_google(user_input)
        # st.success("✅ Pregunta registrada en Google Sheets.")  # Eliminado para que no aparezca en la interfaz
    except Exception as e:
        st.error(f"❌ Error al registrar: {e}")
    if st.session_state.agent is None:
        st.error("El agente no está inicializado. No se puede procesar la pregunta.")
    else:
        st.session_state.history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Inicializar/resetear logs acumulados para la interacción actual
        st.session_state.accumulated_agent_logs = []

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_llm_output = ""
            # intermediate_action_logs ya no se usará de la misma manera, se reemplaza por accumulated_agent_logs
            agent_has_finished = False
            final_cleaned_response_for_history = "No se pudo generar una respuesta."
            
            try:
                start_time = time.time()
                message_placeholder.markdown("▌") # Indicador inicial
                
                with st.spinner("Pensando... 🤔"):
                    response_stream = st.session_state.agent.stream({"input": user_input})
                    stream_did_iterate = False
                    plan_mostrado = False # Para el plan multi-step que se muestra fuera del placeholder
                    # plan_json = None # Ya existen
                    # plan_steps = [] # Ya existen
                    # plan_reasoning = None # Ya existen

                    for chunk_index, chunk in enumerate(response_stream):
                        stream_did_iterate = True
                        print(f"DEBUG streamlit_interface: ----- Chunk {chunk_index} -----")
                        print(f"RAW Stream chunk: {chunk}")
                        
                        current_time_str = datetime.now().strftime("%H:%M:%S") # Podría usarse en logs si se desea
                        formatted_log_entry = None # Para el log formateado de este chunk
                        new_content_from_llm_chunk = None # Contenido de este chunk específico para full_llm_output
                        
                        event = "N/A"
                        name = "N/A"

                        # --- DETECCIÓN Y VISUALIZACIÓN DEL PLAN MULTI-STEP (se mantiene como está, fuera del placeholder) ---
                        if not plan_mostrado:
                            plan_candidate = None
                            if isinstance(chunk, dict):
                                for v in chunk.values():
                                    if isinstance(v, str) and '"plan_required": true' in v: plan_candidate = v; break
                                    if isinstance(v, dict) and v.get("plan_required") is True: plan_candidate = v; break
                                if not plan_candidate and "messages" in chunk:
                                    for msg_item in chunk["messages"]: # msg renombrado
                                        if hasattr(msg_item, "content") and '"plan_required": true' in msg_item.content: plan_candidate = msg_item.content; break
                            elif isinstance(chunk, str) and '"plan_required": true' in chunk:
                                plan_candidate = chunk
                            
                            if plan_candidate:
                                try:
                                    plan_json_data = json.loads(plan_candidate) if isinstance(plan_candidate, str) else plan_candidate # renombrado
                                    if plan_json_data.get("plan_required") and isinstance(plan_json_data.get("plan"), list):
                                        plan_steps_data = plan_json_data["plan"] # renombrado
                                        plan_reasoning_data = plan_json_data.get("reasoning") # renombrado
                                        
                                        plan_display_markdown = "🗂️ **Plan de pasos sugerido por el agente:**\n"
                                        for step in plan_steps_data:
                                            plan_display_markdown += f"**Paso {step.get('step_number', '?')}:** {step.get('step_description', '')}  \n"
                                            plan_display_markdown += f"&nbsp;&nbsp;• Herramienta: `{step.get('tool_to_use', '')}`  \n"
                                            if step.get('inputs'):
                                                plan_display_markdown += f"&nbsp;&nbsp;• Entradas: `{json.dumps(step['inputs'], ensure_ascii=False)}`\n"
                                        if plan_reasoning_data:
                                            st.info(f"Justificación del plan: {plan_reasoning_data}") # Se muestra como st.info
                                        
                                        # Este plan se muestra una vez directamente, no en el placeholder que se refresca.
                                        # Si se quisiera en el placeholder, se añadiría a accumulated_agent_logs.
                                        st.markdown(plan_display_markdown, unsafe_allow_html=True)
                                        plan_mostrado = True
                                except Exception as ex:
                                    st.warning(f"No se pudo analizar el plan multi-step. Error: {ex}")
                                    st.code(str(plan_candidate))
                                    plan_mostrado = True
                        
                        # --- Procesamiento de Chunks para logs y salida LLM ---
                        if isinstance(chunk, dict):
                            event = chunk.get("event")
                            data = chunk.get("data", {})
                            name = chunk.get("name", "N/A")

                            if event is None and ("actions" in chunk or "messages" in chunk):
                                if "actions" in chunk and chunk["actions"]:
                                    for action_item in chunk["actions"]:
                                        if hasattr(action_item, 'log'):
                                            formatted_log_entry = f"#### 🧠 Pensamiento del Agente\n```text\n{action_item.log}\n```"
                                if "messages" in chunk and chunk["messages"]:
                                    for msg_item in chunk["messages"]: # msg renombrado
                                        content = None
                                        if hasattr(msg_item, 'content'): content = msg_item.content
                                        elif isinstance(msg_item, dict) and "content" in msg_item: content = msg_item["content"]
                                        if content: new_content_from_llm_chunk = content
                            
                            elif event == "on_agent_action":
                                action_data = data.get("action", {})
                                thought = action_data.get("log", "").strip()
                                if thought.startswith("Thought:"): thought = thought[len("Thought:"):].strip()
                                tool = action_data.get("tool")
                                tool_input_data = action_data.get("tool_input")
                                tool_input_str = str(tool_input_data)
                                
                                entry_parts = ["#### 🧠 Pensamiento y Acción del Agente"]
                                if thought: entry_parts.append(f"**Pensamiento:**\n```text\n{thought}\n```")
                                if tool:
                                    entry_parts.append(f"**Acción:** Usar herramienta `{tool}`")
                                    if tool_input_data:
                                        input_display = tool_input_str
                                        if len(input_display) > 250: input_display = input_display[:250] + "..."
                                        entry_parts.append(f"**Entrada para `{tool}`:**\n```\n{input_display}\n```")
                                formatted_log_entry = "\n".join(entry_parts)

                            elif event == "on_tool_start":
                                tool_input_data = data.get("input", {}) 
                                input_str = str(tool_input_data)
                                input_preview = (input_str[:250] + '...') if len(input_str) > 250 else input_str
                                entry_parts = [f"#### ⚙️ Ejecutando Herramienta: `{name}`"]
                                entry_parts.append(f"**Entrada:**\n```\n{input_preview}\n```")
                                if name and ("biochat" in name.lower() or "biomedicalchatbotagent" in name.lower()):
                                    entry_parts.append("\n🔬 _Realizando consulta biomédica avanzada... Esto puede tardar varios minutos._")
                                formatted_log_entry = "\n".join(entry_parts)

                            elif event == "on_tool_end":
                                tool_output_data = data.get("output")
                                output_str = str(tool_output_data) if tool_output_data is not None else ""
                                entry_parts = [f"#### ✅ Resultado de Herramienta: `{name}`"]
                                output_display = output_str
                                if len(output_display) > 400: output_display = output_display[:400] + "..."

                                if "RateLimitError" in output_str or "AuthenticationError" in output_str or "Could not parse LLM output" in output_str or "InvalidRequestError" in output_str:
                                    entry_parts.append(f"⚠️ **Error/Advertencia:**\n```text\n{output_display}\n```")
                                elif output_str:
                                    entry_parts.append(f"**Salida:**\n```text\n{output_display}\n```")
                                else:
                                    entry_parts.append("_Sin salida o salida vacía._")
                                formatted_log_entry = "\n".join(entry_parts)
                            
                            elif event in ["on_chat_model_stream", "on_llm_stream"]:
                                content_piece = data.get("chunk")
                                if isinstance(content_piece, str): new_content_from_llm_chunk = content_piece
                                elif isinstance(content_piece, dict) and "content" in content_piece: new_content_from_llm_chunk = content_piece["content"]
                                elif hasattr(content_piece, 'content'): new_content_from_llm_chunk = content_piece.content
                            
                            elif event == "on_agent_finish":
                                agent_has_finished = True
                                agent_final_output_text = None
                                agent_output_payload = data.get("output")
                                if isinstance(agent_output_payload, str): agent_final_output_text = agent_output_payload
                                elif isinstance(agent_output_payload, dict): agent_final_output_text = agent_output_payload.get("output", str(agent_output_payload))
                                elif hasattr(agent_output_payload, 'return_values') and isinstance(agent_output_payload.return_values, dict): agent_final_output_text = agent_output_payload.return_values.get("output", str(agent_output_payload.return_values))
                                elif hasattr(agent_output_payload, 'log') and isinstance(agent_output_payload.log, str): agent_final_output_text = agent_output_payload.log
                                
                                if agent_final_output_text: new_content_from_llm_chunk = agent_final_output_text # Añadir la salida final al stream de LLM
                                formatted_log_entry = "####🏁 Agente ha finalizado."
                        
                        elif isinstance(chunk, str): # Fallback para chunks de string directos
                            new_content_from_llm_chunk = chunk
                        
                        # --- Acumular logs y salida del LLM ---
                        if formatted_log_entry:
                            st.session_state.accumulated_agent_logs.append(formatted_log_entry)
                        
                        if new_content_from_llm_chunk:
                            full_llm_output += new_content_from_llm_chunk
                        
                        # --- Actualizar el placeholder ---
                        display_parts = []
                        if st.session_state.accumulated_agent_logs:
                            display_parts.append("\n\n---\n\n".join(st.session_state.accumulated_agent_logs))
                        
                        # Añadir la salida actual del LLM (que puede ser la respuesta final o pensamientos)
                        # La limpieza final se hará después del bucle.
                        if full_llm_output:
                            display_parts.append(f"**Respuesta del Agente (en progreso):**\n{full_llm_output}")

                        current_display_content = "\n\n---\n\n".join(filter(None, display_parts))
                        
                        if not agent_has_finished:
                            message_placeholder.markdown(current_display_content + " ▌", unsafe_allow_html=True)
                        else:
                            # Cuando el agente ha finalizado, el bucle terminará pronto.
                            # La actualización final del placeholder se hará en el bloque `finally`.
                            message_placeholder.markdown(current_display_content, unsafe_allow_html=True)

                    # --- FIN DEL BUCLE DEL STREAM ---
                    if not stream_did_iterate and not agent_has_finished:
                        # Si el stream no produjo nada y el agente no ha "finalizado" explícitamente
                        full_llm_output = "El agente no produjo ninguna salida o evento."
                        agent_has_finished = True # Forzar finalización para limpiar

            except Exception as e:
                logger.error(f"streamlit_interface.py: Error durante el stream del agente: {e}", exc_info=True)
                full_llm_output += f"\n\nOcurrió un error: {e}"
                st.error(f"Error al procesar la pregunta: {e}")
                agent_has_finished = True # Asegurar que se limpie
            
            finally:
                end_time = time.time()
                logger.info(f"streamlit_interface.py: Respuesta generada en {end_time - start_time:.2f} segundos.")
                
                final_cleaned_response_for_history = limpiar_respuesta_final(full_llm_output)
                if not final_cleaned_response_for_history.strip() and stream_did_iterate:
                     final_cleaned_response_for_history = "El agente procesó la solicitud pero la respuesta final está vacía o fue eliminada por las reglas de limpieza."
                elif not stream_did_iterate and not full_llm_output: # Si no hubo stream ni error capturado
                    final_cleaned_response_for_history = "No se pudo obtener una respuesta del agente."

                message_placeholder.markdown(final_cleaned_response_for_history, unsafe_allow_html=True)
                st.session_state.history.append({"role": "assistant", "content": final_cleaned_response_for_history})
                st.session_state.accumulated_agent_logs = [] # Limpiar para la próxima vez

# --- Instrucciones para ejecutar ---
# st.sidebar.info("""
# **Para ejecutar esta aplicación:**
# 1. Asegúrate de tener Streamlit instalado (`pip install streamlit`).
# 2. Abre una terminal.
# 3. Navega al directorio `sqlite-analyzer/src/`.
#    (Ej: `cd c:\Users\cpascual\PycharmProjects\pythonProject\cursos_actividades\sina_mcp\sqlite-analyzer\src\`)
# 4. Ejecuta: `streamlit run streamlit_interface.py`
# """)
