import streamlit as st
import sys
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
    # Ensure 'import re' is present at the top of your Python file.
    if not texto:
        return ""

    # Pattern for common conversational prefixes before the actual answer
    conversational_prefixes_pattern = re.compile(
        r"^(I now know the final answer\.?\s*|The final answer is:?\s*|Here is the final answer:?\s*|Okay, I have the final answer:?\s*)+", 
        re.IGNORECASE
    )

    # Attempt to find "Final Answer:" marker
    final_answer_marker_search = re.search(r"Final Answer:?", texto, re.IGNORECASE)

    if final_answer_marker_search:
        # If "Final Answer:" is found, take everything after it
        content_after_marker = texto[final_answer_marker_search.end():].strip()
        # Then, remove conversational prefixes from this extracted content
        cleaned_answer = conversational_prefixes_pattern.sub("", content_after_marker).strip()
        return cleaned_answer
    else:
        # If "Final Answer:" is NOT found, first remove conversational prefixes from the whole text
        texto_sin_prefijos = conversational_prefixes_pattern.sub("", texto).strip()

        # Then, apply original patterns to remove thoughts, actions, logs, etc.
        
        # Pattern for thoughts, actions, reasoning blocks
        patron_bloques_intermedios = re.compile(
            r"(^|\n)[\s\u2022\-\*]*((🧠\s*)?Pensamiento:.*|Action:? ?[\w\s]*:? ?.*|Action Input:? ?.*|Razonamiento:? ?.*|\(Mostrando.*registros.*\)|Datos:.*|\[.*?\]:.*|\(y \d+ m[aá]s\)\s*|\(Sin salida o salida vacía\))($|\n)", 
            re.IGNORECASE
        )
        texto_limpio = patron_bloques_intermedios.sub("\n", texto_sin_prefijos)

        # Pattern for logs (e.g., table listings, column info)
        patron_logs = re.compile(
            r"(^|\n)[\s\u2022\-\*]*(Tablas disponibles:.*|Columnas de [A-Z_]+:.*|He encontrado.*resultado.*|Datos:.*|\[\{.*?\}\].*|\(Mostrando.*?\)|No se encontraron.*|Tablas que coinciden.*|Columnas que coinciden.*|El resultado del conteo es:.*|No se encontraron datos para esta consulta\.)($|\n)", 
            re.IGNORECASE
        )
        texto_limpio = patron_logs.sub("\n", texto_limpio)
        
        # Remove code blocks (like JSON outputs)
        texto_limpio = re.sub(r"```[\w]*\n[\s\S]*?```", "\n", texto_limpio)
        
        # Consolidate multiple newlines and strip whitespace
        texto_limpio = re.sub(r"\n{2,}", "\n\n", texto_limpio).strip()
        
        return texto_limpio

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
  <li><i>"¿Cuáles son los avances recientes en el manejo del infarto agudo de miocardio?"</i></li>
  <li><i>"¿Podría explicar el mecanismo de acción y aplicaciones clínicas de la tecnología CRISPR-Cas9?"</i></li>
  <li><i>"¿Cuáles son los avances recientes en el manejo del infarto agudo de miocardio?"</i></li>
  <li><i>"¿Qué terapias génicas emergentes existen para enfermedades cardiovasculares?"</i></li>
</ul>

---

### <span style="color:#ffe066;">🗄️ Consultas a la Base de Datos Médica (SQL Engine)</span>
<ul style="color:#fff;">
  <li><i>"¿Cuál es el número total de pacientes registrados actualmente?"</i></li>
  <li><i>Edad y sexo del paciente con ID 1010.</i></li>
  <li><i>Listado de pacientes hospitalizados en el ultimo mes .</i></li>
  <li><i>¿Qué informes de imagen están disponibles para el paciente Mariana (ID 3163)?</i></li>
  <li><i>¿Cuáles son los diagnósticos principales registrados en episodios de 2022?</i></li>
  <li><i>Resumen clínico del paciente MARIA DOLORES PRUEBAS101904 SINA101904.</i></li>
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
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name("credenciales_google.json", scope)
gc = gspread.authorize(credentials)
sheet = gc.open("RegistroPreguntas").sheet1

def log_pregunta_google(pregunta):
    sheet.append_row([datetime.now().isoformat(), pregunta])


# --- Entrada del usuario ---
user_input = st.chat_input("Escribe tu pregunta aquí...")
if user_input:
    log_pregunta_google(user_input)
    if st.session_state.agent is None:
        st.error("El agente no está inicializado. No se puede procesar la pregunta.")
    else:
        st.session_state.history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_llm_output = ""
            intermediate_action_logs = ""
            agent_has_finished = False
            final_cleaned_response_for_history = "No se pudo generar una respuesta."
            try:
                start_time = time.time()
                message_placeholder.markdown("▌")
                with st.spinner("Pensando... 🤔"):
                    response_stream = st.session_state.agent.stream({"input": user_input})
                    stream_did_iterate = False
                    for chunk_index, chunk in enumerate(response_stream):
                        stream_did_iterate = True
                        print(f"DEBUG streamlit_interface: ----- Chunk {chunk_index} -----")
                        print(f"RAW Stream chunk: {chunk}")
                        current_time_str = datetime.now().strftime("%H:%M:%S")
                        intermediate_log_line = None
                        new_content_from_llm_chunk = None # Contenido de este chunk específico
                        event = "N/A"
                        name = "N/A"

                        if isinstance(chunk, dict):
                            event = chunk.get("event")
                            data = chunk.get("data", {})
                            name = chunk.get("name", "N/A")
                            log_prefix = f"`{current_time_str}` | `{event}` | (__{name}__ )"

                            if event is None and ("actions" in chunk or "messages" in chunk):
                                if "actions" in chunk and chunk["actions"]:
                                    for action_item in chunk["actions"]: # Renombrado para evitar conflicto
                                        if hasattr(action_item, 'log'):
                                            intermediate_log_line = f"🧠 **Pensamiento:** {action_item.log}"
                                if "messages" in chunk and chunk["messages"]:
                                    for msg in chunk["messages"]:
                                        if hasattr(msg, 'content'):
                                            new_content_from_llm_chunk = msg.content
                                        elif isinstance(msg, dict) and "content" in msg:
                                            new_content_from_llm_chunk = msg["content"]

                            if event == "on_agent_action":
                                action_data = data.get("action", {}) # Renombrado para evitar conflicto
                                thought = action_data.get("log", "").strip()
                                if thought.startswith("Thought:"):
                                    thought = thought[len("Thought:"):].strip()
                                
                                tool = action_data.get("tool")
                                tool_input_data = action_data.get("tool_input")
                                tool_input_str = str(tool_input_data)
                                
                                intermediate_log_line = f"{log_prefix}\\\\n"
                                if thought:
                                    intermediate_log_line += f"🧠 **Pensamiento:** {thought}\\\\n"
                                if tool:
                                    intermediate_log_line += f"🛠️ **Acción:** Usar `{tool}`"
                                    if tool_input_data:
                                        intermediate_log_line += f" con entrada: `{tool_input_str[:100]}{'...' if len(tool_input_str) > 100 else ''}`"
                                elif not thought: 
                                    intermediate_log_line += "❓ Acción del agente sin herramienta o pensamiento claro."
                                
                            elif event == "on_tool_start":
                                tool_input_data = data.get("input", {}) 
                                input_str = str(tool_input_data)
                                input_preview = (input_str[:100] + '...') if len(input_str) > 100 else input_str
                                intermediate_log_line = f"{log_prefix}\\\\n⚙️ **Iniciando herramienta:** `{name}` (Entrada: `{input_preview}`)"
                                if name and ("biochat" in name.lower() or "biomedicalchatbotagent" in name.lower()):
                                    intermediate_log_line += "\\\\n🔬 _Realizando consulta biomédica avanzada... Esto puede tardar varios minutos._"

                            elif event == "on_tool_end":
                                tool_output_data = data.get("output")
                                output_str = str(tool_output_data) if tool_output_data is not None else ""
                                intermediate_log_line = f"{log_prefix}\\\\n✅ **Herramienta `{name}` finalizada.**\\\\n"

                                if "RateLimitError" in output_str or "AuthenticationError" in output_str or "Could not parse LLM output" in output_str or "InvalidRequestError" in output_str:
                                    intermediate_log_line += f"⚠️ **Error/Advertencia de la herramienta `{name}`:**\\\\n```text\\\\n{output_str[:500]}{'...' if len(output_str) > 500 else ''}\\\\n```"
                                elif name and ("biochat" in name.lower() or "biomedicalchatbotagent" in name.lower()) and output_str:
                                    try:
                                        # ... (código de formateo JSON existente) ...
                                        if output_str.startswith("[") and output_str.endswith("]") and name.endswith("_tool_func"): # Asumiendo que esto es para BioChat
                                            outputs_list = json.loads(output_str)
                                            if isinstance(outputs_list, list) and all(isinstance(item, str) for item in outputs_list):
                                                parsed_items = [json.loads(item_str) for item_str in outputs_list]
                                                pretty_json = json.dumps(parsed_items, indent=2, ensure_ascii=False)
                                            else: # Si ya es una lista de dicts o similar
                                                pretty_json = json.dumps(outputs_list, indent=2, ensure_ascii=False)
                                        else: # Para otros JSON
                                            parsed_json = json.loads(output_str)
                                            pretty_json = json.dumps(parsed_json, indent=2, ensure_ascii=False)
                                        intermediate_log_line += f"📄 **Resultado de `{name}` (JSON):**\\\\n```json\\\\n{pretty_json}\\\\n```"
                                    except json.JSONDecodeError:
                                        intermediate_log_line += f"📄 **Resultado de `{name}` (texto, no se pudo analizar como JSON):**\\\\n```text\\\\n{output_str[:500]}{'...' if len(output_str) > 500 else ''}\\\\n```"

                                elif output_str:
                                    intermediate_log_line += f"📄 **Resultado de `{name}`:**\\\\n```text\\\\n{output_str[:300]}{'...' if len(output_str) > 300 else ''}\\\\n```"
                                else:
                                    intermediate_log_line += f"📄 **Resultado de `{name}`:** (Sin salida o salida vacía)"
                            
                            elif event in ["on_chat_model_stream", "on_llm_stream"]:
                                content_piece = data.get("chunk")
                                if isinstance(content_piece, str):
                                    new_content_from_llm_chunk = content_piece
                                elif isinstance(content_piece, dict) and "content" in content_piece: # Manejo de dict chunk
                                    new_content_from_llm_chunk = content_piece.get("content")
                                elif hasattr(content_piece, 'content'): # Manejo de objeto AIMessageChunk o similar
                                     new_content_from_llm_chunk = content_piece.content
                            
                            elif event == "on_agent_finish":
                                agent_final_output_text = None
                                agent_output_payload = data.get("output") # Esto puede ser un string o un dict

                                # Intentar extraer la cadena de texto de la respuesta final
                                if isinstance(agent_output_payload, str):
                                    agent_final_output_text = agent_output_payload
                                elif isinstance(agent_output_payload, dict):
                                    if "output" in agent_output_payload:
                                        agent_final_output_text = agent_output_payload["output"]
                                    elif "final_answer" in agent_output_payload:
                                        agent_final_output_text = agent_output_payload["final_answer"]
                                    elif "result" in agent_output_payload:
                                        agent_final_output_text = agent_output_payload["result"]
                                # Para el objeto AgentFinish de Langchain (si data['output'] es el objeto en sí)
                                elif hasattr(agent_output_payload, 'return_values') and isinstance(agent_output_payload.return_values, dict):
                                    agent_final_output_text = agent_output_payload.return_values.get('output')
                                # A veces, el 'log' del AgentFinish puede contener la respuesta formateada completa
                                elif hasattr(agent_output_payload, 'log') and isinstance(agent_output_payload.log, str):
                                     # Usar el log solo si es más sustancial que un simple mensaje de finalización
                                     if "Final Answer:" in agent_output_payload.log or len(agent_output_payload.log) > 100:
                                        agent_final_output_text = agent_output_payload.log

                                if agent_final_output_text and isinstance(agent_final_output_text, str):
                                    # Usar la respuesta final directa del agente.
                                    # Esto sobrescribe cualquier acumulación previa de full_llm_output
                                    # con la conclusión explícita del agente.
                                    full_llm_output = agent_final_output_text
                                    intermediate_log_line = f"{log_prefix}\\n🏁 **Agente finalizado.** Respuesta final directa del agente utilizada."
                                    print(f"DEBUG streamlit_interface: (on_agent_finish) Usando respuesta directa del agente: '{full_llm_output[:300].replace('\\n', ' ')}...'")
                                else:
                                    # Si no se pudo extraer una respuesta final clara del evento on_agent_finish,
                                    # se confiará en el full_llm_output acumulado de los eventos on_llm_stream.
                                    intermediate_log_line = f"{log_prefix}\\n🏁 **Agente finalizado.** (Respuesta final se tomará de la acumulación de LLM streams)"
                                    print(f"DEBUG streamlit_interface: (on_agent_finish) No se extrajo respuesta directa del payload del evento, se usará full_llm_output acumulado: '{data.get('output')}'")
                                
                                agent_has_finished = True
                        
                        elif isinstance(chunk, str): 
                            new_content_from_llm_chunk = chunk
                        # --- FIN DEL PROCESAMIENTO DEL CHUNK ---

                        # --- DEBUGGING INTERNO DEL BUCLE ---
                        print(f"DEBUG streamlit_interface: (Chunk {chunk_index}) event='{event}', name='{name}'")
                        if intermediate_log_line is not None:
                            print(f"DEBUG streamlit_interface: (Chunk {chunk_index}) intermediate_log_line='{intermediate_log_line.strip()}'")
                        if new_content_from_llm_chunk is not None:
                            print(f"DEBUG streamlit_interface: (Chunk {chunk_index}) new_content_from_llm_chunk='{new_content_from_llm_chunk}'")
                        # --- FIN DEBUGGING INTERNO ---

                        if intermediate_log_line:
                            log_to_add = intermediate_log_line.strip()
                            if log_to_add:
                                if intermediate_action_logs and not intermediate_action_logs.strip().endswith("\n\n"):
                                    intermediate_action_logs += "\n\n"
                                intermediate_action_logs += log_to_add + "\n\n"
                        
                        if new_content_from_llm_chunk:
                            full_llm_output += new_content_from_llm_chunk

                        # --- Lógica de visualización dentro del bucle ---
                        # Mostrar logs de acción + salida acumulada del LLM mientras el agente no haya finalizado explícitamente.
                        # Una vez finalizado, solo se mostrará full_llm_output (que luego será limpiado).
                        current_display_content = ""
                        if not agent_has_finished: # Mostrar logs intermedios solo mientras el agente está trabajando activamente
                            current_display_content += intermediate_action_logs
                        
                        current_display_content += full_llm_output # Siempre mostrar la salida del LLM acumulada

                        if current_display_content.strip():
                            message_placeholder.markdown(current_display_content + " ▌")
                        else:
                            message_placeholder.markdown("▌")
                        
                        if agent_has_finished: # Si el evento on_agent_finish ocurrió
                            break # Salir del bucle de streaming

                    print("DEBUG streamlit_interface: Fin del bucle for chunk in response_stream.")
                    if not stream_did_iterate:
                        print("DEBUG streamlit_interface: ADVERTENCIA - El stream no produjo ningún chunk.")
                        message_placeholder.error("❌ El agente no devolvió ninguna respuesta. Puede haber un problema con el modelo, la API o la configuración. Revisa los logs del terminal para más detalles.")
                        logger.error(f"streamlit_interface.py: User: '{user_input}', Agent: El stream no produjo ningún chunk.")
                        final_cleaned_response_for_history = "❌ El agente no devolvió ninguna respuesta."
                        # st.session_state.history.append ya se maneja al final del bloque try/except/finally
                        st.stop() # Detener si no hay chunks

                print("DEBUG streamlit_interface: Saliendo del bloque st.spinner.")
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                print(f"DEBUG streamlit_interface: Contenido de full_llm_output ANTES de limpiar: '{full_llm_output}'")
                final_cleaned_response = limpiar_respuesta_final(full_llm_output.strip())
                print(f"DEBUG streamlit_interface: Respuesta final (limpia DESPUÉS de limpiar full_llm_output): '{final_cleaned_response}'")

                if not final_cleaned_response:
                    message_placeholder.warning("No se generó una respuesta visible o la respuesta estaba vacía. Revisa los logs del terminal para más detalles, especialmente si esperabas una respuesta de BioChat (puede haber problemas con la API de OpenAI).")
                    logger.warning(f"streamlit_interface.py: User: '{user_input}', Agent: Respuesta vacía tras limpieza, Time: {elapsed_time:.2f}s. Raw output: '{full_llm_output[:500]}...'")
                    final_cleaned_response_for_history = "⚠️ No se generó una respuesta visible o la respuesta estaba vacía."
                else:
                    message_placeholder.markdown(final_cleaned_response) # Mostrar la respuesta final limpia
                    final_cleaned_response_for_history = final_cleaned_response
                    # --- NUEVO: Añadir contexto de pregunta anterior si la respuesta es de seguimiento ---
                    respuesta = final_cleaned_response
                    if (
                        "¿Deseas que investigue esta relación?" in respuesta
                        or "¿Quieres que investigue esta relación?" in respuesta
                        or "¿Desea que investigue esta relación?" in respuesta
                    ):
                        # Aquí puedes añadir lógica para el tag de seguimiento si es necesario
                        pass # st.markdown("<span style='color:cyan;font-size:0.8em;'>↪️ Respuesta de seguimiento</span>", unsafe_allow_html=True)
                
                st.caption(f"Tiempo de generación: {elapsed_time:.2f} segundos")

            except Exception as e:
                print(f"ERROR streamlit_interface: Excepción durante el streaming: {e}", file=sys.stderr)
                logger.error(f"streamlit_interface.py: Error al invocar el agente para input '{user_input}': {e}", exc_info=True)
                error_message = f"Error catastrófico al procesar la pregunta: {e}"
                message_placeholder.error(error_message)
                final_cleaned_response_for_history = error_message
            finally:
                st.session_state.history.append({"role": "assistant", "content": final_cleaned_response_for_history})

# --- Instrucciones para ejecutar ---
# st.sidebar.info("""
# **Para ejecutar esta aplicación:**
# 1. Asegúrate de tener Streamlit instalado (`pip install streamlit`).
# 2. Abre una terminal.
# 3. Navega al directorio `sqlite-analyzer/src/`.
#    (Ej: `cd c:\Users\cpascual\PycharmProjects\pythonProject\cursos_actividades\sina_mcp\sqlite-analyzer\src\`)
# 4. Ejecuta: `streamlit run streamlit_interface.py`
# """)
