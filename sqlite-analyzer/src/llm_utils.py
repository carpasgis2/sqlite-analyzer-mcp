"""
Utilidades para comunicación con modelos de lenguaje (LLM)
Este módulo centraliza las funciones relacionadas con llamadas a APIs de LLM
"""
import os
import re
import time
import json
import requests
import logging
from typing import Dict, Any, List, Optional, Union, Tuple

def preserve_rag_context(messages: List[Dict[str, str]]) -> None:
    """
    Identifica y marca mensajes que contienen contexto RAG para evitar su truncamiento.
    Modifica la lista de mensajes in-place.
    
    Args:
        messages: Lista de mensajes en formato chat
    """
    for i, msg in enumerate(messages):
        content = msg.get('content', '')
        
        # Detectar si es contexto RAG (cualquier mensaje que contenga esquema o contexto de base de datos)
        is_rag_context = any([
            "Esquema de la base de datos:" in content,
            "schema_knowledge" in content,
            "table_corpus" in content,
            "TABLAS EXPLÍCITAMENTE IDENTIFICADAS" in content
        ])
        
        if is_rag_context:
            # Añadir marca especial para no truncar este mensaje
            if 'metadata' not in msg:
                msg['metadata'] = {}
            msg['metadata']['preserve_full'] = True
            
            # Opcional: registrar en el log que este mensaje se preservará completo
            logging.debug(f"Mensaje {i+1} marcado como contexto RAG - no será truncado")

def log_detailed_messages(messages: List[Dict[str, str]], step_name: str) -> None:
    """
    Registra información detallada sobre los mensajes que se enviarán al LLM.
    
    Args:
        messages: Lista de mensajes en formato chat
        step_name: Nombre del paso para el logging
    """
    total_chars = sum(len(msg.get('content', '')) for msg in messages)
    logging.debug(f"[{step_name}] DETALLE DE MENSAJES A ENVIAR:")
    logging.debug(f"[{step_name}] Total mensajes: {len(messages)}, Total caracteres: {total_chars}")
    
    for i, msg in enumerate(messages):
        content = msg.get('content', '')
        role = msg.get('role', 'unknown')
        is_preserved = msg.get('metadata', {}).get('preserve_full', False)
        
        # Para mensajes muy largos, mostrar solo una parte
        if len(content) > 200 and not logging.getLogger().level <= logging.NOTSET:
            preview = f"{content[:100]}...{content[-100:]}"
            logging.debug(f"[{step_name}] Mensaje {i+1} ({role}): {len(content)} chars "
                         f"{'[PRESERVADO]' if is_preserved else ''}\n"
                         f"INICIO: {content[:100]}...\n"
                         f"...FIN: {content[-100:]}")
        else:
            logging.debug(f"[{step_name}] Mensaje {i+1} ({role}): {content}")

def call_llm(messages: List[Dict[str, str]], config: Dict[str, Any], step_name: str = "LLM Call") -> str:
    """
    Llama exclusivamente al LLM de Deepseek.
    
    Args:
        messages: Lista de mensajes en formato chat para el LLM
        config: Configuración para la llamada (temperatura, tokens, etc)
        step_name: Nombre del paso para el logging
        
    Returns:
        Contenido de la respuesta del LLM como string
    """
    # Configuración de Deepseek
    max_retries = config.get("max_retries", 2)
    api_key = os.environ.get("DEEPSEEK_API_KEY", "") or config.get("llm_api_key", "")
    api_url = os.environ.get("DEEPSEEK_API_URL", config.get("llm_api_url", "https://api.deepseek.com/v1/chat/completions"))
    model = os.environ.get("DEEPSEEK_MODEL", config.get("llm_model", "deepseek-chat"))
    
    # Verificar que tenemos una API key
    if not api_key:
        logging.error(f"[{step_name}] No hay API key para Deepseek")
        return "ERROR: API key no configurada"
    
    # NUEVO: Identificar y preservar contexto RAG en los mensajes
    preserve_rag_context(messages)
    
    # NUEVO: Log detallado del contenido de los mensajes (solo en modo DEBUG)
    if logging.getLogger().level <= logging.DEBUG:
        log_detailed_messages(messages, step_name)
    
    # Preparar headers y payload
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    # Obtener parámetros de configuración o usar valores predeterminados
    temperature = float(os.environ.get("TEMPERATURE", config.get("temperature", 0.1)))
    max_tokens = int(os.environ.get("MAX_TOKENS", config.get("max_tokens", 500)))
    
    payload = {
        "model": model, 
        "messages": messages, 
        "temperature": temperature, 
        "max_tokens": max_tokens
    }

    # Registrar inicio de la llamada
    logging.info(f"[{step_name}] Llamando a Deepseek con modelo {model}")
    start = time.time()
    
    # Intentar la llamada con reintentos si falla
    for attempt in range(1, max_retries+1):
        try:
            logging.info(f"[{step_name}] Deepseek intento {attempt}/{max_retries}")
            r = requests.post(api_url, headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            body = r.json()
            
            if "choices" in body and body["choices"]:
                content = body["choices"][0]["message"]["content"]
                elapsed = time.time() - start
                logging.info(f"[{step_name}] Respuesta Deepseek recibida en {elapsed:.2f}s")
                return content
            
            logging.warning(f"[{step_name}] Deepseek devolvió formato inesperado: {body}")
        except requests.exceptions.RequestException as ex:
            logging.error(f"[{step_name}] Error en solicitud HTTP: {ex}")
        except json.JSONDecodeError:
            logging.error(f"[{step_name}] Error al decodificar respuesta JSON")
        except Exception as ex:
            logging.error(f"[{step_name}] Error inesperado: {ex}")
        
        # Esperar antes de reintentar (estrategia de backoff exponencial)
        if attempt < max_retries:
            wait_time = 2 ** attempt
            logging.info(f"[{step_name}] Esperando {wait_time}s antes de reintentar...")
            time.sleep(wait_time)

    logging.error(f"[{step_name}] Todos los intentos de Deepseek fallaron")
    return "ERROR: Llamada a LLM fallida después de múltiples intentos"

def call_llm_with_fallbacks(messages: List[Dict[str, str]], config: Dict[str, Any], step_name: str = "LLM Call") -> str:
    """
    Llama al LLM con un sistema de fallbacks automáticos.
    Esta función es un wrapper sobre call_llm que añade manejo adicional de errores.
    
    Args:
        messages: Lista de mensajes en formato chat
        config: Configuración para la llamada
        step_name: Nombre del paso para logging
        
    Returns:
        Contenido de la respuesta del LLM como string
    """
    try:
        return call_llm(messages, config, step_name)
    except Exception as e:
        logging.warning(f"[{step_name}] Error en llamada al LLM: {e}")
        
        # Configuración de respaldo con temperatura más baja (más determinístico)
        fallback_config = config.copy()
        fallback_config["temperature"] = 0.0
        fallback_config["max_tokens"] = 300  # Respuesta más corta para evitar problemas
        
        try:
            logging.info(f"[{step_name}] Intentando con configuración de respaldo...")
            return call_llm(messages, fallback_config, f"{step_name} (fallback)")
        except Exception as e2:
            logging.error(f"[{step_name}] Error en llamada de respaldo: {e2}")
            return f"Error: No se pudo completar la operación LLM después de intentos adicionales."

def extract_json_from_llm_response(response: str) -> Dict[str, Any]:
    """
    Extrae un objeto JSON de la respuesta del LLM.
    
    Args:
        response: Texto de respuesta del LLM
        
    Returns:
        Diccionario con el JSON extraído o diccionario vacío en caso de error
    """
    # Buscar bloques de código JSON con markdown
    json_block = None
    json_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", response, re.IGNORECASE)
    
    if json_blocks:
        for block in json_blocks:
            try:
                # Intentar parsear como JSON
                parsed = json.loads(block)
                if isinstance(parsed, dict):  # Verificar que sea un objeto JSON válido
                    json_block = parsed
                    break
            except json.JSONDecodeError:
                continue
    
    # Si no se encontró un bloque JSON, buscar directamente en el texto
    if not json_block:
        # Buscar texto que parezca JSON (entre llaves { })
        matches = re.findall(r"(\{[\s\S]*\})", response)
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict):
                    json_block = parsed
                    break
            except:
                continue
    
    # Si no se encontró JSON válido, devolver diccionario vacío
    return json_block or {}

def extract_info_from_question_llm(question: str, db_schema_str_full_details: str, db_schema_str_simple: str, relaciones_tablas_str: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Utiliza el LLM para extraer información estructurada de una pregunta del usuario,
    considerando el esquema de la base de datos y las relaciones entre tablas.

    Args:
        question: La pregunta del usuario.
        db_schema_str_full_details: String con el esquema detallado de la BD. (No se usa directamente en el prompt final)
        db_schema_str_simple: String con el esquema simplificado de la BD.
        relaciones_tablas_str: String con las relaciones entre tablas.
        config: Configuración para la llamada al LLM.

    Returns:
        Un diccionario con la información estructurada extraída.
        Ejemplo: {'tables': ['PATI_PATIENT_ALLERGIES', 'ALLE_ALLERGY_TYPES'], 'columns': ['ALLE_ALLERGY_TYPES.ALTY_DESCRIPTION_ES'], 'conditions': [{'column': 'PATI_PATIENT_ALLERGIES.PATI_ID', 'operator': '=', 'value': '1931'}], 'joins': [{'type': 'INNER', 'table1': 'PATI_PATIENT_ALLERGIES', 'table2': 'ALLE_ALLERGY_TYPES', 'on': 'PATI_PATIENT_ALLERGIES.ALTY_ID = ALLE_ALLERGY_TYPES.ALTY_ID'}]}
    """
    if config is None:
        config = {} # Usar configuración por defecto de call_llm_with_fallbacks

    system_message_template = """
Eres un asistente experto en SQL y bases de datos médicas. Tu tarea es analizar la pregunta del usuario
y extraer la siguiente información en formato JSON:
1.  `tables`: Lista de tablas principales necesarias para responder la pregunta. Identifica TODAS las tablas implicadas por las condiciones y columnas solicitadas. La primera tabla de esta lista será la tabla principal en la cláusula FROM.
2.  `columns`: Lista de columnas que se deben seleccionar. Por defecto, selecciona todas las columnas (`["*"]`) de la tabla principal si no se especifican columnas. Si se mencionan explícitamente, inclúyelas con el nombre de la tabla (ej: "TABLE.COLUMN").
3.  `conditions`: Lista de condiciones para la cláusula WHERE. Cada condición debe ser un diccionario con "column", "operator" y "value".
    -   Identifica la columna correcta para cada parte de la pregunta. Por ejemplo, si la pregunta es "Pacientes con síntoma X y diagnóstico Y", X podría estar en `EPIS_DIAGNOSTICS.DIAG_OBSERVATION` y Y podría ser un código en `EPIS_DIAGNOSTICS.CDTE_ID` o también texto en `EPIS_DIAGNOSTICS.DIAG_OBSERVATION`.
    -   Para frases como "reportaron 'síntoma A'", la condición podría ser `{{"column": "TRIA_SYMPTOMS.SYMP_DESCRIPTION", "operator": "LIKE", "value": "%síntoma A%"}}` (asumiendo que los síntomas están en TRIA_SYMPTOMS).
    -   Para frases como "diagnosticados con 'enfermedad B'", si 'enfermedad B' es un texto descriptivo, considera usar la columna de observaciones o notas (ej: `{{"column": "EPIS_DIAGNOSTICS.DIAG_OBSERVATION", "operator": "LIKE", "value": "%enfermedad B%"}}`). Si 'enfermedad B' es un código de diagnóstico, utiliza la columna correspondiente (ej: `{{"column": "EPIS_DIAGNOSTICS.CDTE_ID", "operator": "=", "value": "CODIGO_ENFERMEDAD"}}`).
    -   Extrae valores directamente de la pregunta (ej: PATI_ID = 1931 de "¿Qué alergias tiene el paciente 1931?").
4.  `joins`: Lista de JOINs necesarios para conectar las tablas.
    -   DEBES incluir un JOIN si las columnas en `columns` o `conditions` pertenecen a tablas diferentes de la primera tabla listada en `tables` (la tabla principal del FROM).
    -   DEBES incluir JOINs para conectar cualquier tabla adicional listada en `tables` (después de la primera) con la tabla principal o con otras tablas ya unidas.
    -   Utiliza el esquema y las relaciones proporcionadas para determinar las condiciones ON correctas.
    -   Cada JOIN debe ser un diccionario con "type" (INNER, LEFT, etc.), "table1" (la tabla ya presente en el FROM o unida previamente), "table2" (la nueva tabla a unir), y "on" (la condición del join completa, ej: "TABLE1.ID_RELACION = TABLE2.ID_RELACION").

Contexto de la Base de Datos:
Esquema Simplificado:
{0}

Relaciones entre Tablas:
{1}

Ejemplo de pregunta 1: "Mostrar los síntomas reportados por pacientes diagnosticados con 'gripe común'."
Asumir que los síntomas están en `TRIA_SYMPTOMS (SYMP_DESCRIPTION)` y los diagnósticos en `EPIS_DIAGNOSTICS (DIAG_OBSERVATION, CDTE_ID)`. Ambas tablas pueden tener `EPIS_ID` y/o `PATI_ID`. `PATI_ID` también está en `PATI_PATIENTS`.
Posible JSON (si `TRIA_SYMPTOMS` es la tabla principal y se une con `EPIS_DIAGNOSTICS`):
{{
  "tables": ["TRIA_SYMPTOMS", "EPIS_DIAGNOSTICS"],
  "columns": ["TRIA_SYMPTOMS.SYMP_DESCRIPTION"],
  "conditions": [
    {{"column": "EPIS_DIAGNOSTICS.DIAG_OBSERVATION", "operator": "LIKE", "value": "%gripe común%"}}
  ],
  "joins": [
    {{
      "type": "INNER",
      "table1": "TRIA_SYMPTOMS",
      "table2": "EPIS_DIAGNOSTICS",
      "on": "TRIA_SYMPTOMS.EPIS_ID = EPIS_DIAGNOSTICS.EPIS_ID"
    }}
  ]
}}

Ejemplo de pregunta 2: "Pacientes con nombre 'Juan Pérez' que reportaron 'fuerte dolor de cabeza' como síntoma y fueron diagnosticados con 'migraña crónica'."
Tablas involucradas: `PATI_PATIENTS` (para nombre), `TRIA_SYMPTOMS` (para síntomas), `EPIS_DIAGNOSTICS` (para diagnósticos).
Asumir relaciones: `PATI_PATIENTS.PATI_ID = TRIA_SYMPTOMS.PATI_ID` (si síntomas tiene PATI_ID) o `PATI_PATIENTS.PATI_ID = EPIS_DIAGNOSTICS.PATI_ID` y luego `TRIA_SYMPTOMS` se une a `EPIS_DIAGNOSTICS` por `EPIS_ID`.
Posible JSON (usando `PATI_PATIENTS` como tabla principal):
{{
  "tables": ["PATI_PATIENTS", "TRIA_SYMPTOMS", "EPIS_DIAGNOSTICS"],
  "columns": ["PATI_PATIENTS.PATI_ID", "PATI_PATIENTS.PATI_NAME", "TRIA_SYMPTOMS.SYMP_DESCRIPTION", "EPIS_DIAGNOSTICS.DIAG_OBSERVATION"],
  "conditions": [
    {{"column": "PATI_PATIENTS.PATI_NAME", "operator": "LIKE", "value": "%Juan Pérez%"}},
    {{"column": "TRIA_SYMPTOMS.SYMP_DESCRIPTION", "operator": "LIKE", "value": "%fuerte dolor de cabeza%"}},
    {{"column": "EPIS_DIAGNOSTICS.DIAG_OBSERVATION", "operator": "LIKE", "value": "%migraña crónica%"}}
  ],
  "joins": [
    {{
      "type": "INNER",
      "table1": "PATI_PATIENTS",
      "table2": "EPIS_DIAGNOSTICS",
      "on": "PATI_PATIENTS.PATI_ID = EPIS_DIAGNOSTICS.PATI_ID"
    }},
    {{
      "type": "INNER",
      "table1": "EPIS_DIAGNOSTICS", 
      "table2": "TRIA_SYMPTOMS",
      "on": "EPIS_DIAGNOSTICS.EPIS_ID = TRIA_SYMPTOMS.EPIS_ID"
    }}
  ]
}}

Responde ÚNICAMENTE con el objeto JSON. No incluyas explicaciones adicionales.
"""
    system_message = system_message_template.format(db_schema_str_simple, relaciones_tablas_str)

    user_message = f"Pregunta del usuario: {question}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    logging.info(f"[extract_info_from_question_llm] Enviando pregunta al LLM: {question}")
    
    llm_config = {
        "temperature": 0.0, 
        "max_tokens": 1500, 
        **config
    }

    response_text = call_llm_with_fallbacks(messages, llm_config, step_name="ExtractInfoFromQuestion")
    
    logging.debug(f"[extract_info_from_question_llm] Respuesta cruda del LLM: {response_text}")

    extracted_json = extract_json_from_llm_response(response_text)

    if not extracted_json:
        logging.warning(f"[extract_info_from_question_llm] No se pudo extraer JSON de la respuesta del LLM: {response_text}")
        # Fallback: Si no hay JSON, intentar al menos obtener las tablas si la pregunta es compleja
        if " versus " in question.lower() or " compar" in question.lower() or " desglosado por " in question.lower():
            logging.info(f"[extract_info_from_question_llm] Pregunta compleja detectada, intentando extracción de tablas simple.")
            # Re-llamar al LLM con un prompt más simple solo para tablas
            simple_table_prompt = f'''Dada la pregunta del usuario, identifica las tablas principales de la base de datos que serían necesarias para responderla.
Pregunta: {question}
Esquema Simplificado:
{db_schema_str_simple}
Relaciones entre Tablas:
{relaciones_tablas_str}
Responde ÚNICAMENTE con un objeto JSON que contenga una clave "tables" con la lista de nombres de tablas. Ejemplo: {{"tables": ["TABLE_X", "TABLE_Y"]}}'''
            
            messages_simple_tables = [
                {"role": "system", "content": "Eres un asistente experto en SQL y bases de datos médicas."},
                {"role": "user", "content": simple_table_prompt}
            ]
            config_simple_tables = {**llm_config, "max_tokens": 200}
            
            response_simple_tables = call_llm_with_fallbacks(messages_simple_tables, config_simple_tables, step_name="ExtractTablesForComplexQuery")
            json_simple_tables = extract_json_from_llm_response(response_simple_tables)
            
            if json_simple_tables and "tables" in json_simple_tables:
                logging.info(f"[extract_info_from_question_llm] Tablas extraídas para pregunta compleja (fallback): {json_simple_tables['tables']}")
                return {"tables": json_simple_tables["tables"], "columns": [], "conditions": [], "joins": [], "is_complex_fallback": True} # Marcar que es un fallback
        return {}

    logging.info(f"[extract_info_from_question_llm] JSON extraído: {extracted_json}")
    return extracted_json