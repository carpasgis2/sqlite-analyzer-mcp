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
from openai import OpenAI, APIConnectionError, BadRequestError, APIStatusError # Asegurar que APIStatusError esté importada
import httpx
import tiktoken # Añadir tiktoken

# Intenta obtener la clave API de una variable de entorno
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-aedf531ee17447aa95c9102e595f29ae")
LLM_API_URL = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/v1") # Corregido: URL base sin /chat/completions
LLM_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")  # Corregido el nombre del modelo
LLM_PROVIDER = "deepseek"  # Identificador del proveedor de LLM este es el llm

# Configuración del logger
logger = logging.getLogger(__name__)

# Configuración del cliente LLM global (ahora para DeepSeek)
if not LLM_API_KEY: # Usar la variable LLM_API_KEY
    logger.error("La variable de entorno DEEPSEEK_API_KEY no está configurada.")
    # Podrías levantar un error aquí o tener un comportamiento de fallback
    # Por ahora, se procederá, pero el cliente fallará si se usa sin clave.
    llm_client = None 
else:
    llm_client = OpenAI(
        api_key=LLM_API_KEY, # Usar la variable LLM_API_KEY
        base_url=LLM_API_URL # Usar la variable LLM_API_URL
    )

LLM_MODEL_NAME = LLM_MODEL # Usar la variable LLM_MODEL

# Comentando la configuración anterior de OpenAI para evitar confusión
# OPENAI_API_KEY = "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" # Extraída de llm_test.py
# OPENAI_MODEL_NAME = "gpt-4.5-preview"
# openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Configuración del cliente Deepseek (comentado para usar OpenAI)
# DEEPSEEK_CLIENT = OpenAI(
# api_key=DEEPSEEK_API_KEY,
# base_url="https://api.deepseek.com/v1"
# )

# Modelos disponibles (comentado para usar OpenAI)
# AVAILABLE_MODELS = ["deepseek-chat", "deepseek-coder"]
# FALLBACK_MODELS = ["deepseek-coder"]


def count_tokens(text: str, model_name: str = LLM_MODEL_NAME) -> int:
    """
    Cuenta el número de tokens en un texto usando tiktoken.
    Se asume que los modelos compatibles con OpenAI (como DeepSeek) pueden usar encodings similares.
    """
    if not text:
        return 0
    try:
        # Intenta obtener el encoding para el modelo específico.
        # Para muchos modelos de OpenAI y compatibles, "cl100k_base" es un buen punto de partida.
        # Si DeepSeek tiene un encoding específico recomendado, debería usarse aquí.
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        logger.warning(f"No se encontró un encoding específico para el modelo {model_name}. Usando 'cl100k_base' como fallback.")
        encoding = tiktoken.get_encoding("cl100k_base")
    
    num_tokens = len(encoding.encode(text))
    logger.debug(f"Texto (primeros 50 chars): '{text[:50]}...' tiene {num_tokens} tokens para el modelo {model_name}.")
    return num_tokens

def extract_json_from_llm_response(response_text: str) -> Union[Dict[str, Any], List[Any], None]:
    """
    Extrae un objeto o una lista JSON de la respuesta del LLM.
    Maneja JSON envuelto en bloques de código Markdown.

    Args:
        response_text: Texto de respuesta del LLM.

    Returns:
        Diccionario o lista con el JSON extraído, o None en caso de error.
    """
    logger.debug(f"extract_json_from_llm_response: input repr='{repr(response_text)}', len={len(response_text if response_text else '')}")
    if not response_text:
        return None

    # 1. Intentar encontrar JSON dentro de bloques de código Markdown ```json ... ``` o ``` ... ```
    # Corregido: \\\\s* a \\s*
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text, re.DOTALL)
    
    json_str_to_parse = response_text 

    if match:
        logger.debug("extract_json_from_llm_response: Markdown block FOUND.")
        json_str_to_parse = match.group(1).strip()
    else:
        logger.debug("extract_json_from_llm_response: Markdown block NOT found by regex.")
    
    logger.debug(f"extract_json_from_llm_response: Attempting to parse (first 300 chars): '{json_str_to_parse[:300]}...'")

    try:
        if not json_str_to_parse:
            logger.debug("extract_json_from_llm_response: json_str_to_parse is empty, returning None.")
            return None 

        parsed_json = json.loads(json_str_to_parse)
        logger.debug("extract_json_from_llm_response: JSON parsed successfully.")
        return parsed_json
    except json.JSONDecodeError as e:
        # Log modificado para incluir el error específico de JSONDecodeError
        logger.warning(f"extract_json_from_llm_response: JSONDecodeError. Error: {e}. Content (first 200 chars): '{json_str_to_parse[:200]}...'")
        return None

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

def call_llm(config: dict, messages: list, model_name: str = LLM_MODEL_NAME, client=None, stream=False):
    """
    Llama al LLM especificado con la configuración y mensajes dados.
    Ahora utiliza el cliente LLM global por defecto (configurado para DeepSeek).
    """
    if client is None:
        client = llm_client # Usar el cliente LLM global por defecto

    if client is None:
        logger.error("El cliente LLM no está inicializado. Verifica la configuración de la API KEY.")
        raise ValueError("LLM client is not initialized. Check API KEY configuration.")

    max_tokens = config.get("max_tokens", 4096)
    temperature = config.get("temperature", 0.7)
    # top_p = config.get("top_p", 1.0) # OpenAI usa top_p, Deepseek podría usarlo diferente o no. Ajustar si es necesario.
    # frequency_penalty = config.get("frequency_penalty", 0) # Ajustar si es necesario
    # presence_penalty = config.get("presence_penalty", 0) # Ajustar si es necesario

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream
        )
        
        if stream:
            # Implementar lógica para manejar stream si es necesario
            # Por ahora, asumimos que no se usa stream o se maneja de forma síncrona para obtener el contenido completo
            # Ejemplo básico (puede necesitar ajustes para ensamblar chunks):
            # full_response_content = ""
            # for chunk in response:
            #     if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            #         full_response_content += chunk.choices[0].delta.content
            # return full_response_content.strip()
            # logger.warning("El manejo de stream no está completamente implementado para devolver contenido agregado en call_llm.")
            # return None # O manejar según la lógica de stream
            # Por ahora, si es stream, devolvemos un error o None, ya que el código principal espera una cadena.
            # O, si el stream ya fue consumido por el cliente de OpenAI y 'response' es el objeto final,
            # el siguiente bloque 'else' podría funcionar. Esto depende de cómo el SDK maneje el stream.
            # Para simplificar, asumiremos que si stream=True, esta función no es la adecuada o necesita
            # un manejo específico que no está aquí.
            # Dado que el código que llama no parece usar stream, nos enfocamos en el caso no-stream.
            # Si 'response' ya es el objeto completo después del stream:
            if hasattr(response, 'choices') and response.choices and response.choices[0].message and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                logger.error(f"Respuesta inesperada del LLM (stream) o contenido vacío. Respuesta completa: {response}")
                return None
        else:
            # Caso no-stream
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                content_to_return = response.choices[0].message.content.strip()
                logger.debug(f"Contenido crudo recibido del LLM: '{content_to_return}'") # REGISTRO AÑADIDO
                return content_to_return
            else:
                logger.error(f"Respuesta inesperada del LLM o contenido vacío. Respuesta completa: {response}")
                return None
    except APIStatusError as e:
        logger.error(f"Error de API (estado {e.status_code}) al llamar al LLM: {e.message}")
        logger.error(f"Cuerpo de la respuesta de error de la API: {e.response.text if e.response and hasattr(e.response, 'text') else 'No response body available'}")
        raise
    except APIConnectionError as e:
        logger.error(f"Error de conexión con la API: {e}")
        raise
    except BadRequestError as e:
        logger.error(f"Error de solicitud incorrecta (BadRequestError) al llamar al LLM: {e}")
        logger.error(f"Cuerpo de la respuesta (si disponible): {e.response.text if hasattr(e, 'response') and e.response and hasattr(e.response, 'text') else 'No response body available'}")
        raise
    except json.JSONDecodeError as e: # Capturar explícitamente si se propaga desde la librería cliente
        logger.error(f"Error al decodificar la respuesta JSON del LLM (probablemente la API no devolvió JSON): {e}")
        # 'e' ya contiene información sobre el error de decodificación.
        # No tenemos acceso directo al response.text aquí si el error ocurrió dentro de la librería.
        raise
    except httpx.ConnectTimeout as e:
        logger.error(f"Timeout al conectar con el API de LLM: {e}")
        raise
    except Exception as e: # Catch-all para otros errores inesperados
        logger.error(f"Error inesperado ({type(e).__name__}) al llamar al LLM: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            logger.error(f"Cuerpo de la respuesta (si disponible en excepción genérica): {e.response.text}")
        raise

def call_llm_with_fallbacks(config: dict, messages: list, model_name: str = LLM_MODEL_NAME, fallbacks: list = None):
    """
    Intenta llamar al LLM principal y recurre a los modelos de fallback en caso de error.
    Actualmente configurado para usar el cliente LLM global (DeepSeek) y no tiene fallbacks.
    """
    # El concepto de fallbacks a otros modelos de Deepseek ya no aplica directamente si solo usamos OpenAI.
    # Si se quisiera un fallback a otro modelo de OpenAI (ej. gpt-3.5-turbo), se podría implementar aquí.
    # Por ahora, se llamará directamente al modelo OpenAI principal.
    # current_fallbacks = fallbacks if fallbacks is not None else FALLBACK_MODELS
    
    try:
        # Asegurarse de que se pasa el nombre del modelo correcto
        return call_llm(config, messages, model_name=model_name, client=llm_client)
    except Exception as e:
        logger.warning(f"Error al llamar al modelo principal {model_name}: {e}. No hay fallbacks configurados.")
        # logger.warning(f"Error al llamar al modelo principal {model_name}: {e}. Intentando con fallbacks...")
        # for fallback_model in current_fallbacks:
        #     try:
        #         logger.info(f"Intentando con el modelo de fallback: {fallback_model}")
        #         return call_llm(config, messages, model_name=fallback_model, client=DEEPSEEK_CLIENT)
        #     except Exception as fallback_e:
        #         logger.error(f"Error al llamar al modelo de fallback {fallback_model}: {fallback_e}")
        # logger.error(f"Error al llamar al modelo {model_name} de OpenAI: {e}. No hay fallbacks configurados para OpenAI en esta función.")
        raise # Re-lanza la excepción si todos los intentos fallan (o no hay fallbacks)

def extract_info_from_question_llm(
    question: str, 
    db_schema_str_simple: str, # Cambiado de db_schema_str_full_details
    relaciones_tablas_str: str, 
    conversation_history: Optional[List[Dict[str, str]]] = None, # Añadido historial
    db_type: str = "sqlite", # Añadido tipo de BD
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Utiliza el LLM para extraer información estructurada de una pregunta del usuario,
    considerando el esquema de la base de datos y las relaciones entre tablas.

    Args:
        question: La pregunta del usuario.
        db_schema_str_simple: String con el esquema simplificado de la BD.
        relaciones_tablas_str: String con las relaciones entre tablas.
        conversation_history: Historial de la conversación (opcional).
        db_type: Tipo de base de datos (ej. "sqlite", "postgres").
        config: Configuración para la llamada al LLM.

    Returns:
        Un diccionario con la información estructurada extraída.
    """
    if config is None:
        config = {} 

    # Definir la plantilla del prompt del sistema como una cadena literal.
    # Los placeholders como {conversation_history}, {db_schema_subset_str}, etc., son literales.
    system_prompt_template = """Eres un asistente de IA experto en SQL y bases de datos médicas.\n\n\n***IMPORTANTE: SOLO puedes usar columnas y tablas que aparecen exactamente en el esquema proporcionado (`db_structure`). Si una columna o tabla no existe en el esquema, NO la inventes ni la infieras bajo ninguna circunstancia. Si la información no está disponible en el esquema, indícalo claramente en el JSON de salida. Si tienes dudas entre varias columnas, elige la que más se parezca y explica tu razonamiento. Si no encuentras la columna exacta pero hay una similar, usa la más parecida y justifica el cambio. Si no hay ninguna columna válida, indícalo y pide aclaración al usuario. Si la consulta no devuelve resultados, sugiere alternativas, revisa los criterios y pide al usuario que aclare o modifique su pregunta si es necesario.***\n\n\nTu tarea principal es analizar la pregunta del usuario y la conversación previa para extraer información estructurada en formato JSON que permita construir una consulta SQL válida y precisa, o generar directamente una consulta SQL si la pregunta es compleja o analítica.\nDebes utilizar el esquema de la base de datos proporcionada (`db_structure`) para identificar tablas y columnas relevantes.\nEl dialecto SQL es SQLite.\n\nConsideraciones importantes:
1.  **Identificación de Tablas y Columnas**:
    *   Usa `db_structure` para encontrar las tablas y columnas correctas. Presta atención a los nombres exactos.
    *   Si la pregunta es vaga, intenta inferir las entidades más probables.
    *   Utiliza los mapeos de `table_synonyms` y `column_synonyms` del `terms_dict` para relacionar términos del usuario con nombres de tablas/columnas.
    *   Si se mencionan varias entidades, considera si se necesitan JOINs. Las relaciones están en `table_relationships`.

2.  **Tipos de Preguntas**:
    *   **Preguntas directas sobre datos**: Extrae `tables`, `columns`, `conditions`, `joins`.
        Ejemplo: "¿Cuántos pacientes hombres mayores de 40 años tienen diagnóstico de diabetes?"
        JSON: {{ "tables": ["PATI_PATIENTS", "DIAG_DIAGNOSES"], "columns": ["COUNT(DISTINCT PATI_PATIENTS.PATI_ID)"], "conditions": [{{"column": "PATI_SEX", "operator": "=", "value": "Hombre"}}, {{"column": "PATI_AGE", "operator": ">", "value": 40}}, {{"column": "DIAG_DESCRIPTION", "operator": "LIKE", "value": "%diabetes%"}}], "joins": [{{"from_table": "PATI_PATIENTS", "to_table": "DIAG_DIAGNOSES", "on": "PATI_PATIENTS.PATI_ID = DIAG_DIAGNOSES.PATI_ID"}}], "query_type": "SELECT" }}
    *   **Preguntas de conteo**: Similar, pero la columna principal será `COUNT(*)`, `COUNT(DISTINCT column)`, etc.
    *   **Preguntas complejas/analíticas (GENERAR SQL DIRECTAMENTE)**: Si la pregunta requiere comparaciones complejas, cálculos agrupados, subconsultas, o es una pregunta de "comparar X vs Y", "evolución de Z", etc., es mejor que generes la consulta SQL directamente.
        *   En estos casos, el JSON de salida debe ser: `{{ \"direct_sql_query\": \"SELECT ... FROM ... WHERE ... GROUP BY ...;\" }}`
        *   **Uso de Alias en `GROUP BY` para SQLite**: SQLite no permite usar alias de la cláusula `SELECT` directamente en `GROUP BY`. Si necesitas agrupar por una expresión que tiene un alias en el `SELECT`, debes repetir la expresión completa en el `GROUP BY`.
            Ejemplo: `SELECT CASE WHEN PATI_AGE > 18 THEN \'Adulto\' ELSE \'Menor\' END AS GRUPO_EDAD, COUNT(*) FROM PATI_PATIENTS GROUP BY CASE WHEN PATI_AGE > 18 THEN \'Adulto\' ELSE \'Menor\' END;` (Correcto)
            NO: `SELECT CASE WHEN PATI_AGE > 18 THEN \'Adulto\' ELSE \'Menor\' END AS GRUPO_EDAD, COUNT(*) FROM PATI_PATIENTS GROUP BY GRUPO_EDAD;` (Incorrecto para SQLite)
        *   **Comparativas X vs Y**: Para preguntas como "duración promedio de hospitalización para pacientes con diagnóstico X vs. pacientes con diagnóstico Y", usa `CASE` para categorizar y luego agrupa.
            Ejemplo:
            ```sql
            SELECT
                CASE
                    WHEN d.DIAG_DESCRIPTION_ES LIKE \'%infarto%\' THEN \'Infarto\'
                    WHEN d.DIAG_DESCRIPTION_ES LIKE \'%neumonia%\' THEN \'Neumonía\'
                    ELSE \'Otro\'
                END AS CONDICION_PRINCIPAL,
                AVG(e.EPIS_DAYS_OF_STAY) AS PROMEDIO_DIAS_ESTANCIA
            FROM EPIS_EPISODES e
            JOIN DIAG_DIAGNOSES d ON e.DIAG_ID_ADMISSION = d.DIAG_ID
            WHERE CONDICION_PRINCIPAL IN (\'Infarto\', \'Neumonía\') -- Opcional, si solo quieres esos grupos
            GROUP BY CONDICION_PRINCIPAL;
            ```

3.  **Condiciones (`conditions`)**:
    *   Cada condición es un objeto: `{{"column": "COL_NAME", "operator": "=", "value": "valor"}}`.
    *   Operadores comunes: `=`, `>`, `<`, `>=`, `<=`, `LIKE`, `NOT LIKE`, `IN`, `NOT IN`, `IS NULL`, `IS NOT NULL`.
    *   Para `LIKE`, el valor ya debe incluir los `%` (ej., `\"%diabetes%\"`).
    *   Manejo de fechas: Si es posible, convierte fechas a formato YYYY-MM-DD.

4.  **Joins (`joins`)**:
    *   Cada join es un objeto: `{{"from_table": "TABLE_A", "to_table": "TABLE_B", "on": "TABLE_A.ID = TABLE_B.A_ID"}}`.
    *   Usa `table_relationships` para determinar las condiciones de JOIN correctas.

5.  **Pseudo-campos y Lógica Especial**:
    *   `diagnostico_termino`: Si el usuario menciona un diagnóstico (ej. "diabetes"), la condición debe ser `DIAG_DIAGNOSES.DIAG_DESCRIPTION_ES LIKE \'%diabetes%\'`.
    *   `procedimiento_termino`: Similar, para `PROC_PROCEDURES.PROC_DESCRIPTION_ES LIKE \'%bypass%\'`.
    *   `grupo_terapeutico_termino`: Para consultas sobre grupos terapéuticos de medicamentos (ej. "antibióticos"), la condición debe ser `MEDI_PHARMA_THERAPEUTIC_GROUPS.PHTH_DESCRIPTION_ES LIKE \'%antibiótico%\'`. La tabla principal probablemente será `MEDI_MEDICATIONS` unida a `MEDI_PHARMA_THERAPEUTIC_GROUPS`.
    *   `no_contiene_ingrediente_activo`: Si el usuario pide medicamentos que NO contengan un ingrediente activo (ej. "sin pseudoefedrina"), esto es más complejo.
        *   Si generas SQL directamente: Usa una subconsulta `NOT EXISTS` o `NOT IN` para excluir medicamentos que están vinculados a ese ingrediente activo en `MEDI_MEDI_ACTIVE_INGREDIENTS` y `MEDI_ACTIVE_INGREDIENTS`.
            Ejemplo para "medicamentos que no contienen pseudoefedrina":
            ```sql
            SELECT DISTINCT m.MEDI_DESCRIPTION_ES
            FROM MEDI_MEDICATIONS m
            WHERE NOT EXISTS (
                SELECT 1
                FROM MEDI_MEDI_ACTIVE_INGREDIENTS mai
                JOIN MEDI_ACTIVE_INGREDIENTS ai ON mai.ACIN_ID = ai.ACIN_ID
                WHERE mai.MEDI_ID = m.MEDI_ID AND ai.ACIN_DESCRIPTION_ES LIKE \'%pseudoefedrina%\'
            );
            ```
        *   Si generas JSON estructurado: Puedes definir una condición especial como `{{"pseudo_column": "no_contiene_ingrediente_activo", "value": "pseudoefedrina"}}`. El `sql_generator.py` deberá interpretar esto. (Este es el enfoque preferido si no generas SQL directa).
    *   `contiene_ingrediente_activo`: Similar al anterior, pero con `EXISTS` o `IN`.
        Ejemplo para "medicamentos que contienen amoxicilina":
        ```sql
        SELECT DISTINCT m.MEDI_DESCRIPTION_ES
        FROM MEDI_MEDICATIONS m
        WHERE EXISTS (
            SELECT 1
            FROM MEDI_MEDI_ACTIVE_INGREDIENTS mai
            JOIN MEDI_ACTIVE_INGREDIENTS ai ON mai.ACIN_ID = ai.ACIN_ID
            WHERE mai.MEDI_ID = m.MEDI_ID AND ai.ACIN_DESCRIPTION_ES LIKE \'%amoxicilina%\'
        );
        ```
    - Manejo de consultas sobre idoneidad por edad (ej. "aptos para niños", "pediátrico", "para menores de X años"):
        - PREGUNTA CLAVE: ¿Existe un campo estructurado y fiable en la BD para determinar esto (p.ej., una columna booleana `ES_PEDIATRICO`, o una columna de rango de edad específico según el esquema proporcionado)?
        - SI NO EXISTE TAL CAMPO FIABLE (que es el caso para determinar la idoneidad pediátrica general basada en descripciones textuales como `MEDI_DESCRIPTION_ES`):
            - NO intentes inferir la idoneidad pediátrica buscando palabras clave (como "niños", "infantil", "pediátrico") en campos de texto generales. Este enfoque es demasiado propenso a errores y puede llevar a conclusiones incorrectas.
            - Si estás generando una consulta SQL directamente: OMITE completamente cualquier filtro SQL relacionado con la idoneidad pediátrica basada en palabras clave en descripciones. Genera la consulta SQL solo con los otros criterios explícitos (p.ej., tipo de medicamento, ingredientes a excluir/incluir).
            - Si estás extrayendo información estructurada para que `sql_generator.py` construya la consulta: NO incluyas ninguna condición o filtro para la idoneidad pediátrica si se basa en búsquedas de palabras clave en texto libre.
            - La responsabilidad de comunicar al usuario la limitación de no poder filtrar fiablemente por este criterio recaerá en el módulo que genera la respuesta final en lenguaje natural. Tu tarea aquí es asegurar que la consulta a la BD no contenga este filtro impreciso.
        - SI EXISTIERA UN CAMPO FIABLE Y ESPECÍFICO PARA IDONEIDAD PEDIÁTRICA (revisar el esquema proporcionado): Entonces sí podrías usar ese campo para filtrar. (Pero para la pregunta actual sobre "aptos para niños menores de 18 años", asume que no existe un campo tan directo y preciso a menos que el esquema lo demuestre explícitamente).

6.  **Formato de Salida**:
    *   Siempre devuelve un único bloque de código JSON. No incluyas explicaciones fuera del JSON.
    *   Si la pregunta es ambigua o necesitas aclaraciones, puedes usar: `{{ "clarification_needed": "Tu pregunta es ambigua. ¿Podrías especificar X?" }}`
    *   Si la pregunta no parece relacionada con la base de datos médica: `{{ "out_of_scope": "Lo siento, solo puedo responder preguntas sobre la base de datos médica." }}`
    *   Si no puedes procesar la pregunta por alguna razón: `{{ "error": "No pude procesar tu pregunta." }}`

Contexto de la conversación (últimos mensajes):
{conversation_history}

Esquema de la BD (parcial o completo) (`db_structure`):
```json
{db_schema_subset_str}
```

Diccionario de Términos Enriquecido (`terms_dict` - extracto relevante si aplica):
Contiene `table_synonyms`, `column_synonyms`, `table_descriptions`, `column_descriptions`.
Ejemplo: `{{ \"table_synonyms\": {{\"pacientes\": \"PATI_PATIENTS\"}}, \"column_synonyms\": {{\"edad del paciente\": \"PATI_AGE\"}} }}`
```json
{terms_dict_subset_str}
```

Relaciones entre tablas (`table_relationships` - extracto relevante si aplica):
Ejemplo: `{{ \"PATI_PATIENTS\": {{\"DIAG_DIAGNOSES\": \"PATI_PATIENTS.PATI_ID = DIAG_DIAGNOSES.PATI_ID\"}} }}`
```json
{relationships_subset_str}
```
Analiza la siguiente pregunta del usuario y genera el JSON o SQL correspondiente.
Pregunta del usuario: \"{question}\"
""" # Fin de la plantilla system_prompt_template
    
    # Formatear el historial de conversación para incluirlo en el prompt del sistema
    conversation_history_str = ""
    if conversation_history:
        for entry in conversation_history:
            role_display = "User" if entry.get("role") == "user" else "Assistant"
            content = entry.get('content', '')
            # Asegurar que el contenido es una cadena
            if content is None: content = ""
            conversation_history_str += f"{role_display}: {content}\\n"
            
    if not conversation_history_str.strip(): # Usar strip para verificar si está realmente vacío
        conversation_history_str = "No hay conversación previa."

    # Valor para el placeholder {terms_dict_subset_str}
    # Idealmente, este diccionario/string debería pasarse a la función o generarse aquí si es necesario.
    # Por ahora, se usa un JSON vacío como string.
    terms_dict_placeholder_value = "{}" 

    # Formatear el prompt del sistema con los valores actuales
    # La variable db_type se recibe como parámetro pero no hay un placeholder {db_type} en la plantilla actual.
    # Si se necesitara, se añadiría a la plantilla y aquí.
    system_prompt_content = system_prompt_template.format(
        conversation_history=conversation_history_str,
        db_schema_subset_str=db_schema_str_simple,
        terms_dict_subset_str=terms_dict_placeholder_value,
        relationships_subset_str=relaciones_tablas_str,
        question=question
    )
    
    messages = []
    # Añadir historial de conversación a la lista de mensajes (como en el código original)
    if conversation_history:
        for entry in conversation_history:
            # Asegurarse de que cada entrada tiene 'role' y 'content'
            if "role" in entry and "content" in entry:
                 messages.append(entry)
            elif "type" in entry and "text" in entry: # Adaptar desde formato antiguo si es necesario
                 messages.append({"role": "user" if entry["type"] == "user" else "assistant", "content": entry["text"]})

    # Añadir el prompt del sistema formateado
    messages.append({"role": "system", "content": system_prompt_content})
    
    # La pregunta del usuario ya está incluida en system_prompt_content a través del placeholder {question}.
    # Por lo tanto, la siguiente línea que añade la pregunta de nuevo es redundante y se comenta.
    # messages.append({"role": "user", "content": f"Pregunta del usuario: {question}"})

    logging.info(f"[extract_info_from_question_llm] Enviando pregunta al LLM: {question}")
    # Log del prompt del sistema completo para depuración
    logging.debug(f"[extract_info_from_question_llm] System prompt para LLM:\\n{system_prompt_content}")
    
    llm_config = {
        "temperature": 0.0, 
        "max_tokens": 2000, # Aumentado para prompts más largos y JSONs complejos
        **config 
    }

    # response_text = call_llm_with_fallbacks(messages, llm_config, step_name="ExtractInfoFromQuestion") # Original
    # Asegurarse de que se usa el LLM_MODEL_NAME configurado (DeepSeek)
    response_text = call_llm_with_fallbacks(config=llm_config, messages=messages, model_name=LLM_MODEL_NAME) 
    
    logging.debug(f"[extract_info_from_question_llm] Respuesta cruda del LLM: {response_text}")

    extracted_json = extract_json_from_llm_response(response_text)

    if not extracted_json:
        logging.warning(f"[extract_info_from_question_llm] No se pudo extraer JSON de la respuesta del LLM: {response_text}")
        # ... (lógica de fallback existente, podría necesitar ajustes si el nuevo prompt es muy diferente)
        # Por ahora, se mantiene la lógica de fallback simple para tablas si la pregunta es compleja.
        if " versus " in question.lower() or " compar" in question.lower() or " desglosado por " in question.lower():
            logging.info(f"[extract_info_from_question_llm] Pregunta compleja detectada, intentando extracción de tablas simple como fallback.")
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
            config_simple_tables = {**llm_config, "max_tokens": 300} # Aumentado un poco por si acaso
            
            # response_simple_tables = call_llm_with_fallbacks(messages_simple_tables, config_simple_tables, step_name="ExtractTablesForComplexQueryFallback") # Original
            # Asegurarse de que se usa el LLM_MODEL_NAME configurado (DeepSeek)
            response_simple_tables = call_llm_with_fallbacks(config=config_simple_tables, messages=messages_simple_tables, model_name=LLM_MODEL_NAME)

            json_simple_tables = extract_json_from_llm_response(response_simple_tables)
            
            if json_simple_tables and "tables" in json_simple_tables:
                logging.info(f"[extract_info_from_question_llm] Tablas extraídas para pregunta compleja (fallback): {json_simple_tables['tables']}")
                # Devolver una estructura mínima para que el pipeline pueda intentar algo o fallar controladamente
                return {
                    "tables": json_simple_tables["tables"], 
                    "columns": ["*"], # Por defecto seleccionar todo de la primera tabla
                    "conditions": [], 
                    "joins": [],
                    "group_by": [],
                    "is_complex_fallback": True 
                }
        return {} # Devuelve diccionario vacío si no hay JSON y no es fallback complejo

    logging.info(f"[extract_info_from_question_llm] JSON extraído: {json.dumps(extracted_json, indent=2)}") # Log con indentación
    return extracted_json

# --- FIN CONFIGURACIÓN DE LOGGING ---

# Configuración del cliente OpenAI (reutiliza el existente si ya está configurado globalmente)
# ... (código de inicialización de openai_client existente) ...

# Definición de la nueva función
def create_prompt_for_table_identification(enriched_question: str, db_schema_str_simple: str, relaciones_tablas_str: str) -> list[dict[str, str]]:
    """
    Crea un prompt para que el LLM identifique las tablas relevantes para una pregunta dada.
    """
    logger.debug(f"Creando prompt para identificación de tablas. Pregunta: {enriched_question[:100]}...")

    system_message = f"""Eres un asistente experto en SQL y análisis de bases de datos.
Tu tarea es identificar las tablas de la base de datos que son más relevantes para responder la pregunta del usuario.
Considera el esquema de la base de datos y las relaciones entre tablas que te proporciono.

Esquema de la base de datos (simplificado):
{db_schema_str_simple}

Relaciones entre tablas:
{relaciones_tablas_str}

Analiza la siguiente pregunta del usuario y devuelve ÚNICAMENTE una lista de Python con los nombres de las tablas relevantes.
Por ejemplo: ["tabla1", "tabla2"]
Si no hay tablas claramente relevantes o la pregunta no parece relacionarse con la base de datos, devuelve una lista vacía: [].
No incluyas explicaciones adicionales, solo la lista de Python.
"""
    
    user_message = f"Pregunta del usuario: {enriched_question}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    logger.debug(f"Prompt para identificación de tablas creado: {messages}")
    return messages

def get_llm_config(
    temperature: float = 0.0,
    max_tokens: int = 2000, # Default from extract_info_from_question_llm
    top_p: Optional[float] = None, # OpenAI default is 1.0
    frequency_penalty: Optional[float] = None, # OpenAI default is 0
    presence_penalty: Optional[float] = None, # OpenAI default is 0
    extra_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Devuelve una configuración base para las llamadas al LLM.
    Permite la personalización a través de parámetros.
    """
    config = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if top_p is not None:
        config["top_p"] = top_p
    if frequency_penalty is not None:
        config["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        config["presence_penalty"] = presence_penalty
    
    if extra_params:
        config.update(extra_params)
        
    logger.debug(f"LLM config generated: {config}")
    return config

def create_prompt_for_structured_extraction(
    question_with_context: str,
    db_schema_str: str,
    relaciones_tablas_str: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    terms_dict_str: str = "{}", 
) -> List[Dict[str, str]]:
    """
    Crea un prompt para que el LLM extraiga información estructurada de una pregunta,
    considerando el esquema de la BD, relaciones, historial y diccionario de términos.
    """
    logger.debug(f"Creando prompt para extracción estructurada. Pregunta: {question_with_context[:100]}...")

    conversation_history_formatted_str = ""
    if conversation_history:
        for entry in conversation_history:
            role_display = "User" if entry.get("role") == "user" else "Assistant"
            content = entry.get('content', '')
            if content is None: content = ""
            # Escapar \'\'\' dentro del contenido del historial para no romper el f-string principal que usa f\'\'\'
            content_escaped = str(content).replace("'''", "\\'\\'\\'") # Corrected line
            conversation_history_formatted_str += f"{role_display}: {content_escaped}\\n"
    
    if not conversation_history_formatted_str.strip():
        conversation_history_formatted_str = "No hay conversación previa."

    # Usar f'''...''' para la cadena multilínea del sistema.
    # Las comillas triples internas deben escaparse como \\\'\\\'\\\' o usar un tipo de comilla diferente.
    # Los ejemplos de SQL y JSON usan comillas simples o dobles internamente, lo cual está bien.
    system_message_content = f'''Eres un asistente de IA experto en SQL y bases de datos médicas.
Tu tarea principal es analizar la pregunta del usuario y la conversación previa para extraer información estructurada en formato JSON que permita construir una consulta SQL válida y precisa, o generar directamente una consulta SQL si la pregunta es compleja o analítica.
Debes utilizar el esquema de la base de datos proporcionada (`db_structure`) para identificar tablas y columnas relevantes.
El dialecto SQL es SQLite.

Consideraciones importantes:
1.  **Identificación de Tablas y Columnas**:
    *   Usa `db_structure` para encontrar las tablas y columnas correctas. Presta atención a los nombres exactos.
    *   Si la pregunta es vaga, intenta inferir las entidades más probables.
    *   Utiliza los mapeos de `table_synonyms` y `column_synonyms` del `terms_dict` para relacionar términos del usuario con nombres de tablas/columnas.
    *   Si se mencionan varias entidades, considera si se necesitan JOINs. Las relaciones están en `table_relationships`.

2.  **Tipos de Preguntas**:
    *   **Preguntas directas sobre datos**: Extrae `tables`, `columns`, `conditions`, `joins`.
        Ejemplo: "¿Cuántos pacientes hombres mayores de 40 años tienen diagnóstico de diabetes?"
        JSON: {{ "tables": ["PATI_PATIENTS", "DIAG_DIAGNOSES"], "columns": ["COUNT(DISTINCT PATI_PATIENTS.PATI_ID)"], "conditions": [{{"column": "PATI_SEX", "operator": "=", "value": "Hombre"}}, {{"column": "PATI_AGE", "operator": ">", "value": 40}}, {{"column": "DIAG_DESCRIPTION", "operator": "LIKE", "value": "%diabetes%"}}], "joins": [{{"from_table": "PATI_PATIENTS", "to_table": "DIAG_DIAGNOSES", "on": "PATI_PATIENTS.PATI_ID = DIAG_DIAGNOSES.PATI_ID"}}], "query_type": "SELECT" }}
    *   **Preguntas de conteo**: Similar, pero la columna principal será `COUNT(*)`, `COUNT(DISTINCT column)`, etc.
    *   **Preguntas complejas/analíticas (GENERAR SQL DIRECTAMENTE)**: Si la pregunta requiere comparaciones complejas, cálculos agrupados, subconsultas, o es una pregunta de "comparar X vs Y", "evolución de Z", etc., es mejor que generes la consulta SQL directamente.
        *   En estos casos, el JSON de salida debe ser: `{{ \"direct_sql_query\": \"SELECT ... FROM ... WHERE ... GROUP BY ...;\" }}`
        *   **Uso de Alias en `GROUP BY` para SQLite**: SQLite no permite usar alias de la cláusula `SELECT` directamente en `GROUP BY`. Si necesitas agrupar por una expresión que tiene un alias en el `SELECT`, debes repetir la expresión completa en el `GROUP BY`.
            Ejemplo: `SELECT CASE WHEN PATI_AGE > 18 THEN \'Adulto\' ELSE \'Menor\' END AS GRUPO_EDAD, COUNT(*) FROM PATI_PATIENTS GROUP BY CASE WHEN PATI_AGE > 18 THEN \'Adulto\' ELSE \'Menor\' END;` (Correcto)
            NO: `SELECT CASE WHEN PATI_AGE > 18 THEN \'Adulto\' ELSE \'Menor\' END AS GRUPO_EDAD, COUNT(*) FROM PATI_PATIENTS GROUP BY GRUPO_EDAD;` (Incorrecto para SQLite)
        *   **Comparativas X vs Y**: Para preguntas como "duración promedio de hospitalización para pacientes con diagnóstico X vs. pacientes con diagnóstico Y", usa `CASE` para categorizar y luego agrupa.
            Ejemplo:
            ```sql
            SELECT
                CASE
                    WHEN d.DIAG_DESCRIPTION_ES LIKE \'%infarto%\' THEN \'Infarto\'
                    WHEN d.DIAG_DESCRIPTION_ES LIKE \'%neumonia%\' THEN \'Neumonía\'
                    ELSE \'Otro\'
                END AS CONDICION_PRINCIPAL,
                AVG(e.EPIS_DAYS_OF_STAY) AS PROMEDIO_DIAS_ESTANCIA
            FROM EPIS_EPISODES e
            JOIN DIAG_DIAGNOSES d ON e.DIAG_ID_ADMISSION = d.DIAG_ID
            WHERE CONDICION_PRINCIPAL IN (\'Infarto\', \'Neumonía\') -- Opcional, si solo quieres esos grupos
            GROUP BY CONDICION_PRINCIPAL;
            ```

3.  **Condiciones (`conditions`)**:
    *   Cada condición es un objeto: `{{"column": "COL_NAME", "operator": "=", "value": "valor"}}`.
    *   Operadores comunes: `=`, `>`, `<`, `>=`, `<=`, `LIKE`, `NOT LIKE`, `IN`, `NOT IN`, `IS NULL`, `IS NOT NULL`.
    *   Para `LIKE`, el valor ya debe incluir los `%` (ej., `\"%diabetes%\"`).
    *   Manejo de fechas: Si es posible, convierte fechas a formato YYYY-MM-DD.

4.  **Joins (`joins`)**:
    *   Cada join es un objeto: `{{"from_table": "TABLE_A", "to_table": "TABLE_B", "on": "TABLE_A.ID = TABLE_B.A_ID"}}`.
    *   Usa `table_relationships` para determinar las condiciones de JOIN correctas.

5.  **Pseudo-campos y Lógica Especial**:
    *   `diagnostico_termino`: Si el usuario menciona un diagnóstico (ej. "diabetes"), la condición debe ser `DIAG_DIAGNOSES.DIAG_DESCRIPTION_ES LIKE \'%diabetes%\'`.
    *   `procedimiento_termino`: Similar, para `PROC_PROCEDURES.PROC_DESCRIPTION_ES LIKE \'%bypass%\'`.
    *   `grupo_terapeutico_termino`: Para consultas sobre grupos terapéuticos de medicamentos (ej. "antibióticos"), la condición debe ser `MEDI_PHARMA_THERAPEUTIC_GROUPS.PHTH_DESCRIPTION_ES LIKE \'%antibiótico%\'`. La tabla principal probablemente será `MEDI_MEDICATIONS` unida a `MEDI_PHARMA_THERAPEUTIC_GROUPS`.
    *   `no_contiene_ingrediente_activo`: Si el usuario pide medicamentos que NO contengan un ingrediente activo (ej. "sin pseudoefedrina"), esto es más complejo.
        *   Si generas SQL directamente: Usa una subconsulta `NOT EXISTS` o `NOT IN` para excluir medicamentos que están vinculados a ese ingrediente activo en `MEDI_MEDI_ACTIVE_INGREDIENTS` y `MEDI_ACTIVE_INGREDIENTS`.
            Ejemplo para "medicamentos que no contienen pseudoefedrina":
            ```sql
            SELECT DISTINCT m.MEDI_DESCRIPTION_ES
            FROM MEDI_MEDICATIONS m
            WHERE NOT EXISTS (
                SELECT 1
                FROM MEDI_MEDI_ACTIVE_INGREDIENTS mai
                JOIN MEDI_ACTIVE_INGREDIENTS ai ON mai.ACIN_ID = ai.ACIN_ID
                WHERE mai.MEDI_ID = m.MEDI_ID AND ai.ACIN_DESCRIPTION_ES LIKE \'%pseudoefedrina%\'
            );
            ```
        *   Si generas JSON estructurado: Puedes definir una condición especial como `{{"pseudo_column": "no_contiene_ingrediente_activo", "value": "pseudoefedrina"}}`. El `sql_generator.py` deberá interpretar esto. (Este es el enfoque preferido si no generas SQL directa).
    *   `contiene_ingrediente_activo`: Similar al anterior, pero con `EXISTS` o `IN`.
        Ejemplo para "medicamentos que contienen amoxicilina":
        ```sql
        SELECT DISTINCT m.MEDI_DESCRIPTION_ES
        FROM MEDI_MEDICATIONS m
        WHERE EXISTS (
            SELECT 1
            FROM MEDI_MEDI_ACTIVE_INGREDIENTS mai
            JOIN MEDI_ACTIVE_INGREDIENTS ai ON mai.ACIN_ID = ai.ACIN_ID
            WHERE mai.MEDI_ID = m.MEDI_ID AND ai.ACIN_DESCRIPTION_ES LIKE \'%amoxicilina%\'
        );
        ```
    - Manejo de consultas sobre idoneidad por edad (ej. "aptos para niños", "pediátrico", "para menores de X años"):
        - PREGUNTA CLAVE: ¿Existe un campo estructurado y fiable en la BD para determinar esto (p.ej., una columna booleana `ES_PEDIATRICO`, o una columna de rango de edad específico según el esquema proporcionado)?
        - SI NO EXISTE TAL CAMPO FIABLE (que es el caso para determinar la idoneidad pediátrica general basada en descripciones textuales como `MEDI_DESCRIPTION_ES`):
            - NO intentes inferir la idoneidad pediátrica buscando palabras clave (como "niños", "infantil", "pediátrico") en campos de texto generales. Este enfoque es demasiado propenso a errores y puede llevar a conclusiones incorrectas.
            - Si estás generando una consulta SQL directamente: OMITE completamente cualquier filtro SQL relacionado con la idoneidad pediátrica basada en palabras clave en descripciones. Genera la consulta SQL solo con los otros criterios explícitos (p.ej., tipo de medicamento, ingredientes a excluir/incluir).
            - Si estás extrayendo información estructurada para que `sql_generator.py` construya la consulta: NO incluyas ninguna condición o filtro para la idoneidad pediátrica si se basa en búsquedas de palabras clave en texto libre.
            - La responsabilidad de comunicar al usuario la limitación de no poder filtrar fiablemente por este criterio recaerá en el módulo que genera la respuesta final en lenguaje natural. Tu tarea aquí es asegurar que la consulta a la BD no contenga este filtro impreciso.
        - SI EXISTIERA UN CAMPO FIABLE Y ESPECÍFICO PARA IDONEIDAD PEDIÁTRICA (revisar el esquema proporcionado): Entonces sí podrías usar ese campo para filtrar. (Pero para la pregunta actual sobre "aptos para niños menores de 18 años", asume que no existe un campo tan directo y preciso a menos que el esquema lo demuestre explícitamente).

6.  **Formato de Salida**:
    *   Siempre devuelve un único bloque de código JSON. No incluyas explicaciones fuera del JSON.
    *   Si la pregunta es ambigua o necesitas aclaraciones, puedes usar: `{{ "clarification_needed": "Tu pregunta es ambigua. ¿Podrías especificar X?" }}`
    *   Si la pregunta no parece relacionada con la base de datos médica: `{{ "out_of_scope": "Lo siento, solo puedo responder preguntas sobre la base de datos médica." }}`
    *   Si no puedes procesar la pregunta por alguna razón: `{{ "error": "No pude procesar tu pregunta." }}`

Esquema de la BD (parcial o completo) (`db_structure`):
```json
{db_schema_str}
```

Relaciones entre tablas (`table_relationships`):
```json
{relaciones_tablas_str}
```
Analiza la siguiente pregunta del usuario y genera el JSON o SQL correspondiente.
Pregunta del usuario: "{question_with_context}"
''' # Fin de system_message_content # Corrected closing delimiter and variable
    
    messages = [
        {"role": "system", "content": system_message_content}
        # La pregunta del usuario ya está embebida en el system_message_content
        # a través del placeholder {enriched_question} al final.
        # No es necesario añadir un mensaje de usuario separado con la misma pregunta.
    ]
    logger.debug(f"Prompt para extracción estructurada creado: {system_message_content[:500]}...") # Loguea una parte del prompt
    return messages

# Funciones existentes como get_llm_config, call_llm, call_llm_with_fallbacks, etc.
# ...