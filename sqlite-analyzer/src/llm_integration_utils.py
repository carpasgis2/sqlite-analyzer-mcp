\
import json
import logging
from typing import List, Dict, Any, Optional
import os
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

# Configuración de la API de Deepseek (preferiblemente desde variables de entorno)
LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-aedf531ee17447aa95c9102e595f29ae")
LLM_API_URL = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
LLM_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
LLM_PROVIDER = "deepseek"

# Configuración simulada del LLM (puede usarse para parámetros como temperature, max_tokens si no se especifican de otra forma)
LLM_CONFIGS = {
    "default": {"model": LLM_MODEL, "temperature": 0.5, "max_tokens": 1500},
    "table_identification": {"model": LLM_MODEL, "temperature": 0.2, "max_tokens": 500, "stop_sequences": ["\\n\\n"]},
    "structured_extraction": {"model": LLM_MODEL, "temperature": 0.7, "max_tokens": 2000},
}

def get_llm_config(config_type: str) -> dict:
    """
    Devuelve la configuración para un tipo específico de llamada al LLM,
    asegurándose de que el modelo sea el configurado globalmente.
    """
    config = LLM_CONFIGS.get(config_type, LLM_CONFIGS["default"]).copy() # Copiar para no modificar el original
    config["model"] = LLM_MODEL # Asegurar que el modelo es el de Deepseek
    return config

def create_prompt_for_table_identification(
    enriched_question: str, 
    db_schema_str_simple: str, 
    relaciones_tablas_str: str
) -> List[Dict[str, str]]:
    """
    Crea un prompt para que el LLM identifique las tablas relevantes.
    El prompt instruye al LLM para que devuelva una lista JSON de nombres de tablas.
    """
    prompt_system = f"""Eres un asistente experto en SQL y análisis de bases de datos.
Tu tarea es identificar las tablas de la base de datos que son relevantes para responder la pregunta del usuario.
Considera el esquema de la base de datos y las relaciones entre tablas proporcionadas.

Esquema de la base de datos (simplificado):
{db_schema_str_simple}

Relaciones entre tablas:
{relaciones_tablas_str}

Analiza la siguiente pregunta del usuario y devuelve ÚNICAMENTE una lista JSON con los nombres de las tablas relevantes.
Por ejemplo: ["TABLA1", "TABLA2"]
Si no hay tablas claramente relevantes o la pregunta no parece relacionarse con la base de datos, devuelve una lista vacía [].
"""
    prompt_user = f"Pregunta del usuario: \"{enriched_question}\"\n\nTablas relevantes (lista JSON):"
    
    return [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt_user}
    ]

def create_prompt_for_structured_extraction(
    enriched_question: str, 
    db_schema_str: str, 
    relaciones_tablas_str: str,
    identified_tables: Optional[List[str]] = None
) -> List[Dict[str, str]]:
    """
    Crea un prompt para que el LLM extraiga información estructurada para construir una consulta SQL.
    El prompt instruye al LLM para que devuelva un objeto JSON con claves como:
    "tables", "columns", "conditions", "joins", "query_type", "is_complex_query".
    """
    
    tables_hint = ""
    if identified_tables:
        tables_hint = f"Considera prioritariamente las siguientes tablas ya identificadas como potencialmente relevantes: {json.dumps(identified_tables)}. Si necesitas otras, puedes incluirlas."

    prompt_system = f"""Eres un asistente experto en SQL y análisis de bases de datos.
Tu tarea es analizar la pregunta de un usuario y extraer la información necesaria para construir una consulta SQL.
Proporciona la información en formato JSON.

Esquema de la base de datos:
{db_schema_str}

Relaciones entre tablas (esto te ayudará a definir los JOINs):
{relaciones_tablas_str}

{tables_hint}

Analiza la pregunta del usuario y devuelve un objeto JSON con la siguiente estructura:
{{
  "tables": ["LISTA_DE_TABLAS_RELEVANTES"],
  "columns": ["LISTA_DE_COLUMNAS_A_SELECCIONAR_O_FILTRAR"],
  "conditions": [
    {{
      "column": "NOMBRE_COLUMNA",
      "operator": "OPERADOR_SQL (ej: '=', '>', '<', 'LIKE', 'IN')",
      "value": "VALOR_CONDICION (puede ser string, número, o lista para IN)",
      "logical_operator_to_next": "AND/OR (opcional, para unir con la siguiente condición)"
    }}
  ],
  "joins": [
    {{
      "type": "TIPO_DE_JOIN (ej: INNER JOIN, LEFT JOIN)",
      "from_table": "TABLA_ORIGEN",
      "to_table": "TABLA_DESTINO",
      "on_condition": "CONDICION_DEL_JOIN (ej: 'TABLA_ORIGEN.ID = TABLA_DESTINO.FK_ID')"
    }}
  ],
  "query_type": "TIPO_DE_CONSULTA (ej: SELECT, COUNT, SUM, AVG, o pregunta general que no requiere SQL directo)",
  "is_complex_query": true/false (indica si la pregunta requiere múltiples pasos, subconsultas, o lógica compleja más allá de un SELECT simple),
  "explanation": "Breve explicación de tu razonamiento."
}}

Si una sección no es aplicable (ej. no hay joins), puedes omitirla o usar una lista/valor vacío apropiado.
Presta mucha atención a las entidades mencionadas en la pregunta y cómo se mapean al esquema.
Para las condiciones, intenta extraer el valor exacto. Si es una comparación de fechas, usa el formato 'YYYY-MM-DD'.
Si la pregunta es muy general o no parece una consulta a la base de datos, ajústalo en "query_type" y "is_complex_query".
"""
    prompt_user = f"Pregunta del usuario: \"{enriched_question}\"\n\nInformación estructurada (JSON):"

    return [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt_user}
    ]

def call_llm_with_fallbacks(messages: List[Dict[str, str]], llm_config: dict) -> Dict[str, Any]:
    """
    Realiza una llamada a la API del LLM de DeepSeek.
    """
    logger.info(f"DEBUG: [llm_integration_utils.py] Realizando llamada LLM ({LLM_PROVIDER}) con config: {llm_config}")
    logger.info(f"DEBUG: [llm_integration_utils.py] Mensajes para LLM (último mensaje): {messages[-1]['content'][:200] if messages else 'No messages'}")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}"
    }

    # Parámetros a tomar de llm_config, con valores por defecto si no están.
    temperature = llm_config.get("temperature", 0.5)
    max_tokens = llm_config.get("max_tokens", 1500)
    # Deepseek puede no usar 'stop_sequences', verificar su documentación. Si se usa, debe llamarse 'stop'.
    stop = llm_config.get("stop_sequences") # El API de Deepseek espera "stop" como string o array de strings

    payload = {
        "model": llm_config.get("model", LLM_MODEL), # Usar el modelo de la config, o el global
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if stop:
        payload["stop"] = stop
    
    # Eliminar claves con valor None para no enviarlas si no son necesarias
    payload = {k: v for k, v in payload.items() if v is not None}

    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(LLM_API_URL, data=data, headers=headers, method='POST')

    try:
        with urllib.request.urlopen(req, timeout=30) as response: # Añadido timeout
            response_body = response.read().decode('utf-8')
            response_json = json.loads(response_body)
            
            logger.info(f"DEBUG: [llm_integration_utils.py] Respuesta cruda del LLM: {response_body[:500]}") # Loguear parte de la respuesta

            if response.status == 200:
                # La estructura de respuesta de Deepseek es similar a la de OpenAI:
                # response_json['choices'][0]['message']['content']
                if response_json.get("choices") and len(response_json["choices"]) > 0:
                    content = response_json["choices"][0].get("message", {}).get("content")
                    if content:
                        return {"status": "success", "response_text": content, "raw_response": response_json}
                    else:
                        logger.error(f"DEBUG: [llm_integration_utils.py] LLM no devolvió contenido en la respuesta: {response_json}")
                        return {"status": "error", "message": "LLM response did not contain content.", "raw_response": response_json}
                else:
                    logger.error(f"DEBUG: [llm_integration_utils.py] LLM no devolvió 'choices' en la respuesta: {response_json}")
                    return {"status": "error", "message": "LLM response did not contain 'choices'.", "raw_response": response_json}
            else: # Aunque urlopen suele lanzar error para status != 2xx, por si acaso
                logger.error(f"DEBUG: [llm_integration_utils.py] Error en llamada LLM. Status: {response.status}, Body: {response_body}")
                return {"status": "error", "message": f"LLM API request failed with status {response.status}: {response_body}", "raw_response": response_json if response_body else None}

    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8') if e.fp else str(e)
        logger.error(f"DEBUG: [llm_integration_utils.py] HTTPError durante la llamada al LLM: {e.code} - {error_body}", exc_info=True)
        return {"status": "error", "message": f"HTTPError {e.code}: {error_body}", "raw_response": {"error_details": error_body}}
    except urllib.error.URLError as e:
        logger.error(f"DEBUG: [llm_integration_utils.py] URLError durante la llamada al LLM: {e.reason}", exc_info=True)
        return {"status": "error", "message": f"URLError: {e.reason}", "raw_response": None}
    except json.JSONDecodeError as e:
        logger.error(f"DEBUG: [llm_integration_utils.py] Error al decodificar JSON de la respuesta del LLM: {e}. Respuesta: {response_body if 'response_body' in locals() else 'No response body'}", exc_info=True)
        return {"status": "error", "message": f"JSONDecodeError: {str(e)}", "raw_response": {"response_text": response_body if 'response_body' in locals() else 'No response body'}}
    except Exception as e:
        logger.error(f"DEBUG: [llm_integration_utils.py] Excepción inesperada durante la llamada al LLM: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "raw_response": None}

def parse_llm_table_response(llm_response: Dict[str, Any]) -> Optional[List[str]]:
    """
    Parsea la respuesta del LLM (esperada en formato JSON string) para extraer una lista de tablas.
    """
    if llm_response.get("status") == "success" and llm_response.get("response_text"):
        try:
            # Asumimos que response_text es un string JSON que representa una lista de tablas.
            tables = json.loads(llm_response["response_text"])
            if isinstance(tables, list) and all(isinstance(table, str) for table in tables):
                return tables
            else:
                logger.warning(f"DEBUG: [llm_integration_utils.py] LLM devolvió una lista JSON pero no es una lista de strings: {tables}")
                return None
        except json.JSONDecodeError as e:
            logger.error(f"DEBUG: [llm_integration_utils.py] Error al decodificar JSON de la respuesta del LLM para tablas: {e}. Respuesta: {llm_response['response_text']}")
            return None
        except Exception as e_gen:
            logger.error(f"DEBUG: [llm_integration_utils.py] Error inesperado al parsear respuesta de tablas LLM: {e_gen}")
            return None
    else:
        logger.error(f"DEBUG: [llm_integration_utils.py] La llamada al LLM para tablas no fue exitosa o no devolvió texto: {llm_response}")
        return None

# Podrías añadir más funciones de utilidad aquí, como para parsear la respuesta de extracción estructurada,
# manejar reintentos, etc.
