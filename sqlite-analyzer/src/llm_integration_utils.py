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

Instrucciones adicionales para preguntas complejas:
- **Subconsultas**: Si la pregunta requiere una subconsulta (por ejemplo, en una cláusula WHERE IN, o para obtener un valor agregado que se usa en la consulta principal), intenta describir la subconsulta dentro del campo "value" de una condición si es posible, o detalla la necesidad y lógica de la subconsulta en el campo "explanation". Ejemplo para una condición:
  `"value": "subconsulta: SELECCIONAR PATI_ID DE EPIS_DIAGNOSTICS DONDE DIAG_OTHER_DIAGNOSTIC LIKE \'%diabetes%\'"`.
