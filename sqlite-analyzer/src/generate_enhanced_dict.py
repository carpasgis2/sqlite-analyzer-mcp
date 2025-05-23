import json
import sqlite3
import os
from typing import Dict, Any, List, Tuple
from llm_utils import call_llm_with_fallbacks
from tqdm import tqdm
import re

def parse_llm_response_for_dict(response_text: str, default_name: str) -> Tuple[str, List[str]]:
    """
    Parsea la respuesta del LLM para extraer descripción y términos comunes.
    Devuelve una tupla (descripción, lista_de_terminos).
    """
    description = ""
    terms = []

    # Intentar extraer descripción (parte 1)
    desc_match = re.search(r"1\\.(.*?)(?:2\\.|$)", response_text, re.DOTALL | re.IGNORECASE)
    if desc_match:
        description = desc_match.group(1).strip()
    else:
        desc_match_fallback = re.search(r"^(.*?)(?:2\\.|$)", response_text, re.DOTALL | re.IGNORECASE)
        if desc_match_fallback:
            description = desc_match_fallback.group(1).strip()

    # Intentar extraer términos (parte 2)
    terms_match = re.search(r"2\\.(.*)", response_text, re.DOTALL | re.IGNORECASE)
    if terms_match:
        terms_str = terms_match.group(1).strip()
        terms = [t.strip().lstrip("-").strip() for t in re.split(r"[\\n,]+", terms_str) if t.strip()]

    if not description and not terms and response_text:
        description = response_text.strip()

    if not terms:
        terms = [default_name.lower().replace('_', ' ')]
    
    if not isinstance(description, str):
        description = str(description)
    if not isinstance(terms, list) or not all(isinstance(t, str) for t in terms):
        terms = [default_name.lower().replace('_', ' ')]

    return description, terms

def generate_enhanced_dictionary(db_path: str, output_path: str, llm_config: dict = None) -> None:
    """
    Genera un diccionario plano enriquecido para mapeo de términos y sinónimos, usando LLM para descripciones y términos comunes.
    Args:
        db_path: Ruta a la base de datos SQLite
        output_path: Ruta donde guardar el diccionario generado
        llm_config: Configuración para el LLM (opcional)
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Base de datos no encontrada: {db_path}")
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    llm_config = llm_config if llm_config is not None else {}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    dictionary = {
        "tables": {},
        "columns": {},
        "common_typos": {},
        "table_relationships": []
    }
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row["name"] for row in cursor.fetchall()]

    for table_name in tqdm(tables, desc="Procesando tablas", unit="tabla"):
        cursor.execute(f"PRAGMA table_info(\"{table_name}\");")
        columns_info_raw = cursor.fetchall()
        columns_names_for_prompt = [col_raw["name"] for col_raw in columns_info_raw] if columns_info_raw else []

        table_prompt_messages = [
            {"role": "system", "content": "Eres un experto en bases de datos. Genera descripciones útiles y términos comunes para mapeo de sinónimos. Responde concisamente."},
            {"role": "user", "content": f"Para la tabla '{table_name}' con columnas {columns_names_for_prompt}, genera:\\n1. Descripción concisa (máx 2 frases, en español) de su propósito.\\n2. Lista de 2-5 términos/sinónimos comunes (en español) para esta tabla."}
        ]
        
        table_response_text = call_llm_with_fallbacks(table_prompt_messages, llm_config, step_name=f"Tabla {table_name}")
        
        table_desc, table_terms = parse_llm_response_for_dict(table_response_text, table_name)
        
        dictionary["tables"][table_name] = {
            "description": table_desc,
            "common_terms": table_terms
        }
        
        if columns_info_raw:
            for col_raw_info in tqdm(columns_info_raw, desc=f"Columnas de {table_name}", leave=False, unit="col"):
                col_name = col_raw_info["name"]
                col_type = col_raw_info["type"]
                
                col_prompt_messages = [
                    {"role": "system", "content": "Eres un experto en bases de datos. Genera descripciones útiles y términos comunes para mapeo de sinónimos. Responde concisamente."},
                    {"role": "user", "content": f"Para la columna '{col_name}' (tipo: {col_type}) de la tabla '{table_name}', genera:\\n1. Descripción breve (máx 1 frase, en español) de qué almacena.\\n2. Lista de 2-5 términos/sinónimos comunes (en español) para esta columna."}
                ]
                
                col_response_text = call_llm_with_fallbacks(col_prompt_messages, llm_config, step_name=f"Columna {table_name}.{col_name}")
                
                col_desc, col_terms = parse_llm_response_for_dict(col_response_text, col_name)
                
                dictionary["columns"][f"{table_name}.{col_name}"] = { 
                    "table": table_name,
                    "column_name": col_name,
                    "type": col_type,
                    "description": col_desc,
                    "common_terms": col_terms
                }

    for table_name_fk_check in tables: 
        cursor.execute(f"PRAGMA foreign_key_list(\"{table_name_fk_check}\");")
        fk_list = cursor.fetchall()
        for fk_row in fk_list:
            dictionary["table_relationships"].append({
                "from_table": table_name_fk_check,
                "from_column": fk_row["from"], 
                "to_table": fk_row["table"],    
                "to_column": fk_row["to"]     
            })

    dictionary["common_typos"] = {
        "paicente": "paciente",
        "pacinte": "paciente",
        "pacente": "paciente",
        "medico": "médico",
        "medcio": "médico",
        "doctro": "doctor",
        "medicacion": "medicación",
        "alergia": "alergia", 
        "alegia": "alergia",
        "medicamneto": "medicamento",
        "diagostico": "diagnóstico",
        "diagnostco": "diagnóstico"
    }
    
    conn.close()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)
    
    print(f"Diccionario enriquecido generado y guardado en: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generador de diccionario plano enriquecido para bases de datos SQLite (LLM-driven)')
    parser.add_argument('db_path', help='Ruta a la base de datos SQLite')
    # Argumento de salida opcional, con valor por defecto relativo al script
    parser.add_argument('--output', '-o', default=os.path.join("data", "dictionary.json"), help='Ruta donde guardar dictionary.json (default: data/dictionary.json relativo al script)')
    
    args = parser.parse_args()
    
    # Cargar la configuración del LLM, incluyendo la API key
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    default_api_key = "sk-aedf531ee17447aa95c9102e595f29ae" # Clave por defecto como en langchain_chatbot.py

    if not deepseek_api_key:
        print("Advertencia: La variable de entorno DEEPSEEK_API_KEY no está configurada. Se usará una clave por defecto.")
        print(f"Para configurar la clave API, ejecute: export DEEPSEEK_API_KEY='su_clave_api_aqui'")
        deepseek_api_key = default_api_key
        
    llm_config_example = {
        "llm_api_key": deepseek_api_key
        # Aquí se podrían añadir otras configuraciones del LLM si fueran necesarias,
        # como "llm_api_url" o "llm_model", si se quieren sobrescribir los valores
        # que llm_utils.py toma de las variables de entorno o sus propios defaults.
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.output):
        output_file_path = os.path.join(script_dir, args.output)
    else:
        output_file_path = args.output
        
    output_file_path = os.path.normpath(output_file_path)

    generate_enhanced_dictionary(args.db_path, output_file_path, llm_config=llm_config_example)