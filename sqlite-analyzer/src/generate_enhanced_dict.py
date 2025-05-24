import json
import sqlite3
import os
from typing import Dict, Any, List, Tuple
from llm_utils import call_llm_with_fallbacks
from tqdm import tqdm
import re
import ast  # Importar ast para literal_eval

def parse_llm_response_for_dict(response_text: str, default_name: str) -> Tuple[str, List[str]]:
    """
    Parsea la respuesta del LLM para extraer descripción y términos comunes.
    Devuelve una tupla (descripción, lista_de_terminos).
    """
    description = ""
    terms: List[str] = []

    # Intentar extraer descripción (parte 1)
    # Busca "1." seguido de cualquier texto (no codicioso) hasta ANTES de "2." o el final de la cadena.
    desc_match = re.search(r"1\\.\\s*(.*?)(?=\\n*2\\.|\\Z)", response_text, re.DOTALL | re.IGNORECASE)
    if desc_match:
        description = desc_match.group(1).strip()
    else:
        # Fallback si "1." no está, pero "2." podría estar. Tomar todo ANTES de "2."
        # O si no hay "1." ni "2.", tomar todo como descripción.
        desc_before_2_match = re.search(r"^(.*?)(?=2\\.|$)", response_text, re.DOTALL | re.IGNORECASE)
        if desc_before_2_match:
            potential_desc = desc_before_2_match.group(1).strip()
            # Evitar que la descripción sea una lista de términos si el LLM solo devolvió la lista.
            if not (potential_desc.startswith('[') and potential_desc.endswith(']')):
                description = potential_desc
            elif not re.search(r"2\\.", response_text, re.IGNORECASE): # Si no hay "2." y la respuesta es solo una lista
                 # No asignar a descripción, podría ser solo la lista de términos.
                 pass

        # Si la descripción sigue vacía y no hay "1." ni "2.", tomar toda la respuesta como descripción.
        # Pero solo si no parece una lista de términos.
        elif not (response_text.strip().startswith('[') and response_text.strip().endswith(']')) and \
             not re.search(r"1\\.", response_text, re.IGNORECASE) and \
             not re.search(r"2\\.", response_text, re.IGNORECASE):
             description = response_text.strip()

    # Intentar extraer términos (parte 2)
    # Busca "2." seguido de cualquier texto.
    terms_section_match = re.search(r"2\\.\\s*(.*)", response_text, re.DOTALL | re.IGNORECASE)
    if terms_section_match:
        terms_str = terms_section_match.group(1).strip()
        
        # Intento 1: Parsear como una lista literal de Python (ej. ['term1', 'term2'])
        try:
            list_start_index = terms_str.find('[')
            list_end_index = terms_str.rfind(']')

            if list_start_index != -1 and list_end_index != -1 and list_start_index < list_end_index:
                potential_list_str = terms_str[list_start_index : list_end_index + 1]
                evaluated_terms = ast.literal_eval(potential_list_str)
                if isinstance(evaluated_terms, list) and all(isinstance(t, str) for t in evaluated_terms):
                    terms = [t.lower().strip() for t in evaluated_terms if t.strip()]
                    seen = set()
                    terms = [x for x in terms if not (x in seen or seen.add(x))]
        except (ValueError, SyntaxError):
            pass # Si ast.literal_eval falla, `terms` seguirá vacío y se pasará al Intento 2

        # Intento 2: Si no se parseó como lista literal o el resultado está vacío (y `terms` sigue vacío)
        if not terms: 
            normalized_terms_str = re.sub(r'[\\n\\r]+\\s*-\\s*|[\\n\\r]+', ',', terms_str)
            normalized_terms_str = re.sub(r'\\s*-\\s*', ',', normalized_terms_str)
            # Quitar corchetes solo si están al principio y al final de la cadena normalizada,
            # para no afectar corchetes internos de términos malformados.
            if normalized_terms_str.startswith('[') and normalized_terms_str.endswith(']'):
                normalized_terms_str = normalized_terms_str[1:-1]
            
            raw_terms = re.split(r'\\s*,\\s*', normalized_terms_str)
            processed_terms = []
            for t in raw_terms:
                cleaned_term = t.strip().strip('\'"') 
                if cleaned_term:
                    processed_terms.append(cleaned_term.lower())
            
            terms = [t for t in processed_terms if t]
            seen = set()
            terms = [x for x in terms if not (x in seen or seen.add(x))]
    
    # Si `terms` sigue vacío y la respuesta original (o la descripción capturada) parece una lista, intentar parsearla.
    # Esto es para casos donde el LLM solo devuelve la lista sin "1." o "2."
    if not terms:
        target_text_for_list_parse = response_text.strip()
        if target_text_for_list_parse.startswith('[') and target_text_for_list_parse.endswith(']'):
            try:
                evaluated_terms = ast.literal_eval(target_text_for_list_parse)
                if isinstance(evaluated_terms, list) and all(isinstance(t, str) for t in evaluated_terms):
                    terms = [t.lower().strip() for t in evaluated_terms if t.strip()]
                    seen = set()
                    terms = [x for x in terms if not (x in seen or seen.add(x))]
                    # Si hemos extraído términos así, la descripción debería estar vacía si no se encontró antes.
                    if not description:
                        description = "" 
            except (ValueError, SyntaxError):
                pass

    # Fallback final: si no se extrajeron términos después de todos los intentos
    if not terms:
        terms = [default_name.lower().replace('_', ' ')]
    
    # Asegurar que la descripción sea un string y los términos sean una lista de strings
    if not isinstance(description, str):
        description = str(description)
    if not (isinstance(terms, list) and all(isinstance(t, str) for t in terms)):
        terms = [default_name.lower().replace('_', ' ')]

    return description, terms

def save_dictionary_incrementally(data: Dict[str, Any], output_path: str):
    """Guarda el diccionario en el archivo JSON."""
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"Error al guardar el diccionario incrementalmente en {output_path}: {e}")

def generate_enhanced_dictionary(db_path: str, output_path: str, llm_config: dict = None) -> None:
    """
    Genera un diccionario plano enriquecido para mapeo de términos y sinónimos, usando LLM para descripciones y términos comunes.
    Guarda el progreso incrementalmente.
    Args:
        db_path: Ruta a la base de datos SQLite
        output_path: Ruta donde guardar el diccionario generado
        llm_config: Configuración para el LLM (opcional)
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Base de datos no encontrada: {db_path}")
    
    llm_config = llm_config if llm_config is not None else {}

    # Intentar cargar el diccionario existente para actualizaciones incrementales
    dictionary: Dict[str, Any]
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                dictionary = json.load(f)
            # Asegurarse de que las claves principales existan
            if "tables" not in dictionary: dictionary["tables"] = {}
            if "columns" not in dictionary: dictionary["columns"] = {}
            if "common_typos" not in dictionary: dictionary["common_typos"] = {}
            if "table_relationships" not in dictionary: dictionary["table_relationships"] = []
            print(f"Diccionario existente cargado desde {output_path}")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Advertencia: No se pudo cargar el diccionario existente de {output_path} ({e}). Se creará uno nuevo.")
            dictionary = {
                "tables": {}, "columns": {}, "common_typos": {}, "table_relationships": []
            }
    else:
        print(f"No se encontró diccionario existente en {output_path}. Se creará uno nuevo.")
        dictionary = {
            "tables": {}, "columns": {}, "common_typos": {}, "table_relationships": []
        }

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row["name"] for row in cursor.fetchall()]

    for table_name in tqdm(tables, desc="Procesando tablas", unit="tabla"):
        cursor.execute(f"PRAGMA table_info(\"{table_name}\");")
        columns_info_raw = cursor.fetchall()
        columns_names_for_prompt = [col_raw["name"] for col_raw in columns_info_raw] if columns_info_raw else []

        table_prompt_messages = [
            {"role": "system", "content": "Eres un experto en bases de datos y terminología de usuario. Tu tarea es generar descripciones claras y una lista exhaustiva de términos comunes y sinónimos que los usuarios podrían emplear. Responde concisamente y sigue el formato solicitado."},
            {"role": "user", "content": f"Para la tabla '{table_name}' (columnas de ejemplo: {columns_names_for_prompt if columns_names_for_prompt else 'ninguna'}), por favor genera:\\n1. Descripción concisa (máximo 2 frases, en español) de su propósito principal.\\n2. Lista de 3 a 7 términos o sinónimos comunes en español y estrictamente en minúsculas que un usuario podría usar para referirse a esta tabla. Incluye palabras clave individuales (ej: si la tabla es 'CITAS_MEDICAS', incluye 'cita', 'citas'), frases cortas (ej: 'historial de citas') y variaciones comunes (plurales/singulares). Formato para términos: ['termino1', 'frase corta 2', 'palabra clave 3']"}
        ]
        
        table_response_text = call_llm_with_fallbacks(table_prompt_messages, llm_config, step_name=f"Tabla {table_name}")
        
        table_desc, table_terms = parse_llm_response_for_dict(table_response_text, table_name)
        
        dictionary["tables"][table_name] = {
            "description": table_desc,
            "common_terms": table_terms
        }
        save_dictionary_incrementally(dictionary, output_path) # Guardado incremental
        print(f"Información de tabla '{table_name}' guardada.")
        
        if columns_info_raw:
            for col_raw_info in tqdm(columns_info_raw, desc=f"Columnas de {table_name}", leave=False, unit="col"):
                col_name = col_raw_info["name"]
                col_type = col_raw_info["type"]
                
                col_prompt_messages = [
                    {"role": "system", "content": "Eres un experto en bases de datos y terminología de usuario. Tu tarea es generar descripciones claras y una lista exhaustiva de términos comunes y sinónimos que los usuarios podrían emplear para referirse a una columna. Responde concisamente y sigue el formato solicitado."},
                    {"role": "user", "content": f"Para la columna '{col_name}' (tipo: {col_type}) de la tabla '{table_name}', por favor genera:\\n1. Descripción breve (máximo 1 frase, en español) de la información que almacena.\\n2. Lista de 3 a 7 términos o sinónimos comunes en español y estrictamente en minúsculas que un usuario podría usar para referirse a esta columna. Incluye palabras clave individuales, frases cortas y variaciones comunes si aplican. Formato para términos: ['termino_columna', 'otro nombre columna', 'dato x']"}
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
                save_dictionary_incrementally(dictionary, output_path) # Guardado incremental
                print(f"Información de columna '{table_name}.{col_name}' guardada.")

    print("Procesando relaciones de claves foráneas...")
    current_relationships = {tuple(sorted(rel.items())) for rel in dictionary.get("table_relationships", [])}

    for table_name_fk_check in tqdm(tables, desc="Verificando claves foráneas", unit="tabla"): 
        cursor.execute(f"PRAGMA foreign_key_list(\"{table_name_fk_check}\");")
        fk_list = cursor.fetchall()
        for fk_row in fk_list:
            rel_dict = {
                "from_table": table_name_fk_check,
                "from_column": fk_row["from"], 
                "to_table": fk_row["table"],    
                "to_column": fk_row["to"]     
            }
            # Añadir solo si no existe para evitar duplicados en re-ejecuciones
            if tuple(sorted(rel_dict.items())) not in current_relationships:
                dictionary["table_relationships"].append(rel_dict)
                current_relationships.add(tuple(sorted(rel_dict.items())))
    
    print("Añadiendo typos comunes...")
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
    
    # Guardado final con toda la información
    save_dictionary_incrementally(dictionary, output_path)
    
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