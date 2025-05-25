import os
import json
import requests

# Configuración de la API de Deepseek (preferiblemente desde variables de entorno)
LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-aedf531ee17447aa95c9102e595f29ae")
LLM_API_URL = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
LLM_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# Rutas de archivos (ajusta según sea necesario)
SCHEMA_FILE_PATH = "c:\\Users\\cpascual\\PycharmProjects\\pythonProject\\cursos_actividades\\sina_mcp\\sqlite-analyzer\\src\\schema_rag_enriched.json"
OUTPUT_DICTIONARY_PATH = "c:\\Users\\cpascual\\PycharmProjects\\pythonProject\\cursos_actividades\\sina_mcp\\sqlite-analyzer\\src\\data\\dictionary.json"

def load_schema(file_path):
    """Carga el esquema de la base de datos desde un archivo JSON."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        return schema
    except FileNotFoundError:
        print(f"Error: El archivo de esquema {file_path} no fue encontrado.")
        return None
    except json.JSONDecodeError:
        print(f"Error: El archivo de esquema {file_path} no es un JSON válido.")
        return None
    except Exception as e:
        print(f"Error inesperado al cargar el esquema {file_path}: {e}")
        return None

def load_existing_dictionary(file_path):
    """Carga el diccionario de datos existente si existe."""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Advertencia: El archivo de diccionario existente {file_path} no es un JSON válido. Se creará uno nuevo.")
            return {"tables": {}, "columns": {}}
        except Exception as e:
            print(f"Error inesperado al cargar el diccionario existente {file_path}: {e}. Se creará uno nuevo.")
            return {"tables": {}, "columns": {}}
    return {"tables": {}, "columns": {}}

def generate_table_specific_prompt(table_name, table_details):
    """Genera el prompt para obtener información de una tabla específica."""
    columns_info = {}
    # table_details["columns"] es una LISTA de diccionarios, cada dict es una columna
    # cada dict de columna tiene "name" y "type"
    columns_list = table_details.get("columns", [])
    if isinstance(columns_list, list):
        for col_data in columns_list:
            if isinstance(col_data, dict) and "name" in col_data:
                columns_info[col_data.get("name")] = col_data.get("type", "unknown")
            else:
                print(f"Advertencia: Se encontró un dato de columna con formato inesperado en la tabla {table_name}: {col_data}")
    else:
        print(f"Advertencia: 'columns' para la tabla {table_name} no es una lista: {columns_list}")

    prompt = f"""
Eres un asistente experto en bases de datos. Tu tarea es generar la descripción y términos comunes para UNA tabla específica.

Nombre de la tabla: {table_name}
Columnas y tipos de la tabla:
{json.dumps(columns_info, indent=2, ensure_ascii=False)}

Formato esperado para la respuesta JSON (solo para esta tabla):
Un objeto JSON con:
  - "description": Una descripción concisa y útil de la tabla en ESPAÑOL. Debe explicar el propósito de la tabla.
  - "common_terms": Una lista de cadenas de texto en ESPAÑOL. Estos deben ser sinónimos, palabras clave o términos comúnmente asociados con los datos de la tabla, útiles para búsquedas semánticas. Incluye al menos 3-5 términos relevantes.

Consideraciones importantes:
- TODAS las descripciones y common_terms DEBEN estar en ESPAÑOL.
- Sé conciso pero informativo.
- Para "common_terms", piensa en cómo un usuario podría referirse a esta tabla en lenguaje natural.

Proporciona el resultado como un único bloque de código JSON válido, sin ningún texto explicativo adicional antes o después del JSON.
Comienza tu respuesta directamente con ```json y termínala con ``` o simplemente el JSON.
"""
    return prompt

def generate_column_specific_prompt(table_name, column_name, column_details):
    """Genera el prompt para obtener información de una columna específica."""
    column_type = column_details.get("type", "unknown")
    prompt = f"""
Eres un asistente experto en bases de datos. Tu tarea es generar la descripción y términos comunes para UNA columna específica.

Nombre de la tabla: {table_name}
Nombre de la columna: {column_name}
Tipo de dato de la columna: {column_type}

Formato esperado para la respuesta JSON (solo para esta columna):
Un objeto JSON con:
  - "description": Una descripción concisa y útil de la columna en ESPAÑOL. Debe explicar qué representa el dato en esa columna.
  - "common_terms": Una lista de cadenas de texto en ESPAÑOL. Estos deben ser sinónimos o términos relacionados con el significado de la columna. Incluye al menos 2-3 términos relevantes.

Consideraciones importantes:
- TODAS las descripciones y common_terms DEBEN estar en ESPAÑOL.
- Sé conciso pero informativo.
- Para "common_terms", piensa en cómo un usuario podría referirse a esta columna en lenguaje natural.

Proporciona el resultado como un único bloque de código JSON válido, sin ningún texto explicativo adicional antes o después del JSON.
Comienza tu respuesta directamente con ```json y termínala con ``` o simplemente el JSON.
"""
    return prompt

def query_llm(prompt):
    """Envía el prompt al LLM y devuelve la respuesta."""
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2, # Ligeramente creativo pero mayormente factual
        "max_tokens": 8000  # Aumentado para asegurar que quepa el JSON completo
    }
    print(f"Enviando solicitud a LLM: {LLM_API_URL} con modelo {LLM_MODEL}")
    try:
        response = requests.post(LLM_API_URL, headers=headers, json=data, timeout=300) # Timeout aumentado
        response.raise_for_status()
        
        response_json = response.json()
        content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Limpiar el contenido si está envuelto en ```json ... ```
        if content.strip().startswith("```json"):
            content = content.strip()[7:-3].strip()
        elif content.strip().startswith("```"):
             content = content.strip()[3:-3].strip()
             
        # Intentar parsear el JSON aquí para una validación temprana
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Error: Contenido del LLM no es JSON válido después de la limpieza: {e}")
            print(f"Contenido problemático: {content[:200]}...") # Mostrar parte del contenido
            return None # Devolver None si no es JSON válido
        return content
    except requests.exceptions.Timeout:
        print(f"Error: Timeout al contactar la API del LLM en {LLM_API_URL}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error al contactar la API del LLM: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Respuesta del servidor: {e.response.status_code} - {e.response.text}")
        return None
    except (IndexError, KeyError) as e:
        print(f"Error al parsear la respuesta del LLM: {e}. Respuesta cruda: {response.text if 'response' in locals() else 'No response object'}")
        return None
    except Exception as e:
        print(f"Error inesperado durante la consulta al LLM: {e}")
        return None

def save_dictionary(dictionary_data, output_path):
    """Guarda el diccionario generado en un archivo JSON."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dictionary_data, f, ensure_ascii=False, indent=2)
        # No imprimir aquí, se hará en main después de cada guardado.
        return True
    except Exception as e:
        print(f"Error inesperado al guardar el diccionario: {e}")
        return False

def main():
    print(f"Iniciando la generación del diccionario de datos...")
    print(f"Usando API Key: {'Sí' if LLM_API_KEY.startswith('sk-') else 'No (o formato incorrecto)'}")
    print(f"Usando API URL: {LLM_API_URL}")
    print(f"Usando Modelo: {LLM_MODEL}")

    print(f"Cargando esquema desde: {SCHEMA_FILE_PATH}")
    db_schema = load_schema(SCHEMA_FILE_PATH)
    if not db_schema or "tables" not in db_schema:
        print("No se pudo cargar el esquema o no contiene tablas. Abortando.")
        return

    print(f"Cargando diccionario existente (si existe) desde: {OUTPUT_DICTIONARY_PATH}")
    current_dictionary = load_existing_dictionary(OUTPUT_DICTIONARY_PATH)

    total_tables = len(db_schema["tables"])
    processed_tables_count = 0

    for table_name, table_details in db_schema["tables"].items():
        processed_tables_count += 1
        print(f"\nProcesando Tabla {processed_tables_count}/{total_tables}: {table_name}")

        if table_name not in current_dictionary["tables"]:
            print(f"  Generando información para la tabla {table_name}...")
            table_prompt = generate_table_specific_prompt(table_name, table_details)
            llm_response_content = query_llm(table_prompt)
            if llm_response_content:
                try:
                    table_info = json.loads(llm_response_content)
                    current_dictionary["tables"][table_name] = {
                        "description": table_info.get("description", "Descripción no generada"),
                        "common_terms": table_info.get("common_terms", [])
                    }
                    print(f"    Información de tabla {table_name} generada.")
                except json.JSONDecodeError as e:
                    print(f"    Error al parsear JSON para la tabla {table_name}: {e}")
                    print(f"    Respuesta recibida: {llm_response_content[:200]}...")
                    current_dictionary["tables"][table_name] = {
                        "description": "Error en generación", "common_terms": []
                    }
            else:
                print(f"    No se recibió respuesta del LLM para la tabla {table_name}.")
                current_dictionary["tables"][table_name] = {
                    "description": "Error en comunicación con LLM", "common_terms": []
                }
        else:
            print(f"  Información para la tabla {table_name} ya existe en el diccionario. Saltando.")

        # Procesar columnas para esta tabla
        columns_list_for_table = table_details.get("columns", []) # Obtener la lista de columnas
        if isinstance(columns_list_for_table, list): # Asegurarse de que es una lista
            total_columns = len(columns_list_for_table)
            processed_columns_count = 0
            for col_data in columns_list_for_table: # col_data es ahora un dict de una columna
                if not isinstance(col_data, dict) or "name" not in col_data:
                    print(f"    Advertencia: Elemento de columna inválido o sin nombre en la tabla {table_name}: {col_data}")
                    continue

                processed_columns_count +=1
                col_name = col_data["name"] # Obtener el nombre de la columna del dict
                full_col_name = f"{table_name}.{col_name}"
                print(f"  Procesando Columna {processed_columns_count}/{total_columns}: {full_col_name}")

                if full_col_name not in current_dictionary["columns"]:
                    print(f"    Generando información para la columna {full_col_name}...")
                    # col_data ya es el diccionario correcto para la columna
                    col_prompt = generate_column_specific_prompt(table_name, col_name, col_data)
                    llm_response_content = query_llm(col_prompt)
                    if llm_response_content:
                        try:
                            col_info = json.loads(llm_response_content)
                            current_dictionary["columns"][full_col_name] = {
                                "table": table_name,
                                "column_name": col_name,
                                "type": col_data.get("type", "unknown"), # Usar col_data para el tipo
                                "description": col_info.get("description", "Descripción no generada"),
                                "common_terms": col_info.get("common_terms", [])
                            }
                            print(f"      Información de columna {full_col_name} generada.")
                        except json.JSONDecodeError as e:
                            print(f"      Error al parsear JSON para la columna {full_col_name}: {e}")
                            print(f"      Respuesta recibida: {llm_response_content[:200]}...")
                            current_dictionary["columns"][full_col_name] = {
                                "table": table_name, "column_name": col_name, "type": col_data.get("type", "unknown"),
                                "description": "Error en generación", "common_terms": []
                            }
                    else:
                        print(f"      No se recibió respuesta del LLM para la columna {full_col_name}.")
                        current_dictionary["columns"][full_col_name] = {
                            "table": table_name, "column_name": col_name, "type": col_data.get("type", "unknown"),
                            "description": "Error en comunicación con LLM", "common_terms": []
                        }
                else:
                    print(f"    Información para la columna {full_col_name} ya existe. Saltando.")
        else:
            print(f"Advertencia: 'columns' para la tabla {table_name} no es una lista. Saltando procesamiento de columnas.")
        
        # Guardar después de procesar cada tabla y sus columnas
        print(f"  Guardando progreso en {OUTPUT_DICTIONARY_PATH}...")
        if save_dictionary(current_dictionary, OUTPUT_DICTIONARY_PATH):
            print(f"  Progreso guardado para la tabla {table_name}.")
        else:
            print(f"  Error al guardar el progreso para la tabla {table_name}.")

    print("\nProceso de generación del diccionario de datos completado.")
    if save_dictionary(current_dictionary, OUTPUT_DICTIONARY_PATH):
        print(f"Diccionario final guardado exitosamente en {OUTPUT_DICTIONARY_PATH}")
    else:
        print("Error al guardar el diccionario final.")

if __name__ == "__main__":
    # Pequeña validación de configuración antes de empezar
    if not LLM_API_KEY or not LLM_API_KEY.startswith("sk-"):
        print("Advertencia: DEEPSEEK_API_KEY no parece estar configurada correctamente o no está disponible.")
        print("Por favor, asegúrate de que la variable de entorno DEEPSEEK_API_KEY esté configurada.")
    
    main()
