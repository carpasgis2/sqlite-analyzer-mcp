import json
import os
import re  # Añadido import re

# Directorio base del script actual
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# Rutas de los archivos de esquema
SCHEMA_ENHANCED_PATH = os.path.join(DATA_DIR, "schema_enhanced.json")
SCHEMA_SIMPLE_PATH = os.path.join(DATA_DIR, "schema_simple.json")

def create_simple_schema():
    """
    Lee el archivo schema_enhanced.json y crea schema_simple.json
    conteniendo solo los nombres de las tablas y los nombres de sus columnas.
    """
    try:
        with open(SCHEMA_ENHANCED_PATH, 'r', encoding='utf-8') as f_enhanced:
            enhanced_schema = json.load(f_enhanced)
    except FileNotFoundError:
        print(f"Error: El archivo {SCHEMA_ENHANCED_PATH} no fue encontrado.")
        return
    except json.JSONDecodeError:
        print(f"Error: El archivo {SCHEMA_ENHANCED_PATH} no es un JSON válido.")
        return
    except Exception as e:
        print(f"Error inesperado al leer {SCHEMA_ENHANCED_PATH}: {e}")
        return

    simple_schema = {}
    
    tables_data_iterable = None # Usaremos esto para iterar, ya sea una lista de tablas o los valores de un dict de tablas

    if isinstance(enhanced_schema, list):
        # Caso 1: enhanced_schema es directamente una lista de diccionarios de tabla
        tables_data_iterable = enhanced_schema
        print("Información: enhanced_schema es una lista. Se procesarán sus elementos como tablas.")
    elif isinstance(enhanced_schema, dict):
        # Caso 2: enhanced_schema es un diccionario, buscar la clave 'tables' o similar
        if "tables" in enhanced_schema and isinstance(enhanced_schema["tables"], dict):
            tables_data_iterable = enhanced_schema["tables"].items() # Iterar sobre (table_name, table_details)
            print("Información: Se procesarán las tablas desde enhanced_schema['tables'] (diccionario).")
        elif "tables" in enhanced_schema and isinstance(enhanced_schema["tables"], list):
            # Nuevo subcaso: enhanced_schema tiene una clave "tables" que es una lista
            tables_data_iterable = enhanced_schema["tables"]
            print("Información: Se procesarán las tablas desde enhanced_schema['tables'] (lista).")
        elif "schema_knowledge" in enhanced_schema and \
             isinstance(enhanced_schema.get("schema_knowledge"), dict) and \
             "tables" in enhanced_schema["schema_knowledge"] and \
             isinstance(enhanced_schema["schema_knowledge"]["tables"], dict):
            tables_data_iterable = enhanced_schema["schema_knowledge"]["tables"].items() # Iterar sobre (table_name, table_details)
            print("Información: Se procesarán las tablas desde enhanced_schema['schema_knowledge']['tables'] (diccionario).")
        elif enhanced_schema.values() and isinstance(list(enhanced_schema.values())[0], dict) and "columns" in list(enhanced_schema.values())[0]:
            # Si es un diccionario de tablas directamente y el primer valor parece una tabla
            tables_data_iterable = enhanced_schema.items() # Iterar sobre (table_name, table_details)
            print("Información: Se procesarán las tablas desde enhanced_schema (diccionario de tablas).")
        else: # Intenta asumir que enhanced_schema es el diccionario de tablas directamente si no coincide con los anteriores
            is_likely_tables_dict = True
            if not enhanced_schema: # si está vacío, no es un diccionario de tablas
                is_likely_tables_dict = False
            else:
                for key, value in enhanced_schema.items():
                    if not isinstance(value, dict) or "columns" not in value:
                        is_likely_tables_dict = False
                        break
            if is_likely_tables_dict:
                tables_data_iterable = enhanced_schema.items() # Iterar sobre (table_name, table_details)
                print("Información: Se procesarán las tablas desde enhanced_schema (diccionario de tablas, por heurística).")

    if not tables_data_iterable:
        print("Error: No se pudo encontrar la sección de 'tables' en el esquema mejorado con un formato reconocible o el formato es inesperado.")
        print("Estructura recibida (primeras 500 caracteres):", str(enhanced_schema)[:500])
        return

    for item in tables_data_iterable:
        table_name = None
        table_details = None

        if isinstance(item, tuple) and len(item) == 2:
            # Caso: item es (nombre_tabla, detalles_tabla), típicamente de .items()
            table_name, table_details = item
        elif isinstance(item, dict) and "table_name" in item:
            # Caso: item es un diccionario de tabla que contiene su propio nombre
            # típicamente cuando tables_data_iterable es una lista de estos diccionarios
            table_name = item.get("table_name")
            table_details = item # El item completo es los detalles de la tabla
        else:
            print(f"Advertencia: Elemento en tables_data_iterable tiene un formato inesperado o falta 'table_name': {str(item)[:200]}. Se omite.")
            continue

        if not table_name: # Asegurarse de que table_name se haya extraído correctamente
            print(f"Advertencia: No se pudo determinar table_name para el item: {str(item)[:200]}. Se omite.")
            continue
            
        # Asegurarse de que table_details sea un diccionario para el procesamiento posterior
        if not isinstance(table_details, dict):
            print(f"Advertencia: table_details para la tabla '{table_name}' no es un diccionario: {str(table_details)[:200]}. Se procesará como tabla sin columnas.")
            simple_schema[table_name] = {"columns": []}
            continue
            
        column_names = []
        columns_data = None
        source_of_columns_data_info = "ninguna"  # Para depuración

        if not isinstance(table_details, dict):
            print(f"Advertencia: table_details para {table_name} no es un diccionario. Se omite y se añade con columnas vacías.")
            simple_schema[table_name] = {"columns": []}
            continue

        # Intento 1: Obtener columnas de 'columns_list'
        if "columns_list" in table_details:
            columns_data = table_details["columns_list"]
            source_of_columns_data_info = "clave directa 'columns_list'"
        # Intento 2: Obtener columnas directamente de la clave 'columns' (fallback)
        elif "columns" in table_details:
            columns_data = table_details["columns"]
            source_of_columns_data_info = "clave directa 'columns'"
        else:
            # Intento 3: Obtener columnas de la clave 'description' si 'columns' o 'columns_list' no existen
            description_str = table_details.get("description")
            if isinstance(description_str, str):
                match = re.search(r"```json\s*([\s\S]*?)\s*```", description_str, re.IGNORECASE)
                if match:
                    json_from_description_str = match.group(1)
                    try:
                        inner_json_data = json.loads(json_from_description_str)
                        if isinstance(inner_json_data, dict):
                            if "columns" in inner_json_data:
                                columns_data = inner_json_data["columns"]
                                source_of_columns_data_info = f"JSON en 'description' (clave 'columns')"
                            elif "columnas" in inner_json_data:  # Común en esquemas en español
                                columns_data = inner_json_data["columnas"]
                                source_of_columns_data_info = f"JSON en 'description' (clave 'columnas')"
                            # else: columns_data sigue siendo None
                        # else: inner_json_data no es un diccionario
                    except json.JSONDecodeError:
                        print(f"Advertencia: Error al decodificar JSON de la descripción para la tabla {table_name}.")
                # else: No se encontró bloque ```json ``` en la descripción
            # else: 'description' no es una cadena o no existe

        # Procesar columns_data si se encontró
        if columns_data is not None:
            if isinstance(columns_data, list) and columns_data and isinstance(columns_data[0], dict):
                # Lista de diccionarios (esperando {'name':...} o {'nombre':...})
                for col_dict in columns_data:
                    if isinstance(col_dict, dict):
                        col_name = col_dict.get("name") or col_dict.get("nombre")  # Probar 'name' y luego 'nombre'
                        if col_name:
                            column_names.append(col_name)
                        # else: print(f"Advertencia: Diccionario de columna en {table_name} no tiene 'name' ni 'nombre': {col_dict}")
            elif isinstance(columns_data, dict):
                # Diccionario donde las claves son nombres de columnas
                column_names = list(columns_data.keys())
            elif isinstance(columns_data, list) and all(isinstance(item, str) for item in columns_data):
                # Lista de strings (nombres de columnas directamente)
                column_names = columns_data
            elif columns_data:  # Si columns_data no es None pero no coincide con formatos esperados y no está vacío
                print(f"Advertencia: Formato de columnas (desde {source_of_columns_data_info}) no reconocido para la tabla {table_name}. Detalles: {str(columns_data)[:200]}. Se añade con columnas vacías.")
        
        # Si después de todos los intentos, columns_data sigue siendo None (no se encontró 'columns' ni en 'description')
        if columns_data is None:
            print(f"Advertencia: No se encontraron datos de columnas para la tabla {table_name} ni en la clave 'columns' ni en 'description'. Se añade con columnas vacías.")
            # Log adicional de las claves de table_details para ayudar a diagnosticar
            print(f"  Detalles de la tabla {table_name}: Claves encontradas en table_details: {list(table_details.keys())}")

        simple_schema[table_name] = {"columns": column_names}
        
        if not column_names and source_of_columns_data_info != "ninguna" and columns_data is not None and columns_data != []:
             # Se encontraron datos de columnas pero no se pudieron extraer nombres (ya se debería haber advertido antes)
             print(f"Info: La tabla {table_name} resultó con columnas vacías a pesar de procesar datos desde {source_of_columns_data_info}.")

    try:
        with open(SCHEMA_SIMPLE_PATH, 'w', encoding='utf-8') as f_simple:
            json.dump(simple_schema, f_simple, indent=4, ensure_ascii=False)
        print(f"Archivo {SCHEMA_SIMPLE_PATH} creado exitosamente con {len(simple_schema)} tablas.")
    except Exception as e:
        print(f"Error al escribir {SCHEMA_SIMPLE_PATH}: {e}")

if __name__ == "__main__":
    create_simple_schema()
