'''
Este script genera un archivo schema_enhanced.json más robusto y completo.
Carga un esquema base (schema_simple.json), añade una sección para mejoras generales
y aplica refinamientos críticos a las descripciones para guiar mejor al LLM.
'''
import json
import os
import sqlite3
from llm_utils import call_llm_with_fallbacks
from db_config import DBConnector # Importar la clase DBConnector

# Configuración del LLM (preferiblemente desde variables de entorno o un archivo de config centralizado)
# Estas se usarán si no están ya configuradas en llm_utils o pasadas directamente
LM_CONFIG_DEFAULTS = {
    "llm_api_key": "sk-aedf531ee17447aa95c9102e595f29ae",
    "llm_api_url": "https://api.deepseek.com/v1",
    "llm_model": os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat"),  # Usar variable de entorno, default a deepseek-chat
    "temperature": 0.3,
    "max_tokens": 200
}


# Directorio base del script actual
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Directorio de datos donde se encuentran los archivos de esquema
DATA_DIR = os.path.join(BASE_DIR, "data")

# Ruta a la base de datos, construida a partir de BASE_DIR
DATABASE_PATH = os.path.join(BASE_DIR, "db", "database_new.sqlite3.db")

# SIMPLE_SCHEMA_PATH = os.path.join(DATA_DIR, "schema_simple.json") # Removed
ENHANCED_SCHEMA_PATH = os.path.join(DATA_DIR, "schema_enhanced.json")

def _save_schema(schema_data, path):
    """Guarda el diccionario del esquema en un archivo JSON."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(schema_data, f, ensure_ascii=False, indent=4)
        print(f"Esquema guardado/actualizado en {path}")
    except IOError:
        print(f"Error: No se pudo escribir el archivo en {path}")

def get_llm_generated_description(element_name: str, element_type: str, existing_description: str = None, context_schema: dict = None) -> str:
    """
    Genera una descripción para un elemento del esquema (tabla o columna) usando el LLM.
    """
    prompt_parts = [
        f"Eres un asistente experto en documentación de bases de datos médicas. Tu tarea es generar una descripción clara y concisa para un elemento de un esquema de base de datos."
    ]

    if element_type == "tabla":
        prompt_parts.append(f"El elemento es una TABLA llamada '{element_name}'.")
        prompt_parts.append("Describe su propósito principal, el tipo de entidad o información que almacena, y cualquier característica clave obvia por su nombre.")
    elif element_type == "columna":
        prompt_parts.append(f"El elemento es una COLUMNA llamada '{element_name}'.")
        if context_schema and context_schema.get("table_name"):
            prompt_parts.append(f"Esta columna pertenece a la tabla '{context_schema.get('table_name')}' (descripción de la tabla: '{context_schema.get('description', 'No disponible')}').")
        
        prompt_parts.append("Describe su propósito específico. Presta atención a sufijos comunes y explica su significado:")
        prompt_parts.append("  - '_ID': Generalmente un identificador único (clave primaria o foránea).")
        prompt_parts.append("  - '_CODE', '_CD': Un código o clave alfanumérica.")
        prompt_parts.append("  - '_NAME', '_DESCRIPTION', '_TEXT', '_LABEL': Un nombre o descripción textual. Indica si parece ser la descripción principal/humana de una entidad.")
        prompt_parts.append("  - '_DATE', '_DT', '_TIMESTAMP': Un campo de fecha o fecha/hora.")
        prompt_parts.append("  - '_FLAG', '_IND', 'IS_': Un indicador booleano (flag).")
        prompt_parts.append("  - '_ES', '_EN', '_FR': Indica el idioma del contenido (Español, Inglés, Francés, etc.).")
        prompt_parts.append("  - '_TYPE', '_CATEGORY': Indica un tipo o categoría.")
        prompt_parts.append("Si parece una clave foránea por su nombre (ej. 'PATIENT_ID' en una tabla de episodios), menciónalo.")
        prompt_parts.append("Si el nombre sugiere que es la descripción oficial de un diagnóstico o un término médico codificado, indícalo.")

    if existing_description: # This will likely be None when reading fresh from DB
        prompt_parts.append(f"La descripción actual es: '{existing_description}'. Por favor, revisa y mejora esta descripción si es necesario, o genera una nueva si la actual no es adecuada o está vacía.")
    else:
        prompt_parts.append("Por favor, genera una descripción concisa y útil para este elemento.")

    prompt_parts.append("La descripción debe ser informativa para un analista de datos o un desarrollador que necesite entender el propósito de este elemento.")
    
    # Caso especial para PHTH_DESCRIPTION_ES para asegurar que el LLM entiende la guía crítica
    if element_name == "PHTH_DESCRIPTION_ES" and context_schema and context_schema.get("table_name") == "MEDI_PHARMA_THERAPEUTIC_GROUPS":
        prompt_parts.append("IMPORTANTE: Para la columna 'PHTH_DESCRIPTION_ES' en la tabla 'MEDI_PHARMA_THERAPEUTIC_GROUPS', la descripción DEBE enfatizar que se refiere a una CLASIFICACIÓN FARMACOLÓGICA y NO a una condición médica o enfermedad. Debe guiar al usuario a buscar primero la clasificación farmacológica relevante para una condición y luego usar esa clasificación aquí.")
        prompt_parts.append("Ejemplo de énfasis: 'Descripción en español de la CLASIFICACIÓN FARMACOLÓGICA del grupo terapéutico (ej: \\\\'Inhibidores de la enzima convertidora de angiotensina\\\\'). IMPORTANTE: Esta columna se refiere a una clasificación farmacológica, NO a una condición médica o enfermedad que el grupo trata.'")

    elif element_name == "MEDI_PHARMA_THERAPEUTIC_GROUPS" and element_type == "tabla":
        prompt_parts.append("IMPORTANTE: Para la tabla 'MEDI_PHARMA_THERAPEUTIC_GROUPS', la descripción DEBE aclarar que la columna PHTH_DESCRIPTION_ES describe la CLASIFICACIÓN FARMACOLÓGICA y NO debe usarse para buscar directamente nombres de enfermedades.")

    prompt_parts.append("Responde ÚNICAMENTE con la descripción generada, sin frases introductorias como 'Aquí tienes la descripción:'.")

    system_message = "Eres un asistente de documentación de bases de datos experto."
    user_message = "\\n".join(prompt_parts)

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    # Usar una copia de los defaults y permitir overrides si es necesario
    llm_config = LM_CONFIG_DEFAULTS.copy()
    # Aquí podrías añadir lógica para pasar configuraciones específicas si es necesario

    print(f"Generando descripción para {element_type} '{element_name}'...")
    response = call_llm_with_fallbacks(llm_config, messages) # Corregido el orden de los argumentos

    if response and not response.startswith("ERROR:"):
        # Limpiar la respuesta si el LLM añade frases extra (aunque se le pidió que no)
        cleaned_response = response.strip()
        # Podríamos añadir más lógica de limpieza aquí si es necesario
        print(f"Descripción generada para '{element_name}': '{cleaned_response}'")
        return cleaned_response
    else:
        print(f"Error al generar descripción para '{element_name}': {response}")
        # Return None or a default error message if generation fails, existing_description might be None
        return "Descripción no disponible (error en generación)."


def generate_enhanced_schema():
    """
    Lee el esquema directamente de la base de datos SQLite,
    genera descripciones mejoradas usando LLM para cada tabla y columna,
    y guarda el resultado en schema_enhanced.json.
    """
    enhanced_schema = []
    
    # Instanciar DBConnector con la ruta a la base de datos
    db_connector = DBConnector(DATABASE_PATH)
    conn = None # Inicializar conn

    try:
        # Obtener una conexión usando el método de la instancia de DBConnector
        conn = db_connector.get_connection() 
        if not conn:
            print(f"Error: No se pudo establecer la conexión a la base de datos en {DATABASE_PATH}.")
            return

        cursor = conn.cursor()
        # Obtener todos los nombres de tablas (excluyendo las internas de SQLite)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        db_tables = cursor.fetchall()

        print(f"Se encontraron {len(db_tables)} tablas en la base de datos.")

        for table_idx, db_table_row in enumerate(db_tables):
            table_name = db_table_row[0]
            print(f"Procesando tabla ({table_idx + 1}/{len(db_tables)}): {table_name}")

            table_data = {
                "table_name": table_name,
                "description": "", # Se generará a continuación
                "columns": []
            }

            # Generar descripción para la tabla
            # El contexto para la descripción de la tabla solo incluye su propio nombre y una lista vacía de columnas inicialmente.
            table_context_for_llm = {"table_name": table_name, "columns": []}
            table_description = get_llm_generated_description(
                element_name=table_name,
                element_type="tabla",
                existing_description=None, # Generando desde cero
                context_schema=table_context_for_llm
            )
            table_data["description"] = table_description if table_description else "Descripción no generada por el LLM."

            # Obtener información de las columnas para la tabla actual
            cursor.execute(f"PRAGMA table_info('{table_name}');")
            db_columns_info = cursor.fetchall()
            
            # Obtener información de claves foráneas para la tabla actual UNA VEZ
            cursor.execute(f"PRAGMA foreign_key_list('{table_name}');")
            foreign_keys_for_table = cursor.fetchall()
            # Mapear columnas origen de FK a su información de referencia para búsqueda rápida
            fk_map = {fk_row[3]: {"references_table": fk_row[2], "references_column": fk_row[4]} for fk_row in foreign_keys_for_table}
            
            print(f"  Encontradas {len(db_columns_info)} columnas para la tabla {table_name}.")

            for col_idx, db_col_row in enumerate(db_columns_info):
                col_name = db_col_row[1]  # 'name' está en el índice 1
                col_type = db_col_row[2]  # 'type' está en el índice 2 (afinidad)
                # col_notnull = bool(db_col_row[3]) # 'notnull' está en el índice 3. No la usamos directamente aquí pero es info útil.
                col_pk = bool(db_col_row[5])    # 'pk' está en el índice 5 (si es parte de la PK)
                
                print(f"  Procesando columna ({col_idx + 1}/{len(db_columns_info)}): {col_name} (Tipo: {col_type}) en tabla {table_name}")

                column_data = {
                    "name": col_name,
                    "type": col_type, # Afinidad del tipo de dato
                    "description": "" # Se generará a continuación
                }

                # Generar descripción para la columna
                # El contexto para la descripción de la columna es table_data, que ya tiene
                # el nombre de la tabla y su descripción recién generada, y las columnas procesadas anteriormente.
                column_description = get_llm_generated_description(
                    element_name=col_name,
                    element_type="columna",
                    existing_description=None, # Generando desde cero
                    context_schema=table_data # table_data contiene table_name, table_description, y cols previas
                )
                column_data["description"] = column_description if column_description else "Descripción no generada por el LLM."

                # --- INICIO DE MODIFICACIONES ESPECÍFICAS (ej. EPIS_DIAGNOSTICS) ---
                if table_name == "EPIS_DIAGNOSTICS":
                    if col_name == "DIAG_OTHER_DIAGNOSTIC":
                        column_data["description"] = "Descripción textual libre del diagnóstico, ingresada manualmente. IMPORTANTE: Para análisis y búsquedas de diagnósticos oficiales y codificados, utilice la tabla CODR_DIAGNOSTIC_GROUPS unida a través de EPIS_DIAGNOSTICS.CDTE_ID. Esta columna puede contener información no estandarizada o complementaria."
                        column_data["is_official_diagnostic_description"] = False
                    elif col_name == "CDTE_ID":
                        column_data["description"] = "Identificador único del término diagnóstico codificado. IMPORTANTE: Esta columna es la clave foránea para enlazar con la tabla CODR_DIAGNOSTIC_GROUPS (específicamente con la columna DGGR_ID) y obtener la descripción oficial y estandarizada del diagnóstico."
                        # La información de FK se añade programáticamente abajo, pero la descripción puede reforzarlo.
                    elif col_name == "DIAG_MAIN":
                        column_data["description"] = "Indicador booleano (0 o 1) que señala si el diagnóstico es el principal del episodio. 1 significa que es el diagnóstico principal, 0 que no lo es. IMPORTANTE: Utilice esta columna para filtrar por diagnósticos principales cuando sea relevante para la consulta."
                # --- FIN DE MODIFICACIONES ESPECÍFICAS ---

                # --- INICIO DE ADICIÓN PROGRAMÁTICA DE METADATOS ESTRUCTURALES ---
                column_data["is_primary_key"] = col_pk
                
                is_fk = col_name in fk_map
                column_data["is_foreign_key"] = is_fk
                if is_fk:
                    column_data["references_table"] = fk_map[col_name]["references_table"]
                    column_data["references_column"] = fk_map[col_name]["references_column"]
                else:
                    column_data["references_table"] = None
                    column_data["references_column"] = None
                # --- FIN DE ADICIÓN PROGRAMÁTICA DE METADATOS ESTRUCTURALES ---
                
                table_data["columns"].append(column_data)

            enhanced_schema.append(table_data)
            
            # Guardar el progreso después de procesar cada tabla
            _save_schema(enhanced_schema, ENHANCED_SCHEMA_PATH)
            print(f"Progreso guardado después de procesar y añadir la tabla: {table_name}")

    except sqlite3.Error as e:
        print(f"Error de SQLite durante el procesamiento del esquema: {e}")
    finally:
        if conn:
            # Usar el método close de la instancia de DBConnector si existe,
            # o el método close de la conexión directamente.
            # Asumiendo que DBConnector.close() maneja la lógica de cierre de la conexión que obtuvo.
            # Si DBConnector no tiene un método close explícito para la conexión que devuelve get_connection(),
            # entonces se debería cerrar conn directamente o db_config.py debería proveer una función para ello.
            # Por ahora, intentaremos un método close() en el conector, 
            # o cerramos la conexión directamente si el conector no lo tiene.
            if hasattr(db_connector, 'close_connection') and callable(getattr(db_connector, 'close_connection')):
                db_connector.close_connection(conn) # Suponiendo que existe un método así
            elif hasattr(conn, 'close') and callable(getattr(conn, 'close')):
                 # Esto podría no ser lo ideal si DBConnector gestiona un pool o estado
                 # pero es mejor que no cerrarla.
                 # La clase DBConnector en el contexto no muestra un método close_connection(conn) explícito
                 # ni un close() general. Muestra un _close_all_connections() y un close_current_thread_connection()
                 # Para ser más precisos con la API de DBConnector vista:
                 if hasattr(db_connector, 'close_current_thread_connection') and callable(getattr(db_connector, 'close_current_thread_connection')):
                    db_connector.close_current_thread_connection()
                 else:
                    print("Advertencia: DBConnector no tiene un método close_current_thread_connection. La conexión podría no cerrarse limpiamente por el conector.")
                    # conn.close() # Como fallback, pero idealmente el conector lo maneja.
            else:
                print("Advertencia: No se pudo determinar cómo cerrar la conexión a través de DBConnector.")


    # Guardado final (aunque ya se guarda progresivamente)
    if db_tables: # Solo imprimir si se procesaron tablas
        print(f"Proceso completado. Esquema final guardado en {ENHANCED_SCHEMA_PATH}")
    else:
        print("No se procesaron tablas. Verifique la conexión a la BD y la existencia de tablas.")


if __name__ == "__main__":
    generate_enhanced_schema()
