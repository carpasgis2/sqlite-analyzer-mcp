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
    Se enfoca en concisión y relevancia para la generación de SQL y comprensión del LLM.
    """
    prompt_parts = [
        f"Eres un asistente experto en documentación de bases de datos médicas. Tu tarea es generar una descripción MUY CONCISA (1-2 frases MÁXIMO) para un elemento de un esquema de base de datos, enfocándote en su propósito y semántica para la generación de consultas SQL y la comprensión por un LLM."
    ]

    if element_type == "tabla":
        prompt_parts.append(f"El elemento es una TABLA llamada '{element_name}'.")
        prompt_parts.append("Describe su propósito principal de forma breve. Ejemplo: 'Almacena información de pacientes.' o 'Registros de diagnósticos médicos.'")
    elif element_type == "columna":
        prompt_parts.append(f"El elemento es una COLUMNA llamada '{element_name}'.")
        if context_schema and context_schema.get("table_name"):
            prompt_parts.append(f"Pertenece a la tabla '{context_schema.get('table_name')}' (descripción tabla: '{context_schema.get('description', 'No disponible')}').")
        
        prompt_parts.append("Describe su propósito específico de forma breve. Si es una clave, indica a qué entidad identifica. Si es una descripción, indica de qué. Si es una fecha, indica qué evento representa. Presta atención a sufijos comunes y explica su significado de forma concisa:")
        prompt_parts.append("  - '_ID': Identificador único (PK o FK). Si es FK, menciona la entidad referenciada si es obvio por el nombre.")
        prompt_parts.append("  - '_CODE', '_CD': Código o clave alfanumérica.")
        prompt_parts.append("  - '_NAME', '_DESCRIPTION', '_TEXT', '_LABEL': Nombre o descripción textual. Indica si es la descripción principal de una entidad.")
        prompt_parts.append("  - '_DATE', '_DT', '_TIMESTAMP': Fecha o fecha/hora de un evento específico.")
        prompt_parts.append("  - '_FLAG', '_IND', 'IS_': Indicador booleano.")
        prompt_parts.append("  - '_ES', '_EN': Idioma del contenido (Español, Inglés).")
        prompt_parts.append("  - '_TYPE', '_CATEGORY': Tipo o categoría.")
        prompt_parts.append("Ejemplo para columna 'PATIENT_ID' en tabla 'EPISODES': 'Identificador del paciente asociado a este episodio (FK a la tabla de pacientes).' Ejemplo para 'DIAG_DATE': 'Fecha en que se registró el diagnóstico.'")

    # No se usará existing_description para este nuevo enfoque más directo.
    # else:
    #     prompt_parts.append("Por favor, genera una descripción MUY CONCISA y útil para este elemento.")

    # Caso especial para PHTH_DESCRIPTION_ES
    if element_name == "PHTH_DESCRIPTION_ES" and context_schema and context_schema.get("table_name") == "MEDI_PHARMA_THERAPEUTIC_GROUPS":
        prompt_parts.append("CRÍTICO: Para 'PHTH_DESCRIPTION_ES' en 'MEDI_PHARMA_THERAPEUTIC_GROUPS', la descripción DEBE indicar que es una CLASIFICACIÓN FARMACOLÓGICA, NO una enfermedad. Ejemplo: 'Clasificación farmacológica del grupo terapéutico (ej: Inhibidores ECA). NO es una enfermedad.'")
    elif element_name == "MEDI_PHARMA_THERAPEUTIC_GROUPS" and element_type == "tabla":
        prompt_parts.append("CRÍTICO: Para la tabla 'MEDI_PHARMA_THERAPEUTIC_GROUPS', aclarar que PHTH_DESCRIPTION_ES es la CLASIFICACIÓN FARMACOLÓGICA, no nombres de enfermedades.")

    prompt_parts.append("Responde ÚNICAMENTE con la descripción generada (1-2 frases).")

    system_message = "Eres un asistente de documentación de bases de datos experto en concisión y relevancia para LLMs."
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


def enrich_schema_with_coded_conditions(enhanced_schema, llm_config):
    """
    Añade una sección 'coded_conditions' al esquema enriquecido usando el LLM para identificar diagnósticos médicos codificados y sus mapeos.
    """
    # Construir un resumen del esquema para el LLM (solo nombres de tablas y columnas)
    schema_info = {table['table_name']: [col['name'] for col in table['columns']] for table in enhanced_schema}
    
    system_prompt_content = (
        "Eres un experto en ingeniería de datos biomédicos. Analiza el siguiente esquema de base de datos "
        "(tablas y columnas) y devuelve una lista de diagnósticos o condiciones médicas codificadas "
        "presentes en la base de datos. Para cada diagnóstico, proporciona un objeto JSON con las claves: "
        "'term' (nombre común del diagnóstico o condición), "
        "'table' (tabla principal donde se instancia o registra esta condición para una entidad, ej. EPIS_DIAGNOSTICS), "
        "'code_field' (columna en 'table' que contiene el código de la condición), "
        "'code_values' (lista de ejemplos de códigos si son evidentes o muy limitados; dejar lista vacía [] si no se pueden inferir directamente o son muchos), "
        "'catalog_table' (tabla de catálogo donde se define el código y su descripción oficial, si aplica, ej. CODR_DIAGNOSTIC_GROUPS), "
        "'description_field' (columna en 'catalog_table' que contiene la descripción textual del código, si 'catalog_table' aplica), "
        "'description' (breve descripción del término médico), "
        "'synonyms' (lista de sinónimos o términos relacionados en lenguaje natural, ej. ['HTA', 'Presión alta'])."
        "Asegúrate de que 'table' y 'code_field' se refieran a donde se usa el código, y 'catalog_table' y 'description_field' a donde se define."
        "Ejemplo de estructura de un elemento de la lista:"
        "{"
        "  'term': 'Hipertensión Arterial',"
        "  'table': 'EPIS_DIAGNOSTICS',"
        "  'code_field': 'CDTE_ID',"
        "  'code_values': ['I10', 'I10.9'],"
        "  'catalog_table': 'CODR_DIAGNOSTIC_GROUPS',"
        "  'description_field': 'DGGR_DESCRIPTION_ES',"
        "  'description': 'Condición médica caracterizada por una presión arterial elevada de forma persistente.',"
        "  'synonyms': ['HTA', 'Presión alta']"
        "}"
        "Responde solo con una lista JSON de objetos, sin explicaciones adicionales ni texto introductorio."
    )
    
    user_prompt_content = f"Esquema de la base de datos (tablas y columnas): {json.dumps(schema_info, ensure_ascii=False)}"

    prompt_messages = [
        {"role": "system", "content": system_prompt_content},
        {"role": "user", "content": user_prompt_content}
    ]
    
    llm_response = call_llm_with_fallbacks(llm_config, prompt_messages)
    coded_conditions = {}
    import re
    try:
        # Intenta encontrar el array JSON en la respuesta del LLM
        match = re.search(r'\[\s*\{.*?\}\s*\]', llm_response, re.DOTALL | re.MULTILINE)
        if not match: # Intento más permisivo si el anterior falla
             match = re.search(r'\[.*\]', llm_response, re.DOTALL)

        if match:
            json_str = match.group(0)
            # Un intento adicional de limpieza por si el LLM añade comentarios o texto residual
            json_str = re.sub(r'//.*?\n|/\*.*?\*/', '', json_str, flags=re.DOTALL | re.MULTILINE) # Eliminar comentarios JS/C++
            json_str = json_str.strip()
            
            # A veces el LLM puede envolver la lista en un bloque de código markdown
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            json_str = json_str.strip()

            coded_list = json.loads(json_str)
            for entry in coded_list:
                term = entry.get("term")
                if term:
                    # Validar que los campos esperados estén presentes, aunque sea con valor None o lista vacía
                    validated_entry = {
                        "term": term,
                        "table": entry.get("table"),
                        "code_field": entry.get("code_field"),
                        "code_values": entry.get("code_values", []),
                        "catalog_table": entry.get("catalog_table"),
                        "description_field": entry.get("description_field"),
                        "description": entry.get("description"),
                        "synonyms": entry.get("synonyms", [])
                    }
                    coded_conditions[term] = validated_entry
        else:
            print(f"No se pudo extraer lista JSON de diagnósticos. Respuesta del LLM: {llm_response}")
    except Exception as e:
        print(f"Error procesando la lista de diagnósticos: {e}\nRespuesta del LLM: {llm_response}")
    return coded_conditions


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
        # Enriquecer con diagnósticos codificados
        coded_conditions = enrich_schema_with_coded_conditions(enhanced_schema, LM_CONFIG_DEFAULTS)
        # Guardar el esquema con la nueva sección
        output_schema = {
            "tables": enhanced_schema,
            "coded_conditions": coded_conditions
        }
        _save_schema(output_schema, ENHANCED_SCHEMA_PATH)
        print(f"Proceso completado. Esquema final guardado en {ENHANCED_SCHEMA_PATH}")
    else:
        print("No se procesaron tablas. Verifique la conexión a la BD y la existencia de tablas.")


if __name__ == "__main__":
    generate_enhanced_schema()
