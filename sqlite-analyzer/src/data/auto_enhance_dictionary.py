\
import argparse
import json
import os
import shutil
import sqlite3 # Añadido
import logging # Añadido para logging del LLM
import ast # Añadido para evaluar listas si el LLM no devuelve JSON puro
import re # Añadido para el re.search en el manejo de listas
import sys # Añadido para modificar sys.path

# Bloque para manejar la importación de llm_utils de forma flexible
try:
    # Intento de importación relativa (ideal cuando se ejecuta como parte de un paquete)
    from ..llm_utils import call_llm_with_fallbacks, LLM_MODEL_NAME, extract_json_from_llm_response
except ImportError:
    # Fallback para cuando el script se ejecuta directamente (ej. python path/to/script.py)
    # Modificamos sys.path para encontrar llm_utils.py en el directorio 'src'
    current_script_dir = os.path.dirname(os.path.abspath(__file__)) # Directorio 'data'
    # parent_dir_of_current es el directorio 'src', que contiene llm_utils.py
    parent_dir_of_current = os.path.abspath(os.path.join(current_script_dir, '..')) 
    
    if parent_dir_of_current not in sys.path:
        sys.path.insert(0, parent_dir_of_current)
        
    try:
        from llm_utils import call_llm_with_fallbacks, LLM_MODEL_NAME, extract_json_from_llm_response
    except ImportError as e:
        # Si aún falla, podría haber un problema más profundo.
        # Usar print aquí ya que el logger podría no estar configurado si llm_utils falla.
        print(f"ERROR CRÍTICO: No se pudo importar llm_utils después de modificar sys.path. {e}")
        print(f"sys.path actual: {sys.path}")
        print("Asegúrese de que llm_utils.py exista en el directorio 'src/' (relativo a la ubicación del script) y que la estructura del proyecto sea la esperada.")
        raise # Re-lanzar el error para detener la ejecución.

# Configurar un logger básico para este script si es necesario, o confiar en el de llm_utils
logger = logging.getLogger(__name__)
# Asegurarse de que la configuración básica del logging solo se llama una vez
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Ruta por defecto al archivo dictionary.json
DEFAULT_DICT_PATH = os.path.join(os.path.dirname(__file__), 'dictionary.json')
# Ruta por defecto a la base de datos SQLite
DEFAULT_DB_PATH = r'c:\Users\cpascual\PycharmProjects\pythonProject\cursos_actividades\sina_mcp\sqlite-analyzer\src\db\database_new.sqlite3.db' # Añadido y corregido

# --- Funciones para cargar/guardar el diccionario (adaptadas de update_dictionary.py) ---
def load_dictionary(file_path=None):
    current_path = file_path if file_path else DEFAULT_DICT_PATH
    if not os.path.exists(current_path) and file_path is None:
        dir_name = os.path.dirname(current_path)
        if not os.path.exists(dir_name):
            try:
                os.makedirs(dir_name)
                print(f"Directorio {dir_name} creado.")
            except OSError as e:
                print(f"Error al crear el directorio {dir_name}: {e}")
                return {"tables": {}, "concepts": {}}
    
    if not os.path.exists(current_path):
        print(f"Archivo {current_path} no encontrado. Se creará uno vacío si se guarda.")
        return {"tables": {}, "concepts": {}} # Devuelve estructura base si no existe
    
    try:
        with open(current_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content:
                print(f"Advertencia: El archivo {current_path} está vacío. Inicializando con estructura base.")
                data = {"tables": {}, "concepts": {}}
            else:
                data = json.loads(content)

            if not isinstance(data, dict):
                raise json.JSONDecodeError("El contenido no es un diccionario JSON válido.", content, 0)

            if 'tables' not in data or not isinstance(data['tables'], dict):
                data['tables'] = {}
            if 'concepts' not in data or not isinstance(data['concepts'], dict):
                data['concepts'] = {}
            return data
    except json.JSONDecodeError as e:
        print(f"Error: El archivo {current_path} está corrupto o no es un JSON válido: {e}. Se retornará una estructura base.")
        return {"tables": {}, "concepts": {}}
    except IOError as e:
        print(f"Error al leer el archivo {current_path}: {e}")
        return {"tables": {}, "concepts": {}}

def save_dictionary(data, file_path=None, backup=True):
    current_path = file_path if file_path else DEFAULT_DICT_PATH
    
    if backup and os.path.exists(current_path):
        backup_path = current_path + '.bak'
        try:
            shutil.copy2(current_path, backup_path)
            print(f"Backup creado en {backup_path}")
        except IOError as e:
            print(f"Error al crear el backup {backup_path}: {e}")

    try:
        with open(current_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Diccionario guardado exitosamente en {current_path}")
    except IOError as e:
        print(f"Error al guardar el diccionario en {current_path}: {e}")

# --- Placeholder para la llamada al LLM ---
def call_llm_for_enhancement(prompt_text: str, task_type: str, context: dict = None) -> any:
    """
    Llama a un LLM para generar mejoras basadas en el prompt y el tipo de tarea.
    Utiliza las funciones de llm_utils.py.
    
    Args:
        prompt_text (str): El prompt principal para el LLM.
        task_type (str): 'description_es', 'common_terms', o 'examples'.
        context (dict): Información adicional como nombre de tabla/columna.

    Returns:
        str: Para 'description_es'.
        list[str]: Para 'common_terms' o 'examples'.
        None: Si falla.
    """
    logger.info(f"Solicitando al LLM: Task='{task_type}', Context='{context}', Prompt (inicio)='{prompt_text[:100]}...'")

    messages = [
        {"role": "system", "content": "Eres un asistente experto en bases de datos y generación de metadatos. Proporciona respuestas concisas y precisas en el formato solicitado."},
        {"role": "user", "content": prompt_text}
    ]

    llm_config = {
        "temperature": 0.5, # Un poco de creatividad pero no demasiada
        "max_tokens": 500   # Ajustar según la longitud esperada de la respuesta
    }
    raw_response = None # Inicializar raw_response
    try:
        # Asegúrate de que llm_utils.llm_client está inicializado y LLM_MODEL_NAME es correcto
        raw_response = call_llm_with_fallbacks(config=llm_config, messages=messages, model_name=LLM_MODEL_NAME)

        if not raw_response:
            logger.warning(f"LLM no devolvió respuesta para la tarea {task_type} con contexto {context}")
            return None

        logger.debug(f"Respuesta cruda del LLM para {task_type} ({context}): {raw_response}")

        if task_type == "description_es":
            # Eliminar comillas si el LLM las añade innecesariamente alrededor de la descripción
            return raw_response.strip().strip('"').strip("'") # Quita comillas dobles y luego simples de los extremos
        
        elif task_type == "common_terms" or task_type == "examples":
            # Intentar extraer JSON primero
            parsed_json = extract_json_from_llm_response(raw_response)
            if isinstance(parsed_json, list):
                # Asegurarse de que todos los elementos son strings
                return [str(item) for item in parsed_json]
            
            # Si no es JSON, intentar evaluar como literal de Python (para listas como "['a', 'b']")
            logger.debug(f"No se pudo parsear como JSON, intentando ast.literal_eval para: {raw_response}")
            try:
                # Limpiar la respuesta si está envuelta en texto adicional antes de la lista
                match = re.search(r'\\[(.*?)\\]', raw_response, re.DOTALL) # Busca contenido entre corchetes
                list_to_eval = raw_response
                if match:
                    list_to_eval = match.group(0) # Obtener la subcadena que es la lista, ej: "['a', 'b']"
                
                evaluated_list = ast.literal_eval(list_to_eval)
                if isinstance(evaluated_list, list):
                        # Asegurarse de que todos los elementos son strings
                    return [str(item) for item in evaluated_list]

            except (ValueError, SyntaxError, TypeError) as e_eval:
                logger.warning(f"Error al evaluar la respuesta del LLM como lista para {task_type} ({context}): {e_eval}. Respuesta: {raw_response}")
                # Como último recurso, si es una sola cadena, devolverla como una lista de un solo elemento
                if isinstance(raw_response, str) and raw_response.strip():
                    # Evitar que una descripción larga se convierta en una lista de un solo elemento si no es lo esperado.
                    # Esto es más para common_terms o examples si el LLM devuelve una sola cadena en lugar de una lista.
                    # Si la respuesta no empieza con '[' y termina con ']', es probable que no sea una lista malformada.
                    if not (raw_response.strip().startswith('[') and raw_response.strip().endswith(']')):
                        return [raw_response.strip()]
            return None # Si todo falla después de los intentos de parseo

    except Exception as e_call:
        logger.error(f"Error durante la llamada al LLM o procesamiento de su respuesta para {task_type} ({context}): {e_call}. Respuesta cruda fue: {raw_response}")
        return None
    # Esta línea estaba causando el error de try sin except/finally si se descomentaba o si el flujo llegaba aquí incorrectamente.
    # return None # Asegurarse de que todas las rutas de código retornan explícitamente.
# --- Nueva función para obtener valores de la base de datos ---
def get_distinct_values_from_db(db_path: str, table_name: str, column_name: str, limit: int = 5) -> list[str] | None:
    """
    Obtiene valores distintos de una columna específica en una tabla de la base de datos.

    Args:
        db_path (str): Ruta al archivo de la base de datos SQLite.
        table_name (str): Nombre de la tabla.
        column_name (str): Nombre de la columna.
        limit (int): Número máximo de valores distintos a devolver.

    Returns:
        list[str] | None: Una lista de valores distintos como strings, o None si ocurre un error.
    """
    if not os.path.exists(db_path):
        print(f"Error: La base de datos no se encuentra en {db_path}")
        return None
    
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Es importante sanitizar los nombres de tabla y columna si vinieran de fuentes no confiables,
        # pero aquí asumimos que son controlados internamente o validados previamente.
        query = f"SELECT DISTINCT \"{column_name}\" FROM \"{table_name}\" WHERE \"{column_name}\" IS NOT NULL LIMIT {limit};"
        print(f"Ejecutando consulta: {query}")
        cursor.execute(query)
        rows = cursor.fetchall()
        # Convertir todos los valores a string, ya que examples en JSON suelen ser strings.
        return [str(row[0]) for row in rows if row[0] is not None]
    except sqlite3.Error as e:
        print(f"Error al consultar la base de datos para {table_name}.{column_name}: {e}")
        return None
    finally:
        if conn:
            conn.close()

def main():
    parser = argparse.ArgumentParser(description="Mejora automáticamente dictionary.json usando un LLM simulado y datos de la BD.")
    parser.add_argument('--dict-path', default=DEFAULT_DICT_PATH,
                        help=f'Ruta al archivo dictionary.json de SALIDA (default: {DEFAULT_DICT_PATH})')
    parser.add_argument('--base-dict-path', default=None, # Nuevo argumento
                        help='Ruta al archivo dictionary.json de ENTRADA (base). Si no se provee, se usa --dict-path como entrada.')
    parser.add_argument('--db-path', default=DEFAULT_DB_PATH, 
                        help=f'Ruta a la base de datos SQLite (default: {DEFAULT_DB_PATH})')
    parser.add_argument('--overwrite-descriptions', action='store_true', 
                        help="Sobrescribir descripciones existentes con las generadas por el LLM.")
    parser.add_argument('--overwrite-common-terms', action='store_true',
                        help="Sobrescribir common_terms existentes.")
    parser.add_argument('--overwrite-examples', action='store_true',
                        help="Sobrescribir ejemplos existentes.")
    parser.add_argument('--process-tables', type=str, default=None,
                        help="Lista de tablas específicas a procesar, separadas por coma. Si no se especifica, procesa todas.")

    args = parser.parse_args()

    # Determinar la ruta del diccionario de entrada
    input_dict_path = args.base_dict_path if args.base_dict_path else args.dict_path

    print(f"Cargando diccionario base desde: {input_dict_path}")
    dictionary_data = load_dictionary(input_dict_path)
    # La siguiente condición hacía que el script saliera si el diccionario
    # se inicializaba como {"tables": {}, "concepts": {}} (p.ej. si el archivo no existía).
    # Se elimina para permitir la creación de un diccionario nuevo o el procesamiento de uno vacío.
    # load_dictionary ya imprime errores si el archivo está corrupto, pero devuelve una estructura base.
    # if not dictionary_data or (not dictionary_data.get('tables') and not dictionary_data.get('concepts')):
    #     print(f"No se pudo cargar un diccionario válido desde {input_dict_path} o está vacío. Saliendo.")
    #     return

    tables_to_process = None
    if args.process_tables:
        tables_to_process = [table.strip() for table in args.process_tables.split(',')]

    for table_name, table_info in dictionary_data.get('tables', {}).items():
        if tables_to_process and table_name not in tables_to_process:
            continue

        print(f"\\nProcesando tabla: {table_name}") # Corrected from \\\\n to \\n
        table_info = dictionary_data['tables'][table_name] # Asegurar que es la referencia correcta

        # 1. Mejorar descripción de la tabla
        if args.overwrite_descriptions or not table_info.get('description_es'):
            prompt = f"Genera una descripción concisa y profesional en español para la tabla de base de datos llamada '{table_name}'. La tabla podría contener columnas como: {list(table_info.get('columns', {}).keys()) if table_info.get('columns') else 'desconocidas'}. La descripción no debe exceder las 2 frases y debe ser puramente descriptiva del contenido y propósito de la tabla. Devuelve solo el texto de la descripción, sin frases introductorias como 'Aquí tienes una descripción:'." # Corrected f\\" and \\'
            generated_desc = call_llm_for_enhancement(prompt, "description_es", {"name": table_name, "type": "table"})
            if generated_desc:
                table_info['description_es'] = generated_desc
                print(f"  Descripción de tabla '{table_name}' actualizada.") # Corrected f\\" and \\'

        # 2. Mejorar common_terms de la tabla
        if args.overwrite_common_terms or not table_info.get('common_terms'):
            prompt = f"Genera una lista JSON de 2-3 términos comunes o sinónimos en español para la tabla '{table_name}', cuya descripción es: '{table_info.get('description_es', '(sin descripción)')}'. Los términos deben ser útiles para buscar o referirse a esta tabla. Devuelve solo la lista JSON. Ejemplo de formato de salida: [\\\"término1\\\", \\\"término2\\\"]" # Corrected f\\" and \\'
            generated_terms = call_llm_for_enhancement(prompt, "common_terms", {"name": table_name, "type": "table"})
            if generated_terms:
                table_info['common_terms'] = generated_terms
                print(f"  Common_terms de tabla '{table_name}' actualizados.") # Corrected f\\" and \\'

        # 3. Mejorar examples de la tabla (ejemplos de uso o propósito)
        if args.overwrite_examples or not table_info.get('examples'):
            prompt = f"Genera una lista JSON con 1-2 ejemplos conceptuales breves en español que ilustren el propósito o el tipo de información que se encuentra en la tabla '{table_name}', descrita como: '{table_info.get('description_es', '(sin descripción)')}'. Cada ejemplo debe ser una cadena de texto. Devuelve solo la lista JSON. Ejemplo de formato de salida: [\\\"Ejemplo conceptual A.\\\", \\\"Ejemplo conceptual B.\\\"]" # Corrected f\\" and \\'
            generated_examples = call_llm_for_enhancement(prompt, "examples", {"name": table_name, "type": "table"})
            if generated_examples:
                table_info['examples'] = generated_examples
                print(f"  Ejemplos de tabla '{table_name}' actualizados.") # Corrected f\\" and \\'

        # Procesar columnas
        if 'columns' in table_info and isinstance(table_info['columns'], dict):
            for column_name, column_info_ref in table_info['columns'].items():
                column_info = table_info['columns'][column_name] # Asegurar referencia
                print(f"  Procesando columna: {column_name}") # Corrected f\\"

                # 3.1. Mejorar descripción de columna
                if args.overwrite_descriptions or not column_info.get('description_es'):
                    prompt_col_desc = f"Genera una descripción concisa y profesional en español para la columna '{column_name}' de la tabla '{table_name}'. La tabla se describe como: '{table_info.get('description_es', '(sin descripción de tabla)')}'. La descripción de la columna no debe exceder 1-2 frases y debe ser puramente descriptiva. Devuelve solo el texto de la descripción, sin frases introductorias." # Corrected f\\" and \\'
                    gen_col_desc = call_llm_for_enhancement(prompt_col_desc, "description_es", {"name": column_name, "type": "column", "table": table_name})
                    if gen_col_desc:
                        column_info['description_es'] = gen_col_desc
                        print(f"    Descripción de columna '{column_name}' actualizada.") # Corrected f\\" and \\'
                
                # 3.2. Mejorar common_terms de columna
                if args.overwrite_common_terms or not column_info.get('common_terms'):
                    prompt_col_terms = f"Genera una lista JSON de 2-3 términos comunes o sinónimos en español para la columna '{column_name}' (en tabla '{table_name}'), cuya descripción es: '{column_info.get('description_es', '(sin descripción)')}'. Los términos deben ser útiles para referirse a esta columna. Devuelve solo la lista JSON. Ejemplo de formato de salida: [\\\"término_col1\\\", \\\"término_col2\\\"]" # Corrected f\\" and \\'
                    gen_col_terms = call_llm_for_enhancement(prompt_col_terms, "common_terms", {"name": column_name, "type": "column", "table": table_name})
                    if gen_col_terms:
                        column_info['common_terms'] = gen_col_terms
                        print(f"    Common_terms de columna '{column_name}' actualizados.") # Corrected f\\" and \\'

                # 3.3. Obtener examples de columna DESDE LA BASE DE DATOS
                # La clave 'examples' para columnas ahora se llenará prioritariamente desde la BD.
                if args.overwrite_examples or not column_info.get('examples'):
                    print(f"    Intentando obtener ejemplos para {table_name}.{column_name} desde la base de datos...")
                    db_examples = get_distinct_values_from_db(args.db_path, table_name, column_name, limit=5)
                    
                    if db_examples: # Si se obtuvieron ejemplos de la BD
                        column_info['examples'] = db_examples
                        print(f"    Ejemplos de columna '{column_name}' actualizados desde la BD: {db_examples}")
                    elif not column_info.get('examples'): # Si no hay de la BD y no había antes
                        # Según la nueva directriz, no usamos LLM para ejemplos de columnas si la BD no los da.
                        # Podríamos dejarlo vacío o con un mensaje, o usar LLM si se permite explícitamente como fallback.
                        # Por ahora, si la BD no da nada y no había nada, se queda sin ejemplos o con los que tuviera.
                        # Si se quisiera usar LLM como fallback aquí, se añadiría la lógica de call_llm_for_enhancement.
                        print(f"    No se encontraron ejemplos en la BD para '{column_name}' y no hay ejemplos previos. Se omite la generación por LLM para ejemplos de columna.")
                    elif column_info.get('examples') and not db_examples:
                         print(f"    No se encontraron ejemplos en la BD para '{column_name}'. Se conservan los ejemplos existentes si los hay y no se especifica --overwrite-examples.")
        
        # Guardar el diccionario después de procesar cada tabla
        # Esto asegura que el progreso se guarda incrementalmente.
        # La función save_dictionary ya maneja la creación de backups.
        save_dictionary(dictionary_data, args.dict_path, backup=True) # Aseguramos backup=True aquí también
        print(f"Progreso guardado para la tabla: {table_name}")

    # Guardar una última vez al final por si acaso, aunque ya se guarda en cada iteración.
    # save_dictionary(dictionary_data, args.dict_path) # Comentado ya que se guarda en el bucle
    print(f"\\nProceso de mejora completado. Diccionario final guardado en: {args.dict_path}")

if __name__ == '__main__':
    main()
