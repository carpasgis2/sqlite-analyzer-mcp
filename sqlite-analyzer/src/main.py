import sqlite3
import os
import sys # Añadido por si se necesita para sys.path, aunque las importaciones directas deberían funcionar
import json # NUEVO
from pathlib import Path # NUEVO
from dotenv import load_dotenv # NUEVO

# Importaciones de la aplicación
from db.connection import create_connection
from db.queries import get_all_tables, get_table_schema # Usadas por la lógica original de main
from analysis.schema_analyzer import SchemaAnalyzer # Usadas por la lógica original de main
from analysis.data_analyzer import DataAnalyzer # Usadas por la lógica original de main
from utils.helpers import print_results # Usadas por la lógica original de main

# NUEVAS importaciones para la prueba del pipeline
from db_config import DBConnector, DEFAULT_DB_CONFIG
from pipeline import process_natural_language_query, ChatMemory, load_schema_as_string 
from table_relationship import load_table_relationships
from sql_utils import load_terms_mapping 
from rag_enhancements import initialize_enhanced_rag


def main():
    # Cargar .env al inicio de main()
    # Asume que main.py está en .../src, entonces .parent es .../src, .parent.parent es .../sqlite-analyzer
    env_path = Path(__file__).resolve().parent.parent / ".env" 
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"Variables de entorno cargadas desde {env_path}")
    else:
        print(f"Archivo .env no encontrado en {env_path}. Asegúrese de que las variables de entorno (ej. API keys) estén configuradas.")

    # Usar rutas absolutas basadas en la ubicación del script
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # La base de datos está dentro del directorio db en el directorio src
    db_dir = os.path.join(current_script_dir, "db")
    database = os.path.join(db_dir, "database.sqlite3.db")
    
    # Asegurarse de que el directorio db exista
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
        print(f"Directorio creado: {db_dir}")
    
    print(f"Intentando conectar a la base de datos: {database}")
    
    # Si el archivo existe pero está corrupto, preguntar si se desea crear uno nuevo
    if os.path.isfile(database):
        try:
            # Intentar abrir la base de datos para verificar si es válida
            test_conn = sqlite3.connect(database)
            test_conn.cursor().execute("SELECT 1")
            test_conn.close()
        except sqlite3.DatabaseError:
            print("Error: El archivo de base de datos parece estar corrupto.")
            response = input("¿Desea crear una nueva base de datos? (s/n): ")
            if response.lower() == 's':
                try:
                    os.remove(database)
                    print(f"Archivo corrupto eliminado. Creando nueva base de datos en: {database}")
                except Exception as e:
                    print(f"No se pudo eliminar el archivo corrupto: {e}")
                    return
            else:
                print("Operación cancelada.")
                return
    
    # Establecer conexión a la base de datos
    conn = create_connection(database)
    
    if conn is not None:
        try:
            # Analizar el esquema de la base de datos
            schema_analyzer = SchemaAnalyzer(conn)
            tables = get_all_tables(conn)
            
            # Si no hay tablas, podríamos crear algunas de ejemplo
            if not tables:
                print("La base de datos no contiene tablas.")
                response = input("¿Desea crear algunas tablas de ejemplo? (s/n): ")
                if response.lower() == 's':
                    create_example_tables(conn)
                    tables = get_all_tables(conn)
                else:
                    print("No hay tablas para analizar. Saliendo.")
                    conn.close()
                    return
            
            schema_info = {}
            for table in tables:
                schema_info[table] = get_table_schema(conn, table)
            
            print_results("Esquema de la base de datos:", schema_info)
            
            # Analizar los datos en las tablas
            data_analyzer = DataAnalyzer(conn)
            for table in tables:
                data_analysis = data_analyzer.analyze_data(table)
                print_results(f"Análisis de datos para la tabla {table}:", data_analysis)
            

            # --- Inicio de la sección de prueba del pipeline ---
            print("\\n\\n--- Iniciando prueba del pipeline de consulta ---")
            
            # Definir rutas (consistente con pipeline.py y la estructura del proyecto)
            # main.py está en src. current_script_dir es .../src
            SCRIPT_DIR_paths = current_script_dir # .../src
            PROJECT_ROOT_DIR_paths = os.path.abspath(os.path.join(SCRIPT_DIR_paths, "..")) # .../sqlite-analyzer
            WORKSPACE_ROOT_DIR_paths = os.path.abspath(os.path.join(PROJECT_ROOT_DIR_paths, "..")) # .../sina_mcp
            
            SCHEMA_DATA_DIR_paths = os.path.join(SCRIPT_DIR_paths, "data") # .../src/data
            
            SCHEMA_FULL_PATH_val = os.path.join(SCHEMA_DATA_DIR_paths, "schema_enhanced.json")
            SCHEMA_SIMPLE_PATH_val = os.path.join(SCHEMA_DATA_DIR_paths, "schema_simple.json")
            # Usar la definición de ruta de pipeline.py para table_relationships.json (esperado en WORKSPACE_ROOT_DIR)
            RELACIONES_PATH_FOR_LLM_val = os.path.join(WORKSPACE_ROOT_DIR_paths, "table_relationships.json")
            TERMS_DICT_PATH_val = os.path.join(SCHEMA_DATA_DIR_paths, "dictionary.json")

            # Verificar existencia de archivos clave
            required_files = {
                "Schema completo": SCHEMA_FULL_PATH_val,
                "Schema simple": SCHEMA_SIMPLE_PATH_val,
                "Relaciones LLM": RELACIONES_PATH_FOR_LLM_val,
                "Diccionario términos": TERMS_DICT_PATH_val,
                "Base de datos": database 
            }
            missing_files = False
            print("\\nVerificando archivos requeridos para el pipeline:")
            for name, path_val in required_files.items():
                if not os.path.exists(path_val):
                    print(f"Error: Archivo requerido no encontrado - {name}: {path_val}")
                    missing_files = True
                else:
                    print(f"OK: {name} encontrado en {path_val}")

            if missing_files:
                print("\\nNo se puede continuar con la prueba del pipeline debido a archivos faltantes.")
                # No usamos return aquí para permitir que el finally se ejecute, 
                # pero el pipeline no se ejecutará.
            else:
                print(f"\\nUsando DBConnector con base de datos: {database}")
                db_connector = DBConnector(db_path=database, config=DEFAULT_DB_CONFIG)
                
                print(f"Cargando schema completo desde: {SCHEMA_FULL_PATH_val}")
                schema_full_str = load_schema_as_string(SCHEMA_FULL_PATH_val)
                print(f"Cargando schema simple desde: {SCHEMA_SIMPLE_PATH_val}")
                schema_simple_str = load_schema_as_string(SCHEMA_SIMPLE_PATH_val)
                
                print(f"Cargando relaciones desde: {RELACIONES_PATH_FOR_LLM_val}")
                relationships_data = load_table_relationships(RELACIONES_PATH_FOR_LLM_val)
                table_relationships_str = json.dumps(relationships_data)
                
                print(f"Cargando diccionario de términos desde: {TERMS_DICT_PATH_val}")
                terms_data = load_terms_mapping(TERMS_DICT_PATH_val)
                terms_dict_str = json.dumps(terms_data)
                
                chat_memory = ChatMemory()
                
                print("Inicializando EnhancedSchemaRAG...")
                enhanced_rag = initialize_enhanced_rag(db_connector, SCHEMA_FULL_PATH_val, RELACIONES_PATH_FOR_LLM_val)
                
                pregunta_test = "¿Qué médicos han tratado a pacientes diagnosticados con 'diabetes tipo 2' que también están tomando 'metformina'?"
                print(f"\\nProcesando pregunta: {pregunta_test}")
                
                resultado_pipeline = process_natural_language_query(
                    user_question=pregunta_test,
                    db_connector=db_connector,
                    schema_full_str=schema_full_str,
                    schema_simple_str=schema_simple_str,
                    table_relationships_str=table_relationships_str,
                    terms_dict_str=terms_dict_str,
                    chat_memory=chat_memory,
                    enhanced_rag_instance=enhanced_rag,
                    is_direct_sql=False
                )
                
                print("\\n--- Resultado del Pipeline ---")
                if resultado_pipeline.get("error"):
                    print(f"Error en el pipeline: {resultado_pipeline['error']}")
                if resultado_pipeline.get("sql_query"):
                    print(f"SQL Generado: {resultado_pipeline['sql_query']}")
                if resultado_pipeline.get("results"):
                    print(f"Resultados de la consulta: {resultado_pipeline['results']}")
                if resultado_pipeline.get("explanation"):
                    print(f"Explicación: {resultado_pipeline['explanation']}")

                print("\\n--- Detalle completo del resultado (JSON) ---")
                print(json.dumps(resultado_pipeline, indent=2, ensure_ascii=False))
            # --- Fin de la sección de prueba del pipeline ---

        except Exception as e:
            print(f"Error al analizar la base de datos o ejecutar el pipeline: {e}")
        finally:
            conn.close()
    else:
        print("Error! No se pudo establecer la conexión a la base de datos.")

def create_example_tables(conn):
    """Crea tablas de ejemplo en la base de datos"""
    cursor = conn.cursor()
    
    # Crear tabla de usuarios
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS usuarios (
        id INTEGER PRIMARY KEY,
        nombre TEXT NOT NULL,
        email TEXT UNIQUE,
        edad INTEGER,
        fecha_registro DATE
    )
    ''')
    
    # Insertar algunos datos de ejemplo
    cursor.execute('''
    INSERT INTO usuarios (nombre, email, edad, fecha_registro) VALUES 
        ('Ana García', 'ana@ejemplo.com', 28, '2022-01-15'),
        ('Juan López', 'juan@ejemplo.com', 35, '2022-02-20'),
        ('María Rodríguez', 'maria@ejemplo.com', 42, '2022-01-10'),
        ('Carlos Martínez', 'carlos@ejemplo.com', 19, '2022-03-05')
    ''')
    
    # Crear tabla de productos
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS productos (
        id INTEGER PRIMARY KEY,
        nombre TEXT NOT NULL,
        precio REAL,
        stock INTEGER,
        categoria TEXT
    )
    ''')
    
    # Insertar algunos datos de ejemplo
    cursor.execute('''
    INSERT INTO productos (nombre, precio, stock, categoria) VALUES 
        ('Laptop', 1200.50, 10, 'Electrónica'),
        ('Teléfono', 850.75, 25, 'Electrónica'),
        ('Mesa', 299.99, 5, 'Muebles'),
        ('Silla', 150.00, 20, 'Muebles')
    ''')
    
    conn.commit()
    print("Tablas de ejemplo creadas con éxito.")

if __name__ == "__main__":
    main()