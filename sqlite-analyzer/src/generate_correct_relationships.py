import sqlite3
import json
import argparse
import os
import logging
import sys

# Configurar logging básico
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Añadir el directorio actual al sys.path para permitir importaciones relativas
# si db_config.py está en el mismo directorio.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Intentar importar la función original de db_config.py
    from db_config import extract_foreign_keys as extract_fks_from_db_config
    logging.info("Se importó 'extract_foreign_keys' desde 'db_config' exitosamente.")
except ImportError as e:
    logging.warning(f"No se pudo importar 'extract_foreign_keys' desde 'db_config': {e}. Se usará una implementación local.")
    
    def extract_fks_from_db_config(db_path_local: str) -> dict:
        logging.info(f"Usando implementación local de extract_foreign_keys para {db_path_local}")
        conn = sqlite3.connect(db_path_local)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row[0] for row in cursor.fetchall()]
        
        foreign_keys_data = {}
        for table_name_local in tables:
            # PRAGMA foreign_key_list devuelve una fila por cada columna en una FK.
            # Columnas: id, seq, table (referenciada), from (col en actual), to (col en referenciada), ...
            cursor.execute(f"PRAGMA foreign_key_list('{table_name_local}');")
            fks = cursor.fetchall()
            if fks:
                if table_name_local not in foreign_keys_data:
                    foreign_keys_data[table_name_local] = []
                for fk_row in fks:
                    # fk_row es una tupla: (id, seq, referenced_table, current_table_column, referenced_table_column, ...)
                    # Índices: referenced_table=2, current_table_column=3, referenced_table_column=4
                    foreign_keys_data[table_name_local].append({
                        "referenced_table": fk_row[2], # Tabla a la que se referencia
                        "foreign_key": fk_row[3],      # Columna FK en la tabla actual
                        "referenced_column": fk_row[4], # Columna PK/Unique en la tabla referenciada
                        "confidence": "high" 
                    })
        conn.close()
        logging.info(f"Extracción local de FKs completada. Encontradas relaciones para {len(foreign_keys_data)} tablas.")
        return foreign_keys_data

def generate_correct_relationships_file(db_path: str, output_path: str):
    """
    Genera un archivo JSON con las relaciones de clave foránea de una base de datos SQLite.
    El formato de salida es un diccionario donde las claves son nombres de tablas y los
    valores son listas de diccionarios que describen las relaciones salientes.
    """
    logging.info(f"Iniciando generación de relaciones para la base de datos: {db_path}")
    
    if not os.path.exists(db_path):
        logging.error(f"El archivo de base de datos no existe: {db_path}")
        print(f"Error: El archivo de base de datos no existe: {db_path}")
        return

    raw_relationships = extract_fks_from_db_config(db_path)
    
    if not raw_relationships:
        logging.warning(f"No se encontraron relaciones de clave foránea en {db_path} o la función de extracción falló.")
        final_relationships = {}
    else:
        logging.info(f"Se encontraron {len(raw_relationships)} tablas con relaciones de clave foránea.")
        final_relationships = {}
        for table_name, fk_list in raw_relationships.items():
            if table_name not in final_relationships:
                final_relationships[table_name] = []
            for fk_detail in fk_list:
                final_relationships[table_name].append({
                    "column": fk_detail["foreign_key"], # Columna en la tabla actual que es FK
                    "foreign_table": fk_detail["referenced_table"], # Tabla a la que apunta la FK
                    "foreign_column": fk_detail["referenced_column"] # Columna en la tabla referenciada
                })
        logging.info("Transformación de formato de relaciones completada.")

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_relationships, f, indent=4, ensure_ascii=False)
        logging.info(f"Archivo de relaciones generado correctamente en: {output_path}")
        print(f"Archivo de relaciones generado correctamente en: {output_path}")
    except IOError as e:
        logging.error(f"Error al escribir el archivo JSON en {output_path}: {e}")
        print(f"Error al escribir el archivo JSON en {output_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de archivo de relaciones de tabla para SQLite.")
    parser.add_argument(
        "--db", 
        required=True, 
        help="Ruta al archivo de base de datos SQLite."
    )
    parser.add_argument(
        "--output", 
        required=True, 
        help="Ruta al archivo JSON de salida para las relaciones."
    )
    
    args = parser.parse_args()
    
    db_file_path = os.path.abspath(args.db)
    output_file_path = os.path.abspath(args.output)

    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logging.info(f"Directorio de salida creado: {output_dir}")
        except OSError as e:
            logging.error(f"No se pudo crear el directorio de salida {output_dir}: {e}")
            print(f"Error: No se pudo crear el directorio de salida {output_dir}: {e}")
            sys.exit(1)
            
    generate_correct_relationships_file(db_file_path, output_file_path)
