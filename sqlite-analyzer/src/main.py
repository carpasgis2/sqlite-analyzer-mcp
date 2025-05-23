import sqlite3
import os
import sys
from db.connection import create_connection
from db.queries import get_all_tables, get_table_schema
from analysis.schema_analyzer import SchemaAnalyzer
from analysis.data_analyzer import DataAnalyzer
from utils.helpers import print_results


def main():
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
        
        except Exception as e:
            print(f"Error al analizar la base de datos: {e}")
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