# diagnóstico_db.py - Guardar este archivo en la misma carpeta que langchain_chatbot.py
import sqlite3
import os
import time
import sys

# Ruta a la base de datos
db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "db", "database_new.sqlite3.db"))

print(f"Verificando existencia de la base de datos: {db_path}")
if not os.path.exists(db_path):
    print(f"ERROR: El archivo de base de datos no existe en la ruta: {db_path}")
    sys.exit(1)

print(f"Archivo de base de datos encontrado. Tamaño: {os.path.getsize(db_path) / (1024*1024):.2f} MB")

# Intentar una conexión simple sin procesos complejos
print("Intentando conexión directa a SQLite...")
try:
    start = time.time()
    # Usar timeout corto y modo autocommit
    conn = sqlite3.connect(db_path, timeout=2.0, isolation_level=None)
    print(f"Conexión establecida en {time.time() - start:.3f} segundos")
    
    # Verificar tablas
    print("Verificando tablas disponibles...")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"Encontradas {len(tables)} tablas en la base de datos.")
    
    # Probar la tabla específica
    print("\nProbando consulta a PATI_PATIENT_ALLERGIES...")
    try:
        start = time.time()
        cursor.execute("SELECT COUNT(*) FROM PATI_PATIENT_ALLERGIES")
        count = cursor.fetchone()[0]
        print(f"La tabla PATI_PATIENT_ALLERGIES contiene {count} registros (consulta completada en {time.time() - start:.3f} segundos)")
        
        # Probar la consulta específica que está fallando
        print("\nProbando la consulta problemática directamente:")
        start = time.time()
        cursor.execute("SELECT * FROM PATI_PATIENT_ALLERGIES WHERE PATI_ID = ?", [1001])
        rows = cursor.fetchall()
        print(f"Consulta completada en {time.time() - start:.3f} segundos")
        print(f"Encontrados {len(rows)} resultados para PATI_ID = 1001")
        
        if rows:
            print("\nCabeceras de la tabla:")
            col_names = [description[0] for description in cursor.description]
            print(", ".join(col_names))
            
            print("\nPrimer resultado:")
            for i, col in enumerate(col_names):
                print(f"{col}: {rows[0][i]}")
    except Exception as e:
        print(f"ERROR al consultar PATI_PATIENT_ALLERGIES: {e}")
    
    conn.close()
    print("\nDiagnóstico completado con éxito.")
    
except Exception as e:
    print(f"ERROR al conectar a la base de datos: {e}")