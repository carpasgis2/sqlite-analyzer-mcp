import os
import sqlite3
import re
import json
import requests
from typing import Dict, List, Any
from datetime import datetime

# Funciones extraídas del notebook para poder importarlas en pruebas

def connect_to_database() -> sqlite3.Connection:
    import glob

    db_path = "db/database.sqlite3.db"
    if not os.path.isfile(db_path):
        sqlite_files = glob.glob("**/*.db", recursive=True) + glob.glob("**/*.sqlite3", recursive=True)
        if sqlite_files:
            db_path = sqlite_files[0]
    conn = sqlite3.connect(db_path)
    return conn


def load_database_metadata(conn: sqlite3.Connection) -> dict:
    """
    Devuelve la estructura de la base de datos en el formato esperado por el pipeline:
    {
        "tables": [
            {
                "name": "NOMBRE_TABLA",
                "columns": [
                    {"name": "NOMBRE_COLUMNA"}, ...
                ]
            },
            ...
        ]
    }
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    structure = {"tables": []}
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table});")
        columns = cursor.fetchall()
        structure["tables"].append({
            "name": table,
            "columns": [{"name": col[1]} for col in columns]
        })
    return structure


def find_numeric_column(db_structure: Dict[str, Dict[str, List[str]]], table_name: str) -> str:
    if table_name not in db_structure:
        return None
    cols = db_structure[table_name]['columns']
    types = db_structure[table_name]['types']
    # Buscar primer tipo numérico
    numeric_types = ['INT', 'REAL', 'NUMERIC', 'DECIMAL', 'FLOAT', 'DOUBLE']
    for col, typ in zip(cols, types):
        if any(num in typ.upper() for num in numeric_types):
            return col
    # Fallback: columna _ID no PK
    for col in cols:
        if '_ID' in col and col not in db_structure[table_name].get('primary_key', []):
            return col
    return None


def generate_sql_query(keywords: Dict[str, List[str]], db_structure: Dict[str, Dict[str, List[str]]]) -> str:
    # Determinar acción
    action = 'SELECT'
    if keywords.get('actions'):
        for a in keywords['actions']:
            if a.upper() in ['SELECT', 'COUNT']:
                action = a.upper()
                break

    main_table = keywords.get('tables', [None])[0] or 'PATI_PATIENTS'
    # Generar SQL según acción
    if action == 'COUNT':
        return f"SELECT COUNT(*) as COUNT_TOTAL FROM {main_table}"

    # SELECT
    selected_columns = '*'
    cols = keywords.get('columns', [])
    if cols:
        selected_columns = ', '.join(cols)
    sql = f"SELECT {selected_columns} FROM {main_table}"
    if action == 'SELECT':
        sql += " LIMIT 10"
    return sql
