import pytest
import sqlite3
from src.medical_database_chatbot import (
    connect_to_database,
    load_database_metadata,
    generate_sql_query,
    find_numeric_column
)

def test_load_database_metadata():
    # Usar una DB en memoria para metadata
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE PATI_PATIENTS (
            PATI_ID INTEGER PRIMARY KEY,
            NAME TEXT,
            AGE INTEGER
        )
    """)
    cursor.execute("INSERT INTO PATI_PATIENTS (NAME, AGE) VALUES ('John', 30), ('Jane', 25)")
    conn.commit()

    metadata = load_database_metadata(conn)
    assert "PATI_PATIENTS" in metadata
    cols = metadata["PATI_PATIENTS"]["columns"]
    types = metadata["PATI_PATIENTS"]["types"]
    pk = metadata["PATI_PATIENTS"]["primary_key"]
    assert cols == ["PATI_ID", "NAME", "AGE"]
    assert types == ["INTEGER", "TEXT", "INTEGER"]
    assert pk == ["PATI_ID"]

@ pytest.mark.parametrize("keywords, expected_sql", [
    (
        {'actions':['COUNT'], 'tables':['PATI_PATIENTS'], 'columns':[], 'conditions':[], 'values':[]},
        "SELECT COUNT(*) as COUNT_TOTAL FROM PATI_PATIENTS"
    ),
    (
        {'actions':['SELECT'], 'tables':['PATI_PATIENTS'], 'columns':[], 'conditions':[], 'values':[]},
        "SELECT * FROM PATI_PATIENTS LIMIT 10"
    ),
    (
        {'actions':['SELECT'], 'tables':['PATI_PATIENTS'], 'columns':['PATI_ID'], 'conditions':[], 'values':[]},
        "SELECT PATI_ID FROM PATI_PATIENTS LIMIT 10"
    )
])
def test_generate_sql_query(keywords, expected_sql):
    db_structure = {
        "PATI_PATIENTS": {
            "columns": ["PATI_ID", "NAME"],
            "types": ["INTEGER", "TEXT"],
            "primary_key": ["PATI_ID"]
        }
    }
    sql = generate_sql_query(keywords, db_structure)
    assert sql == expected_sql


def test_find_numeric_column_with_integer_type():
    fake_structure = {
        "table": {
            "columns": ["A", "B", "C"],
            "types": ["TEXT", "INTEGER", "REAL"],
            "primary_key": []
        }
    }
    col = find_numeric_column(fake_structure, "table")
    assert col == "B"


def test_find_numeric_column_fallback_id():
    fake_structure = {
        "table": {
            "columns": ["A_ID", "B", "C"],
            "types": ["TEXT", "TEXT", "TEXT"],
            "primary_key": ["A_ID"]
        }
    }
    col = find_numeric_column(fake_structure, "table")
    # Sin tipos numéricos y único _ID es clave primaria, no hay fallback => None
    assert col is None
