import sqlite3

db_path = r"c:\Users\cpascual\PycharmProjects\pythonProject\cursos_actividades\sina_mcp\sqlite-analyzer\src\db\database_new.sqlite3.db"
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Modifica la consulta para que sea la consulta original completa
cur.execute("""
    SELECT mai.ACIN_DESCRIPTION_ES, COUNT(*) as frecuencia
    FROM PATI_USUAL_MEDICATION pum
    JOIN MEDI_ACTIVE_INGREDIENTS mai ON pum.ACIN_ID = mai.ACIN_ID
    GROUP BY mai.ACIN_DESCRIPTION_ES
    ORDER BY frecuencia DESC
    LIMIT 5;
""")

# Obtener todos los resultados si la consulta devuelve algo
results = cur.fetchall()

if results:
    print("Resultados de la consulta completa:")
    # Imprimir encabezados de columna (opcional, pero útil)
    column_names = [description[0] for description in cur.description]
    print(column_names)
    for row in results:
        print(row)
else:
    print("La consulta completa no devolvió resultados.")

conn.close()