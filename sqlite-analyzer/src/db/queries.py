def get_all_tables(connection):
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    cursor = connection.cursor()
    cursor.execute(query)
    tables = cursor.fetchall()
    cursor.close()
    return [table[0] for table in tables]

def get_table_schema(connection, table_name):
    query = f"PRAGMA table_info({table_name});"
    cursor = connection.cursor()
    cursor.execute(query)
    schema = cursor.fetchall()
    cursor.close()
    return schema