def create_connection(db_file):
    """Establece una conexión a la base de datos SQLite especificada por db_file."""
    import sqlite3
    from sqlite3 import Error

    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print("Conexión establecida con la base de datos.")
    except Error as e:
        print(f"Error al conectar a la base de datos: {e}")
    
    return conn