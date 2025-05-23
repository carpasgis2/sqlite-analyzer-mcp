class SchemaAnalyzer:
    def __init__(self, connection):
        self.connection = connection

    def analyze_schema(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        schema_info = {}
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            schema_info[table_name] = [
                {"name": column[1], "type": column[2], "notnull": column[3], "default": column[4]}
                for column in columns
            ]
        
        cursor.close()
        return schema_info