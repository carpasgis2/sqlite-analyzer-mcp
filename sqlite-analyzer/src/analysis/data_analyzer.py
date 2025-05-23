class DataAnalyzer:
    def __init__(self, connection):
        self.connection = connection

    def analyze_data(self, table_name):
        cursor = self.connection.cursor()
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()

        if not rows:
            return {}

        analysis_results = {
            "total_rows": len(rows),
            "columns": {column: [] for column in cursor.description},
        }

        for row in rows:
            for column, value in zip(cursor.description, row):
                analysis_results["columns"][column[0]].append(value)

        return analysis_results

    def get_statistics(self, data):
        statistics = {}
        for column, values in data["columns"].items():
            if isinstance(values[0], (int, float)):
                statistics[column] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }
        return statistics