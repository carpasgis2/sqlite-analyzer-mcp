import unittest
import os
import sys
import re

# Obtener la ruta al directorio raíz del proyecto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.sql_generator import SQLGenerator

class TestSQLGenerator(unittest.TestCase):
    def test_generate_sql_with_joins(self):
        """Prueba la generación de consultas SQL con JOINs y aliasado robusto."""
        
        # Configurar el generador con tablas y columnas permitidas
        allowed_tables = ["A", "B", "C"]
        allowed_columns = {
            "A": ["id", "name", "x"],
            "B": ["id", "a_id", "y"],
            "C": ["id", "b_id", "z"]
        }
        
        sql_generator = SQLGenerator(allowed_tables, allowed_columns)
        
        # Caso 1: JOIN con diccionario
        info = {
            "tables": ["A"],
            "joins": [{"table": "B", "on": "A.id=B.a_id"}],
            "columns": ["A.x", "B.y"],
            "conditions": {"column": "A.id", "operator": "=", "value": "123"}
        }
        
        sql, params = sql_generator.generate_sql(info)
        
        # Validar que hay un FROM con alias generado para A
        self.assertRegex(sql, r"FROM A t[0-9]+_a")
        # Validar que hay un JOIN con alias generado para B
        self.assertRegex(sql, r"JOIN B t[0-9]+_b ON t[0-9]+_a.id = t[0-9]+_b.a_id")
        # Validar que las columnas están totalmente calificadas
        self.assertRegex(sql, r"SELECT t[0-9]+_a.x, t[0-9]+_b.y")
        self.assertEqual(params, ["123"])
        
        # Caso 2: JOIN con string que debe ser parseado
        info = {
            "tables": ["A"],
            "joins": ['{"table": "B", "on": "A.id=B.a_id"}'],
            "columns": ["A.x", "B.y"],
            "conditions": {"column": "A.id", "operator": "=", "value": "123"}
        }
        
        sql, params = sql_generator.generate_sql(info)
        
        self.assertRegex(sql, r"FROM A t[0-9]+_a")
        self.assertRegex(sql, r"JOIN B t[0-9]+_b ON t[0-9]+_a.id = t[0-9]+_b.a_id")
        
        # Caso 3: JOIN múltiple
        info = {
            "tables": ["A"],
            "joins": [
                {"table": "B", "on": "A.id=B.a_id"},
                {"table": "C", "on": "B.id=C.b_id"}
            ],
            "columns": ["A.x", "B.y", "C.z"],
            "conditions": {"column": "A.id", "operator": "=", "value": "123"}
        }
        
        sql, params = sql_generator.generate_sql(info)
        
        self.assertRegex(sql, r"FROM A t[0-9]+_a")
        self.assertRegex(sql, r"JOIN B t[0-9]+_b ON t[0-9]+_a.id = t[0-9]+_b.a_id")
        self.assertRegex(sql, r"JOIN C t[0-9]+_c ON t[0-9]+_b.id = t[0-9]+_c.b_id")

class TestSQLAliasingEdgeCases(unittest.TestCase):
    def setUp(self):
        self.allowed_tables = ["A", "B", "ON", "SELECT"]
        self.allowed_columns = {
            "A": ["id", "name"],
            "B": ["id", "a_id"],
            "ON": ["id", "b_id"],
            "SELECT": ["id", "on_id"]
        }
        self.sql_generator = SQLGenerator(self.allowed_tables, self.allowed_columns)

    def test_self_join_aliasing(self):
        info = {
            "tables": ["A", "A"],
            "joins": [{"table": "A", "on": "A.id=A.a_id"}],
            "columns": ["A.id", "A.a_id"],
            "conditions": {"column": "A.id", "operator": "=", "value": "1"}
        }
        sql, params = self.sql_generator.generate_sql(info)
        # Debe haber dos aliases distintos para A (ej: t1_a y t2_a)
        aliases = re.findall(r"FROM A (t[0-9]+_a).*JOIN A (t[0-9]+_a)", sql)
        self.assertTrue(aliases or re.search(r"FROM A t[0-9]+_a.*JOIN A t[0-9]+_a", sql))
        # No debe haber alias duplicado ni palabras reservadas
        self.assertNotIn("FROM A A", sql)
        self.assertNotIn("JOIN A A", sql)
        self.assertNotIn("palabra reservada", sql.lower())
        # Las columnas deben estar totalmente calificadas
        self.assertRegex(sql, r"SELECT t[0-9]+_a.id, t[0-9]+_a.a_id")

    def test_reserved_word_alias(self):
        info = {
            "tables": ["ON", "B"],
            "joins": [{"table": "B", "on": "ON.id=B.b_id"}],
            "columns": ["ON.id", "B.b_id"],
            "conditions": {"column": "ON.id", "operator": "=", "value": "2"}
        }
        sql, params = self.sql_generator.generate_sql(info)
        # El alias para ON no debe ser 'ON', debe ser tX_on
        self.assertNotRegex(sql, r"FROM ON ON")
        self.assertRegex(sql, r"FROM ON t[0-9]+_on")
        self.assertRegex(sql, r"JOIN B t[0-9]+_b")
        self.assertNotIn("palabra reservada", sql.lower())
        # Columnas totalmente calificadas
        self.assertRegex(sql, r"SELECT t[0-9]+_on.id, t[0-9]+_b.b_id")

    def test_duplicate_alias(self):
        info = {
            "tables": ["A", "B", "A"],
            "joins": [
                {"table": "B", "on": "A.id=B.a_id"},
                {"table": "A", "on": "B.id=A.id"}
            ],
            "columns": ["A.id", "B.a_id"],
            "conditions": {"column": "A.id", "operator": "=", "value": "3"}
        }
        sql, params = self.sql_generator.generate_sql(info)
        # No debe haber alias duplicado para A
        self.assertNotIn("FROM A A", sql)
        self.assertNotIn("JOIN A A", sql)
        self.assertNotIn("palabra reservada", sql.lower())
        # Debe haber al menos dos alias distintos para A
        self.assertGreaterEqual(len(re.findall(r"A t[0-9]+_a", sql)), 2)
        # Columnas totalmente calificadas
        self.assertRegex(sql, r"SELECT t[0-9]+_a.id, t[0-9]+_b.a_id")

    def test_join_missing_on(self):
        info = {
            "tables": ["A", "B"],
            "joins": ["{\"table\": \"B\"}"],  # JOIN sin ON
            "columns": ["A.id", "B.a_id"],
            "conditions": {"column": "A.id", "operator": "=", "value": "4"}
        }
        sql, params = self.sql_generator.generate_sql(info)
        # Debe corregirse y contener ON
        self.assertIn("JOIN B", sql)
        self.assertIn("ON", sql)
        self.assertNotIn("JOIN B B", sql)
        # Alias robustos
        self.assertRegex(sql, r"FROM A t[0-9]+_a")
        self.assertRegex(sql, r"JOIN B t[0-9]+_b")
        # Columnas totalmente calificadas
        self.assertRegex(sql, r"SELECT t[0-9]+_a.id, t[0-9]+_b.a_id")

    def test_ambiguous_column(self):
        info = {
            "tables": ["A", "B"],
            "joins": [{"table": "B", "on": "A.id=B.a_id"}],
            "columns": ["id"],  # Ambiguo
            "conditions": {"column": "id", "operator": "=", "value": "5"}
        }
        sql, params = self.sql_generator.generate_sql(info)
        # Debe forzar el uso de alias o tabla
        self.assertRegex(sql, r"SELECT t[0-9]+_a.id|t[0-9]+_b.id")
        # La condición también debe estar calificada
        self.assertRegex(sql, r"WHERE t[0-9]+_a.id =|t[0-9]+_b.id =")

# Nota: Los asserts han sido adaptados para aceptar alias generados robustos (tX_tabla) y referencias totalmente calificadas, según la política documentada en ALIASING_POLICY.md

if __name__ == "__main__":
    unittest.main()