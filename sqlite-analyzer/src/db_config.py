from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import sqlite3
import os
import json
import re
import time

class DBConnector:
    """Clase para gestionar conexiones a la base de datos y ejecutar consultas"""
    
    def __init__(self, connection_string: str):
        """
        Inicializa el conector de base de datos.
        
        Args:
            connection_string: Cadena de conexión a la base de datos (p.ej., ruta al archivo SQLite)
        """
        self.connection_string = connection_string
        self.connection: Optional[sqlite3.Connection] = None
        self.logger = logging.getLogger(__name__)
    
    def connect(self):
        """Establece conexión con la base de datos SQLite."""
        if self.connection is None: # Solo intentar conectar si no hay conexión activa
            try:
                self.logger.info(f"Intentando conectar a la base de datos: {self.connection_string}")
                
                # Crear directorio para la base de datos si es un path y no existe
                db_dir = os.path.dirname(self.connection_string)
                if db_dir and not os.path.exists(db_dir): # db_dir no es vacío y no existe
                    os.makedirs(db_dir, exist_ok=True)
                    self.logger.info(f"Directorio creado para la base de datos: {db_dir}")

                # Forzar timeout bajo para evitar bloqueos largos
                self.connection = sqlite3.connect(self.connection_string, check_same_thread=False, timeout=2.0, isolation_level=None)
                # Evitar bloqueos indefinidos estableciendo busy_timeout (ms)
                try:
                    self.connection.execute('PRAGMA busy_timeout = 2000')
                    self.logger.info('PRAGMA busy_timeout establecido a 2000 ms.')
                except Exception as e_prag:
                    self.logger.warning(f'No se pudo establecer busy_timeout: {e_prag}')
                self.logger.info(f"Conexión a la base de datos {self.connection_string} establecida exitosamente (timeout=2s).")
            except sqlite3.Error as e:
                self.logger.error(f"Error al conectar a la BD SQLite ({self.connection_string}): {str(e)}")
                self.connection = None # Asegurar que la conexión es None si falla
                raise # Re-lanzar la excepción para que el llamador sepa del fallo crítico
            except Exception as e: # Capturar otros posibles errores (p.ej. problemas de path)
                self.logger.error(f"Error inesperado durante la conexión a {self.connection_string}: {str(e)}")
                self.connection = None
                raise
    
    def validate_sql_query(self, query: str) -> Tuple[bool, str]:
        """
        Valida que la consulta SQL sea correcta en términos de sintaxis básica.
        
        Args:
            query: Consulta SQL a validar
            
        Returns:
            Tupla de (es_válida, mensaje_error)
        """
        # --- NUEVO: Manejar query None ---
        if query is None:
            self.logger.warning("validate_sql_query recibió una consulta None.")
            return False, "Error: La consulta SQL es None"

        query = query.strip()
        
        # Verificar si está vacía
        if not query:
            return False, "Error: La consulta SQL está vacía"
        
        # Verificar si es una consulta SELECT sin FROM
        if query.upper().startswith("SELECT"):
            # Buscar la cláusula FROM
            if " FROM " not in query.upper():
                return False, "Error: No se especificó ninguna tabla (falta cláusula FROM)"
            
            # Verificar que haya algo después del FROM
            from_parts = query.upper().split(" FROM ")
            if len(from_parts) > 1:
                after_from = from_parts[1].strip()
                if not after_from or after_from.startswith(";"):
                    return False, "Error: No se especificó ninguna tabla después de FROM"
        
        # Verificar si es una consulta UPDATE sin tabla
        if query.upper().startswith("UPDATE"):
            parts = query.split()
            if len(parts) < 2:
                return False, "Error: No se especificó ninguna tabla después de UPDATE"
        
        # Verificar si es una consulta DELETE sin FROM
        if query.upper().startswith("DELETE") and " FROM " not in query.upper():
            return False, "Error: No se especificó ninguna tabla (falta cláusula FROM en DELETE)"
        
        return True, ""

    def execute_query(self, query: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """
        Ejecuta una consulta SQL y devuelve los resultados.
        
        Args:
            query: Consulta SQL a ejecutar
            params: Parámetros para la consulta parametrizada (opcional)
            
        Returns:
            Lista de diccionarios con los resultados, o lista con un diccionario de error.
        """
        self.logger.debug(f"execute_query llamada con query: '{query}', params: {params}")

        is_valid, error_message = self.validate_sql_query(query)
        if not is_valid:
            self.logger.warning(f"Consulta SQL inválida: {query} - Error: {error_message}")
            return [{"error": error_message}]
            
        try:
            if self.connection is None:
                self.logger.info("Conexión no existente, intentando conectar...")
                self.connect() # Intentar conectar. connect() ahora eleva excepción si falla.
            
            if self.connection is None: # Debería ser redundante si connect() eleva excepción en fallo
                self.logger.error("Fallo crítico: No se pudo establecer la conexión a la base de datos.")
                return [{"error": "Fallo crítico al conectar con la base de datos."}]

            cursor = self.connection.cursor()
            self.logger.info(f"Ejecutando consulta: \"{query}\" con parámetros: {params}")
            
            import time
            start_time = time.time()
            self.logger.info(f"[TIMER] Inicio de ejecución de consulta SQL.")
            self.logger.info(f"[EXEC_DETAIL] Antes de cursor.execute() para la consulta: '{query}'")
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            self.logger.info(f"[EXEC_DETAIL] Después de cursor.execute() para la consulta: '{query}'")
            exec_time = time.time() - start_time
            self.logger.info(f"[TIMER] Consulta ejecutada en {exec_time:.3f} segundos.")
            
            query_upper = query.strip().upper()
            results: List[Dict[str, Any]] = []

            if query_upper.startswith("SELECT") or query_upper.startswith("PRAGMA"):
                if cursor.description:
                    column_names = [description[0] for description in cursor.description]
                    self.logger.info(f"[EXEC_DETAIL] Antes de cursor.fetchall() para la consulta: '{query}'")
                    fetched_rows = cursor.fetchall()
                    self.logger.info(f"[EXEC_DETAIL] Después de cursor.fetchall(). Filas obtenidas: {len(fetched_rows)}")
                    results = [dict(zip(column_names, row)) for row in fetched_rows]
                    self.logger.info(f"Consulta SELECT/PRAGMA ejecutada. Filas devueltas: {len(results)}")
                else:
                    self.logger.info("Consulta SELECT/PRAGMA ejecutada, pero no se encontraron descripciones de columnas (posiblemente no hay resultados o es una consulta especial).")
                    results = []
            elif any(query_upper.startswith(stmt) for stmt in ["INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"]):
                try:
                    self.logger.info(f"[EXEC_DETAIL] Antes de self.connection.commit() para la consulta DML/DDL: '{query}'")
                    self.connection.commit()
                    self.logger.info(f"[EXEC_DETAIL] Después de self.connection.commit()")
                    rows_affected = cursor.rowcount if cursor.rowcount != -1 else 'N/A'
                    self.logger.info(f"Consulta DML/DDL ejecutada y cambios confirmados. Filas afectadas: {rows_affected}")
                    results = [{"status": "success", "rows_affected": rows_affected}]
                except sqlite3.Error as commit_err:
                    self.logger.error(f"Error al hacer commit de los cambios para la consulta '{query}': {commit_err}")
                    try:
                        self.logger.info("[EXEC_DETAIL] Antes de self.connection.rollback() debido a error en commit.")
                        self.connection.rollback()
                        self.logger.info("[EXEC_DETAIL] Después de self.connection.rollback().")
                        self.logger.info("Rollback realizado debido a error en commit.")
                    except sqlite3.Error as rb_err:
                        self.logger.error(f"Error durante el rollback: {rb_err}")
                    raise commit_err
            else: # Otros tipos de sentencias
                try:
                    self.logger.info(f"[EXEC_DETAIL] Antes de self.connection.commit() para consulta de tipo desconocido: '{query}'")
                    self.connection.commit() # Intentar commit por si acaso (ej. PRAGMA que modifica estado)
                    self.logger.info(f"[EXEC_DETAIL] Después de self.connection.commit() para consulta de tipo desconocido.")
                    rows_affected = cursor.rowcount if cursor.rowcount != -1 else 'N/A'
                    self.logger.info(f"Consulta de tipo desconocido ejecutada. Se intentó commit. Filas afectadas: {rows_affected}")
                    results = [{"status": "executed", "rows_affected": rows_affected}]
                except sqlite3.Error as e: # Ej: "cannot commit - no transaction is active"
                    self.logger.warning(f"Commit fallido para consulta de tipo desconocido (puede ser normal para algunas sentencias): {e}. Consulta: {query}")
                    if cursor.description: # Aún podría haber resultados para leer
                         column_names = [description[0] for description in cursor.description]
                         fetched_rows = cursor.fetchall()
                         results = [dict(zip(column_names, row)) for row in fetched_rows]
                         self.logger.info(f"Consulta de tipo desconocido ejecutada. Filas devueltas: {len(results)}")
                    else:
                         results = [{"status": "executed_no_results_or_commit_not_applicable"}]
            
            return results

        except sqlite3.Error as e:
            self.logger.error(f"Error de SQLite al ejecutar consulta '{query}' con params {params}: {str(e)}")
            if self.connection: # Intentar rollback si hay una conexión y posible transacción activa
                try:
                    self.logger.info("[EXEC_DETAIL] Antes de self.connection.rollback() debido a error en ejecución.")
                    self.connection.rollback()
                    self.logger.info("[EXEC_DETAIL] Después de self.connection.rollback().")
                    self.logger.info("Rollback realizado debido a error en ejecución de consulta.")
                except sqlite3.Error as rb_err:
                    self.logger.error(f"Error durante el rollback tras fallo de ejecución: {rb_err}")
            return [{"error": f"Error de SQLite: {str(e)}"}]
        except Exception as e: # Captura general para otros errores inesperados
            self.logger.error(f"Error inesperado al ejecutar consulta '{query}' con params {params}: {str(e)}")
            return [{"error": f"Error inesperado: {str(e)}"}]

    def execute_sql(self, query: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """Alias para execute_query para compatibilidad con el pipeline"""
        return self.execute_query(query, params)

    def get_database_structure(self) -> Dict[str, Any]:
        """
        Obtiene la estructura completa de la base de datos conectada.
        
        Returns:
            Diccionario con la estructura de la BD (tablas, columnas, tipos, etc.)
        """
        structure: Dict[str, Any] = {}
        conn_to_use: Optional[sqlite3.Connection] = None
        
        try:
            if self.connection:
                self.logger.debug("get_database_structure: Reutilizando conexión existente.")
                conn_to_use = self.connection
            else:
                self.logger.info("get_database_structure: Conexión no existente, intentando conectar...")
                self.connect() # Esto establecerá self.connection o lanzará una excepción
                conn_to_use = self.connection

            if not conn_to_use:
                self.logger.error("get_database_structure: No se pudo establecer o reutilizar la conexión a la base de datos.")
                # Intentar cargar estructura desde caché si la conexión falla
                cache_path = "schema_rag_cache.json"
                if os.path.exists(cache_path):
                    self.logger.warning(f"Cargando estructura desde caché debido a fallo de conexión: {cache_path}")
                    try:
                        with open(cache_path, 'r', encoding='utf-8') as f:
                            cache_data = json.load(f)
                            if "db_structure" in cache_data:
                                return cache_data["db_structure"]
                    except Exception as e_cache:
                        self.logger.error(f"Error al cargar estructura desde caché: {e_cache}")
                return structure # Devuelve estructura vacía si no hay conexión ni caché válida

            cursor = conn_to_use.cursor()
            
            # Obtener lista de tablas
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Log detallado de las tablas encontradas
            self.logger.debug(f"Tablas encontradas en la BD para estructura: {tables}")
            
            for table_name in tables:
                structure[table_name] = {
                    "name": table_name,
                    "columns": [],
                    "indexes": [],
                    "foreign_keys": []
                }
                
                # Obtener información de columnas
                cursor.execute(f"PRAGMA table_info(\"{table_name}\")")  # Asegurar comillas para nombres de tabla
                for col in cursor.fetchall():
                    column_info = {
                        "name": col[1],
                        "type": col[2],
                        "not_null": col[3] == 1,
                        "default_value": col[4],
                        "primary_key": col[5] == 1
                    }
                    structure[table_name]["columns"].append(column_info)
                
                # Obtener información de índices
                cursor.execute(f"PRAGMA index_list(\"{table_name}\")")
                for idx in cursor.fetchall():
                    idx_name = idx[1]
                    idx_info_cursor = conn_to_use.execute(f"PRAGMA index_info(\"{idx_name}\")")
                    idx_cols = [col_info[2] for col_info_cursor in idx_info_cursor for col_info in conn_to_use.execute(f"PRAGMA index_info(\"{idx_name}\")")] # Re-ejecutar para obtener todas las columnas
                    
                    structure[table_name]["indexes"].append({
                        "name": idx_name,
                        "columns": idx_cols,
                        "unique": idx[2] == 1
                    })

                # Obtener información de claves foráneas
                cursor.execute(f"PRAGMA foreign_key_list(\"{table_name}\")")
                for fk in cursor.fetchall():
                    structure[table_name]["foreign_keys"].append({
                        "from_column": fk[3],
                        "to_table": fk[2],
                        "to_column": fk[4],
                        "on_update": fk[5],
                        "on_delete": fk[6],
                    })
            
            num_tables_found = len(tables)
            self.logger.info(f"Estructura de base de datos obtenida: {num_tables_found} tablas procesadas.")
            return structure
            
        except sqlite3.Error as e_sqlite:
            self.logger.error(f"Error de SQLite al obtener la estructura de la BD: {e_sqlite}")
        except Exception as e_generic:
            self.logger.error(f"Error inesperado al obtener la estructura de la BD: {e_generic}")
        
        # Fallback a caché si cualquier excepción ocurre durante la obtención de la estructura
        self.logger.warning("Intentando cargar estructura desde caché debido a error durante la obtención.")
        cache_path = "schema_rag_cache.json"
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    if "db_structure" in cache_data:
                        self.logger.info("Estructura cargada desde caché como fallback.")
                        return cache_data["db_structure"]
            except Exception as e_cache_fallback:
                self.logger.error(f"Error al cargar estructura desde caché (fallback): {e_cache_fallback}")
        
        return {} # Devuelve estructura vacía si todo falla

    def close(self):
        """Cierra la conexión a la base de datos si está abierta."""
        if self.connection:
            try:
                self.logger.info(f"Cerrando conexión a la base de datos: {self.connection_string}")
                self.connection.close()
                self.connection = None
                self.logger.info("Conexión a la base de datos cerrada exitosamente.")
            except sqlite3.Error as e:
                self.logger.error(f"Error al cerrar la conexión a la BD ({self.connection_string}): {str(e)}")
            except Exception as e: # Otros posibles errores
                self.logger.error(f"Error inesperado al cerrar la conexión ({self.connection_string}): {str(e)}")

# Ruta por defecto a la base de datos SQLite
DEFAULT_DB_CONNECTION_STRING = r"C:\\Users\\cpascual\\PycharmProjects\\pythonProject\\cursos_actividades\\sina_mcp\\sqlite-analyzer\\src\\db\\database_new.sqlite3.db"

# Para compatibilidad con la importación en pipeline.py, aunque usemos DEFAULT_DB_CONNECTION_STRING
DEFAULT_DB_CONFIG = {
    "connection_string": DEFAULT_DB_CONNECTION_STRING
}

# Variable global para mantener una única instancia del conector si se desea (Singleton pattern)
_db_connector_instance: Optional[DBConnector] = None

def get_db_connector(connection_string: Optional[str] = None) -> DBConnector:
    """
    Obtiene una instancia del conector de base de datos.
    Si no se proporciona connection_string, utiliza DEFAULT_DB_CONNECTION_STRING.
    Implementa un patrón Singleton simple para reutilizar la instancia.
    """
    global _db_connector_instance
    
    resolved_connection_string = connection_string if connection_string is not None else DEFAULT_DB_CONNECTION_STRING
    
    if _db_connector_instance is None or _db_connector_instance.connection_string != resolved_connection_string:
        logging.info(f"Creando nueva instancia de DBConnector para: {resolved_connection_string}")
        _db_connector_instance = DBConnector(resolved_connection_string)
        # Opcionalmente, se podría llamar a _db_connector_instance.connect() aquí mismo
        # para asegurar que la conexión se establece al obtener el conector,
        # pero la implementación actual de DBConnector.execute_query ya lo maneja.
    else:
        logging.info(f"Reutilizando instancia existente de DBConnector para: {resolved_connection_string}")
        
    return _db_connector_instance

def get_database_path_from_config(config: Optional[Dict[str, Any]] = None) -> str:
    """
    Obtiene la ruta de la base de datos desde una configuración o usa el valor por defecto.
    """
    if config and "connection_string" in config:
        return config["connection_string"]
    return DEFAULT_DB_CONNECTION_STRING
