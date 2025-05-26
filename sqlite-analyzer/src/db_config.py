from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import sqlite3
import os
import json
import re
import time
import threading
import datetime  # Añadir para manejo correcto de tiempo

class DBConnector:
    """Clase para gestionar conexiones a la base de datos y ejecutar consultas"""
    
    # Añadir esta propiedad de clase
    _thread_connections = {}  # Diccionario para almacenar conexiones por thread
    
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
        """Establece la conexión a la base de datos para el thread actual."""
        current_thread_id = threading.get_ident()
        
        try:
            self.logger.info(f"Intentando conectar a la base de datos para thread {current_thread_id}: {self.connection_string}")
            # Usar isolation_level=None para modo autocommit y evitar bloqueos
            connection = sqlite3.connect(
                self.connection_string, 
                timeout=2.0,  # 2 segundos de timeout
                isolation_level=None  # Modo autocommit
            )
            # Configurar timeout adicional
            connection.execute("PRAGMA busy_timeout = 2000")
            
            # Guardar la conexión en el diccionario de threads
            DBConnector._thread_connections[current_thread_id] = connection
            
            self.logger.info(f"Conexión a la base de datos {self.connection_string} establecida exitosamente para thread {current_thread_id}.")
            return connection
        except sqlite3.Error as e:
            self.logger.error(f"Error al conectar a la base de datos: {e}", exc_info=True)
            raise
    
    # Método para obtener la conexión del thread actual
    def get_connection(self):
        """Obtiene la conexión para el thread actual o crea una nueva."""
        current_thread_id = threading.get_ident()
        
        # Si no hay conexión para este thread, crearla
        if current_thread_id not in DBConnector._thread_connections:
            return self.connect()
            
        # Verificar si la conexión existente sigue activa
        try:
            DBConnector._thread_connections[current_thread_id].execute("SELECT 1")
            return DBConnector._thread_connections[current_thread_id]
        except sqlite3.Error:
            # Si hay error, la conexión no es válida, crear nueva
            return self.connect()
    
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

    def execute_query(self, query: str, params: List[Any] = None) -> Tuple[Optional[Union[List[Dict[str, Any]], int]], Optional[str]]:
        """
        Ejecuta una consulta con control robusto de errores y garantiza nunca bloquearse.
        Devuelve una tupla: (resultados, mensaje_de_error).
        Si la consulta es exitosa, mensaje_de_error es None.
        Si hay un error, resultados es None.
        """
        if not query:
            self.logger.error("Se intentó ejecutar una consulta vacía")
            return None, "Consulta vacía"

        current_thread_id = threading.get_ident()
        # Corregido: Paréntesis cerrado en la f-string
        self.logger.info(f'Ejecutando consulta en thread {current_thread_id}: "{query}" con parámetros: {params}')

        connection = self.get_connection()
        if not connection:
            self.logger.error("No se pudo obtener una conexión a la base de datos.")
            return None, "Error de conexión a la base de datos"
            
        connection.execute("PRAGMA busy_timeout = 10000")  # 10 segundos en milisegundos

        cursor = None
        try:
            cursor = connection.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            if query.strip().upper().startswith(("SELECT", "PRAGMA")):
                try:
                    rows = cursor.fetchall()
                    MAX_ROWS = 10000
                    if len(rows) > MAX_ROWS:
                        self.logger.warning(f"Demasiadas filas devueltas ({len(rows)}), truncando a {MAX_ROWS}")
                        rows = rows[:MAX_ROWS]
                    column_names = [col[0] for col in cursor.description] if cursor.description else []
                    results = []
                    for row in rows:
                        try:
                            row_dict = {column_names[i]: value for i, value in enumerate(row)}
                        except Exception as conv_err:
                            self.logger.error(f"Error convirtiendo fila a dict: {conv_err}")
                            row_dict = {str(i): value for i, value in enumerate(row)}
                        results.append(row_dict)
                    self.logger.info(f"Consulta SELECT/PRAGMA ejecutada. Filas devueltas: {len(results)}")
                    return results, None
                except Exception as fetch_err:
                    self.logger.error(f"Error al obtener o procesar resultados: {fetch_err}", exc_info=True)
                    return None, f"Error al procesar resultados: {str(fetch_err)}"
            else:
                rowcount = cursor.rowcount
                self.logger.info(f"Consulta de modificación ejecutada. Filas afectadas: {rowcount}")
                return rowcount, None
        except sqlite3.Error as e:
            self.logger.error(f"Error de SQLite al ejecutar la consulta: {e}", exc_info=True)
            error_message = f"Error de SQLite: {str(e)}"
            try:
                if connection.in_transaction:
                    connection.rollback()
            except Exception as rollback_error:
                self.logger.error(f"Error adicional durante rollback: {rollback_error}")
            return None, error_message
        except Exception as e:
            self.logger.error(f"Error no esperado al ejecutar la consulta: {e}", exc_info=True)
            return None, f"Error inesperado: {str(e)}"
        finally:
            self.logger.info("DBConnector.execute_query: Entering finally block.")
            if cursor is not None:
                try:
                    cursor.close()
                    self.logger.info("DBConnector.execute_query: Cursor closed successfully.")
                except Exception as close_err:
                    self.logger.error(f"DBConnector.execute_query: Error closing cursor: {close_err}")
            self.logger.info("DBConnector.execute_query: Exiting finally block.")

    def execute_sql(self, query: str, params: Optional[List[Any]] = None) -> Tuple[Optional[Union[List[Dict[str, Any]], int]], Optional[str]]:
        """Alias para execute_query para compatibilidad con el pipeline. Devuelve (datos, error_msg)."""
        # Esta función ahora debe coincidir con la nueva firma de execute_query
        result_data, error_msg = self.execute_query(query, params)
        
        # Si execute_query devuelve (int, None) para rowcount, lo convertimos a ([], None) o similar
        # si el pipeline espera una lista para operaciones no SELECT.
        # O, mejor aún, el pipeline debería manejar un int como resultado de operaciones no SELECT.
        # Por ahora, si es un int (rowcount), lo devolvemos tal cual, y el pipeline debe adaptarse.
        if isinstance(result_data, int) and error_msg is None:
             # Para mantener la expectativa de una lista de dicts para SELECTs,
             # y un int para DML, esto está bien.
             # Si el pipeline *siempre* espera una lista, incluso para DML, necesitaríamos:
             # return [{"filas_afectadas": result_data}], None
             pass # Devolver (int, None) como está.

        # Si es una lista (para SELECT) o None (en caso de error), se devuelve tal cual.
        return result_data, error_msg

    def get_database_structure(self) -> Dict[str, Any]:
        """Obtiene la estructura de la base de datos (tablas, columnas, índices, etc.)"""
        structure = {}
        
        try:
            # Obtener la conexión thread-safe en lugar de usar self.connection directamente
            conn_to_use = self.get_connection()
            
            if not conn_to_use:
                self.logger.error("get_database_structure: No se pudo establecer o reutilizar la conexión a la base de datos.")
                return {}
                
            cursor = conn_to_use.cursor()
            
            # Obtener lista de tablas
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            self.logger.debug(f"Tablas encontradas: {tables}")
            
            # Para cada tabla, obtener su estructura
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
            
            self.logger.info(f"Estructura de base de datos obtenida: {len(tables)} tablas procesadas.")
            return structure
            
        except sqlite3.Error as e_sqlite:
            self.logger.error(f"Error SQLite en get_database_structure: {e_sqlite}", exc_info=True)
            return {}
        except Exception as e_generic:
            self.logger.error(f"Error genérico en get_database_structure: {e_generic}", exc_info=True)
            return {}
    
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
    
    def close_all_connections(self):
        """Cierra todas las conexiones abiertas en todos los threads."""
        try:
            for thread_id, conn in list(DBConnector._thread_connections.items()):
                try:
                    self.logger.info(f"Cerrando conexión para thread {thread_id}")
                    conn.commit()  # Intentar hacer commit antes de cerrar
                    conn.close()
                    del DBConnector._thread_connections[thread_id]
                except Exception as e:
                    self.logger.error(f"Error al cerrar conexión para thread {thread_id}: {str(e)}")
        
            # Si hay una conexión principal también cerrarla
            if self.connection:
                try:
                    self.connection.close()
                    self.connection = None
                except Exception as e:
                    self.logger.error(f"Error al cerrar conexión principal: {str(e)}")
                    
            self.logger.info("Todas las conexiones cerradas")
        except Exception as e:
            self.logger.error(f"Error al cerrar conexiones: {str(e)}")

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
        # Si hay una instancia previa con diferente connection_string, cerrar sus conexiones
        if _db_connector_instance is not None:
            _db_connector_instance.close_all_connections()  # Nuevo método que implementaremos
            
        logging.info(f"Creando nueva instancia de DBConnector para: {resolved_connection_string}")
        _db_connector_instance = DBConnector(resolved_connection_string)
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
