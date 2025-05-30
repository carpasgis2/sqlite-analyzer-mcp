\
import sqlite3
import json
import logging
import os
import threading
import math # Importar math para sqrt

logger = logging.getLogger(__name__)

# Ruta absoluta al directorio que contiene este script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT_SINA_MCP = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..")) # Sube dos niveles desde src

# Rutas predeterminadas relativas a la ubicación de este script o raíz del proyecto
_DEFAULT_DB_PATH = os.path.join(_SCRIPT_DIR, "db", "database_new.sqlite3.db")
_DEFAULT_SCHEMA_PATH = os.path.join(_SCRIPT_DIR, "data", "schema_simple.json")
# MODIFICADO: Ruta predeterminada para table_relationships.json ahora en la raíz de sina_mcp
_DEFAULT_RELATIONSHIPS_PATH = os.path.join(_PROJECT_ROOT_SINA_MCP, "table_relationships.json")


class StdevFunc:
    def __init__(self):
        self.M = 0.0
        self.S = 0.0
        self.k = 0

    def step(self, value):
        if value is None:
            return
        # Convertir a float, ya que MEDI_DOSE puede ser numérico pero no necesariamente float
        try:
            float_value = float(value)
        except (ValueError, TypeError):
            logger.warning(f"STDEV: No se pudo convertir el valor '{value}' a float. Se omitirá.")
            return
            
        self.k += 1
        t = self.M
        self.M += (float_value - t) / self.k
        self.S += (float_value - t) * (float_value - self.M)

    def finalize(self):
        if self.k < 2: # La desviación estándar no está definida para menos de 2 muestras
            return None
        return math.sqrt(self.S / (self.k -1)) # Corrección para desviación estándar muestral (k-1)

class DBConnector:
    def __init__(self, db_path: str = None, schema_path: str = None, relationships_path: str = None): # MODIFICADO: Añadido relationships_path
        """
        Inicializa el conector de la base de datos.
        Args:
            db_path: Ruta al archivo de la base de datos SQLite.
            schema_path: Ruta al archivo JSON del esquema de la base de datos.
            relationships_path: Ruta al archivo JSON de relaciones entre tablas.
        """
        # Resolver db_path
        resolved_db_path_input = db_path or os.getenv("DB_PATH")
        if resolved_db_path_input:
            if not os.path.isabs(resolved_db_path_input):
                self.db_path = os.path.abspath(os.path.join(_SCRIPT_DIR, resolved_db_path_input))
                logger.info(f"Ruta relativa de BD '{resolved_db_path_input}' resuelta a: {self.db_path} (relativa al script)")
            else:
                self.db_path = resolved_db_path_input
                logger.info(f"Usando ruta absoluta de BD proporcionada: {self.db_path}")
        else:
            self.db_path = _DEFAULT_DB_PATH
            logger.info(f"Usando ruta de BD predeterminada: {self.db_path}")

        # Resolver schema_path
        resolved_schema_path_input = schema_path or os.getenv("SCHEMA_PATH")
        if resolved_schema_path_input:
            if not os.path.isabs(resolved_schema_path_input):
                self.schema_path = os.path.abspath(os.path.join(_SCRIPT_DIR, resolved_schema_path_input))
                logger.info(f"Ruta relativa de esquema '{resolved_schema_path_input}' resuelta a: {self.schema_path} (relativa al script)")
            else:
                self.schema_path = resolved_schema_path_input
                logger.info(f"Usando ruta absoluta de esquema proporcionada: {self.schema_path}")
        else:
            self.schema_path = _DEFAULT_SCHEMA_PATH
            logger.info(f"Usando ruta de esquema predeterminada: {self.schema_path}")
        
        # MODIFICADO: Resolver relationships_path
        resolved_relationships_path_input = relationships_path or os.getenv("RELATIONSHIPS_PATH")
        if resolved_relationships_path_input:
            if not os.path.isabs(resolved_relationships_path_input):
                # Intentar resolver relativo al script primero, luego relativo a la raíz del proyecto si no existe
                path_rel_to_script = os.path.abspath(os.path.join(_SCRIPT_DIR, resolved_relationships_path_input))
                path_rel_to_project_root = os.path.abspath(os.path.join(_PROJECT_ROOT_SINA_MCP, resolved_relationships_path_input))
                
                if os.path.exists(path_rel_to_script):
                    self.relationships_path = path_rel_to_script
                    logger.info(f"Ruta relativa de relaciones '{resolved_relationships_path_input}' resuelta a: {self.relationships_path} (relativa al script)")
                elif os.path.exists(path_rel_to_project_root):
                    self.relationships_path = path_rel_to_project_root
                    logger.info(f"Ruta relativa de relaciones '{resolved_relationships_path_input}' resuelta a: {self.relationships_path} (relativa a la raíz del proyecto)")
                else:
                    # Si no se encuentra en ninguna de las ubicaciones relativas comunes, se asume relativa al script como fallback
                    self.relationships_path = path_rel_to_script 
                    logger.warning(f"Ruta relativa de relaciones '{resolved_relationships_path_input}' no encontrada ni relativa al script ni a la raíz del proyecto. Usando por defecto: {self.relationships_path} (relativa al script)")
            else:
                self.relationships_path = resolved_relationships_path_input
                logger.info(f"Usando ruta absoluta de relaciones proporcionada: {self.relationships_path}")
        else:
            self.relationships_path = _DEFAULT_RELATIONSHIPS_PATH
            logger.info(f"Usando ruta de relaciones predeterminada: {self.relationships_path}")

        logger.info(f"DBConnector inicializado con db_path: {self.db_path}")
        logger.info(f"DBConnector inicializado con schema_path: {self.schema_path}")
        logger.info(f"DBConnector inicializado con relationships_path: {self.relationships_path}")
        
        # self.conn = None # MODIFICADO: Eliminado, las conexiones se gestionan por llamada
        self.db_structure = None
        self.table_relationships_json_str = None # Cambiado para almacenar el string JSON

    def _create_connection(self) -> sqlite3.Connection: # MODIFICADO: Renombrado a _create_connection y hecho privado
        """Establece y devuelve una nueva conexión con la base de datos SQLite."""
        try:
            if not os.path.exists(self.db_path):
                logger.error(f"El archivo de base de datos no existe en la ruta: {self.db_path}")
                raise FileNotFoundError(f"El archivo de base de datos no existe: {self.db_path}")
            
            conn = sqlite3.connect(self.db_path)
            # Registrar la función STDEV
            conn.create_aggregate("STDEV", 1, StdevFunc)
            logger.info(f"Nueva conexión creada a la base de datos: {self.db_path} (Thread ID: {threading.get_ident()}) y función STDEV registrada.")
            return conn
        except sqlite3.Error as e:
            logger.error(f"Error al crear nueva conexión a la base de datos {self.db_path}: {e}", exc_info=True)
            raise # Re-lanzar la excepción para que el llamador la maneje
        except FileNotFoundError as e:
            logger.error(f"FileNotFoundError al intentar conectar: {e}")
            raise # Re-lanzar
    
    # MODIFICADO: Eliminado el método close(), las conexiones se cierran en execute_query

    def execute_query(self, query: str, params: tuple = None) -> tuple[list[tuple] | None, list[str] | None]:
        """
        Ejecuta una consulta SQL y devuelve los resultados.
        Establece una nueva conexión para cada llamada para asegurar la seguridad en hilos.
        Args:
            query: La consulta SQL a ejecutar.
            params: Parámetros para la consulta SQL (opcional).
        Returns:
            Una tupla (results, column_names).
            results: Lista de tuplas con los resultados, o None si hay error.
            column_names: Lista de nombres de columnas, o None si hay error.
        """
        conn = None
        cursor = None
        try:
            conn = self._create_connection() # MODIFICADO: Usar _create_connection

            cursor = conn.cursor()
            logger.info(f"Ejecutando consulta: {query} con params: {params} (Thread ID: {threading.get_ident()})")
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            results = cursor.fetchall()
            column_names = [description[0] for description in cursor.description] if cursor.description else []
            
            logger.info(f"Consulta ejecutada exitosamente. {len(results)} filas devueltas. (Thread ID: {threading.get_ident()})")
            return results, column_names
        except (sqlite3.Error, FileNotFoundError) as e:
            logger.error(f"Error al ejecutar la consulta SQL '{query}' o al conectar: {e} (Thread ID: {threading.get_ident()})", exc_info=True)
            return None, None
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
                logger.info(f"Conexión a la base de datos cerrada después de execute_query para: {self.db_path} (Thread ID: {threading.get_ident()})")

    def get_db_structure_dict(self, schema_type: str = 'simple') -> dict:
        """
        Carga y devuelve la estructura de la base de datos desde un archivo JSON.
        Args:
            schema_type: 'simple' o 'full'/'enhanced' para cargar el esquema correspondiente.
                         Actualmente, solo usa self.schema_path.
        Returns:
            Un diccionario representando la estructura de la base de datos.
        """
        if self.db_structure is None:
            try:
                # Determinar qué archivo de esquema cargar basado en schema_type podría ser más complejo.
                # Por ahora, se asume que self.schema_path apunta al archivo correcto.
                # Si necesitas cambiar entre schema_simple.json y schema_enhanced.json,
                # esta lógica necesitaría ajustarse o el schema_path debería ser más dinámico.
                logger.info(f"Cargando estructura de BD desde: {self.schema_path} (tipo solicitado: {schema_type})")
                if not os.path.exists(self.schema_path):
                    logger.error(f"El archivo de esquema no existe en la ruta: {self.schema_path}")
                    raise FileNotFoundError(f"El archivo de esquema no existe: {self.schema_path}")

                with open(self.schema_path, 'r', encoding='utf-8') as f:
                    self.db_structure = json.load(f)
                logger.info(f"Estructura de BD cargada desde {self.schema_path}. {len(self.db_structure.get('tables', {}))} tablas encontradas.")
            except FileNotFoundError:
                logger.error(f"Archivo de esquema no encontrado en {self.schema_path}.", exc_info=True)
                self.db_structure = {} # Devolver vacío para evitar errores posteriores
                raise # Re-lanzar para que el pipeline principal sepa que falló
            except json.JSONDecodeError:
                logger.error(f"Error al decodificar el JSON del archivo de esquema: {self.schema_path}.", exc_info=True)
                self.db_structure = {}
                raise
            except Exception as e:
                logger.error(f"Error inesperado al cargar la estructura de la BD: {e}", exc_info=True)
                self.db_structure = {}
                raise
        return self.db_structure

    def get_table_relationships_str(self) -> str | None:
        """
        Carga y devuelve el contenido del archivo de relaciones de tablas como un string JSON.
        Utiliza la ruta configurada en self.relationships_path.
        Cachea el resultado después de la primera carga exitosa.
        Returns:
            Un string JSON con las relaciones de tablas, o None si hay error.
        """
        if self.table_relationships_json_str is None:
            try:
                # MODIFICADO: Usar self.relationships_path
                actual_relationships_path = self.relationships_path 
                logger.info(f"Cargando relaciones de tablas desde: {actual_relationships_path}")
                
                if not os.path.exists(actual_relationships_path):
                    logger.error(f"El archivo de relaciones no existe en: {actual_relationships_path}")
                    # No lanzar excepción aquí, devolver None para que SQLGenerator pueda manejarlo
                    return None 

                with open(actual_relationships_path, 'r', encoding='utf-8') as f:
                    # Leer el archivo directamente como string
                    self.table_relationships_json_str = f.read()
                logger.info(f"Relaciones de tablas cargadas y cacheadas desde {actual_relationships_path} (longitud: {len(self.table_relationships_json_str)} caracteres).")
            except FileNotFoundError: # Aunque ya se verifica con os.path.exists, por si acaso
                logger.error(f"Archivo de relaciones no encontrado en {actual_relationships_path} al intentar leer.", exc_info=True)
                self.table_relationships_json_str = None # Asegurar que es None en caso de error
            except Exception as e:
                logger.error(f"Error inesperado al cargar o leer el archivo de relaciones desde {actual_relationships_path}: {e}", exc_info=True)
                self.table_relationships_json_str = None # Asegurar que es None en caso de error
        
        return self.table_relationships_json_str

    def get_database_structure(self) -> dict:
        """
        Devuelve la estructura completa de la base de datos, incluyendo tablas y columnas.
        Este método es un alias o wrapper para get_db_structure_dict con schema_type='full' (o el que sea por defecto).
        Se utiliza principalmente por SQLGenerator.
        """
        # Asumimos que el schema_path configurado en __init__ ya apunta al esquema "full" o "enhanced"
        # si SQLGenerator lo necesita. Si no, get_db_structure_dict cargará el que esté en self.schema_path.
        # Si se necesita una distinción más clara, se podría añadir un parámetro a __init__ para el "enhanced_schema_path"
        # o modificar get_db_structure_dict para manejar diferentes rutas basadas en schema_type.
        return self.get_db_structure_dict(schema_type='full') # O 'enhanced', según convención

    def get_table_columns(self, table_name: str) -> list:
        """
        Obtiene las columnas para una tabla específica desde la estructura cacheada.
        Args:
            table_name: El nombre de la tabla.
        Returns:
            Una lista de nombres de columnas, o una lista vacía si la tabla no se encuentra.
        """
        if self.db_structure is None:
            self.get_db_structure_dict() # Cargar si aún no se ha hecho
        
        # Buscar la tabla (insensible a mayúsculas/minúsculas para la clave del diccionario)
        found_table_key = None
        for key in self.db_structure.get("tables", {}).keys():
            if key.upper() == table_name.upper():
                found_table_key = key
                break
        
        if found_table_key:
            table_info = self.db_structure["tables"].get(found_table_key, {})
            columns = table_info.get("columns", [])
            # Las columnas pueden ser una lista de strings o una lista de dicts con una clave 'name'
            if columns and isinstance(columns[0], dict):
                return [col.get("name") for col in columns if col.get("name")]
            elif columns and isinstance(columns[0], str):
                return columns
            else:
                logger.warning(f"Formato de columnas inesperado para la tabla '{table_name}' en la estructura cacheada.")
                return []
        else:
            logger.warning(f"Tabla '{table_name}' no encontrada en la estructura de BD cacheada.")
            return []

    def get_all_tables(self) -> list:
        """
        Obtiene todos los nombres de las tablas desde la estructura cacheada.
        Returns:
            Una lista de nombres de tablas.
        """
        if self.db_structure is None:
            self.get_db_structure_dict()
        return list(self.db_structure.get("tables", {}).keys())

if __name__ == '__main__':
    # Ejemplo de uso básico (requiere que los archivos existan en las rutas esperadas)
    # Ajustar rutas si es necesario para la prueba directa.
    # Estas rutas son relativas a donde se ejecute este script directamente.
    # Si db_connector.py está en sqlite-analyzer/src/, y la db está en sqlite-analyzer/src/db/
    # y el schema en sqlite-analyzer/src/data/
    
    # Para probar desde sina_mcp/sqlite-analyzer/src:
    # python db_connector.py
    
    # Calcular rutas relativas desde la ubicación de este archivo
    # current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # db_file = os.path.join(current_script_dir, "db", "database_new.sqlite3.db")
    # schema_file = os.path.join(current_script_dir, "data", "schema_simple.json") # o enhanced
    # relationships_file = os.path.join(current_script_dir, "data", "table_relationships.json")

    # Usar las rutas predeterminadas globales para la prueba, o pasar rutas explícitas si es necesario.
    db_file_test = _DEFAULT_DB_PATH
    schema_file_test = _DEFAULT_SCHEMA_PATH
    relationships_file_test = _DEFAULT_RELATIONSHIPS_PATH
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info(f"Probando DBConnector con DB: {db_file_test}, Schema: {schema_file_test}, Relationships: {relationships_file_test}")

    if not os.path.exists(db_file_test):
        logger.error(f"ERROR EN PRUEBA: El archivo de base de datos NO existe: {db_file_test}")
        logger.error("Por favor, asegúrate de que la base de datos 'database_new.sqlite3.db' exista en el subdirectorio 'db/'")
        logger.error("y que 'schema_simple.json' y 'table_relationships.json' existan en 'data/' relativo a la ubicación de db_connector.py")
    else:
        connector = None
        try:
            # Probar con rutas predeterminadas (que ahora son absolutas y relativas al script)
            connector = DBConnector() # Usa _DEFAULT_DB_PATH y _DEFAULT_SCHEMA_PATH
            # O pasar rutas específicas si se quiere probar algo diferente:
            # connector = DBConnector(db_path=db_file_test, schema_path=schema_file_test)
            connector.connect()
            
            # Probar get_db_structure_dict
            structure = connector.get_db_structure_dict()
            if structure:
                logger.info(f"Estructura de BD cargada (primeras claves): {list(structure.keys())}")
                # logger.info(json.dumps(structure, indent=2, ensure_ascii=False)) # Puede ser muy grande
            else:
                logger.warning("No se pudo cargar la estructura de la BD.")

            # Probar get_table_relationships_str
            # Ya no es necesario pasar relationships_file_path si queremos probar el default
            relations_str = connector.get_table_relationships_str() 
            # O pasar explícitamente:
            # relations_str = connector.get_table_relationships_str(relationships_file_path=relationships_file_test)
            if relations_str and relations_str != "{}":
                logger.info(f"Relaciones entre tablas (JSON string): {relations_str[:200]}...") # Mostrar solo una parte
                try:
                    # Validar que es un JSON parseable
                    json.loads(relations_str)
                    logger.info("El string de relaciones es JSON válido.")
                except json.JSONDecodeError:
                    logger.error("ERROR: El string de relaciones devuelto NO es JSON válido.")
            else:
                logger.info("No se cargaron relaciones entre tablas, el archivo no existe o está vacío.")

            # Probar execute_query (ejemplo)
            # Reemplazar con una tabla y columna que sepas que existe
            # Por ejemplo, si tienes una tabla 'PATI_PATIENTS'
            query_ejemplo = "SELECT COUNT(*) FROM PATI_PATIENTS;" 
            try:
                # Verificar si la tabla PATI_PATIENTS existe en el esquema cargado
                # Asegurarse de que structure['tables'] existe y es un diccionario
                tables_in_structure = structure.get("tables", {}) if isinstance(structure, dict) else {}

                if "PATI_PATIENTS" in (key.upper() for key in tables_in_structure.keys()):
                    results, cols = connector.execute_query(query_ejemplo)
                    if results is not None:
                        logger.info(f"Resultado de '{query_ejemplo}': {results} (Columnas: {cols})")
                    else:
                        logger.warning(f"La consulta '{query_ejemplo}' falló o no devolvió resultados.")
                else:
                    logger.warning(f"La tabla PATI_PATIENTS no parece estar en el esquema cargado ({list(tables_in_structure.keys())}). Omitiendo consulta de ejemplo.")
            except Exception as e_query:
                 logger.error(f"Error al ejecutar consulta de ejemplo: {e_query}")
        except Exception as e:
            logger.error(f"Error durante la prueba de DBConnector: {e}", exc_info=True)
        finally:
            if connector:
                connector.close()
