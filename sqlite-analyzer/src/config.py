\
import os

# Directorio base del script actual (config.py)
# Asumiendo que config.py está en sqlite-analyzer/src/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Directorio raíz del proyecto (sqlite-analyzer/)
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Directorio raíz del espacio de trabajo (sina_mcp/)
# Esto es una suposición basada en la estructura previa, ajustar si es necesario.
WORKSPACE_ROOT_DIR = os.path.abspath(os.path.join(PROJECT_ROOT_DIR, "..", "..")) # Ajustado para subir dos niveles desde src

# Directorio de datos para esquemas dentro de src
SCHEMA_DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# Rutas de los esquemas y archivos de relaciones
SCHEMA_FULL_PATH = os.path.join(SCHEMA_DATA_DIR, "schema_enhanced.json")
SCHEMA_SIMPLE_PATH = os.path.join(SCHEMA_DATA_DIR, "schema_simple.json")

# Rutas para los archivos de relaciones
# Usando WORKSPACE_ROOT_DIR para archivos que podrían estar fuera de la carpeta 'src'
RELACIONES_PATH_FOR_LLM = os.path.join(WORKSPACE_ROOT_DIR, "sina_mcp", "table_relationships.json")
RELACIONES_PATH = os.path.join(WORKSPACE_ROOT_DIR, "sina_mcp", "table_relationships.json")

# Ruta para el diccionario de términos
TERMS_DICT_PATH = os.path.join(SCHEMA_DATA_DIR, "dictionary.json")

# Configuración para el número máximo de filas en la respuesta
MAX_RESPONSE_DATA_ROWS_CONFIG = 100  # Un valor por defecto, ajústalo según sea necesario

# Podrías añadir otras configuraciones globales aquí, por ejemplo:
# LOG_LEVEL = "INFO"
# DATABASE_PATH = os.path.join(PROJECT_ROOT_DIR, "data", "database.sqlite") # Ejemplo

MAX_CONTEXT_TOKENS_FOR_SCHEMA_RELATIONS_APPROX = 60000  # O el valor que consideres apropiado

# Configuración para OpenAI
LLM_PROVIDER_FOR_TOKEN_COUNT = "openai"
LLM_MODEL_NAME_FOR_QUERY_GENERATION = "gpt-3.5-turbo"
MAX_TOKENS_FOR_QUERY_GENERATION = 4096 # Máximos tokens para la generación de consultas
TEMPERATURE_FOR_QUERY_GENERATION = 0.0 # Temperatura para la generación de consultas

MAX_SQL_EXECUTION_TIME_PIPELINE = 90 # Tiempo máximo de ejecución de SQL en el pipeline en segundos
