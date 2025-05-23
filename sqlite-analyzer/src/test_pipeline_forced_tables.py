import sys
import os
import logging
import re

# Configuración básica de logging para capturar problemas tempranos
log_file_path_basic = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_run.log')
logging.basicConfig(filename=log_file_path_basic, level=logging.DEBUG, filemode='w',
                    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s")
logging.info("Inicio del script test_pipeline_forced_tables.py")
print("DEBUG: Inicio del script test_pipeline_forced_tables.py (stdout)")

try:
    # --- Start of new diagnostic prints ---
    path_logger = logging.getLogger('PathDiagnostics')
    path_logger.setLevel(logging.DEBUG) # Asegurarse de que el logger capture DEBUG
    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'path_diagnostics.log')
    # Asegurarse de que el directorio para el log existe
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    file_handler = logging.FileHandler(log_file_path, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    path_logger.addHandler(file_handler)
    # También añadir un StreamHandler para ver los logs de path_logger en la consola si es necesario
    # stream_handler = logging.StreamHandler(sys.stdout)
    # stream_handler.setFormatter(formatter)
    # path_logger.addHandler(stream_handler)


    path_logger.info(f"--- test_pipeline_forced_tables.py ---")
    path_logger.info(f"Initial sys.path: {sys.path}")
    path_logger.info(f"Current working directory: {os.getcwd()}")
    path_logger.info(f"__file__: {__file__}")
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    path_logger.info(f"Directory of current file (test_pipeline_forced_tables.py): {current_file_dir}")
    project_root = os.path.abspath(os.path.join(current_file_dir, '..'))
    path_logger.info(f"Calculated project_root (should be 'sqlite-analyzer' directory): {project_root}")

    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        path_logger.info(f"sys.path AFTER inserting project_root: {sys.path}")
    else:
        path_logger.info(f"project_root '{project_root}' already in sys.path.")
    path_logger.info(f"Current sys.path: {sys.path}") # Movido para que siempre se loguee
    path_logger.info(f"--- End of path diagnostic prints ---")
    print("DEBUG: Path diagnostics complete (stdout)")


    logging.info("Intentando importar 'pipeline.chatbot_pipeline'")
    print("DEBUG: Intentando importar 'pipeline.chatbot_pipeline' (stdout)")
    from pipeline import chatbot_pipeline
    logging.info("'pipeline.chatbot_pipeline' importado exitosamente")
    print("DEBUG: 'pipeline.chatbot_pipeline' importado exitosamente (stdout)")

    logging.info("Intentando importar 'db_config.SQLiteConnector'")
    print("DEBUG: Intentando importar 'db_config.SQLiteConnector' (stdout)")
    from db_config import SQLiteConnector
    logging.info("'db_config.SQLiteConnector' importado exitosamente")
    print("DEBUG: 'db_config.SQLiteConnector' importado exitosamente (stdout)")
    
    # Configurar logging para ver el flujo interno del pipeline (puede ser redundante si ya está configurado arriba)
    # logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s")
    # Usar el logger raíz o uno específico para el pipeline
    pipeline_logger = logging.getLogger() # Obtiene el logger raíz
    pipeline_logger.setLevel(logging.INFO) # Asegura que INFO y superior se capturen
    # Si quieres un handler específico para la consola para el pipeline logger:
    # console_handler = logging.StreamHandler(sys.stdout)
    # console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s"))
    # pipeline_logger.addHandler(console_handler)


    # Obtener la ruta del directorio actual del script
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construir la ruta a la base de datos relativa al directorio actual del script
    db_name = "database_new.sqlite3.db"
    # Priorizar la base de datos en la raíz del proyecto /db, luego src/db
    db_path_root = os.path.join(project_root, "db", db_name) 
    db_path_src = os.path.join(current_script_dir, "db", db_name) 

    db_path = None
    logging.info(f"Buscando base de datos. Raíz: {db_path_root}, Src: {db_path_src}")
    print(f"DEBUG: Buscando base de datos. Raíz: {db_path_root}, Src: {db_path_src} (stdout)")

    if os.path.exists(db_path_root): # Chequear primero la ruta raíz del proyecto
        db_path = db_path_root
        logging.info(f"Base de datos encontrada en project_root/db: {db_path}")
        print(f"DEBUG: Base de datos encontrada en project_root/db: {db_path} (stdout)")
    elif os.path.exists(db_path_src): # Luego la ruta dentro de src/db
        db_path = db_path_src
        logging.info(f"Base de datos encontrada en src/db: {db_path}")
        print(f"DEBUG: Base de datos encontrada en src/db: {db_path} (stdout)")
    else:
        logging.error(f"La base de datos no se encontró en {db_path_root} ni en {db_path_src}")
        print(f"ERROR: La base de datos no se encontró en {db_path_root} ni en {db_path_src} (stdout)")
        # Fallback a una ruta absoluta si es necesario (descomentar y ajustar si es preciso)
        # db_path_abs = "c:\Users\cpascual\PycharmProjects\pythonProject\cursos_actividades\sina_mcp\sqlite-analyzer\db\database_new.sqlite3.db"
        # if os.path.exists(db_path_abs):
        #     db_path = db_path_abs
        #     logging.info(f"Usando ruta absoluta para la base de datos: {db_path}")
        # else:
        #     logging.error(f"La ruta absoluta {db_path_abs} tampoco funciona.")
        #     raise FileNotFoundError("No se pudo encontrar el archivo de la base de datos.")
        raise FileNotFoundError(f"No se pudo encontrar el archivo de la base de datos en las rutas esperadas: {db_path_root} o {db_path_src}")


    logging.info(f"Usando base de datos en: {db_path}")
    print(f"DEBUG: Usando base de datos en: {db_path} (stdout)")
    
    logging.info("Intentando inicializar SQLiteConnector")
    print("DEBUG: Intentando inicializar SQLiteConnector (stdout)")
    db = SQLiteConnector(db_path)
    logging.info("SQLiteConnector inicializado exitosamente")
    print("DEBUG: SQLiteConnector inicializado exitosamente (stdout)")

    # Forzamos unas tablas de ejemplo:
    tablas_forzadas_caso1 = ["PATI_PATIENT_ALLERGIES", "ALLE_ALLERGY_TYPES"]
    pregunta_caso1 = "¿Qué alergias tiene el paciente 1931?"
    
    # Extracción de ID de paciente y creación de condiciones relevantes para el Caso 1
    match_id_paciente = re.search(r"paciente (\d+)", pregunta_caso1)
    condiciones_relevantes_caso1 = []
    if match_id_paciente:
        patient_id = match_id_paciente.group(1)
        condiciones_relevantes_caso1.append({
            "column": "PATI_PATIENT_ALLERGIES.PATI_ID", # Asumiendo que el ID está en esta tabla/columna
            "operator": "=",
            "value": patient_id
        })
        # Podríamos añadir más condiciones si fueran extraíbles o predefinidas
        # Por ejemplo, si la pregunta implicara una condición sobre ALLE_ALLERGY_TYPES
        # y supiéramos la columna, la añadiríamos aquí.

    logging.info(f"Ejecutando chatbot_pipeline (prueba 1) con pregunta: '{pregunta_caso1}', tablas forzadas: {tablas_forzadas_caso1}, condiciones_relevantes: {condiciones_relevantes_caso1}")
    print(f"DEBUG: Ejecutando chatbot_pipeline (prueba 1) con pregunta: '{pregunta_caso1}', tablas forzadas: {tablas_forzadas_caso1}, condiciones_relevantes: {condiciones_relevantes_caso1} (stdout)")
    
    # Configuración simulada, ya que el error indica que 'config' es None en la segunda llamada
    simulated_config = {
        'llm_config': {
            # ... posibles configuraciones de LLM aquí ...
        },
        'sql_generation_retries': 1
        # ... otras configuraciones que el pipeline podría esperar ...
    }

    result_caso1 = chatbot_pipeline(
        pregunta_caso1,
        db,
        tablas_relevantes=tablas_forzadas_caso1,
        condiciones_relevantes=condiciones_relevantes_caso1, # Nuevo parámetro
        config=simulated_config # Pasar la configuración simulada
    )
    logging.info("chatbot_pipeline (prueba 1) completado")
    print("\n--- Resultados de la Prueba Aislada (Caso 1) ---")
    print(f"SQL generado: {result_caso1.get('sql')}")
    print(f"Respuesta: {result_caso1.get('response')}")
    print(f"Tablas usadas (según pipeline): {result_caso1.get('tables')}")

    # Prueba adicional: ¿Qué pasa si la pregunta no parece necesitar ambas tablas?
    pregunta_caso2 = "¿Cuáles son los tipos de alergia?"
    tablas_forzadas_caso2 = ["ALLE_ALLERGY_TYPES"]
    # Para el caso 2, no hay condiciones relevantes extraíbles de la pregunta de forma simple
    condiciones_relevantes_caso2 = [] 
    
    logging.info(f"Ejecutando chatbot_pipeline (prueba 2) con pregunta: '{pregunta_caso2}' y tablas forzadas: {tablas_forzadas_caso2}")
    print(f"DEBUG: Ejecutando chatbot_pipeline (prueba 2) con pregunta: '{pregunta_caso2}' y tablas forzadas: {tablas_forzadas_caso2} (stdout)")
    
    result_caso2 = chatbot_pipeline(
        pregunta_caso2,
        db,
        tablas_relevantes=tablas_forzadas_caso2,
        condiciones_relevantes=condiciones_relevantes_caso2, # Nuevo parámetro (vacío aquí)
        config=simulated_config # Reutilizar config simulada
    )
    logging.info("chatbot_pipeline (prueba 2) completado")
    print("DEBUG: chatbot_pipeline (prueba 2) completado (stdout)")

    print("\n--- Resultados de la Prueba Aislada (Caso 2) ---")
    print(f"SQL generado: {result_caso2.get('sql')}")
    print(f"Respuesta: {result_caso2.get('response')}")
    print(f"Tablas usadas (según pipeline): {result_caso2.get('tables')}")
    logging.info("Fin del script test_pipeline_forced_tables.py (éxito)")
    print("DEBUG: Fin del script test_pipeline_forced_tables.py (éxito) (stdout)")

except ModuleNotFoundError as e:
    logging.error(f"ModuleNotFoundError: {e}", exc_info=True) # Añadido exc_info
    logging.error(f"sys.path al momento del error: {sys.path}")
    print(f"ERROR: ModuleNotFoundError - {e}", file=sys.stderr) # Imprimir a stderr
    raise # Relanzar la excepción para que la ejecución falle y se vea el error
except FileNotFoundError as e:
    logging.error(f"FileNotFoundError: {e}", exc_info=True) # Añadido exc_info
    print(f"ERROR: FileNotFoundError - {e}", file=sys.stderr) # Imprimir a stderr
    raise
except Exception as e:
    logging.error(f"Ocurrió una excepción no esperada: {e}", exc_info=True) # exc_info=True para traceback
    print(f"ERROR: Ocurrió una excepción no esperada - {e}", file=sys.stderr) # Imprimir a stderr
    raise
