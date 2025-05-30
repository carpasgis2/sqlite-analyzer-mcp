import json
import logging
import time
import os # AÑADIDO: Para manejo de rutas de archivos
import re # AÑADIDO: Para expresiones regulares

# Asumir que estos módulos existen y son importables
from sql_generator import SQLGenerator # MODIFICADO: Importar SQLGenerator directamente
import whitelist_validator 
from db_connector import DBConnector

# Configuración del logger
logger = logging.getLogger(__name__)
# Para ver los mensajes si no hay otra configuración de logging:
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

MAX_RETRIES_LLM_EXTRACT = 2 

# Funciones auxiliares (deben estar definidas o importadas)
def preprocess_question(question: str, terms_dictionary: dict) -> str:
    # logger.debug(f"Original question for preprocess: {question}")
    # Esta es una implementación placeholder, la real debería estar en el código.
    # Aquí se podrían normalizar términos, etc.
    # Ejemplo: Reemplazar sinónimos o corregir errores comunes basados en terms_dictionary
    # for term, replacement in terms_dictionary.get("synonyms", {}).items():
    #    question = question.replace(term, replacement)
    return question

def log_benchmark(start_time, process_name, logger_instance=None):
    current_logger_bm = logger_instance if logger_instance else logger
    end_time = time.time()
    current_logger_bm.info(f"BENCHMARK: {process_name} took {end_time - start_time:.4f} seconds.")

# Asumir que extract_info_from_question_llm está definida en otra parte.
# Firma esperada: def extract_info_from_question_llm(question, llm, terms_dict_str, db_structure_dict, max_retries)
# y que devuelve (structured_info_dict, error_message_str_or_none)
# from your_llm_module import extract_info_from_question_llm # Ejemplo de importación

# MODIFICADO: Firma de la función y lógica interna
def generate_sql_and_validate_whitelist(
    structured_info, 
    sql_generator_instance: SQLGenerator, 
    db_connector_for_sql_gen: DBConnector,
    current_allowed_columns_map, 
    terms_dict_str_for_validation, 
    question_for_logging,
    logger_param: logging.Logger = None # Añadido logger_param
) -> tuple[str, list, str, any]:
    current_logger_sql_gen = logger_param if logger_param else logger
    current_logger_sql_gen.debug(f"DEBUG: [pipeline.py] Entrando en generate_sql_and_validate_whitelist con question: {question_for_logging}")
    
    # Si la pregunta original era SQL directo, structured_info podría contenerlo.
    if isinstance(structured_info, dict) and "direct_sql_query" in structured_info:
        direct_sql = structured_info["direct_sql_query"]
        current_logger_sql_gen.info(f"DEBUG: [pipeline.py] generate_sql_and_validate_whitelist recibió direct_sql_query: '{direct_sql}'. Se procesará este SQL directamente.")

        # INICIO: Parche robusto para convertir INNER JOIN a LEFT JOIN para MEDI_MEDICATIONS y MEDI_PHARMA_THERAPEUTIC_GROUPS
        # Este parche se aplica específicamente a la relación entre estas dos tablas a través de PHTH_ID.
        
        # Nombres de las tablas y columna de unión
        table_meds_name = "MEDI_MEDICATIONS"
        table_pharma_name = "MEDI_PHARMA_THERAPEUTIC_GROUPS"
        join_column_name = "PHTH_ID"

        # Patrones para encontrar las tablas y sus alias
        # Busca "TABLE_NAME opcional_AS alias" o "TABLE_NAME alias" (sin AS)
        # El alias es \\w+ (uno o más caracteres alfanuméricos o guion bajo)
        # \\s+ asegura uno o más espacios, (?:\\s+AS\\s+)? hace que " AS " sea opcional
        pattern_meds_alias = re.compile(fr"{table_meds_name}\\s+(?:AS\\s+)?(\\w+)", re.IGNORECASE)
        pattern_pharma_alias = re.compile(fr"{table_pharma_name}\\s+(?:AS\\s+)?(\\w+)", re.IGNORECASE)

        match_meds = pattern_meds_alias.search(direct_sql)
        alias_meds = match_meds.group(1) if match_meds else table_meds_name
        
        match_pharma = pattern_pharma_alias.search(direct_sql)
        alias_pharma = match_pharma.group(1) if match_pharma else table_pharma_name

        current_logger_sql_gen.debug(f"Parche LEFT JOIN: Alias detectados: {table_meds_name} -> {alias_meds}, {table_pharma_name} -> {alias_pharma}")

        # Construir el patrón de JOIN para reemplazar.
        # Buscamos:
        #   FROM MEDI_MEDICATIONS [AS] m ... 
        #   [INNER] JOIN MEDI_PHARMA_THERAPEUTIC_GROUPS [AS] p
        #   ON m.PHTH_ID = p.PHTH_ID (o viceversa)
        # El patrón debe ser flexible con los espacios y el uso opcional de "AS".
        # (\\bJOIN\\b|\\bINNER\\s+JOIN\\b) captura "JOIN" o "INNER JOIN" como grupo 1 (el tipo de join)
        # \\s+ asegura espacios.
        # {re.escape(alias_pharma)} usa el alias detectado para la tabla pharma.
        # (?:\\s+AS\\s+)? es opcional " AS ".
        # ON\\s+ hace match con "ON "
        # La condición de join es más compleja:
        #   (?:{re.escape(alias_meds)}\\.{join_column_name}\\s*=\\s*{re.escape(alias_pharma)}\\.{join_column_name}|{re.escape(alias_pharma)}\\.{join_column_name}\\s*=\\s*{re.escape(alias_meds)}\\.{join_column_name})
        #   Esto permite m.PHTH_ID = p.PHTH_ID O p.PHTH_ID = m.PHTH_ID
        
        # Patrón para encontrar el JOIN entre MEDI_MEDICATIONS y MEDI_PHARMA_THERAPEUTIC_GROUPS
        # Este patrón busca específicamente el JOIN que involucra a MEDI_PHARMA_THERAPEUTIC_GROUPS
        # y la condición ON que relaciona PHTH_ID de ambas tablas (usando sus alias).
        # Grupo 1: El tipo de JOIN (JOIN o INNER JOIN)
        # Grupo 2: El resto de la cláusula JOIN (MEDI_PHARMA_THERAPEUTIC_GROUPS [AS] alias ON ...)
        join_search_pattern = re.compile(
            r"(\bJOIN\b|\bINNER\s+JOIN\b)\s+" +  # Grupo 1: Tipo de JOIN
            r"(" + # Inicio Grupo 2: Resto de la cláusula JOIN
            re.escape(table_pharma_name) + r"\s+(?:AS\s+)?(?P<alias_pharma_match>\w+)\s+" + # Tabla Pharma con su alias
            r"ON\s+" +
            r"(?:" + # Grupo para las condiciones ON (alternativas)
            r"(?P=alias_pharma_match)\." + re.escape(join_column_name) + r"\s*=\s*" + re.escape(alias_meds) + r"\." + re.escape(join_column_name) + # p.PHTH_ID = m.PHTH_ID
            r"|" + 
            re.escape(alias_meds) + r"\." + re.escape(join_column_name) + r"\s*=\s*(?P=alias_pharma_match)\." + re.escape(join_column_name) + # m.PHTH_ID = p.PHTH_ID
            r")" + # Fin grupo condiciones ON
            r")", # Fin Grupo 2
            re.IGNORECASE
        )
        
        # Segunda búsqueda, ahora que alias_meds está definido.
        # Este es un patrón más específico para la tabla MEDI_PHARMA_THERAPEUTIC_GROUPS
        # y su condición ON con MEDI_MEDICATIONS.
        
        # Primero, intentamos encontrar un JOIN que una MEDI_PHARMA_THERAPEUTIC_GROUPS (alias_pharma)
        # a MEDI_MEDICATIONS (alias_meds)
        
        # Patrón para el JOIN de MEDI_PHARMA_THERAPEUTIC_GROUPS a MEDI_MEDICATIONS
        # (?:FROM|JOIN)\s+MEDI_MEDICATIONS\s+(?:AS\s+)?(\w+)\s+ # Captura alias de MEDI_MEDICATIONS
        # (INNER\s+JOIN|JOIN)\s+ # El tipo de JOIN
        # MEDI_PHARMA_THERAPEUTIC_GROUPS\s+(?:AS\s+)?(\w+)\s+ # Captura alias de MEDI_PHARMA_THERAPEUTIC_GROUPS
        # ON\s+\1\.PHTH_ID\s*=\s*\2\.PHTH_ID # Condición usando los alias capturados
        
        # Simplificamos: Buscamos "JOIN MEDI_PHARMA_THERAPEUTIC_GROUPS [AS] p ON m.PHTH_ID = p.PHTH_ID"
        # o "JOIN MEDI_PHARMA_THERAPEUTIC_GROUPS [AS] p ON p.PHTH_ID = m.PHTH_ID"
        # donde 'm' es el alias de MEDI_MEDICATIONS y 'p' el de MEDI_PHARMA_THERAPEUTIC_GROUPS.

        # Patrón para encontrar el JOIN específico y reemplazarlo
        # Este patrón busca:
        # 1. Un JOIN o INNER JOIN (case-insensitive)
        # 2. Seguido por la tabla MEDI_PHARMA_THERAPEUTIC_GROUPS (con su alias)
        # 3. Seguido por la condición ON que relaciona PHTH_ID de MEDI_MEDICATIONS (con su alias)
        #    y MEDI_PHARMA_THERAPEUTIC_GROUPS (con su alias).
        
        # El alias_pharma_in_on se refiere al alias de MEDI_PHARMA_THERAPEUTIC_GROUPS usado en la condición ON
        # El alias_meds_in_on se refiere al alias de MEDI_MEDICATIONS usado en la condición ON
        
        # Patrón: (JOIN_KEYWORD) (TABLE_PHARMA_NAME [AS] alias_pharma) ON (alias_meds.PHTH_ID = alias_pharma.PHTH_ID OR alias_pharma.PHTH_ID = alias_meds.PHTH_ID)
        # Grupo 1: (JOIN | INNER JOIN)
        # Grupo 2: MEDI_PHARMA_THERAPEUTIC_GROUPS\\s+(?:AS\\s+)?(\\w+)  -- El alias es el subgrupo \w+
        # Grupo 3: La condición ON completa
        
        # Necesitamos encontrar el alias de MEDI_PHARMA_THERAPEUTIC_GROUPS en su declaración
        # y luego usar ese alias y el alias_meds en la condición ON.

        pharma_table_declaration_pattern = re.compile(fr"{table_pharma_name}\s+(?:AS\s+)?(\w+)", re.IGNORECASE)
        pharma_alias_match_in_declaration = pharma_table_declaration_pattern.search(direct_sql)

        if pharma_alias_match_in_declaration:
            actual_pharma_alias = pharma_alias_match_in_declaration.group(1)
            current_logger_sql_gen.debug(f"Parche LEFT JOIN: Alias de {table_pharma_name} en su declaración: {actual_pharma_alias}")

            # Ahora construimos el patrón de JOIN completo
            # (JOIN | INNER JOIN) + MEDI_PHARMA_THERAPEUTIC_GROUPS [AS] actual_pharma_alias + ON + (condición con actual_pharma_alias y alias_meds)
            specific_join_pattern = re.compile(
                r"(\bJOIN\b|\bINNER\s+JOIN\b)\s+" + # Grupo 1: El tipo de JOIN
                re.escape(table_pharma_name) + r"\s+(?:AS\s+)?" + re.escape(actual_pharma_alias) + r"\s+" + # La tabla PHARMA con su alias exacto
                r"ON\s+" +
                r"(" + # Grupo 2: La condición ON
                    re.escape(alias_meds) + r"\." + re.escape(join_column_name) + r"\s*=\s*" + re.escape(actual_pharma_alias) + r"\." + re.escape(join_column_name) +
                    r"|" +
                    re.escape(actual_pharma_alias) + r"\." + re.escape(join_column_name) + r"\s*=\s*" + re.escape(alias_meds) + r"\." + re.escape(join_column_name) +
                r")",
                re.IGNORECASE
            )

            match = specific_join_pattern.search(direct_sql)
            if match:
                join_type_found = match.group(1) # "JOIN" o "INNER JOIN"
                
                # Construir la cláusula LEFT JOIN
                # LEFT JOIN MEDI_PHARMA_THERAPEUTIC_GROUPS [AS] actual_pharma_alias ON (condición original)
                # La condición original está en match.group(2) pero ya la tenemos por partes.
                # Es importante mantener la tabla MEDI_MEDICATIONS como la "izquierda" conceptualmente.
                
                original_join_clause = match.group(0) # La cláusula JOIN completa encontrada
                
                # Reemplazar el tipo de JOIN por LEFT JOIN
                # Si es "INNER JOIN", se reemplaza por "LEFT JOIN"
                # Si es "JOIN", se reemplaza por "LEFT JOIN"
                
                modified_join_clause = original_join_clause.upper().replace(join_type_found.upper(), "LEFT JOIN", 1)
                
                # Para asegurar que el case del resto de la cláusula se mantenga, aplicamos el replace sobre la original
                # Encontramos el índice del tipo de join y reemplazamos
                start_index = original_join_clause.upper().find(join_type_found.upper())
                end_index = start_index + len(join_type_found)
                
                # Reconstruimos con LEFT JOIN
                # Esto es para preservar el case original de la tabla y alias, solo cambiando el tipo de JOIN
                reconstructed_left_join = "LEFT JOIN" + original_join_clause[end_index:]
                
                modified_sql = direct_sql.replace(original_join_clause, reconstructed_left_join, 1)

                if modified_sql != direct_sql:
                    current_logger_sql_gen.warning(f"DEBUG: [pipeline.py] Consulta SQL modificada por parche robusto para usar LEFT JOIN. Original: '{direct_sql}', Modificada: '{modified_sql}'")
                    direct_sql = modified_sql
                else: # pragma: no cover
                    current_logger_sql_gen.warning(f"DEBUG: [pipeline.py] Parche LEFT JOIN robusto: No se pudo reemplazar el JOIN específico aunque el patrón coincidió. Original: '{direct_sql}' Patrón: '{specific_join_pattern.pattern}'")
            else:
                current_logger_sql_gen.debug(f"DEBUG: [pipeline.py] Parche LEFT JOIN robusto: No se encontró el patrón de JOIN específico para reemplazar. SQL: '{direct_sql}'. Patrón: '{specific_join_pattern.pattern}'")
        else:
            current_logger_sql_gen.debug(f"DEBUG: [pipeline.py] Parche LEFT JOIN robusto: No se pudo encontrar el alias de declaración para {table_pharma_name}. No se aplicará el parche.")
        
        # FIN: Parche robusto

        # Para SQL directo, después del parche, retornamos directamente.
        # No pasa por la validación de whitelist estructurada ni por el SQLGenerator.
        return direct_sql, [], "", None # SQL, params, error_msg, context

    # Si no es SQL directo, o si structured_info no es un dict, procedemos como antes.
    if not structured_info or not isinstance(structured_info, dict): # Asegurarse que structured_info es un dict si no es SQL directo
        current_logger_sql_gen.warning("  structured_info está vacío, no es un diccionario, o no es SQL directo al entrar a generate_sql_and_validate_whitelist.")
        return "", [], "structured_info estaba vacío o no era un diccionario, no se puede generar SQL.", None
    
    current_logger_sql_gen.debug(f"  structured_info para validación y SQL-gen: {json.dumps(structured_info, indent=2, ensure_ascii=False)}")

    # Whitelist validation
    try:
        is_valid_structured_info, validation_msg = whitelist_validator.validate_structured_info_whitelist(
            structured_info,
            allowed_columns_map=current_allowed_columns_map,
            terms_dictionary=terms_dict_str_for_validation,
            logger_param=current_logger_sql_gen # Pasar logger
        )
    except Exception as e_val:
        current_logger_sql_gen.error(f"Excepción durante validate_structured_info_whitelist: {e_val}", exc_info=True)
        return "", [], f"Error en la validación de whitelist: {e_val}", None

    if not is_valid_structured_info:
        error_message = f"Validación de whitelist de partes estructuradas fallida: {validation_msg}"
        current_logger_sql_gen.error(f"{error_message} para structured_info: {structured_info}")
        return "", [], error_message, None

    sql_query_generated = ""
    query_params = []
    error_message_generation = ""
    context_from_sql_gen = None

    try:
        sql_query_generated, query_params = sql_generator_instance.generate_sql(
            question_data=structured_info,
            db_connector=db_connector_for_sql_gen
        )
        if sql_query_generated and "SELECT 'Error:" in sql_query_generated:
            error_message_generation = sql_query_generated
            current_logger_sql_gen.error(f"DEBUG: [pipeline.py] SQLGenerator devolvió una query de error: {sql_query_generated}")
        elif not sql_query_generated:
            error_message_generation = "SQLGenerator devolvió una consulta vacía sin un mensaje de error explícito."
            current_logger_sql_gen.error(f"DEBUG: [pipeline.py] {error_message_generation}")
        if sql_query_generated and not error_message_generation:
             current_logger_sql_gen.info(f"DEBUG: [pipeline.py] SQL generado por SQLGenerator: {sql_query_generated}, Params: {query_params}")
    except Exception as e_gensql_call:
        error_message_generation = f"EXCEPCIÓN al llamar a SQLGenerator.generate_sql: {str(e_gensql_call)}"
        sql_query_generated = ""
        query_params = []
        current_logger_sql_gen.error(f"DEBUG: [pipeline.py] {error_message_generation}", exc_info=True)
    return sql_query_generated, query_params, error_message_generation, context_from_sql_gen

def chatbot_pipeline(question: str, db_connector_param: DBConnector, llm_param, terms_dict_path_param: str, schema_path_param: str, logger_param: logging.Logger = None):
    pipeline_start_time = time.time()
    
    current_logger = logger_param if logger_param else logger
    
    current_logger.info(f"--- Iniciando chatbot_pipeline para la pregunta: '{question}' ---")

    # Carga de diccionarios y estructuras (usando atributos de función para caché simple)
    actual_terms_dict_path = terms_dict_path_param
    if not actual_terms_dict_path:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        actual_terms_dict_path = os.path.join(base_dir, "data", "dictionary.json")
        current_logger.info(f"terms_dict_path_param no proporcionado, usando por defecto: {actual_terms_dict_path}")

    if not hasattr(chatbot_pipeline, 'terms_dictionary_common') or \
       getattr(chatbot_pipeline, "terms_dictionary_path_cache", None) != actual_terms_dict_path or \
       not chatbot_pipeline.terms_dictionary_common:
        current_logger.info(f"Cargando terms_dictionary_common desde {actual_terms_dict_path}...")
        try:
            with open(actual_terms_dict_path, 'r', encoding='utf-8') as f:
                chatbot_pipeline.terms_dictionary_common = json.load(f)
            chatbot_pipeline.terms_dictionary_str_common = json.dumps(chatbot_pipeline.terms_dictionary_common, ensure_ascii=False)
            chatbot_pipeline.terms_dictionary_path_cache = actual_terms_dict_path
            current_logger.info(f"terms_dictionary_common cargado desde {actual_terms_dict_path} con {len(chatbot_pipeline.terms_dictionary_common)} entradas.")
        except FileNotFoundError:
            current_logger.error(f"Error CRÍTICO al cargar el diccionario de términos: [Errno 2] No such file or directory: '{actual_terms_dict_path}'. El pipeline no puede continuar sin este archivo.")
            return {"response": f"Error crítico: No se pudo cargar el archivo de configuración esencial '{os.path.basename(actual_terms_dict_path)}'. Verifique la ruta y la existencia del archivo.", "data": None, "executed_sql_query_info": None}
        except json.JSONDecodeError as e_json:
            current_logger.error(f"Error CRÍTICO al decodificar JSON del diccionario de términos en: '{actual_terms_dict_path}'. Error: {e_json}", exc_info=True)
            return {"response": f"Error crítico: El archivo de configuración '{os.path.basename(actual_terms_dict_path)}' está corrupto o no es un JSON válido.", "data": None, "executed_sql_query_info": None}
        except Exception as e:
            current_logger.error(f"Error CRÍTICO inesperado al cargar el diccionario de términos desde '{actual_terms_dict_path}': {e}", exc_info=True)
            return {"response": f"Error crítico: No se pudo cargar el diccionario de términos debido a un error inesperado. {e}", "data": None, "executed_sql_query_info": None}
    
    if not chatbot_pipeline.terms_dictionary_common:
        current_logger.error(f"Error CRÍTICO: terms_dictionary_common está vacío después del intento de carga desde {actual_terms_dict_path}. El pipeline no puede continuar.")
        return {"response": "Error crítico: El diccionario de términos está vacío después de la carga. Verifique el archivo.", "data": None, "executed_sql_query_info": None}

    if not hasattr(chatbot_pipeline, 'db_structure_dict_common') or \
       not hasattr(chatbot_pipeline, 'allowed_columns_map_common') or \
       not hasattr(chatbot_pipeline, 'table_relationships_str_common') or \
       not hasattr(chatbot_pipeline, 'sql_generator_instance_common'):
        current_logger.info("Inicializando db_structure_dict_common, allowed_columns_map_common, table_relationships_str_common y SQLGenerator instance...")
        try:
            chatbot_pipeline.db_structure_dict_common = db_connector_param.get_db_structure_dict(schema_type='full')
            if not chatbot_pipeline.db_structure_dict_common:
                 current_logger.error("db_connector_param.get_db_structure_dict(schema_type='full') devolvió vacío.")
                 return {"response": "Error crítico: No se pudo obtener la estructura de la base de datos.", "data": None, "executed_sql_query_info": None}
            current_logger.info(f"db_structure_dict_common obtenida con {len(chatbot_pipeline.db_structure_dict_common)} tablas.")
            
            chatbot_pipeline.allowed_columns_map_common = {}
            problematic_tables_for_logging = []
            for table_name_for_map, cols_data in chatbot_pipeline.db_structure_dict_common.items():
                table_name_upper = table_name_for_map.upper()
                if not isinstance(table_name_for_map, str) or not table_name_for_map.strip():
                    current_logger.warning(f"Nombre de tabla inválido en db_structure: '{table_name_for_map}'. Se omite.")
                    continue
                col_names_for_map = []
                original_cols_data_repr = str(cols_data)[:200]
                if isinstance(cols_data, dict) and 'columns' in cols_data:
                    columns_list_or_dict = cols_data['columns']
                    if isinstance(columns_list_or_dict, list):
                        for col_entry in columns_list_or_dict:
                            if isinstance(col_entry, dict) and isinstance(col_entry.get('name'), str) and col_entry.get('name').strip():
                                col_names_for_map.append(col_entry['name'])
                            elif isinstance(col_entry, str) and col_entry.strip():
                                col_names_for_map.append(col_entry)
                    elif isinstance(columns_list_or_dict, str) and columns_list_or_dict.strip():
                        try:
                            parsed_columns = json.loads(columns_list_or_dict)
                            if isinstance(parsed_columns, list):
                                for item in parsed_columns:
                                    if isinstance(item, str) and item.strip(): col_names_for_map.append(item)
                                    elif isinstance(item, dict) and isinstance(item.get('name'), str) and item.get('name').strip(): col_names_for_map.append(item.get('name'))
                            else: problematic_tables_for_logging.append(f"{table_name_for_map} (JSON 'columns' no es lista)")
                        except json.JSONDecodeError: problematic_tables_for_logging.append(f"{table_name_for_map} (JSON 'columns' inválido)")
                    else: problematic_tables_for_logging.append(f"{table_name_for_map} (cols_data['columns'] tipo inesperado: {type(columns_list_or_dict)})")
                elif isinstance(cols_data, list):
                    for item in cols_data:
                        if isinstance(item, str) and item.strip(): col_names_for_map.append(item)
                        elif isinstance(item, dict) and isinstance(item.get('name'), str) and item.get('name').strip(): col_names_for_map.append(item.get('name'))
                    if not col_names_for_map and cols_data: problematic_tables_for_logging.append(f"{table_name_for_map} (cols_data lista pero no produjo columnas)")
                else: problematic_tables_for_logging.append(f"{table_name_for_map} (cols_data estructura inesperada: {type(cols_data)})")
                current_table_cols = [c.upper() for c in col_names_for_map if c]
                chatbot_pipeline.allowed_columns_map_common[table_name_upper] = current_table_cols
                if not current_table_cols and ((isinstance(cols_data, dict) and cols_data.get('columns')) or (isinstance(cols_data, list) and cols_data)):
                    reason = next((s for s in problematic_tables_for_logging if table_name_for_map in s), "razón no especificada")
                    current_logger.warning(f"Tabla '{table_name_for_map}' resultó en 0 columnas en allowed_columns_map. Razón/Tipo original: {reason}. Datos originales: {original_cols_data_repr}")
                    if not any(table_name_for_map in entry for entry in problematic_tables_for_logging):
                         problematic_tables_for_logging.append(f"{table_name_for_map} (datos entrada pero 0 columnas procesadas)")
            if problematic_tables_for_logging:
                current_logger.warning(f"--- Resumen tablas problemáticas/advertencias (allowed_columns_map): {problematic_tables_for_logging} ---")
            current_logger.info("DEBUG PIPELINE: Muestra allowed_columns_map_common (3 primeras tablas, 5 cols c/u):")
            sample_map = {k: v[:5] for k, v in list(chatbot_pipeline.allowed_columns_map_common.items())[:3]}
            current_logger.info(f"{json.dumps(sample_map, indent=2, ensure_ascii=False)}")
            chatbot_pipeline.table_relationships_str_common = db_connector_param.get_table_relationships_str()
            current_logger.info("table_relationships_str_common obtenido.")
            current_logger.info("Inicializando SQLGenerator instance...")
            allowed_tables_list = list(chatbot_pipeline.allowed_columns_map_common.keys())
            chatbot_pipeline.sql_generator_instance_common = SQLGenerator(
                allowed_tables=allowed_tables_list,
                allowed_columns=chatbot_pipeline.allowed_columns_map_common,
                enhanced_schema_path=schema_path_param, 
                relationships_str=chatbot_pipeline.table_relationships_str_common,
                dictionary_str=chatbot_pipeline.terms_dictionary_str_common,
                logger=current_logger
            )
            current_logger.info("SQLGenerator instance inicializada.")
        except Exception as e:
            current_logger.error(f"Error al inicializar estructuras de BD, mapa de columnas o SQLGenerator: {e}", exc_info=True)
            return {"response": f"Error crítico: No se pudo inicializar la configuración de la BD. {e}", "data": None, "executed_sql_query_info": None}

    s_time = time.time()
    preprocessed_question = preprocess_question(question, chatbot_pipeline.terms_dictionary_common)
    log_benchmark(s_time, "preprocess_question", logger_instance=current_logger)

    s_time = time.time()
    structured_info_dict = None
    error_llm_extract = "Error no inicializado en extract_info_from_question_llm"
    current_logger.info(f"DEBUG: [pipeline.py] Antes de llamar a extract_info_from_question_llm con question: '{preprocessed_question}'")
    try:
        structured_info_dict, error_llm_extract = extract_info_from_question_llm(
            preprocessed_question,
            llm_param,
            chatbot_pipeline.terms_dictionary_str_common,
            chatbot_pipeline.db_structure_dict_common,
            max_retries=MAX_RETRIES_LLM_EXTRACT,
            logger_param=current_logger
        )
        current_logger.info(f"DEBUG: [pipeline.py] Después de llamar a extract_info_from_question_llm. structured_info_dict: {json.dumps(structured_info_dict, indent=2) if structured_info_dict else 'None'}, error_llm_extract: '{error_llm_extract}'")
    except NameError:
        current_logger.error("La función 'extract_info_from_question_llm' no está definida o importada.", exc_info=True)
        return {"response": "Error de configuración interna: componente de extracción de información no encontrado.", "data": None, "executed_sql_query_info": None}
    except Exception as e_extract:
        current_logger.error(f"Excepción durante extract_info_from_question_llm: {e_extract}", exc_info=True)
        return {"response": f"Error al extraer información de la pregunta: {e_extract}", "data": None, "executed_sql_query_info": None}

    log_benchmark(s_time, "extract_info_from_question_llm", logger_instance=current_logger)

    if error_llm_extract or not structured_info_dict:
        if isinstance(structured_info_dict, dict) and "direct_sql_query" in structured_info_dict:
            current_logger.info(f"Bypass de extracción LLM: Se usará SQL directo: {structured_info_dict['direct_sql_query']}")
        else:
            current_logger.error(f"DEBUG: [pipeline.py] extract_info_from_question_llm falló o no devolvió info. Error: {error_llm_extract}. Info: {structured_info_dict}")
            answer = f"No se pudo extraer información de la pregunta. {error_llm_extract or ''}".strip()
            return {"response": answer, "data": None, "executed_sql_query_info": {"error_extracting_info": True, "details": error_llm_extract}}

    s_time = time.time()
    sql_query = None
    query_params = None
    error_msg_sql_gen = "Error no inicializado en generate_sql_and_validate_whitelist"
    current_logger.info(f"DEBUG: [pipeline.py] Antes de llamar a generate_sql_and_validate_whitelist. structured_info_dict: {json.dumps(structured_info_dict, indent=2)}")
    sql_query, query_params, error_msg_sql_gen, context_for_nl_response = generate_sql_and_validate_whitelist(
        structured_info_dict,
        chatbot_pipeline.sql_generator_instance_common,
        db_connector_param,
        chatbot_pipeline.allowed_columns_map_common,
        chatbot_pipeline.terms_dictionary_str_common,
        preprocessed_question,
        logger_param=current_logger
    )
    current_logger.info(f"DEBUG: [pipeline.py] Después de llamar a generate_sql_and_validate_whitelist. sql_query: '{sql_query}', query_params: {query_params}, error_msg_sql_gen: '{error_msg_sql_gen}'")
    log_benchmark(s_time, "generate_sql_and_validate_whitelist", logger_instance=current_logger)
    
    final_response_message = ""
    data_results_for_output = None
    executed_sql_query_info_for_output = {
        "original_question": question,
        "preprocessed_question": preprocessed_question,
        "structured_info": structured_info_dict,
        "generated_sql": sql_query or "No generada",
        "query_params": query_params or [],
        "error_in_sql_generation": error_msg_sql_gen if error_msg_sql_gen else None,
        "execution_error": None,
        "nl_generation_error": None,
        "final_executed_query": None
    }

    if error_msg_sql_gen or not sql_query:
        current_logger.error(f"DEBUG: [pipeline.py] Error en SQL-gen/validation o SQL no generado: {error_msg_sql_gen}. SQL Query era: '{sql_query}'")
        user_error_message = f"Lo siento, no pude procesar tu pregunta para generar una consulta SQL. Detalle: {error_msg_sql_gen}"
        if error_msg_sql_gen:
            if "No se especificó ninguna tabla válida" in error_msg_sql_gen:
                user_error_message = "No pude identificar las tablas correctas en la base de datos para tu consulta. Intenta reformularla."
            elif "Validación de whitelist" in error_msg_sql_gen:
                user_error_message = "Tu pregunta contiene términos o estructuras que no puedo procesar de forma segura. Intenta simplificarla."
        final_response_message = user_error_message
    else:
        current_logger.info(f"DEBUG: [pipeline.py] SQL Query a ejecutar: {sql_query}, Params: {query_params}")
        results_from_db = None
        column_names_from_db = None
        try:
            s_time = time.time()
            current_logger.info(f"DEBUG: [pipeline.py] Antes de llamar a db_connector_param.execute_query con SQL: {sql_query}")
            results_from_db, column_names_from_db = db_connector_param.execute_query(sql_query, query_params)
            data_results_for_output = results_from_db 
            executed_sql_query_info_for_output["final_executed_query"] = sql_query 
            current_logger.info(f"DEBUG: [pipeline.py] Después de llamar a db_connector_param.execute_query. Results (primeras 3 filas si hay): {str(results_from_db[:3]) if results_from_db else 'None/Empty'}, Column names: {column_names_from_db}")
            log_benchmark(s_time, "db_connector.execute_query", logger_instance=current_logger)

            if results_from_db is None: 
                current_logger.error(f"DEBUG: [pipeline.py] La ejecución de SQL falló (results_from_db is None). Query: {sql_query}, Params: {query_params}")
                final_response_message = "Error al ejecutar la consulta SQL en la base de datos."
                executed_sql_query_info_for_output["execution_error"] = "La consulta no devolvió resultados (None)."
            else:
                current_logger.info(f"DEBUG: [pipeline.py] Resultados ({len(results_from_db)} filas): {str(results_from_db[:3])[:500]}...")
                s_time = time.time()
                # nl_response_from_llm = "Respuesta no generada por LLM (placeholder)" # Eliminado placeholder antiguo
                current_logger.info(f"DEBUG: [pipeline.py] Antes de llamar a generate_natural_language_response_via_llm. Pregunta: '{preprocessed_question}', SQL: '{sql_query}', Results (primeras 3): {str(results_from_db[:3]) if results_from_db else 'None/Empty'}")
                try:
                    # MODIFICADO: Llamar a la nueva función que usa el LLM directamente
                    nl_response_from_llm = generate_natural_language_response_via_llm(
                        llm=llm_param, # Pasar la instancia del LLM
                        original_question=preprocessed_question,
                        sql_query=sql_query,
                        results=results_from_db,
                        column_names=column_names_from_db,
                        context=context_for_nl_response, 
                        error_extract_msg=error_llm_extract, # error_message_from_llm_extract
                        logger_param=current_logger
                    )
                    final_response_message = nl_response_from_llm
                    current_logger.info(f"DEBUG: [pipeline.py] Después de llamar a generate_natural_language_response_via_llm. Respuesta: '{final_response_message}'")
                except Exception as e_nl_resp:
                    current_logger.error(f"DEBUG: [pipeline.py] Excepción durante generate_natural_language_response_via_llm: {e_nl_resp}", exc_info=True)
                    final_response_message = f"Error al generar la respuesta en lenguaje natural: {e_nl_resp}"
                    executed_sql_query_info_for_output["nl_generation_error"] = str(e_nl_resp)
                log_benchmark(s_time, "generate_natural_language_response_via_llm", logger_instance=current_logger)
        except Exception as e_exec_or_nl:

            current_logger.error(f"DEBUG: [pipeline.py] EXCEPCIÓN durante ejecución SQL o generación NL: {e_exec_or_nl}", exc_info=True)
            final_response_message = f"Error inesperado al procesar consulta o generar respuesta: {e_exec_or_nl}"
            executed_sql_query_info_for_output["execution_error"] = str(e_exec_or_nl)

    log_benchmark(pipeline_start_time, "TOTAL chatbot_pipeline", logger_instance=current_logger)
    current_logger.info(f"DEBUG: [pipeline.py] Retornando desde chatbot_pipeline. Response: '{final_response_message}', Data: (presente: {data_results_for_output is not None}), SQL Info: (presente: {executed_sql_query_info_for_output is not None})")
    
    return {
        "response": final_response_message,
        "data": data_results_for_output,
        "executed_sql_query_info": executed_sql_query_info_for_output
    }


# MODIFICADO: Lógica interna para simular llamada a LLM y manejar preguntas específicas
def extract_info_from_question_llm(question, llm, terms_dict_str, db_structure_dict, max_retries, logger_param=None):
    current_logger = logger_param if logger_param else logger
    
    # SIMULACIÓN DE DETECCIÓN DE SQL DIRECTO O ENVUELTO EN MARKDOWN
    # Esta lógica intenta identificar si la pregunta es una consulta SQL directa
    # o una consulta SQL envuelta en bloques de código Markdown.
    
    stripped_question = question.strip()
    # Palabras clave SQL para ayudar a identificar consultas. Añadir más si es necesario.
    sql_keywords = ("SELECT", "INSERT", "UPDATE", "DELETE", "WITH", "CREATE", "ALTER", "DROP", "EXPLAIN", "--")

    # Caso 1: SQL directo (comienza con una palabra clave SQL y no está envuelto en ```)
    # Se considera que si empieza con ``` no es SQL "crudo" directo.
    # Convertir a mayúsculas para la comparación insensible a mayúsculas/minúsculas.
    # También se permite que comience con un comentario SQL.
    is_direct_sql_candidate = False
    for keyword in sql_keywords:
        if stripped_question.upper().startswith(keyword):
            is_direct_sql_candidate = True
            break
            
    if is_direct_sql_candidate and not stripped_question.startswith("```"):
        current_logger.info(f"DEBUG: [pipeline.py] extract_info_from_question_llm detectó SQL directo (sin Markdown): '{stripped_question}'")
        return {"direct_sql_query": stripped_question}, None

    # Caso 2: SQL envuelto en Markdown (ej. ```sql ... ``` o ``` ... ```)
    extracted_sql_from_markdown = None
    if stripped_question.startswith("```") and stripped_question.endswith("```"):
        # Quitar los backticks externos ```
        content_inside_backticks = stripped_question[3:-3].strip()
        
        # Verificar si tiene el identificador de lenguaje 'sql' (ej: ```sql SELECT...)
        # Se hace en minúsculas para ser insensible al caso de 'sql'.
        if content_inside_backticks.lower().startswith("sql"):
            # Quitar 'sql' y cualquier espacio/salto de línea posterior para obtener la consulta real.
            potential_sql = content_inside_backticks[3:].strip()
            if potential_sql: # Asegurarse de que haya contenido después de 'sql'
                extracted_sql_from_markdown = potential_sql
                current_logger.info(f"DEBUG: [pipeline.py] extract_info_from_question_llm detectó SQL en Markdown con 'sql' tag. Original: '{question}', Extraído: '{extracted_sql_from_markdown}'")
        else:
            # No tiene 'sql' tag, pero el contenido podría ser SQL si empieza con palabras clave SQL.
            # Esto maneja casos como ```SELECT ...```
            is_generic_markdown_sql_candidate = False
            for keyword in sql_keywords:
                if content_inside_backticks.upper().startswith(keyword):
                    is_generic_markdown_sql_candidate = True
                    break
            if is_generic_markdown_sql_candidate:
                extracted_sql_from_markdown = content_inside_backticks
                current_logger.info(f"DEBUG: [pipeline.py] extract_info_from_question_llm detectó SQL en Markdown genérico (sin 'sql' tag). Original: '{question}', Extraído: '{extracted_sql_from_markdown}'")

    # Si se extrajo SQL del Markdown y no está vacío, usarlo.
    if extracted_sql_from_markdown and extracted_sql_from_markdown.strip():
        actual_extracted_sql = extracted_sql_from_markdown.strip()
        current_logger.info(f"DEBUG: [pipeline.py] SQL extraído de Markdown para usar como direct_sql_query: '{actual_extracted_sql}'")
        return {"direct_sql_query": actual_extracted_sql}, None
    # else: # Opcional: log si se detectó markdown pero no se pudo extraer SQL válido
        # if stripped_question.startswith("```"): # Solo loguear si era un bloque de código
            # current_logger.debug(f"DEBUG: [pipeline.py] Se detectó bloque Markdown pero no se extrajo SQL válido o estaba vacío. Query: '{question}', Contenido interno: '{content_inside_backticks if 'content_inside_backticks' in locals() else 'N/A'}'")

    # Si no es SQL directo ni SQL en Markdown detectado arriba, proceder con la simulación de LLM / llamada real
    current_logger.info(f"DEBUG: [pipeline.py] No se detectó SQL directo ni en Markdown válido. Procediendo con lógica LLM para: '{question}'")

    # --- INICIO SIMULACIÓN LLAMADA A LLM ---
    # Aquí es donde se construiría el prompt y se llamaría al LLM
    # Ejemplo de prompt (muy simplificado):
    # prompt_text = f"""Dada la pregunta del usuario, el diccionario de términos y la estructura de la base de datos, 
    # extrae la información necesaria para construir una consulta SQL.
    # Pregunta: {question}
    # Diccionario de términos: {terms_dict_str} # Podría ser muy largo, considerar resúmenes o partes relevantes.
    # Estructura de la BD: {json.dumps(db_structure_dict)} # También podría ser muy largo.
    #
    # Devuelve un JSON con la siguiente estructura (o un subconjunto si no toda la info es necesaria):
    # {{
    #     "tables": ["TABLE_NAME_1", "TABLE_NAME_2"],
    #     "columns": ["TABLE_NAME_1.COLUMN_NAME_1", "TABLE_NAME_2.COLUMN_NAME_2"],
    #     "conditions": [
    #         {{"field": "TABLE_NAME.COLUMN_NAME", "operator": "=", "value": "some_value"}},
    #         {{"field": "ANOTHER_TABLE.COLUMN_NAME", "operator": "LIKE", "value": "%another_value%"}}
    #     ],
    #     "joins": [ # Opcional, si el LLM puede inferir joins complejos
    #        {{"type": "INNER", "from_table": "T1", "to_table": "T2", "on": "T1.id = T2.t1_id"}}
    #     ],
    #     "order_by": [{{"field": "TABLE_NAME.COLUMN_NAME", "direction": "ASC"}}],
    #     "limit": 10,
    #     "aggregation": [{{"function": "COUNT", "field": "*", "alias": "total_count"}}] # Ejemplo
    # }}
    # Si no puedes extraer la información, devuelve un JSON vacío {{}} y un mensaje de error.
    # Asegúrate de que los nombres de tablas y columnas coincidan exactamente con los de la estructura de la BD.
    # """
    # current_logger.debug(f"DEBUG: [pipeline.py] Prompt para LLM (extracción info) para pregunta '{question}' sería: {prompt_text[:500]}...") # Loguear solo una parte

    # structured_info_str = None # Para evitar UnboundLocalError si el try falla antes de la asignación
    # for attempt in range(max_retries + 1):
    #     try:
    #         current_logger.info(f"Attempt {attempt + 1} of {max_retries + 1} to call LLM for structure extraction for question: '{question}'.")
    #         # response_from_llm = llm.invoke(prompt_text) # LLAMADA REAL AL LLM (ej. ChatOpenAI)
    #         # structured_info_str = response_from_llm.content 
    #         # current_logger.info(f"DEBUG: [pipeline.py] LLM (extracción) raw response: {structured_info_str}")
    #         # structured_info = json.loads(structured_info_str)
    #         # if not structured_info: # Si el LLM devuelve un JSON vacío explícitamente
    #         #    current_logger.warning(f"LLM devolvió un JSON vacío para la pregunta: {question}")
    #         #    return {}, "El LLM no pudo extraer información estructurada (devolvió JSON vacío)."
    #         # return structured_info, None # Asumiendo que el LLM devuelve el JSON directamente
    #         # --- SIMULACIÓN DE LLAMADA --- 
    #         current_logger.warning("LLM call for structure extraction is COMMENTED OUT. Simulating failure.")
    #         # Para probar el flujo como si el LLM no pudiera extraer:
    #         if attempt < max_retries: # Simular fallo y reintento
    #             # raise Exception(f"Simulated LLM error on attempt {attempt + 1}") 
    #             pass # Simular que no se obtiene respuesta y se reintenta
    #         # En el último intento, devolver error simulado
    #         # structured_info_str = "{}" # Simular JSON vacío del LLM
    #         # structured_info = json.loads(structured_info_str)
    #         # return structured_info, "El LLM no pudo extraer información (simulación de JSON vacío)."
    #         # O simular un error más genérico:
    #         return {}, "El LLM no pudo extraer información estructurada (simulación de fallo tras reintentos)."
    #         # --- FIN SIMULACIÓN --- 
    #     except json.JSONDecodeError as e_json_dec:
    #         current_logger.error(f"Error al decodificar JSON de LLM en el intento {attempt + 1}: {e_json_dec}. Respuesta: {structured_info_str}")
    #         if attempt == max_retries:
    #             return {}, f"Error al decodificar JSON de LLM tras {max_retries + 1} intentos: {e_json_dec}. Respuesta original: {structured_info_str}"
    #         time.sleep(1) # Esperar antes de reintentar
    #     except Exception as e_llm:
    #         current_logger.error(f"Excepción al llamar a LLM para extracción en el intento {attempt + 1}: {e_llm}", exc_info=True)
    #         if attempt == max_retries:
    #            return {}, f"Excepción en LLM tras {max_retries + 1} intentos: {e_llm}"
    #         time.sleep(1) # Esperar antes de reintentar
    # --- FIN LLAMADA REAL AL LLM ---

    # Si la llamada al LLM (actualmente comentada y simulando fallo) no devuelve structured_info:
    current_logger.warning(f"DEBUG: [pipeline.py] LLM (real o simulado) no pudo extraer información para la pregunta: '{question}'. Se devuelve estructura vacía y mensaje de error.")
    return {}, "No se pudo extraer información de la pregunta mediante el LLM."


# NUEVA función para generar respuesta NL con LLM
def generate_natural_language_response_via_llm(llm, original_question, sql_query, results, column_names, context, error_extract_msg, logger_param=None):
    current_logger_nl = logger_param if logger_param else logger
    current_logger_nl.info(f"DEBUG: [pipeline.py] Entrando en generate_natural_language_response_via_llm con LLM: {type(llm)}")

    # --- INICIO COMENTARIOS SOBRE LLAMADA REAL AL LLM ---
    # Aquí es donde se construiría el prompt y se llamaría al LLM
    # prompt_parts = [f"Pregunta original del usuario: {original_question}"]
    # if sql_query and not ("Error:" in sql_query or (error_extract_msg and "Se proporcionó SQL directamente" not in error_extract_msg)):
    #     prompt_parts.append(f"Consulta SQL ejecutada con éxito: {sql_query}")
    
    # if results is not None:
    #     prompt_parts.append(f"Resultados ({len(results)} filas encontradas):")
    #     if column_names:
    #         prompt_parts.append("Columnas: " + " | ".join(column_names))
    #     # Mostrar algunas filas de ejemplo, no todas si son muchas
    #     for i, row in enumerate(results[:3]): # Mostrar hasta 3 filas
    #         prompt_parts.append(f"Fila {i+1}: " + " | ".join(map(str, row)))
    #     if len(results) > 3:
    #         prompt_parts.append(f"... y {len(results) - 3} fila(s) más.")
    #     if not results:
    #          prompt_parts.append("No se encontraron resultados que coincidan con tu consulta.")
    # elif sql_query and not ("Error:" in sql_query):
    #     prompt_parts.append("La consulta se ejecutó pero no devolvió filas.")


    # if error_extract_msg and "Se proporcionó SQL directamente" not in error_extract_msg:
    #      prompt_parts.append(f"Hubo un problema al procesar la pregunta antes de generar SQL: {error_extract_msg}")
    # elif sql_query and "Error:" in sql_query : # Si SQLGenerator devolvió un error
    #      prompt_parts.append(f"Hubo un error al intentar generar la consulta SQL: {sql_query}")


    # prompt_parts.append("Por favor, genera una respuesta concisa, amigable y útil en lenguaje natural para el usuario, resumiendo los hallazgos o explicando el problema.")
    # final_prompt = "\\n".join(prompt_parts)
    # current_logger_nl.debug(f"DEBUG: [pipeline.py] Prompt para LLM (generación NL):\\n{final_prompt[:1000]}...") # Loguear parte del prompt

    # try:
    #     # Asumiendo que llm es una instancia de Langchain (como ChatOpenAI) que tiene el método .invoke()
    #     # response = llm.invoke(final_prompt) 
    #     # generated_text = response.content # o la forma apropiada de obtener el texto
    # --- FIN COMENTARIOS SOBRE LLAMADA REAL AL LLM ---

    # --- INICIO LLM SIMULADO para generate_natural_language_response_via_llm ---
    current_logger_nl.warning("USANDO LLM SIMULADO para generate_natural_language_response_via_llm")
    generated_text = ""
    if results is not None: # results puede ser una lista vacía
        if results: # Hay filas
            generated_text = f"He encontrado {len(results)} resultado(s) para tu consulta sobre '{original_question}'. "
            if column_names:
                generated_text += f"Las columnas son: {', '.join(column_names)}. "
            generated_text += f"Aquí tienes algunos ejemplos: {str(results[:2])}" # Muestra los 2 primeros
            if len(results) > 2:
                generated_text += f" (y {len(results)-2} más)."
        else: # No hay filas
             generated_text = f"No se encontraron resultados para tu consulta: '{original_question}'."
    elif sql_query and "Error:" in sql_query : # Error de SQLGenerator
         generated_text = f"Hubo un problema al generar la consulta para tu pregunta '{original_question}'. Detalle: {sql_query}"
    elif error_extract_msg and "Se proporcionó SQL directamente" not in error_extract_msg: # Error de extracción
        generated_text = f"No pude entender completamente tu pregunta sobre '{original_question}'. Error: {error_extract_msg}"
    elif error_extract_msg and "Se proporcionó SQL directamente" in error_extract_msg and sql_query and results is None: # SQL directo, pero sin resultados (results es None)
         generated_text = f"Ejecuté la consulta SQL que proporcionaste, pero no obtuve resultados o hubo un error en la ejecución."
    else: # Caso genérico
        generated_text = f"He procesado tu pregunta '{original_question}'. Si esperabas datos específicos y no los ves, por favor, reformula tu pregunta o verifica los términos."
    # --- FIN LLM SIMULADO ---

    current_logger_nl.info(f"DEBUG: [pipeline.py] Respuesta generada por LLM (simulada): {generated_text}")
    return generated_text
    # except Exception as e: # Comentado porque la llamada real al LLM está comentada
    #     current_logger_nl.error(f"Excepción al llamar al LLM para generar respuesta NL: {e}", exc_info=True)
    #     return f"Lo siento, tuve un problema al generar la respuesta en lenguaje natural. Error: {e}"


class MockLLMForPipeline:
    def __init__(self, logger_instance=None):
        self.logger = logger_instance if logger_instance else logging.getLogger(__name__ + ".MockLLMForPipeline")
        if not logger_instance:
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        self.logger.info("MockLLMForPipeline instanciado.")

    def generate_natural_language_response(self, question, sql, results, columns, context, error_extract_msg, logger_param=None):
        current_logger_nl = logger_param if logger_param else self.logger 
        current_logger_nl.warning("DEBUG: [pipeline.py] Usando MockLLMForPipeline.generate_natural_language_response")
        if error_extract_msg and "Se proporcionó SQL directamente" not in error_extract_msg:
            return f"Error al extraer información: {error_extract_msg}"
        if sql and results:
            response_parts = [f"Resultados para la consulta: {sql}"]
            if isinstance(results, list):
                if not results:
                    response_parts.append("No se encontraron filas.")
                else:
                    if columns and results and isinstance(results[0], tuple):
                        response_parts.append(" | ".join(columns))
                        response_parts.append(" | ".join(["---"] * len(columns)))
                        for row in results[:5]: 
                            response_parts.append(" | ".join(map(str, row)))
                    elif results and isinstance(results[0], dict):
                        actual_columns = list(results[0].keys())
                        response_parts.append(" | ".join(actual_columns))
                        response_parts.append(" | ".join(["---"] * len(actual_columns)))
                        for row in results[:5]: 
                            response_parts.append(" | ".join(str(row.get(col, '')) for col in actual_columns))
                    else: 
                        for row in results[:5]:
                            response_parts.append(str(row))
                if len(results) > 5:
                    response_parts.append(f"... y {len(results) - 5} más filas.")
            else: 
                response_parts.append(str(results))
            return "\n".join(response_parts)
        elif sql:
            return f"Se ejecutó la consulta {sql} pero no devolvió resultados."
        elif error_extract_msg and "Se proporcionó SQL directamente" in error_extract_msg:
            return f"La entrada fue un SQL directo. No se generó respuesta en lenguaje natural por el mock. SQL: {question}"
        return "No se pudo generar una consulta SQL o no hubo resultados."

def generate_sql_and_validate_whitelist(
    structured_info, 
    sql_generator_instance: SQLGenerator, 
    db_connector_for_sql_gen: DBConnector,
    current_allowed_columns_map, 
    terms_dict_str_for_validation, 
    question_for_logging,
    logger_param: logging.Logger = None
) -> tuple[str, list, str, any]:
    current_logger_sql_gen = logger_param if logger_param else logger
    current_logger_sql_gen.debug(f"DEBUG: [pipeline.py] Entrando en generate_sql_and_validate_whitelist con question: {question_for_logging}")
    
    # Si la pregunta original era SQL directo, structured_info podría contenerlo.
    if isinstance(structured_info, dict) and "direct_sql_query" in structured_info:
        direct_sql = structured_info["direct_sql_query"]
        current_logger_sql_gen.info(f"DEBUG: [pipeline.py] generate_sql_and_validate_whitelist recibió direct_sql_query: '{direct_sql}'. Se procesará este SQL directamente.")

        # INICIO: Parche robusto para convertir INNER JOIN a LEFT JOIN para MEDI_MEDICATIONS y MEDI_PHARMA_THERAPEUTIC_GROUPS
        # Este parche se aplica específicamente a la relación entre estas dos tablas a través de PHTH_ID.
        
        # Nombres de las tablas y columna de unión
        table_meds_name = "MEDI_MEDICATIONS"
        table_pharma_name = "MEDI_PHARMA_THERAPEUTIC_GROUPS"
        join_column_name = "PHTH_ID"

        # Patrones para encontrar las tablas y sus alias
        # Busca "TABLE_NAME opcional_AS alias" o "TABLE_NAME alias" (sin AS)
        # El alias es \\w+ (uno o más caracteres alfanuméricos o guion bajo)
        # \\s+ asegura uno o más espacios, (?:\\s+AS\\s+)? hace que " AS " sea opcional
        pattern_meds_alias = re.compile(fr"{table_meds_name}\\s+(?:AS\\s+)?(\\w+)", re.IGNORECASE)
        pattern_pharma_alias = re.compile(fr"{table_pharma_name}\\s+(?:AS\\s+)?(\\w+)", re.IGNORECASE)

        match_meds = pattern_meds_alias.search(direct_sql)
        alias_meds = match_meds.group(1) if match_meds else table_meds_name
        
        match_pharma = pattern_pharma_alias.search(direct_sql)
        alias_pharma = match_pharma.group(1) if match_pharma else table_pharma_name

        current_logger_sql_gen.debug(f"Parche LEFT JOIN: Alias detectados: {table_meds_name} -> {alias_meds}, {table_pharma_name} -> {alias_pharma}")

        # Construir el patrón de JOIN para reemplazar.
        # Buscamos:
        #   FROM MEDI_MEDICATIONS [AS] m ... 
        #   [INNER] JOIN MEDI_PHARMA_THERAPEUTIC_GROUPS [AS] p
        #   ON m.PHTH_ID = p.PHTH_ID (o viceversa)
        # El patrón debe ser flexible con los espacios y el uso opcional de "AS".
        # (\\bJOIN\\b|\\bINNER\\s+JOIN\\b) captura "JOIN" o "INNER JOIN" como grupo 1 (el tipo de join)
        # \\s+ asegura espacios.
        # {re.escape(alias_pharma)} usa el alias detectado para la tabla pharma.
        # (?:\\s+AS\\s+)? es opcional " AS ".
        # ON\\s+ hace match con "ON "
        # La condición de join es más compleja:
        #   (?:{re.escape(alias_meds)}\\.{join_column_name}\\s*=\\s*{re.escape(alias_pharma)}\\.{join_column_name}|{re.escape(alias_pharma)}\\.{join_column_name}\\s*=\\s*{re.escape(alias_meds)}\\.{join_column_name})
        #   Esto permite m.PHTH_ID = p.PHTH_ID O p.PHTH_ID = m.PHTH_ID
        
        # Patrón para encontrar el JOIN entre MEDI_MEDICATIONS y MEDI_PHARMA_THERAPEUTIC_GROUPS
        # Este patrón busca específicamente el JOIN que involucra a MEDI_PHARMA_THERAPEUTIC_GROUPS
        # y la condición ON que relaciona PHTH_ID de ambas tablas (usando sus alias).
        # Grupo 1: El tipo de JOIN (JOIN o INNER JOIN)
        # Grupo 2: El resto de la cláusula JOIN (MEDI_PHARMA_THERAPEUTIC_GROUPS [AS] alias ON ...)
        join_search_pattern = re.compile(
            r"(\bJOIN\b|\bINNER\s+JOIN\b)\s+" +  # Grupo 1: Tipo de JOIN
            r"(" + # Inicio Grupo 2: Resto de la cláusula JOIN
            re.escape(table_pharma_name) + r"\s+(?:AS\s+)?(?P<alias_pharma_match>\w+)\s+" + # Tabla Pharma con su alias
            r"ON\s+" +
            r"(?:" + # Grupo para las condiciones ON (alternativas)
            r"(?P=alias_pharma_match)\." + re.escape(join_column_name) + r"\s*=\s*" + re.escape(alias_meds) + r"\." + re.escape(join_column_name) + # p.PHTH_ID = m.PHTH_ID
            r"|" + 
            re.escape(alias_meds) + r"\." + re.escape(join_column_name) + r"\s*=\s*(?P=alias_pharma_match)\." + re.escape(join_column_name) + # m.PHTH_ID = p.PHTH_ID
            r")" + # Fin grupo condiciones ON
            r")", # Fin Grupo 2
            re.IGNORECASE
        )
        
        # Segunda búsqueda, ahora que alias_meds está definido.
        # Este es un patrón más específico para la tabla MEDI_PHARMA_THERAPEUTIC_GROUPS
        # y su condición ON con MEDI_MEDICATIONS.
        
        # Primero, intentamos encontrar un JOIN que una MEDI_PHARMA_THERAPEUTIC_GROUPS (alias_pharma)
        # a MEDI_MEDICATIONS (alias_meds)
        
        # Patrón para el JOIN de MEDI_PHARMA_THERAPEUTIC_GROUPS a MEDI_MEDICATIONS
        # (?:FROM|JOIN)\s+MEDI_MEDICATIONS\s+(?:AS\s+)?(\w+)\s+ # Captura alias de MEDI_MEDICATIONS
        # (INNER\s+JOIN|JOIN)\s+ # El tipo de JOIN
        # MEDI_PHARMA_THERAPEUTIC_GROUPS\s+(?:AS\s+)?(\w+)\s+ # Captura alias de MEDI_PHARMA_THERAPEUTIC_GROUPS
        # ON\s+\1\.PHTH_ID\s*=\s*\2\.PHTH_ID # Condición usando los alias capturados
        
        # Simplificamos: Buscamos "JOIN MEDI_PHARMA_THERAPEUTIC_GROUPS [AS] p ON m.PHTH_ID = p.PHTH_ID"
        # o "JOIN MEDI_PHARMA_THERAPEUTIC_GROUPS [AS] p ON p.PHTH_ID = m.PHTH_ID"
        # donde 'm' es el alias de MEDI_MEDICATIONS y 'p' el de MEDI_PHARMA_THERAPEUTIC_GROUPS.

        # Patrón para encontrar el JOIN específico y reemplazarlo
        # Este patrón busca:
        # 1. Un JOIN o INNER JOIN (case-insensitive)
        # 2. Seguido por la tabla MEDI_PHARMA_THERAPEUTIC_GROUPS (con su alias)
        # 3. Seguido por la condición ON que relaciona PHTH_ID de MEDI_MEDICATIONS (con su alias)
        #    y MEDI_PHARMA_THERAPEUTIC_GROUPS (con su alias).
        
        # El alias_pharma_in_on se refiere al alias de MEDI_PHARMA_THERAPEUTIC_GROUPS usado en la condición ON
        # El alias_meds_in_on se refiere al alias de MEDI_MEDICATIONS usado en la condición ON
        
        # Patrón: (JOIN_KEYWORD) (TABLE_PHARMA_NAME [AS] alias_pharma) ON (alias_meds.PHTH_ID = alias_pharma.PHTH_ID OR alias_pharma.PHTH_ID = alias_meds.PHTH_ID)
        # Grupo 1: (JOIN | INNER JOIN)
        # Grupo 2: MEDI_PHARMA_THERAPEUTIC_GROUPS\\s+(?:AS\\s+)?(\\w+)  -- El alias es el subgrupo \w+
        # Grupo 3: La condición ON completa
        
        # Necesitamos encontrar el alias de MEDI_PHARMA_THERAPEUTIC_GROUPS en su declaración
        # y luego usar ese alias y el alias_meds en la condición ON.

        pharma_table_declaration_pattern = re.compile(fr"{table_pharma_name}\s+(?:AS\s+)?(\w+)", re.IGNORECASE)
        pharma_alias_match_in_declaration = pharma_table_declaration_pattern.search(direct_sql)

        if pharma_alias_match_in_declaration:
            actual_pharma_alias = pharma_alias_match_in_declaration.group(1)
            current_logger_sql_gen.debug(f"Parche LEFT JOIN: Alias de {table_pharma_name} en su declaración: {actual_pharma_alias}")

            # Ahora construimos el patrón de JOIN completo
            # (JOIN | INNER JOIN) + MEDI_PHARMA_THERAPEUTIC_GROUPS [AS] actual_pharma_alias + ON + (condición con actual_pharma_alias y alias_meds)
            specific_join_pattern = re.compile(
                r"(\bJOIN\b|\bINNER\s+JOIN\b)\s+" + # Grupo 1: El tipo de JOIN
                re.escape(table_pharma_name) + r"\s+(?:AS\s+)?" + re.escape(actual_pharma_alias) + r"\s+" + # La tabla PHARMA con su alias exacto
                r"ON\s+" +
                r"(" + # Grupo 2: La condición ON
                    re.escape(alias_meds) + r"\." + re.escape(join_column_name) + r"\s*=\s*" + re.escape(actual_pharma_alias) + r"\." + re.escape(join_column_name) +
                    r"|" +
                    re.escape(actual_pharma_alias) + r"\." + re.escape(join_column_name) + r"\s*=\s*" + re.escape(alias_meds) + r"\." + re.escape(join_column_name) +
                r")",
                re.IGNORECASE
            )

            match = specific_join_pattern.search(direct_sql)
            if match:
                join_type_found = match.group(1) # "JOIN" o "INNER JOIN"
                
                # Construir la cláusula LEFT JOIN
                # LEFT JOIN MEDI_PHARMA_THERAPEUTIC_GROUPS [AS] actual_pharma_alias ON (condición original)
                # La condición original está en match.group(2) pero ya la tenemos por partes.
                # Es importante mantener la tabla MEDI_MEDICATIONS como la "izquierda" conceptualmente.
                
                original_join_clause = match.group(0) # La cláusula JOIN completa encontrada
                
                # Reemplazar el tipo de JOIN por LEFT JOIN
                # Si es "INNER JOIN", se reemplaza por "LEFT JOIN"
                # Si es "JOIN", se reemplaza por "LEFT JOIN"
                
                modified_join_clause = original_join_clause.upper().replace(join_type_found.upper(), "LEFT JOIN", 1)
                
                # Para asegurar que el case del resto de la cláusula se mantenga, aplicamos el replace sobre la original
                # Encontramos el índice del tipo de join y reemplazamos
                start_index = original_join_clause.upper().find(join_type_found.upper())
                end_index = start_index + len(join_type_found)
                
                # Reconstruimos con LEFT JOIN
                # Esto es para preservar el case original de la tabla y alias, solo cambiando el tipo de JOIN
                reconstructed_left_join = "LEFT JOIN" + original_join_clause[end_index:]
                
                modified_sql = direct_sql.replace(original_join_clause, reconstructed_left_join, 1)

                if modified_sql != direct_sql:
                    current_logger_sql_gen.warning(f"DEBUG: [pipeline.py] Consulta SQL modificada por parche robusto para usar LEFT JOIN. Original: '{direct_sql}', Modificada: '{modified_sql}'")
                    direct_sql = modified_sql
                else: # pragma: no cover
                    current_logger_sql_gen.warning(f"DEBUG: [pipeline.py] Parche LEFT JOIN robusto: No se pudo reemplazar el JOIN específico aunque el patrón coincidió. Original: '{direct_sql}' Patrón: '{specific_join_pattern.pattern}'")
            else:
                current_logger_sql_gen.debug(f"DEBUG: [pipeline.py] Parche LEFT JOIN robusto: No se encontró el patrón de JOIN específico para reemplazar. SQL: '{direct_sql}'. Patrón: '{specific_join_pattern.pattern}'")
        else:
            current_logger_sql_gen.debug(f"DEBUG: [pipeline.py] Parche LEFT JOIN robusto: No se pudo encontrar el alias de declaración para {table_pharma_name}. No se aplicará el parche.")
        
        # FIN: Parche robusto

        # Para SQL directo, después del parche, retornamos directamente.
        # No pasa por la validación de whitelist estructurada ni por el SQLGenerator.
        return direct_sql, [], "", None # SQL, params, error_msg, context

    # Si no es SQL directo, o si structured_info no es un dict, procedemos como antes.
    if not structured_info or not isinstance(structured_info, dict): # Asegurarse que structured_info es un dict si no es SQL directo
        current_logger_sql_gen.warning("  structured_info está vacío, no es un diccionario, o no es SQL directo al entrar a generate_sql_and_validate_whitelist.")
        return "", [], "structured_info estaba vacío o no era un diccionario, no se puede generar SQL.", None
    
    current_logger_sql_gen.debug(f"  structured_info para validación y SQL-gen: {json.dumps(structured_info, indent=2, ensure_ascii=False)}")

    # Whitelist validation
    try:
        is_valid_structured_info, validation_msg = whitelist_validator.validate_structured_info_whitelist(
            structured_info,
            allowed_columns_map=current_allowed_columns_map,
            terms_dictionary=terms_dict_str_for_validation,
            logger_param=current_logger_sql_gen # Pasar logger
        )
    except Exception as e_val:
        current_logger_sql_gen.error(f"Excepción durante validate_structured_info_whitelist: {e_val}", exc_info=True)
        return "", [], f"Error en la validación de whitelist: {e_val}", None

    if not is_valid_structured_info:
        error_message = f"Validación de whitelist de partes estructuradas fallida: {validation_msg}"
        current_logger_sql_gen.error(f"{error_message} para structured_info: {structured_info}")
        return "", [], error_message, None

    sql_query_generated = ""
    query_params = []
    error_message_generation = ""
    context_from_sql_gen = None

    try:
        sql_query_generated, query_params = sql_generator_instance.generate_sql(
            question_data=structured_info,
            db_connector=db_connector_for_sql_gen
        )
        if sql_query_generated and "SELECT 'Error:" in sql_query_generated:
            error_message_generation = sql_query_generated
            current_logger_sql_gen.error(f"DEBUG: [pipeline.py] SQLGenerator devolvió una query de error: {sql_query_generated}")
        elif not sql_query_generated:
            error_message_generation = "SQLGenerator devolvió una consulta vacía sin un mensaje de error explícito."
            current_logger_sql_gen.error(f"DEBUG: [pipeline.py] {error_message_generation}")
        if sql_query_generated and not error_message_generation:
             current_logger_sql_gen.info(f"DEBUG: [pipeline.py] SQL generado por SQLGenerator: {sql_query_generated}, Params: {query_params}")
    except Exception as e_gensql_call:
        error_message_generation = f"EXCEPCIÓN al llamar a SQLGenerator.generate_sql: {str(e_gensql_call)}"
        sql_query_generated = ""
        query_params = []
        current_logger_sql_gen.error(f"DEBUG: [pipeline.py] {error_message_generation}", exc_info=True)
    return sql_query_generated, query_params, error_message_generation, context_from_sql_gen

def chatbot_pipeline(question: str, db_connector_param: DBConnector, llm_param, terms_dict_path_param: str, schema_path_param: str, logger_param: logging.Logger = None):
    pipeline_start_time = time.time()
    
    current_logger = logger_param if logger_param else logger
    
    current_logger.info(f"--- Iniciando chatbot_pipeline para la pregunta: '{question}' ---")

    # Carga de diccionarios y estructuras (usando atributos de función para caché simple)
    actual_terms_dict_path = terms_dict_path_param
    if not actual_terms_dict_path:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        actual_terms_dict_path = os.path.join(base_dir, "data", "dictionary.json")
        current_logger.info(f"terms_dict_path_param no proporcionado, usando por defecto: {actual_terms_dict_path}")

    if not hasattr(chatbot_pipeline, 'terms_dictionary_common') or \
       getattr(chatbot_pipeline, "terms_dictionary_path_cache", None) != actual_terms_dict_path or \
       not chatbot_pipeline.terms_dictionary_common:
        current_logger.info(f"Cargando terms_dictionary_common desde {actual_terms_dict_path}...")
        try:
            with open(actual_terms_dict_path, 'r', encoding='utf-8') as f:
                chatbot_pipeline.terms_dictionary_common = json.load(f)
            chatbot_pipeline.terms_dictionary_str_common = json.dumps(chatbot_pipeline.terms_dictionary_common, ensure_ascii=False)
            chatbot_pipeline.terms_dictionary_path_cache = actual_terms_dict_path
            current_logger.info(f"terms_dictionary_common cargado desde {actual_terms_dict_path} con {len(chatbot_pipeline.terms_dictionary_common)} entradas.")
        except FileNotFoundError:
            current_logger.error(f"Error CRÍTICO al cargar el diccionario de términos: [Errno 2] No such file or directory: '{actual_terms_dict_path}'. El pipeline no puede continuar sin este archivo.")
            return {"response": f"Error crítico: No se pudo cargar el archivo de configuración esencial '{os.path.basename(actual_terms_dict_path)}'. Verifique la ruta y la existencia del archivo.", "data": None, "executed_sql_query_info": None}
        except json.JSONDecodeError as e_json:
            current_logger.error(f"Error CRÍTICO al decodificar JSON del diccionario de términos en: '{actual_terms_dict_path}'. Error: {e_json}", exc_info=True)
            return {"response": f"Error crítico: El archivo de configuración '{os.path.basename(actual_terms_dict_path)}' está corrupto o no es un JSON válido.", "data": None, "executed_sql_query_info": None}
        except Exception as e:
            current_logger.error(f"Error CRÍTICO inesperado al cargar el diccionario de términos desde '{actual_terms_dict_path}': {e}", exc_info=True)
            return {"response": f"Error crítico: No se pudo cargar el diccionario de términos debido a un error inesperado. {e}", "data": None, "executed_sql_query_info": None}
    
    if not chatbot_pipeline.terms_dictionary_common:
        current_logger.error(f"Error CRÍTICO: terms_dictionary_common está vacío después del intento de carga desde {actual_terms_dict_path}. El pipeline no puede continuar.")
        return {"response": "Error crítico: El diccionario de términos está vacío después de la carga. Verifique el archivo.", "data": None, "executed_sql_query_info": None}

    if not hasattr(chatbot_pipeline, 'db_structure_dict_common') or \
       not hasattr(chatbot_pipeline, 'allowed_columns_map_common') or \
       not hasattr(chatbot_pipeline, 'table_relationships_str_common') or \
       not hasattr(chatbot_pipeline, 'sql_generator_instance_common'):
        current_logger.info("Inicializando db_structure_dict_common, allowed_columns_map_common, table_relationships_str_common y SQLGenerator instance...")
        try:
            chatbot_pipeline.db_structure_dict_common = db_connector_param.get_db_structure_dict(schema_type='full')
            if not chatbot_pipeline.db_structure_dict_common:
                 current_logger.error("db_connector_param.get_db_structure_dict(schema_type='full') devolvió vacío.")
                 return {"response": "Error crítico: No se pudo obtener la estructura de la base de datos.", "data": None, "executed_sql_query_info": None}
            current_logger.info(f"db_structure_dict_common obtenida con {len(chatbot_pipeline.db_structure_dict_common)} tablas.")
            
            chatbot_pipeline.allowed_columns_map_common = {}
            problematic_tables_for_logging = []
            for table_name_for_map, cols_data in chatbot_pipeline.db_structure_dict_common.items():
                table_name_upper = table_name_for_map.upper()
                if not isinstance(table_name_for_map, str) or not table_name_for_map.strip():
                    current_logger.warning(f"Nombre de tabla inválido en db_structure: '{table_name_for_map}'. Se omite.")
                    continue
                col_names_for_map = []
                original_cols_data_repr = str(cols_data)[:200]
                if isinstance(cols_data, dict) and 'columns' in cols_data:
                    columns_list_or_dict = cols_data['columns']
                    if isinstance(columns_list_or_dict, list):
                        for col_entry in columns_list_or_dict:
                            if isinstance(col_entry, dict) and isinstance(col_entry.get('name'), str) and col_entry.get('name').strip():
                                col_names_for_map.append(col_entry['name'])
                            elif isinstance(col_entry, str) and col_entry.strip():
                                col_names_for_map.append(col_entry)
                    elif isinstance(columns_list_or_dict, str) and columns_list_or_dict.strip():
                        try:
                            parsed_columns = json.loads(columns_list_or_dict)
                            if isinstance(parsed_columns, list):
                                for item in parsed_columns:
                                    if isinstance(item, str) and item.strip(): col_names_for_map.append(item)
                                    elif isinstance(item, dict) and isinstance(item.get('name'), str) and item.get('name').strip(): col_names_for_map.append(item.get('name'))
                            else: problematic_tables_for_logging.append(f"{table_name_for_map} (JSON 'columns' no es lista)")
                        except json.JSONDecodeError: problematic_tables_for_logging.append(f"{table_name_for_map} (JSON 'columns' inválido)")
                    else: problematic_tables_for_logging.append(f"{table_name_for_map} (cols_data['columns'] tipo inesperado: {type(columns_list_or_dict)})")
                elif isinstance(cols_data, list):
                    for item in cols_data:
                        if isinstance(item, str) and item.strip(): col_names_for_map.append(item)
                        elif isinstance(item, dict) and isinstance(item.get('name'), str) and item.get('name').strip(): col_names_for_map.append(item.get('name'))
                    if not col_names_for_map and cols_data: problematic_tables_for_logging.append(f"{table_name_for_map} (cols_data lista pero no produjo columnas)")
                else: problematic_tables_for_logging.append(f"{table_name_for_map} (cols_data estructura inesperada: {type(cols_data)})")
                current_table_cols = [c.upper() for c in col_names_for_map if c]
                chatbot_pipeline.allowed_columns_map_common[table_name_upper] = current_table_cols
                if not current_table_cols and ((isinstance(cols_data, dict) and cols_data.get('columns')) or (isinstance(cols_data, list) and cols_data)):
                    reason = next((s for s in problematic_tables_for_logging if table_name_for_map in s), "razón no especificada")
                    current_logger.warning(f"Tabla '{table_name_for_map}' resultó en 0 columnas en allowed_columns_map. Razón/Tipo original: {reason}. Datos originales: {original_cols_data_repr}")
                    if not any(table_name_for_map in entry for entry in problematic_tables_for_logging):
                         problematic_tables_for_logging.append(f"{table_name_for_map} (datos entrada pero 0 columnas procesadas)")
            if problematic_tables_for_logging:
                current_logger.warning(f"--- Resumen tablas problemáticas/advertencias (allowed_columns_map): {problematic_tables_for_logging} ---")
            current_logger.info("DEBUG PIPELINE: Muestra allowed_columns_map_common (3 primeras tablas, 5 cols c/u):")
            sample_map = {k: v[:5] for k, v in list(chatbot_pipeline.allowed_columns_map_common.items())[:3]}
            current_logger.info(f"{json.dumps(sample_map, indent=2, ensure_ascii=False)}")
            chatbot_pipeline.table_relationships_str_common = db_connector_param.get_table_relationships_str()
            current_logger.info("table_relationships_str_common obtenido.")
            current_logger.info("Inicializando SQLGenerator instance...")
            allowed_tables_list = list(chatbot_pipeline.allowed_columns_map_common.keys())
            chatbot_pipeline.sql_generator_instance_common = SQLGenerator(
                allowed_tables=allowed_tables_list,
                allowed_columns=chatbot_pipeline.allowed_columns_map_common,
                enhanced_schema_path=schema_path_param, 
                relationships_str=chatbot_pipeline.table_relationships_str_common,
                dictionary_str=chatbot_pipeline.terms_dictionary_str_common,
                logger=current_logger
            )
            current_logger.info("SQLGenerator instance inicializada.")
        except Exception as e:
            current_logger.error(f"Error al inicializar estructuras de BD, mapa de columnas o SQLGenerator: {e}", exc_info=True)
            return {"response": f"Error crítico: No se pudo inicializar la configuración de la BD. {e}", "data": None, "executed_sql_query_info": None}

    s_time = time.time()
    preprocessed_question = preprocess_question(question, chatbot_pipeline.terms_dictionary_common)
    log_benchmark(s_time, "preprocess_question", logger_instance=current_logger)

    s_time = time.time()
    structured_info_dict = None
    error_llm_extract = "Error no inicializado en extract_info_from_question_llm"
    current_logger.info(f"DEBUG: [pipeline.py] Antes de llamar a extract_info_from_question_llm con question: '{preprocessed_question}'")
    try:
        structured_info_dict, error_llm_extract = extract_info_from_question_llm(
            preprocessed_question,
            llm_param,
            chatbot_pipeline.terms_dictionary_str_common,
            chatbot_pipeline.db_structure_dict_common,
            max_retries=MAX_RETRIES_LLM_EXTRACT,
            logger_param=current_logger
        )
        current_logger.info(f"DEBUG: [pipeline.py] Después de llamar a extract_info_from_question_llm. structured_info_dict: {json.dumps(structured_info_dict, indent=2) if structured_info_dict else 'None'}, error_llm_extract: '{error_llm_extract}'")
    except NameError:
        current_logger.error("La función 'extract_info_from_question_llm' no está definida o importada.", exc_info=True)
        return {"response": "Error de configuración interna: componente de extracción de información no encontrado.", "data": None, "executed_sql_query_info": None}
    except Exception as e_extract:
        current_logger.error(f"Excepción durante extract_info_from_question_llm: {e_extract}", exc_info=True)
        return {"response": f"Error al extraer información de la pregunta: {e_extract}", "data": None, "executed_sql_query_info": None}

    log_benchmark(s_time, "extract_info_from_question_llm", logger_instance=current_logger)

    if error_llm_extract or not structured_info_dict:
        if isinstance(structured_info_dict, dict) and "direct_sql_query" in structured_info_dict:
            current_logger.info(f"Bypass de extracción LLM: Se usará SQL directo: {structured_info_dict['direct_sql_query']}")
        else:
            current_logger.error(f"DEBUG: [pipeline.py] extract_info_from_question_llm falló o no devolvió info. Error: {error_llm_extract}. Info: {structured_info_dict}")
            answer = f"No se pudo extraer información de la pregunta. {error_llm_extract or ''}".strip()
            return {"response": answer, "data": None, "executed_sql_query_info": {"error_extracting_info": True, "details": error_llm_extract}}

    s_time = time.time()
    sql_query = None
    query_params = None
    error_msg_sql_gen = "Error no inicializado en generate_sql_and_validate_whitelist"
    current_logger.info(f"DEBUG: [pipeline.py] Antes de llamar a generate_sql_and_validate_whitelist. structured_info_dict: {json.dumps(structured_info_dict, indent=2)}")
    sql_query, query_params, error_msg_sql_gen, context_for_nl_response = generate_sql_and_validate_whitelist(
        structured_info_dict,
        chatbot_pipeline.sql_generator_instance_common,
        db_connector_param,
        chatbot_pipeline.allowed_columns_map_common,
        chatbot_pipeline.terms_dictionary_str_common,
        preprocessed_question,
        logger_param=current_logger
    )
    current_logger.info(f"DEBUG: [pipeline.py] Después de llamar a generate_sql_and_validate_whitelist. sql_query: '{sql_query}', query_params: {query_params}, error_msg_sql_gen: '{error_msg_sql_gen}'")
    log_benchmark(s_time, "generate_sql_and_validate_whitelist", logger_instance=current_logger)
    
    final_response_message = ""
    data_results_for_output = None
    executed_sql_query_info_for_output = {
        "original_question": question,
        "preprocessed_question": preprocessed_question,
        "structured_info": structured_info_dict,
        "generated_sql": sql_query or "No generada",
        "query_params": query_params or [],
        "error_in_sql_generation": error_msg_sql_gen if error_msg_sql_gen else None,
        "execution_error": None,
        "nl_generation_error": None,
        "final_executed_query": None
    }

    if error_msg_sql_gen or not sql_query:
        current_logger.error(f"DEBUG: [pipeline.py] Error en SQL-gen/validation o SQL no generado: {error_msg_sql_gen}. SQL Query era: '{sql_query}'")
        user_error_message = f"Lo siento, no pude procesar tu pregunta para generar una consulta SQL. Detalle: {error_msg_sql_gen}"
        if error_msg_sql_gen:
            if "No se especificó ninguna tabla válida" in error_msg_sql_gen:
                user_error_message = "No pude identificar las tablas correctas en la base de datos para tu consulta. Intenta reformularla."
            elif "Validación de whitelist" in error_msg_sql_gen:
                user_error_message = "Tu pregunta contiene términos o estructuras que no puedo procesar de forma segura. Intenta simplificarla."
        final_response_message = user_error_message
    else:
        current_logger.info(f"DEBUG: [pipeline.py] SQL Query a ejecutar: {sql_query}, Params: {query_params}")
        results_from_db = None
        column_names_from_db = None
        try:
            s_time = time.time()
            current_logger.info(f"DEBUG: [pipeline.py] Antes de llamar a db_connector_param.execute_query con SQL: {sql_query}")
            results_from_db, column_names_from_db = db_connector_param.execute_query(sql_query, query_params)
            data_results_for_output = results_from_db 
            executed_sql_query_info_for_output["final_executed_query"] = sql_query 
            current_logger.info(f"DEBUG: [pipeline.py] Después de llamar a db_connector_param.execute_query. Results (primeras 3 filas si hay): {str(results_from_db[:3]) if results_from_db else 'None/Empty'}, Column names: {column_names_from_db}")
            log_benchmark(s_time, "db_connector.execute_query", logger_instance=current_logger)

            if results_from_db is None: 
                current_logger.error(f"DEBUG: [pipeline.py] La ejecución de SQL falló (results_from_db is None). Query: {sql_query}, Params: {query_params}")
                final_response_message = "Error al ejecutar la consulta SQL en la base de datos."
                executed_sql_query_info_for_output["execution_error"] = "La consulta no devolvió resultados (None)."
            else:
                current_logger.info(f"DEBUG: [pipeline.py] Resultados ({len(results_from_db)} filas): {str(results_from_db[:3])[:500]}...")
                s_time = time.time()
                # nl_response_from_llm = "Respuesta no generada por LLM (placeholder)" # Eliminado placeholder antiguo
                current_logger.info(f"DEBUG: [pipeline.py] Antes de llamar a generate_natural_language_response_via_llm. Pregunta: '{preprocessed_question}', SQL: '{sql_query}', Results (primeras 3): {str(results_from_db[:3]) if results_from_db else 'None/Empty'}")
                try:
                    # MODIFICADO: Llamar a la nueva función que usa el LLM directamente
                    nl_response_from_llm = generate_natural_language_response_via_llm(
                        llm=llm_param, # Pasar la instancia del LLM
                        original_question=preprocessed_question,
                        sql_query=sql_query,
                        results=results_from_db,
                        column_names=column_names_from_db,
                        context=context_for_nl_response, 
                        error_extract_msg=error_llm_extract, # error_message_from_llm_extract
                        logger_param=current_logger
                    )
                    final_response_message = nl_response_from_llm
                    current_logger.info(f"DEBUG: [pipeline.py] Después de llamar a generate_natural_language_response_via_llm. Respuesta: '{final_response_message}'")
                except Exception as e_nl_resp:
                    current_logger.error(f"DEBUG: [pipeline.py] Excepción durante generate_natural_language_response_via_llm: {e_nl_resp}", exc_info=True)
                    final_response_message = f"Error al generar la respuesta en lenguaje natural: {e_nl_resp}"
                    executed_sql_query_info_for_output["nl_generation_error"] = str(e_nl_resp)
                log_benchmark(s_time, "generate_natural_language_response_via_llm", logger_instance=current_logger)
        except Exception as e_exec_or_nl:

            current_logger.error(f"DEBUG: [pipeline.py] EXCEPCIÓN durante ejecución SQL o generación NL: {e_exec_or_nl}", exc_info=True)
            final_response_message = f"Error inesperado al procesar consulta o generar respuesta: {e_exec_or_nl}"
            executed_sql_query_info_for_output["execution_error"] = str(e_exec_or_nl)

    log_benchmark(pipeline_start_time, "TOTAL chatbot_pipeline", logger_instance=current_logger)
    current_logger.info(f"DEBUG: [pipeline.py] Retornando desde chatbot_pipeline. Response: '{final_response_message}', Data: (presente: {data_results_for_output is not None}), SQL Info: (presente: {executed_sql_query_info_for_output is not None})")
    
    return {
        "response": final_response_message,
        "data": data_results_for_output,
        "executed_sql_query_info": executed_sql_query_info_for_output
    }


# MODIFICADO: Lógica interna para simular llamada a LLM y manejar preguntas específicas
def extract_info_from_question_llm(question, llm, terms_dict_str, db_structure_dict, max_retries, logger_param=None):
    current_logger = logger_param if logger_param else logger
    
    # SIMULACIÓN DE DETECCIÓN DE SQL DIRECTO O ENVUELTO EN MARKDOWN
    # Esta lógica intenta identificar si la pregunta es una consulta SQL directa
    # o una consulta SQL envuelta en bloques de código Markdown.
    
    stripped_question = question.strip()
    # Palabras clave SQL para ayudar a identificar consultas. Añadir más si es necesario.
    sql_keywords = ("SELECT", "INSERT", "UPDATE", "DELETE", "WITH", "CREATE", "ALTER", "DROP", "EXPLAIN", "--")

    # Caso 1: SQL directo (comienza con una palabra clave SQL y no está envuelto en ```)
    # Se considera que si empieza con ``` no es SQL "crudo" directo.
    # Convertir a mayúsculas para la comparación insensible a mayúsculas/minúsculas.
    # También se permite que comience con un comentario SQL.
    is_direct_sql_candidate = False
    for keyword in sql_keywords:
        if stripped_question.upper().startswith(keyword):
            is_direct_sql_candidate = True
            break
            
    if is_direct_sql_candidate and not stripped_question.startswith("```"):
        current_logger.info(f"DEBUG: [pipeline.py] extract_info_from_question_llm detectó SQL directo (sin Markdown): '{stripped_question}'")
        return {"direct_sql_query": stripped_question}, None

    # Caso 2: SQL envuelto en Markdown (ej. ```sql ... ``` o ``` ... ```)
    extracted_sql_from_markdown = None
    if stripped_question.startswith("```") and stripped_question.endswith("```"):
        # Quitar los backticks externos ```
        content_inside_backticks = stripped_question[3:-3].strip()
        
        # Verificar si tiene el identificador de lenguaje 'sql' (ej: ```sql SELECT...)
        # Se hace en minúsculas para ser insensible al caso de 'sql'.
        if content_inside_backticks.lower().startswith("sql"):
            # Quitar 'sql' y cualquier espacio/salto de línea posterior para obtener la consulta real.
            potential_sql = content_inside_backticks[3:].strip()
            if potential_sql: # Asegurarse de que haya contenido después de 'sql'
                extracted_sql_from_markdown = potential_sql
                current_logger.info(f"DEBUG: [pipeline.py] extract_info_from_question_llm detectó SQL en Markdown con 'sql' tag. Original: '{question}', Extraído: '{extracted_sql_from_markdown}'")
        else:
            # No tiene 'sql' tag, pero el contenido podría ser SQL si empieza con palabras clave SQL.
            # Esto maneja casos como ```SELECT ...```
            is_generic_markdown_sql_candidate = False
            for keyword in sql_keywords:
                if content_inside_backticks.upper().startswith(keyword):
                    is_generic_markdown_sql_candidate = True
                    break
            if is_generic_markdown_sql_candidate:
                extracted_sql_from_markdown = content_inside_backticks
                current_logger.info(f"DEBUG: [pipeline.py] extract_info_from_question_llm detectó SQL en Markdown genérico (sin 'sql' tag). Original: '{question}', Extraído: '{extracted_sql_from_markdown}'")

    # Si se extrajo SQL del Markdown y no está vacío, usarlo.
    if extracted_sql_from_markdown and extracted_sql_from_markdown.strip():
        actual_extracted_sql = extracted_sql_from_markdown.strip()
        current_logger.info(f"DEBUG: [pipeline.py] SQL extraído de Markdown para usar como direct_sql_query: '{actual_extracted_sql}'")
        return {"direct_sql_query": actual_extracted_sql}, None
    # else: # Opcional: log si se detectó markdown pero no se pudo extraer SQL válido
        # if stripped_question.startswith("```"): # Solo loguear si era un bloque de código
            # current_logger.debug(f"DEBUG: [pipeline.py] Se detectó bloque Markdown pero no se extrajo SQL válido o estaba vacío. Query: '{question}', Contenido interno: '{content_inside_backticks if 'content_inside_backticks' in locals() else 'N/A'}'")

    # Si no es SQL directo ni SQL en Markdown detectado arriba, proceder con la simulación de LLM / llamada real
    current_logger.info(f"DEBUG: [pipeline.py] No se detectó SQL directo ni en Markdown válido. Procediendo con lógica LLM para: '{question}'")

    # --- INICIO SIMULACIÓN LLAMADA A LLM ---
    # Aquí es donde se construiría el prompt y se llamaría al LLM
    # Ejemplo de prompt (muy simplificado):
    # prompt_text = f"""Dada la pregunta del usuario, el diccionario de términos y la estructura de la base de datos, 
    # extrae la información necesaria para construir una consulta SQL.
    # Pregunta: {question}
    # Diccionario de términos: {terms_dict_str} # Podría ser muy largo, considerar resúmenes o partes relevantes.
    # Estructura de la BD: {json.dumps(db_structure_dict)} # También podría ser muy largo.
    #
    # Devuelve un JSON con la siguiente estructura (o un subconjunto si no toda la info es necesaria):
    # {{
    #     "tables": ["TABLE_NAME_1", "TABLE_NAME_2"],
    #     "columns": ["TABLE_NAME_1.COLUMN_NAME_1", "TABLE_NAME_2.COLUMN_NAME_2"],
    #     "conditions": [
    #         {{"field": "TABLE_NAME.COLUMN_NAME", "operator": "=", "value": "some_value"}},
    #         {{"field": "ANOTHER_TABLE.COLUMN_NAME", "operator": "LIKE", "value": "%another_value%"}}
    #     ],
    #     "joins": [ # Opcional, si el LLM puede inferir joins complejos
    #        {{"type": "INNER", "from_table": "T1", "to_table": "T2", "on": "T1.id = T2.t1_id"}}
    #     ],
    #     "order_by": [{{"field": "TABLE_NAME.COLUMN_NAME", "direction": "ASC"}}],
    #     "limit": 10,
    #     "aggregation": [{{"function": "COUNT", "field": "*", "alias": "total_count"}}] # Ejemplo
    # }}
    # Si no puedes extraer la información, devuelve un JSON vacío {{}} y un mensaje de error.
    # Asegúrate de que los nombres de tablas y columnas coincidan exactamente con los de la estructura de la BD.
    # """
    # current_logger.debug(f"DEBUG: [pipeline.py] Prompt para LLM (extracción info) para pregunta '{question}' sería: {prompt_text[:500]}...") # Loguear solo una parte

    # structured_info_str = None # Para evitar UnboundLocalError si el try falla antes de la asignación
    # for attempt in range(max_retries + 1):
    #     try:
    #         current_logger.info(f"Attempt {attempt + 1} of {max_retries + 1} to call LLM for structure extraction for question: '{question}'.")
    #         # response_from_llm = llm.invoke(prompt_text) # LLAMADA REAL AL LLM (ej. ChatOpenAI)
    #         # structured_info_str = response_from_llm.content 
    #         # current_logger.info(f"DEBUG: [pipeline.py] LLM (extracción) raw response: {structured_info_str}")
    #         # structured_info = json.loads(structured_info_str)
    #         # if not structured_info: # Si el LLM devuelve un JSON vacío explícitamente
    #         #    current_logger.warning(f"LLM devolvió un JSON vacío para la pregunta: {question}")
    #         #    return {}, "El LLM no pudo extraer información estructurada (devolvió JSON vacío)."
    #         # return structured_info, None # Asumiendo que el LLM devuelve el JSON directamente
    #         # --- SIMULACIÓN DE LLAMADA --- 
    #         current_logger.warning("LLM call for structure extraction is COMMENTED OUT. Simulating failure.")
    #         # Para probar el flujo como si el LLM no pudiera extraer:
    #         if attempt < max_retries: # Simular fallo y reintento
    #             # raise Exception(f"Simulated LLM error on attempt {attempt + 1}") 
    #             pass # Simular que no se obtiene respuesta y se reintenta
    #         # En el último intento, devolver error simulado
    #         # structured_info_str = "{}" # Simular JSON vacío del LLM
    #         # structured_info = json.loads(structured_info_str)
    #         # return structured_info, "El LLM no pudo extraer información (simulación de JSON vacío)."
    #         # O simular un error más genérico:
    #         return {}, "El LLM no pudo extraer información estructurada (simulación de fallo tras reintentos)."
    #         # --- FIN SIMULACIÓN --- 
    #     except json.JSONDecodeError as e_json_dec:
    #         current_logger.error(f"Error al decodificar JSON de LLM en el intento {attempt + 1}: {e_json_dec}. Respuesta: {structured_info_str}")
    #         if attempt == max_retries:
    #             return {}, f"Error al decodificar JSON de LLM tras {max_retries + 1} intentos: {e_json_dec}. Respuesta original: {structured_info_str}"
    #         time.sleep(1) # Esperar antes de reintentar
    #     except Exception as e_llm:
    #         current_logger.error(f"Excepción al llamar a LLM para extracción en el intento {attempt + 1}: {e_llm}", exc_info=True)
    #         if attempt == max_retries:
    #            return {}, f"Excepción en LLM tras {max_retries + 1} intentos: {e_llm}"
    #         time.sleep(1) # Esperar antes de reintentar
    # --- FIN LLAMADA REAL AL LLM ---

    # Si la llamada al LLM (actualmente comentada y simulando fallo) no devuelve structured_info:
    current_logger.warning(f"DEBUG: [pipeline.py] LLM (real o simulado) no pudo extraer información para la pregunta: '{question}'. Se devuelve estructura vacía y mensaje de error.")
    return {}, "No se pudo extraer información de la pregunta mediante el LLM."


# NUEVA función para generar respuesta NL con LLM
def generate_natural_language_response_via_llm(llm, original_question, sql_query, results, column_names, context, error_extract_msg, logger_param=None):
    current_logger_nl = logger_param if logger_param else logger
    current_logger_nl.info(f"DEBUG: [pipeline.py] Entrando en generate_natural_language_response_via_llm con LLM: {type(llm)}")

    # --- INICIO COMENTARIOS SOBRE LLAMADA REAL AL LLM ---
    # Aquí es donde se construiría el prompt y se llamaría al LLM
    # prompt_parts = [f"Pregunta original del usuario: {original_question}"]
    # if sql_query and not ("Error:" in sql_query or (error_extract_msg and "Se proporcionó SQL directamente" not in error_extract_msg)):
    #     prompt_parts.append(f"Consulta SQL ejecutada con éxito: {sql_query}")
    
    # if results is not None:
    #     prompt_parts.append(f"Resultados ({len(results)} filas encontradas):")
    #     if column_names:
    #         prompt_parts.append("Columnas: " + " | ".join(column_names))
    #     # Mostrar algunas filas de ejemplo, no todas si son muchas
    #     for i, row in enumerate(results[:3]): # Mostrar hasta 3 filas
    #         prompt_parts.append(f"Fila {i+1}: " + " | ".join(map(str, row)))
    #     if len(results) > 3:
    #         prompt_parts.append(f"... y {len(results) - 3} fila(s) más.")
    #     if not results:
    #          prompt_parts.append("No se encontraron resultados que coincidan con tu consulta.")
    # elif sql_query and not ("Error:" in sql_query):
    #     prompt_parts.append("La consulta se ejecutó pero no devolvió filas.")


    # if error_extract_msg and "Se proporcionó SQL directamente" not in error_extract_msg:
    #      prompt_parts.append(f"Hubo un problema al procesar la pregunta antes de generar SQL: {error_extract_msg}")
    # elif sql_query and "Error:" in sql_query : # Si SQLGenerator devolvió un error
    #      prompt_parts.append(f"Hubo un error al intentar generar la consulta SQL: {sql_query}")


    # prompt_parts.append("Por favor, genera una respuesta concisa, amigable y útil en lenguaje natural para el usuario, resumiendo los hallazgos o explicando el problema.")
    # final_prompt = "\\n".join(prompt_parts)
    # current_logger_nl.debug(f"DEBUG: [pipeline.py] Prompt para LLM (generación NL):\\n{final_prompt[:1000]}...") # Loguear parte del prompt

    # try:
    #     # Asumiendo que llm es una instancia de Langchain (como ChatOpenAI) que tiene el método .invoke()
    #     # response = llm.invoke(final_prompt) 
    #     # generated_text = response.content # o la forma apropiada de obtener el texto
    # --- FIN COMENTARIOS SOBRE LLAMADA REAL AL LLM ---

    # --- INICIO LLM SIMULADO para generate_natural_language_response_via_llm ---
    current_logger_nl.warning("USANDO LLM SIMULADO para generate_natural_language_response_via_llm")
    generated_text = ""
    if results is not None: # results puede ser una lista vacía
        if results: # Hay filas
            generated_text = f"He encontrado {len(results)} resultado(s) para tu consulta sobre '{original_question}'. "
            if column_names:
                generated_text += f"Las columnas son: {', '.join(column_names)}. "
            generated_text += f"Aquí tienes algunos ejemplos: {str(results[:2])}" # Muestra los 2 primeros
            if len(results) > 2:
                generated_text += f" (y {len(results)-2} más)."
        else: # No hay filas
             generated_text = f"No se encontraron resultados para tu consulta: '{original_question}'."
    elif sql_query and "Error:" in sql_query : # Error de SQLGenerator
         generated_text = f"Hubo un problema al generar la consulta para tu pregunta '{original_question}'. Detalle: {sql_query}"
    elif error_extract_msg and "Se proporcionó SQL directamente" not in error_extract_msg: # Error de extracción
        generated_text = f"No pude entender completamente tu pregunta sobre '{original_question}'. Error: {error_extract_msg}"
    elif error_extract_msg and "Se proporcionó SQL directamente" in error_extract_msg and sql_query and results is None: # SQL directo, pero sin resultados (results es None)
         generated_text = f"Ejecuté la consulta SQL que proporcionaste, pero no obtuve resultados o hubo un error en la ejecución."
    else: # Caso genérico
        generated_text = f"He procesado tu pregunta '{original_question}'. Si esperabas datos específicos y no los ves, por favor, reformula tu pregunta o verifica los términos."
    # --- FIN LLM SIMULADO ---

    current_logger_nl.info(f"DEBUG: [pipeline.py] Respuesta generada por LLM (simulada): {generated_text}")
    return generated_text
    # except Exception as e: # Comentado porque la llamada real al LLM está comentada
    #     current_logger_nl.error(f"Excepción al llamar al LLM para generar respuesta NL: {e}", exc_info=True)
    #     return f"Lo siento, tuve un problema al generar la respuesta en lenguaje natural. Error: {e}"


class MockLLMForPipeline:
    def __init__(self, logger_instance=None):
        self.logger = logger_instance if logger_instance else logging.getLogger(__name__ + ".MockLLMForPipeline")
        if not logger_instance:
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        self.logger.info("MockLLMForPipeline instanciado.")

    def generate_natural_language_response(self, question, sql, results, columns, context, error_extract_msg, logger_param=None):
        current_logger_nl = logger_param if logger_param else self.logger 
        current_logger_nl.warning("DEBUG: [pipeline.py] Usando MockLLMForPipeline.generate_natural_language_response")
        if error_extract_msg and "Se proporcionó SQL directamente" not in error_extract_msg:
            return f"Error al extraer información: {error_extract_msg}"
        if sql and results:
            response_parts = [f"Resultados para la consulta: {sql}"]
            if isinstance(results, list):
                if not results:
                    response_parts.append("No se encontraron filas.")
                else:
                    if columns and results and isinstance(results[0], tuple):
                        response_parts.append(" | ".join(columns))
                        response_parts.append(" | ".join(["---"] * len(columns)))
                        for row in results[:5]: 
                            response_parts.append(" | ".join(map(str, row)))
                    elif results and isinstance(results[0], dict):
                        actual_columns = list(results[0].keys())
                        response_parts.append(" | ".join(actual_columns))
                        response_parts.append(" | ".join(["---"] * len(actual_columns)))
                        for row in results[:5]: 
                            response_parts.append(" | ".join(str(row.get(col, '')) for col in actual_columns))
                    else: 
                        for row in results[:5]:
                            response_parts.append(str(row))
                if len(results) > 5:
                    response_parts.append(f"... y {len(results) - 5} más filas.")
            else: 
                response_parts.append(str(results))
            return "\n".join(response_parts)
        elif sql:
            return f"Se ejecutó la consulta {sql} pero no devolvió resultados."
        elif error_extract_msg and "Se proporcionó SQL directamente" in error_extract_msg:
            return f"La entrada fue un SQL directo. No se generó respuesta en lenguaje natural por el mock. SQL: {question}"
        return "No se pudo generar una consulta SQL o no hubo resultados."