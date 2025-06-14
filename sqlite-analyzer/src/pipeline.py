import json
import logging
import re
import time
import os
import sys
from typing import Any, Dict, Optional, Set, List # Asegúrate de tener Set y List

import difflib


import openai
from langchain_core.messages import HumanMessage, SystemMessage

# Asegurarse de que el directorio raíz del proyecto esté en sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Cambiar importaciones relativas a absolutas y eliminar table_identifier

from sql_generator import SQLGenerator
from sql_utils import extract_sql_from_markdown
from whitelist_validator import WhitelistValidator
from db_connector import DBConnector
from db_relationship_graph import build_relationship_graph, find_join_path, generate_join_path

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
            r"(?P=alias_pharma_match)\." + re.escape(join_column_name) + r"\s*=\s*" + re.escape(alias_meds) + r"\." + re.escape(join_column_name) +
            r"|" + 
            re.escape(alias_meds) + r"\." + re.escape(join_column_name) + r"\s*=\s*(?P=alias_pharma_match)\." + re.escape(join_column_name) +
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

        pharma_table_declaration_pattern = re.compile(fr"{table_pharma_name}\\s+(?:AS\\s+)?(\\w+)", re.IGNORECASE)
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

    # Whitelist validation of structured_info (REMOVED as validate_structured_info_whitelist is no longer available)
    # try:
    #     is_valid_structured_info, validation_msg = validate_structured_info_whitelist(
    #         structured_info,
    #         allowed_columns_map=current_allowed_columns_map,
    #         terms_dictionary=terms_dict_str_for_validation,
    #         logger_param=current_logger_sql_gen # Pasar logger
    #     )
    # except Exception as e_val:
    #     current_logger_sql_gen.error(f\"Excepción durante validate_structured_info_whitelist: {e_val}\", exc_info=True)
    #     return \"\", [], f\"Error en la validación de whitelist: {e_val}\", None

    # if not is_valid_structured_info:
    #     error_message = f\"Validación de whitelist de partes estructuradas fallida: {validation_msg}\"
    #     current_logger_sql_gen.error(f\"{error_message} para structured_info: {structured_info}\")
    #     return \"\", [], error_message, None

    # --- INICIO: INFERENCIA DE JOINs MULTI-HOP GENÉRICA ---
    if isinstance(structured_info, dict) and "tables" in structured_info and isinstance(structured_info["tables"], list):
        tables = [t for t in structured_info["tables"] if isinstance(t, str)]
        # Extraer también tablas de columnas y condiciones
        cols = structured_info.get("columns", [])
        conds = structured_info.get("conditions", [])
        tables_from_cols = set()
        for c in cols:
            if isinstance(c, str) and "." in c:
                tables_from_cols.add(c.split(".")[0])
            elif isinstance(c, dict) and "name" in c and "." in c["name"]:
                tables_from_cols.add(c["name"].split(".")[0])
        tables_from_conds = set()
        for cond in conds:
            if isinstance(cond, dict) and "column" in cond and "." in cond["column"]:
                tables_from_conds.add(cond["column"].split(".")[0])
        all_tables = set(tables) | tables_from_cols | tables_from_conds
        all_tables = [t for t in all_tables if t]
        # Construir grafo de relaciones
        try:
            # Cargar relaciones desde JSON si existe
            import os, json
            rel_path = os.path.join(os.path.dirname(__file__), '../../table_relationships.json')
            if os.path.exists(rel_path):
                with open(rel_path, 'r', encoding='utf-8') as f:
                    rel_map = json.load(f)
            else:
                rel_map = None
        except Exception as e:
            rel_map = None
            current_logger_sql_gen.warning(f"No se pudo cargar table_relationships.json: {e}")
        try:
            relationship_graph = build_relationship_graph(db_connector_for_sql_gen, table_relationships=rel_map)
        except Exception as e:
            current_logger_sql_gen.error(f"Error al construir el grafo de relaciones: {e}")
            relationship_graph = None
        # Inferir JOINs multi-hop si hay más de una tabla
        join_path_defs = []
        join_error = None
        if relationship_graph and len(all_tables) > 1:
            # Buscar caminos entre todas las tablas relevantes
            from itertools import combinations
            joins_needed = []
            for t1, t2 in combinations(all_tables, 2):
                path = find_join_path(relationship_graph, t1, t2, max_depth=6)
                if not path:
                    join_error = f"No se puede conectar {t1} y {t2} en la base de datos. No existe un camino de JOIN válido."
                    break
                joins_needed.append(path)
            if not join_error:
                # Unir todos los caminos en una secuencia de tablas (camino mínimo de unión)
                # Para simplicidad, usar la unión de todos los nodos de los caminos
                tables_join_order = [all_tables[0]]
                for path in joins_needed:
                    for t in path[1:]:
                        if t not in tables_join_order:
                            tables_join_order.append(t)
                # Generar la secuencia de JOINs
                join_path_defs = generate_join_path(relationship_graph, tables_join_order)
                # Añadir a structured_info['joins']
                structured_info["joins"] = join_path_defs
        if join_error:
            return "", [], join_error, None
    # --- FIN INFERENCIA JOINs MULTI-HOP ---

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
    
    # Añadir logs detallados en generate_sql_and_validate_whitelist
    logger_param.debug(f"DEBUG: [pipeline.py] structured_info recibido: {structured_info}")
    logger_param.debug(f"DEBUG: [pipeline.py] current_allowed_columns_map recibido: {current_allowed_columns_map}")
    logger_param.debug(f"DEBUG: [pipeline.py] terms_dict_str_for_validation recibido: {terms_dict_str_for_validation}")

    # Log después de generar SQL
    logger_param.debug(f"DEBUG: [pipeline.py] SQL generado: {sql_query_generated}")
    logger_param.debug(f"DEBUG: [pipeline.py] Parámetros de la consulta: {query_params}")
    logger_param.debug(f"DEBUG: [pipeline.py] Mensaje de error en generación SQL: {error_message_generation}")

    return sql_query_generated, query_params, error_message_generation, context_from_sql_gen

def extract_info_from_question_llm(
    question: str,
    db_structure_dict: dict,
    terms_dict: dict,
    llm_client,
    current_logger,
    relationship_graph=None,  # <--- descomentado aquí
    max_retries=1
) -> Optional[Dict[str, Any]]:
    """
    Utiliza un LLM para extraer información estructurada de la pregunta del usuario,
    considerando la estructura de la base de datos y el diccionario de términos.
    Intenta generar una consulta SQL si la pregunta parece requerirlo.

    Args:
        question (str): La pregunta del usuario.
        db_structure_dict (dict): El diccionario con la estructura de la BD.
        terms_dict (dict): El diccionario de términos.
        llm_client: El cliente del LLM inicializado (se asume ChatOpenAI).
        current_logger: La instancia del logger.
        relationship_graph: El grafo de relaciones (opcional).
        max_retries (int): Número máximo de reintentos para la llamada al LLM.

    Returns:
        tuple: (dict, str) donde el dict es la información estructurada (o consulta SQL)
               y str es un mensaje de error/estado.
    """
    current_logger.info(f"DEBUG: [pipeline.py] Iniciando extract_info_from_question_llm para la pregunta: '{question}'")

    # Comprobación simple para SQL directo o en Markdown
    sql_directo = extract_sql_from_markdown(question)
    if sql_directo:
        current_logger.info(f"DEBUG: [pipeline.py] SQL detectado directamente o en Markdown: {sql_directo}")
        return {"sql_query": sql_directo}, "SQL extraído directamente de la pregunta."

    current_logger.info(f"DEBUG: [pipeline.py] No se detectó SQL directo ni en Markdown válido. Procediendo con lógica LLM para: '{question}'")
    
    # --- INICIO LLAMADA REAL AL LLM ---
    
    # Detectar entidades clave (tablas) mencionadas en la pregunta
    forced_tables = set()
    # Intenta extraer nombres de tablas de la pregunta (ej. PALABRA_EN_MAYUSCULAS_CON_GUIONES_BAJOS)
    # Esto es una heurística simple, podría necesitar refinamiento.
    # Busca palabras que parezcan nombres de tablas (mayúsculas, guiones bajos, mínimo 3 caracteres)
    potential_tables_in_question = re.findall(r'\b([A-Z][A-Z0-9_]{2,})\b', question)
    for pt_table_name in potential_tables_in_question:
        if pt_table_name in db_structure_dict:
            forced_tables.add(pt_table_name)
            current_logger.info(f"DEBUG: [pipeline.py] Tabla '{pt_table_name}' añadida a forced_tables desde la pregunta.")
    
    # Añadir automáticamente tablas puente usando el grafo de relaciones
    try:
        rel_path = os.path.join(os.path.dirname(__file__), '../../table_relationships.json')
        if os.path.exists(rel_path):
            with open(rel_path, 'r', encoding='utf-8') as f:
                rel_map = json.load(f)
        else:
            rel_map = None
    except Exception as e:
        rel_map = None
        current_logger.warning(f"No se pudo cargar table_relationships.json para robustez: {e}")
    try:
        from db_relationship_graph import build_relationship_graph, find_join_path
        relationship_graph = build_relationship_graph(None, table_relationships=rel_map, db_structure=db_structure_dict)
    except Exception as e:
        relationship_graph = None
        current_logger.warning(f"No se pudo construir el grafo de relaciones: {e}")
    if relationship_graph and len(forced_tables) > 1:
        from itertools import combinations
        for t1, t2 in combinations(forced_tables, 2):
            path = find_join_path(relationship_graph, t1, t2, max_depth=6)
            if path:
                forced_tables.update(path)
    forced_tables = list(forced_tables)

    # Construir contexto relevante directamente del esquema (sin RAG)
    # Para simplificar, pasamos todo el esquema y los términos relevantes (si existen)
    relevant_schema_dict = db_structure_dict
    # --- OPTIMIZACIÓN: Extraer solo el subesquema relevante para la pregunta ---
    def extract_relevant_schema(
        question_text: str,
        current_db_structure_dict: dict,
        explicitly_forced_tables: List[str], # Cambiado a List para la firma
        graph # Grafo de relaciones
    ) -> dict:
        relevant_sub_schema: Dict[str, Any] = {}
        tables_to_include: Set[str] = set(explicitly_forced_tables)

        # 2a. Añadir tablas forzadas explícitamente
        for forced_table_name in explicitly_forced_tables:
            if forced_table_name in current_db_structure_dict:
                if forced_table_name not in relevant_sub_schema:
                    relevant_sub_schema[forced_table_name] = current_db_structure_dict[forced_table_name]
                    current_logger.info(f"DEBUG: [pipeline.py] Tabla forzada '{forced_table_name}' añadida al subesquema.")
            else:
                current_logger.warning(f"DEBUG: [pipeline.py] Tabla forzada '{forced_table_name}' no encontrada en current_db_structure_dict.")

        # 2b. Añadir tablas que contengan 'PATI_ID' (común en esquemas médicos)
        #     y que no estén ya incluidas.
        for table_name, table_data in current_db_structure_dict.items():
            if "PATI_ID" in table_data.get("columns", []) and table_name not in relevant_sub_schema:
                relevant_sub_schema[table_name] = table_data
                tables_to_include.add(table_name)
                current_logger.info(f"DEBUG: [pipeline.py] Tabla '{table_name}' con PATI_ID añadida al subesquema.")

        # 2c. (Opcional Avanzado) Usar el grafo de relaciones para añadir tablas puente o directamente conectadas
        #     a las tablas ya seleccionadas, si es necesario y el grafo está disponible.
        #     Esto ayuda si una tabla de catálogo no tiene PATI_ID y no fue forzada, pero está vinculada.
        if graph and tables_to_include and hasattr(graph, "neighbors"):
            expanded_by_graph: Set[str] = set()
            for core_table in list(tables_to_include): # Iterar sobre una copia
                if core_table in graph:
                    for neighbor in graph.neighbors(core_table):
                        if neighbor in current_db_structure_dict and neighbor not in relevant_sub_schema:
                            relevant_sub_schema[neighbor] = current_db_structure_dict[neighbor]
                            expanded_by_graph.add(neighbor)
                            current_logger.info(f"DEBUG: [pipeline.py] Tabla vecina '{neighbor}' (conectada a '{core_table}') añadida al subesquema vía grafo.")
            if expanded_by_graph:
                 current_logger.info(f"DEBUG: [pipeline.py] Tablas añadidas por expansión de grafo: {expanded_by_graph}")
        elif graph and tables_to_include:
            current_logger.warning("El objeto 'graph' no tiene el método 'neighbors'. ¿build_relationship_graph está devolviendo un grafo de networkx?")

        # 2d. Fallback: Si después de todos los intentos el sub-esquema está vacío,
        #     y no había tablas forzadas inicialmente, podría ser una pregunta muy genérica.
        #     En este caso, devolver el esquema completo puede ser una opción, pero con advertencia.
        #     Si había tablas forzadas pero no se encontraron, relevant_sub_schema estaría vacío,
        #     lo que indicaría un problema (ya advertido antes).
        if not relevant_sub_schema:
            if not explicitly_forced_tables:
                current_logger.warning(f"DEBUG: [pipeline.py] Subesquema relevante está vacío y no hubo tablas forzadas para la pregunta: '{question_text}'. Considera si esto es esperado. Devolviendo esquema completo como fallback.")
                return current_db_structure_dict # Fallback MUY CAUTELOSO
            else:
                current_logger.warning(f"DEBUG: [pipeline.py] Subesquema relevante está vacío A PESAR de tablas forzadas: {explicitly_forced_tables} para la pregunta: '{question_text}'. Esto indica que las tablas forzadas no se encontraron o no se pudieron procesar.")
                return {} # Devolver vacío si las tablas forzadas no resultaron en un esquema

        return relevant_sub_schema

    # 3. Obtener el diccionario del sub-esquema y convertirlo a string
    relevant_schema_dict = extract_relevant_schema(question, db_structure_dict, list(forced_tables), relationship_graph)
    relevant_schema_str = json.dumps(relevant_schema_dict, ensure_ascii=False, indent=2) # indent para mejor logging

    # --- CORRECCIÓN: Definir relevant_terms_str ---
    relevant_terms_str = json.dumps(terms_dict, ensure_ascii=False, indent=2) if terms_dict else ""
    # --- FIN CORRECCIÓN ---

    current_logger.info(f"DEBUG: [pipeline.py] Subesquema final para LLM (pregunta: '{question[:100]}...'): {list(relevant_schema_dict.keys())}")
    # --- FIN: Lógica para determinar el sub-esquema relevante ---

    # --- CONTEXTO GENÉRICO Y ROBUSTO PARA EL LLM SOBRE CONSULTAS SQL MÉDICAS ---
    contexto_generico = (
        "ATENCIÓN: Eres un asistente experto en SQL médico. Tu tarea es generar una consulta SQL SQLite válida y ejecutable para responder a la pregunta del usuario, usando ÚNICAMENTE la información proporcionada en el esquema de la base de datos y el diccionario de términos relevantes (si se proporciona).\n"
        "REGLAS FUNDAMENTALES:\n"
        "1.  **USA SOLO EL ESQUEMA PROPORCIONADO**: SOLO puedes usar columnas y tablas que aparecen exactamente en el esquema. NO inventes ni infieras columnas o tablas. Si la información no está disponible en el esquema, indícalo claramente.\n"
        "2.  **SINTAXIS SQLITE**: Asegúrate de que la consulta SQL generada sea compatible con SQLite.\n"
        "3.  **MANEJO DE ERRORES**: Si no puedes generar una consulta válida, explica el motivo y pide aclaración al usuario. Si la consulta podría no devolver resultados o ser ambigua, advierte al usuario.\n"
        "4.  **FILTROS Y CONDICIONES**: Si la pregunta requiere filtrar por un paciente, diagnóstico, fecha, etc., utiliza los campos más relevantes del esquema. Para búsquedas de texto con múltiples `LIKE` y `OR`, AGRUPA los `OR` entre paréntesis: `WHERE ... AND (campo LIKE '%a%' OR campo LIKE '%b%')`.\n"
        "5.  **TABLAS DE DICCIONARIO/CATÁLOGO**: Si existen tablas que parecen ser diccionarios de códigos o catálogos (ej., `_DICTIONARIES`, `ALLE_ALLERGY_TYPES`), considera usarlas para mapear descripciones a códigos si es necesario, pero tu objetivo principal es generar la consulta que responda a la pregunta del usuario, no explorar los catálogos a menos que sea imprescindible para la consulta final.\n"
        "6.  **CÁLCULO DE EDAD**: Si necesitas calcular la edad a partir de dos fechas (fecha_final, fecha_inicial), usa la fórmula: `CAST((julianday(fecha_final) - julianday(fecha_inicial)) / 365.25 AS REAL)`. No pongas el alias dentro del `CAST`.\n"
        "7.  **RESPUESTA**: Devuelve la consulta SQL. Si es necesario, puedes añadir una breve explicación de tus supuestos o limitaciones.\n\n"
        "SOBRE EL USO DE HERRAMIENTAS DE BÚSQUEDA EXTERNA (como BioChatMedicalInquiry o similar para PubMed):\n"
        "1. INTENTA PRIMERO RESPONDER CON TU CONOCIMIENTO GENERAL: Para preguntas sobre conceptos médicos establecidos, listas comunes (ej. medicamentos de alto riesgo en geriatría según criterios Beers o STOPP/START), definiciones, o información que es ampliamente conocida en el ámbito médico, utiliza tu conocimiento interno ANTES de recurrir a herramientas de búsqueda externa.\n"
        "2. USA BÚSQUEDA EXTERNA PARA EVIDENCIA CIENTÍFICA RECIENTE O ESPECIALIZADA: La herramienta BioChatMedicalInquiry (o similar para PubMed) está diseñada para encontrar artículos de investigación, estudios clínicos, revisiones sistemáticas y evidencia científica específica que podría no estar en tu conocimiento base o que requiera la información más actualizada de la literatura.\n"
        "3. NO USES BÚSQUEDA EXTERNA PARA PREGUNTAS SIMPLES DE LA BASE DE DATOS: Si la pregunta se puede responder con una consulta SQL a la base de datos local, usa SQLMedicalChatbot.\n"
        "4. EJEMPLO DE CUÁNDO NO USAR BioChatMedicalInquiry INICIALMENTE: Si se pregunta por 'medicamentos de alto riesgo en geriatría', primero intenta generar una lista basada en criterios conocidos (Beers, STOPP/START). Solo si no puedes y la pregunta explícitamente pide 'investigación reciente sobre...' o 'estudios que comparan...' entonces considera BioChatMedicalInquiry.\n"
        "5. SI UNA HERRAMIENTA DE BÚSQUEDA EXTERNA NO DA RESULTADOS ÚTILES TRAS UN INTENTO RAZONABLE: Considera si la pregunta puede ser respondida de otra manera (conocimiento general, reformulación) o si necesitas pedir aclaración al usuario, en lugar de insistir con la misma herramienta y ligeras variaciones de la pregunta.\n"
    )
    prompt_text = f"""{contexto_generico}Pregunta del usuario: {question}\n\nDiccionario de términos relevantes (si aplica): {relevant_terms_str}\n\nEsquema de base de datos relevante: {relevant_schema_str}\n\nRespuesta esperada: (Consulta SQL SQLite válida o explicación clara de por qué no se puede generar)\n"""

    current_logger.info(f"DEBUG: [pipeline.py] PROMPT COMPLETO ENVIADO AL LLM PARA PREGUNTA '{question}':\n{prompt_text}")
    # AÑADE ESTA LÍNEA PARA LOGGING COMPLETO:
    current_logger.info(f"DEBUG: [pipeline.py] PROMPT COMPLETO ENVIADO AL LLM PARA PREGUNTA '{question}':\n{prompt_text}")

    structured_info = {}
    error_message = ""

    # Cambiar max_retries a 0 para que solo se haga un intento
    max_retries = 0
    for attempt in range(max_retries + 1):
        try:
            current_logger.info(f"Intento {attempt + 1} de {max_retries + 1} para llamar al LLM para extracción de estructura para la pregunta: '{question}'.")
            system_message_prompt = "Eres un asistente experto en SQL que ayuda a generar consultas SQLite basadas en la estructura de la base de datos y un diccionario de términos. Tu objetivo es devolver ÚNICAMENTE la consulta SQL o un mensaje de error claro si no puedes generarla."
            messages_for_llm = [
                SystemMessage(content=system_message_prompt),
                HumanMessage(content=prompt_text)
            ]
            response_from_llm = llm_client.invoke(messages_for_llm)
            structured_info_str = None
            if hasattr(response_from_llm, 'content'):
                structured_info_str = response_from_llm.content
            elif isinstance(response_from_llm, str):
                structured_info_str = response_from_llm
            if structured_info_str is None:
                current_logger.error(f"Respuesta del LLM no tiene atributo 'content' o es None, y no es un string. Tipo: {type(response_from_llm)}. Contenido: {str(response_from_llm)[:200]}")
                if attempt == max_retries:
                     return {}, "Formato de respuesta del LLM inesperado tras {max_retries + 1} intentos."
                time.sleep(2)
                continue
            lower_structured = structured_info_str.lower()
            missing_context_keywords = [
                "missing context", "insufficient context", "insufficient information", "información insuficiente", "contexto insuficiente", "no se encuentra suficiente información", "no se encuentra suficiente contexto", "no dispongo del contexto necesario", "no dispongo de la información necesaria", "no tengo suficiente contexto", "no tengo suficiente información"
            ]
            if any(kw in lower_structured for kw in missing_context_keywords):
                current_logger.error(f"LLM devolvió mensaje de contexto insuficiente (intento {attempt + 1}): {structured_info_str}")
                final_answer_explanation = structured_info_str.strip()
                if not final_answer_explanation.lower().startswith("final answer:"):
                    final_answer_explanation = f"Final Answer: {final_answer_explanation}"
                return {"non_sql_response": final_answer_explanation}, "El LLM indicó falta de contexto o información insuficiente (formateado y retornado)."
            current_logger.info(f"DEBUG: [pipeline.py] LLM (extracción) respuesta raw: {structured_info_str}")
            sql_query_match = extract_sql_from_markdown(structured_info_str)
            if sql_query_match:
                current_logger.info(f"SQL extraído de la respuesta del LLM: {sql_query_match}")
                return {"sql_query": sql_query_match}, "Consulta SQL generada por LLM."
            try:
                parsed_json = json.loads(structured_info_str)
                if isinstance(parsed_json, dict) and "error_message" in parsed_json:
                    current_logger.error(f"LLM devolvió un mensaje de error estructurado: {parsed_json.get('error_message')}")
                    error_message = parsed_json.get('error_message')
                    if attempt == max_retries: return {}, error_message
                    time.sleep(2)
                    continue
            except json.JSONDecodeError:
                if ("SELECT".lower() in structured_info_str.lower() or
                    "INSERT".lower() in structured_info_str.lower() or
                    "UPDATE".lower() in structured_info_str.lower() or
                    "DELETE".lower() in structured_info_str.lower()):
                    current_logger.info(f"Respuesta del LLM parece SQL directo (no Markdown, no JSON de error): {sql_query_match}")
                    return {"sql_query": structured_info_str.strip()}, "Consulta SQL (directa) generada por LLM."
                if attempt == max_retries:
                    idx_final_answer = structured_info_str.lower().find("final answer:")
                    if idx_final_answer != -1:
                        final_answer = structured_info_str[idx_final_answer:].strip()
                        return {"non_sql_response": final_answer}, "Respuesta no SQL del LLM (Final Answer extraído)."
                    if not structured_info_str.strip().lower().startswith("final answer:"):
                        current_logger.warning(f"Respuesta del LLM no es SQL ni error ni contiene 'Final Answer:'. Se forzará el formato Final Answer.")
                        final_answer = f"Final Answer: {structured_info_str.strip()}"
                        return {"non_sql_response": final_answer}, "Respuesta no SQL del LLM (formato Final Answer forzado)."
                    else:
                        return {"non_sql_response": structured_info_str.strip()}, "Respuesta no SQL del LLM."
                error_message = "Respuesta no SQL/JSON inesperada del LLM."
        except openai.APIConnectionError as e:
            error_message = f"Error de conexión con la API del LLM: {e}"
            current_logger.error(error_message, exc_info=True)
            if attempt == max_retries: return {}, error_message
            time.sleep(5)
        except openai.BadRequestError as e:
            error_message = f"Error en la solicitud al LLM (BadRequestError): {e}"
            current_logger.error(error_message, exc_info=True)
            return {}, error_message
        except openai.APIStatusError as e:
            error_message = f"Error de estado de la API del LLM: {e}"
            current_logger.error(error_message, exc_info=True)
            if attempt == max_retries: return {}, error_message
            time.sleep(5)
        except Exception as e:
            error_message = f"Excepción inesperada al llamar al LLM o procesar su respuesta: {e}"
            current_logger.error(error_message, exc_info=True)
            if attempt == max_retries:
                return {}, error_message
            time.sleep(2)
    return {}, f"LLM no pudo extraer información después de {max_retries + 1} intentos. Último error: {error_message}"


# NUEVA función para generar respuesta NL con LLM
def generate_natural_language_response_via_llm(llm, original_question, sql_query, results, column_names, context, error_extract_msg, logger_param=None):
    current_logger_nl = logger_param if logger_param else logger
    current_logger_nl.info(f"DEBUG: [pipeline.py] Entrando en generate_natural_language_response_via_llm con LLM: {type(llm)}")

    # --- INICIO LLM SIMULADO para generate_natural_language_response_via_llm ---
    current_logger_nl.warning("USANDO LLM SIMULADO para generate_natural_language_response_via_llm")
    generated_text = ""
    if results is not None: # results puede ser una lista vacía
        if results:
            generated_text = f"He encontrado {len(results)} resultado(s) para tu consulta sobre '{original_question}'. "
            if column_names:
                generated_text += f"Las columnas son: {', '.join(column_names)}. "
                # Mostrar los primeros 2 resultados de forma legible
                ejemplos = "\\n".join([
                    ", ".join(f"{col}: {val}" for col, val in zip(column_names, row))
                    for row in results[:2]
                ])
                generated_text += f"Aquí tienes algunos ejemplos:\\n{ejemplos}"
                if len(results) > 2:
                    generated_text += f"\\n(y {len(results)-2} más)."
            else:
                # Si no hay nombres de columnas, mostrar los resultados tal cual
                generated_text += f"Aquí tienes algunos ejemplos: {str(results[:2])}"
                if len(results) > 2:
                    generated_text += f" (y {len(results)-2} más)."
        else:
         # No hay filas
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

def agrupar_or_like_en_parentesis(sql: str) -> str:
    """
    Agrupa automáticamente los OR de varios LIKE sobre el mismo campo en paréntesis,
    para asegurar que los AND se apliquen correctamente fuera del grupo.
    """
    # Busca el WHERE y separa condiciones
    pattern = re.compile(
        r"(WHERE\s+)(.*?)(\s+ORDER\s+BY|\s+GROUP\s+BY|\s+LIMIT|\s*$)", re.IGNORECASE | re.DOTALL
    )
    match = pattern.search(sql)
    if not match:
        return sql
    where_start, conditions, where_end = match.groups()
    # Busca secuencias de OR con LIKE sobre el mismo campo
    like_or_pattern = re.compile(
        r"((?:\w+\.\w+|\w+)\s+LIKE\s+'[^']+'\s*(?:OR\s+(?:\w+\.\w+|\w+)\s+LIKE\s+'[^']+'\s*)+)",
        re.IGNORECASE
    )
    def agrupar(match_like):
        bloque = match_like.group(1)
        return f"({bloque.strip()})"
    new_conditions = like_or_pattern.sub(agrupar, conditions)
    nueva_where = f"{where_start}{new_conditions}{where_end}"
    return sql.replace(f"{where_start}{conditions}{where_end}", nueva_where)


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


# Inicializar instancia de DBConnector
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DB_PATH = os.path.join(_SCRIPT_DIR, "db", "database_new.sqlite3.db")
# _DEFAULT_SCHEMA_PATH = os.path.join(_SCRIPT_DIR, "data", "schema_simple.json") # ELIMINADO

db_connector = DBConnector(
    db_path=_DEFAULT_DB_PATH
    # schema_path=_DEFAULT_SCHEMA_PATH # ELIMINADO
)
logger.info(f"Instancia de DBConnector creada con db_path: {_DEFAULT_DB_PATH}") # MODIFICADO

# Definir el número máximo de reintentos para LLM
max_retries_llm = MAX_RETRIES_LLM_EXTRACT

# --- INICIO: Definir sql_to_execute y variables críticas por defecto para evitar NameError ---
sql_to_execute = None
sql_fallback = None
params_to_execute = []
results = None
columns = None
error_execution_msg = None
whitelist_error_msg = None
context_sql_gen = None
# --- FIN: Definir sql_to_execute y variables críticas por defecto para evitar NameError ---

def chatbot_pipeline_entrypoint( # MODIFICADO: Renombrar la función principal
    user_question: str, 
    db_connector: DBConnector,
    llm_param: Any,
    logger_param: Optional[logging.Logger] = None,
    max_retries_llm: int = MAX_RETRIES_LLM_EXTRACT
) -> Dict[str, Any]:
    """
    Punto de entrada para el pipeline del chatbot.
    Procesa la pregunta del usuario, genera y valida SQL, y devuelve resultados en lenguaje natural.

    Args:
        user_question (str): La pregunta formulada por el usuario.
        db_connector (DBConnector): Conector a la base de datos.
        llm_param (Any): Parámetros para el modelo de lenguaje (LLM).
        logger_param (Optional[logging.Logger]): Logger opcional para registrar información.
        max_retries_llm (int): Número máximo de reintentos para la extracción de información con LLM.

    Returns:
        Dict[str, Any]: Un diccionario con la respuesta en lenguaje natural, posibles resultados, y mensajes de error.
    """
    current_pipeline_logger = logger_param if logger_param else logger
    current_pipeline_logger.info(f"DEBUG: [pipeline.py] Iniciando chatbot_pipeline_entrypoint con pregunta: '{user_question}'")

    # --- INICIO: Carga de recursos ---
    try:
        # Obtener el esquema dinámicamente
        db_schema_dict = db_connector.get_db_structure_dict()
        if not db_schema_dict:
            current_pipeline_logger.error("ERROR: [pipeline.py] db_connector.get_db_structure_dict() devolvió un esquema vacío o None.")
            raise ValueError("El esquema de la base de datos obtenido dinámicamente está vacío o es None.")
        current_pipeline_logger.info(f"DEBUG: [pipeline.py] Esquema de base de datos obtenido dinámicamente desde DBConnector.")
    except Exception as e_schema:
        current_pipeline_logger.error(f"ERROR: [pipeline.py] No se pudo obtener el esquema de la base de datos desde DBConnector: {e_schema}", exc_info=True)
        return {"error": f"No se pudo obtener el esquema de la base de datos: {e_schema}", "response_message": "Error crítico: no se pudo cargar/obtener el esquema de la base de datos.", "data_results": [], "sql_query_generated": None, "params_used": [], "data_results_empty": True, "error_message": f"No se pudo obtener el esquema de la base de datos: {e_schema}"}

    # Transformar db_schema_dict al formato esperado por WhitelistValidator
    transformed_schema = {"tables": []}
    if isinstance(db_schema_dict, dict):
        for table_name, table_data in db_schema_dict.items():
            if isinstance(table_data, dict) and "columns" in table_data and isinstance(table_data["columns"], list):
                columns = [{"name": col_name} for col_name in table_data.get("columns", [])]
                transformed_schema["tables"].append({"name": table_name, "columns": columns})
            else:
                current_pipeline_logger.warning(f"Formato inesperado para la tabla {table_name} en schema_simple.json. Se omitirá.")
    else:
        current_pipeline_logger.error("ERROR: [pipeline.py] db_schema_dict no es un diccionario como se esperaba después de cargar el JSON.")
        pass

    whitelist_validator_instance = WhitelistValidator(
        db_schema_dict=transformed_schema,
        case_sensitive=True
    )
    current_pipeline_logger.info("DEBUG: [pipeline.py] WhitelistValidator instanciado.")
    # --- FIN: Carga de recursos ---

    # --- INICIO: Variables para el flujo de SQL y resultados ---
    sql_to_execute: Optional[str] = None
    params_to_execute: Optional[list] = [] 
    error_sql_gen: Optional[str] = None
    context_sql_gen: str = "" 
    results: list = []
    data_results_empty: bool = True
    respuesta_final_chatbot: str = ""
    error_execution_msg: Optional[str] = None
    whitelist_error_msg: Optional[str] = None
    is_whitelisted: bool = True
    # --- FIN: Variables para el flujo de SQL y resultados ---

    preprocessed_question = user_question # Aquí puedes aplicar normalización dinámica si es necesario
    current_pipeline_logger.info(f"DEBUG: [pipeline.py] Pregunta preprocesada: '{preprocessed_question}'")

    def is_sql_query(text):
        text = text.strip()
        # Caso 1: SQL directo
        if text.upper().startswith(("SELECT", "WITH", "INSERT", "UPDATE", "DELETE")):
            return True
        # Caso 2: JSON con clave 'query' que contiene SQL
        if text.startswith("{") and text.endswith("}"):
            try:
           
                obj = json.loads(text)
                if isinstance(obj, dict) and "query" in obj:
                    query_val = obj["query"]
                    if isinstance(query_val, str) and query_val.strip().upper().startswith(("SELECT", "WITH", "INSERT", "UPDATE", "DELETE")):
                        return True
            except Exception:
                pass
        return False

    structured_info: Dict[str, Any] = {}
    error_extract_msg: str = ""
    try:
        # --- INICIO: Inicializar relationship_graph ---
        try:
            rel_path = os.path.join(os.path.dirname(__file__), '../../table_relationships.json')
            if os.path.exists(rel_path):
                with open(rel_path, 'r', encoding='utf-8') as f:
                    rel_map = json.load(f)
            else:
                rel_map = None
        except Exception as e:
            rel_map = None
            current_pipeline_logger.warning(f"No se pudo cargar table_relationships.json en entrypoint: {e}")
        try:
            relationship_graph = build_relationship_graph(
                db_connector, 
                table_relationships=rel_map, 
                db_structure=db_schema_dict
            )
            current_pipeline_logger.info("DEBUG: [pipeline.py] Grafo de relaciones construido en entrypoint.")
        except Exception as e:
            relationship_graph = None
            current_pipeline_logger.warning(f"No se pudo construir grafo de relaciones en entrypoint: {e}")
        # --- FIN: Inicializar relationship_graph ---

        # Llamada dinámica al LLM para extracción estructurada
        structured_info, error_extract_msg = extract_info_from_question_llm(
            preprocessed_question, 
            db_schema_dict, 
            None, 
            llm_client=llm_param, 
            current_logger=current_pipeline_logger, 
            relationship_graph=relationship_graph,  
            max_retries=max_retries_llm
        )
        current_pipeline_logger.info(f"DEBUG: Información estructurada extraída: {structured_info}, error: '{error_extract_msg}'")
    except Exception as e_extract:
        current_pipeline_logger.error(f"Excepción durante extract_info_from_question_llm: {e_extract}", exc_info=True)
        error_extract_msg = f"Excepción crítica al extraer info: {e_extract}"
        flow_decision = "extraction_failed_return"

    # Expansión dinámica de sinónimos/diagnósticos (ejemplo con flexible_search_config)
    from flexible_search_config import extract_diagnosis_variants_from_hint, get_llm_generated_synonyms, LLM_CALLER_FUNCTION
    if not is_sql_query(user_question):
        diagnosis_variants = extract_diagnosis_variants_from_hint(user_question)
        if not diagnosis_variants and LLM_CALLER_FUNCTION:
            synonyms_dict = get_llm_generated_synonyms([user_question], llm_caller=LLM_CALLER_FUNCTION)
            if synonyms_dict and user_question in synonyms_dict:
                diagnosis_variants = synonyms_dict[user_question]
                logger.info(f"[pipeline] Sinónimos generados dinámicamente para '{user_question}': {diagnosis_variants}")
        if diagnosis_variants:
            logger.info(f"[pipeline] Detectadas variantes de diagnóstico: {diagnosis_variants}")
            structured_info['diagnosis_variants'] = diagnosis_variants
            preprocessed_question += f" (Buscar diagnósticos: {'; '.join(diagnosis_variants)})"

    # --- INICIO: Decidir si usar SQL extraído, generar SQL, o manejar error de extracción ---
    flow_decision = "generate_new_sql"

    # Permitir respuestas mixtas: si structured_info contiene sql_query y explanation, procesar ambos
    explanation_from_llm = None
    if structured_info:
        # Soportar claves 'explanation', 'explicacion', o 'explanation_text' (flexibilidad)
        for k in ["explanation", "explicacion", "explanation_text"]:
            if k in structured_info and isinstance(structured_info[k], str):
                explanation_from_llm = structured_info[k].strip()
                break

    # Refuerzo: si structured_info contiene una SQL válida, ejecutarla directamente
    if structured_info and "sql_query" in structured_info:
        extracted_sql_candidate = structured_info.get("sql_query")
        if extracted_sql_candidate and isinstance(extracted_sql_candidate, str):
            # Extraer de markdown si es necesario
            if "```" in extracted_sql_candidate:
                import re
                match = re.search(r"```sql(.*?)```", extracted_sql_candidate, re.DOTALL | re.IGNORECASE)
                if match:
                    extracted_sql_candidate = match.group(1).strip()
            if extracted_sql_candidate and extracted_sql_candidate.strip().upper().startswith(("SELECT", "WITH")):
                sql_to_execute = extracted_sql_candidate.strip()
                flow_decision = "use_extracted_sql"

    # Refuerzo: si hay SQL válida, ejecutarla aunque haya error_extract_msg
    if error_extract_msg and structured_info and "sql_query" in structured_info:
        extracted_sql_candidate = structured_info.get("sql_query")
        if extracted_sql_candidate and isinstance(extracted_sql_candidate, str):
            if "```" in extracted_sql_candidate:
                import re
                match = re.search(r"```sql(.*?)```", extracted_sql_candidate, re.DOTALL | re.IGNORECASE)
                if match:
                    extracted_sql_candidate = match.group(1).strip()
            if extracted_sql_candidate and extracted_sql_candidate.strip().upper().startswith(("SELECT", "WITH")):
                sql_to_execute = extracted_sql_candidate.strip()
                flow_decision = "use_extracted_sql"

    if error_extract_msg:
        logger.warning(f"DEBUG: [pipeline.py] Mensaje de la extracción presente: '{error_extract_msg}'")
        is_direct_sql_note = "Consulta SQL (directa) generada por LLM" in error_extract_msg or \
                             "SQL extraído directamente de la pregunta" in error_extract_msg or \
                             "La consulta SQL parece ser directa" in error_extract_msg
        if structured_info and "non_sql_response" in structured_info:
            respuesta_final_chatbot = structured_info["non_sql_response"].strip()
            # Si además hay explicación, la concatenamos
            if explanation_from_llm:
                respuesta_final_chatbot = f"{respuesta_final_chatbot}\n\nExplicación: {explanation_from_llm}"
            logger.error(f"DEBUG: [pipeline.py] Respuesta no SQL del LLM detectada, retornando Final Answer: {respuesta_final_chatbot}")
            return {
                "response_message": respuesta_final_chatbot,
                "data_results": [],
                "sql_query_generated": None,
                "params_used": [],
                "data_results_empty": True,
                "error_message": error_extract_msg
            }
        elif structured_info and "sql_query" in structured_info and is_direct_sql_note:
            # Ejecutar SQL si aplica
            if sql_to_execute:
                sql_to_execute = agrupar_or_like_en_parentesis(sql_to_execute)
                # Ejecutar SQL y devolver explicación si existe
                try:
                    results, columns = db_connector.execute_query(sql_to_execute)
                    data_results_empty = not results
                except Exception as e:
                    error_execution_msg = str(e)
                    results = []
                    columns = []
                    data_results_empty = True
                response_message = ""
                if explanation_from_llm:
                    response_message += f"Explicación: {explanation_from_llm}\n"
                response_message += f"Consulta SQL ejecutada."
                return {
                    "response_message": response_message,
                    "data_results": results,
                    "sql_query_generated": sql_to_execute,
                    "params_used": [],
                    "data_results_empty": data_results_empty,
                    "error_message": error_execution_msg
                }
        elif not (structured_info and "sql_query" in structured_info):
            # Retornar explicación si existe
            if explanation_from_llm:
                return {
                    "response_message": f"Explicación: {explanation_from_llm}",
                    "data_results": [],
                    "sql_query_generated": None,
                    "params_used": [],
                    "data_results_empty": True,
                    "error_message": error_extract_msg
                }
            # Fallback genérico
            return {
                "response_message": f"No se pudo procesar la pregunta. Detalle: {error_extract_msg}",
                "data_results": [],
                "sql_query_generated": None,
                "params_used": [],
                "data_results_empty": True,
                "error_message": error_extract_msg
            }
    # --- FIN: Decidir si usar SQL extraído, generar SQL, o manejar error de extracción ---

    # --- INICIO: Ejecución de SQL y obtención de resultados ---
    if flow_decision == "use_extracted_sql":
        logger.info(f"DEBUG: [pipeline.py] Usando SQL extraído directamente: {sql_to_execute}")
        try:
            rows, columns = db_connector.execute_query(sql_to_execute)
            results = rows
            data_results_empty = len(rows) == 0
            logger.info(f"DEBUG: [pipeline.py] Resultados obtenidos: (filas: {len(rows)}, columnas: {columns})")
        except Exception as e_execute:
            error_execution_msg = f"Error al ejecutar la consulta SQL extraída: {e_execute}"
            logger.error(f"DEBUG: [pipeline.py] {error_execution_msg}", exc_info=True)
            # --- INICIO FALLBACK AUTOMÁTICO ---
            # Reenviar el error y la consulta al LLM para que la corrija
            correction_prompt = (
                f"La consulta SQL generada produjo el siguiente error: {e_execute}. "
                f"Consulta original: '''{sql_to_execute}'''. "
                "Corrige la consulta SQL para que sea válida en SQLite, usando solo tablas y columnas existentes. "
                "Si es necesario, consulta el esquema antes de reintentar."
            )
            try:
                # Llama al LLM para corregir la consulta (puedes usar extract_info_from_question_llm o tu función LLM preferida)
                structured_info_fallback, error_fallback = extract_info_from_question_llm(
                    correction_prompt,
                    db_schema_dict,
                    None,
                    llm_client=llm_param,
                    current_logger=current_pipeline_logger,
                    relationship_graph=relationship_graph,
                    max_retries=max_retries_llm
                )
                # Intenta extraer la nueva SQL del resultado del LLM
                sql_fallback = None
                if structured_info_fallback and "sql_query" in structured_info_fallback:
                    sql_fallback = structured_info_fallback["sql_query"]
                if sql_fallback:
                    logger.info(f"DEBUG: [pipeline.py] Intentando ejecutar SQL corregida por fallback: {sql_fallback}")
                    try:
                        rows, columns = db_connector.execute_query(sql_fallback)
                        results = rows
                        data_results_empty = len(rows) == 0
                        sql_to_execute = sql_fallback  # Actualiza la consulta usada
                        error_execution_msg = None
                        logger.info(f"DEBUG: [pipeline.py] Resultados obtenidos tras fallback: (filas: {len(rows)}, columnas: {columns})")
                    except Exception as e_fallback_execute:
                        error_execution_msg = f"Error también en el fallback/corrección: {e_fallback_execute}"
                        logger.error(f"DEBUG: [pipeline.py] {error_execution_msg}", exc_info=True)
                        flow_decision = "extraction_failed_return"
                else:
                    error_execution_msg = f"No se pudo obtener una consulta SQL corregida del LLM. Error: {error_fallback}"
                    logger.error(f"DEBUG: [pipeline.py] {error_execution_msg}")
                    flow_decision = "extraction_failed_return"
            except Exception as e_fallback_llm:
                error_execution_msg = f"Error al intentar corregir la consulta SQL con el LLM: {e_fallback_llm}"
                logger.error(f"DEBUG: [pipeline.py] {error_execution_msg}", exc_info=True)
                flow_decision = "extraction_failed_return"
            # --- FIN FALLBACK AUTOMÁTICO ---
    elif flow_decision == "generate_new_sql":
        logger.info(f"DEBUG: [pipeline.py] Generando nueva SQL a partir de información estructurada: {json.dumps(structured_info, indent=2, ensure_ascii=False)}")
        sql_to_execute, params_to_execute, error_sql_gen, context_sql_gen = generate_sql_and_validate_whitelist(
            structured_info=structured_info,
            sql_generator_instance=sql_generator_instance,
            db_connector_for_sql_gen=db_connector,
            current_allowed_columns_map=whitelist_validator_instance.allowed_columns_map,
            terms_dict_str_for_validation="{}", # Diccionario vacío, lógica dinámica
            question_for_logging=user_question,
            logger_param=logger
        )
        if error_sql_gen:
            error_execution_msg = f"Error al generar la consulta SQL: {error_sql_gen}"
            logger.error(f"DEBUG: [pipeline.py] {error_execution_msg}")
            flow_decision = "extraction_failed_return"
        else:
            logger.info(f"DEBUG: [pipeline.py] SQL generado exitosamente: {sql_to_execute}, Params: {params_to_execute}")
            try:
                is_whitelisted = whitelist_validator_instance.validate_sql(
                    sql_query=sql_to_execute,
                    params=params_to_execute
                )
                if not is_whitelisted:
                    # Intentar detectar columna inventada y sugerir la más parecida
                    columnas_validas = []
                    for t in db_schema_dict.values():
                        columnas_validas.extend(t.get('columns', []))
                    # Buscar columnas inventadas en el SQL generado
                    columnas_en_sql = re.findall(r'\b([A-Z_]+)\b', sql_to_execute)
                    columna_sugerida = None
                    for col in columnas_en_sql:
                        if col not in columnas_validas:
                            sugerida = sugerir_columna_parecida(col, columnas_validas)
                            if sugerida:
                                columna_sugerida = sugerida
                                columna_inventada = col
                                break
                    if columna_sugerida:
                        logger.warning(f"Columna inventada '{columna_inventada}' detectada. Sugerida: '{columna_sugerida}'. Reintentando consulta con la columna sugerida.")
                        # Reemplazar la columna inventada por la sugerida en la SQL y reintentar
                        sql_to_execute_corregida = sql_to_execute.replace(columna_inventada, columna_sugerida)
                        try:
                            is_whitelisted_2 = whitelist_validator_instance.validate_sql(sql_query=sql_to_execute_corregida, params=params_to_execute)
                            if is_whitelisted_2:
                                sql_to_execute = sql_to_execute_corregida
                                whitelist_error_msg = None
                                logger.info(f"Consulta corregida con columna sugerida: {sql_to_execute}")
                                # Continuar flujo normal
                            else:
                                whitelist_error_msg = f"El SQL corregido con la columna sugerida tampoco está permitido por la whitelist. Columna inventada: {columna_inventada}, sugerida: {columna_sugerida}."
                                flow_decision = "extraction_failed_return"
                        except Exception as e:
                            whitelist_error_msg = f"Error al validar SQL corregido: {e}"
                            flow_decision = "extraction_failed_return"
                    else:
                        whitelist_error_msg = "El SQL generado no está permitido por la whitelist y no se encontró columna parecida para sugerir."
                        logger.warning(f"DEBUG: [pipeline.py] {whitelist_error_msg} SQL: {sql_to_execute}, Params: {params_to_execute}")
                        flow_decision = "extraction_failed_return"
                else:
                    logger.info(f"DEBUG: [pipeline.py] SQL generado está permitido por la whitelist.")
            except Exception as e_whitelist:
                logger.error(f"DEBUG: [pipeline.py] Excepción al validar whitelist: {e_whitelist}", exc_info=True)
                whitelist_error_msg = f"Error al validar whitelist: {e_whitelist}"
                flow_decision = "extraction_failed_return"
    # --- FIN: Ejecución de SQL y obtención de resultados ---

    # --- INICIO: Manejo de resultados y generación de respuesta final ---
    if flow_decision == "extraction_failed_return":
        logger.warning(f"DEBUG: [pipeline.py] Flujo de extracción falló, retornando mensaje de error: {error_extract_msg}")
        respuesta_final_chatbot = f"No pude procesar tu solicitud. Detalle: {error_extract_msg}"
    else:
        # Aquí se asume que si no hubo error, se generó una respuesta válida
        respuesta_final_chatbot = f"He procesado tu pregunta. "
        if results is not None:
            respuesta_final_chatbot += f"Se encontraron {len(results)} resultado(s). "
        if sql_to_execute:
            respuesta_final_chatbot += f"Consulta SQL utilizada: {sql_to_execute}. "
        if params_to_execute:
            respuesta_final_chatbot += f"Parámetros: {params_to_execute}. "
        if context_sql_gen:
            respuesta_final_chatbot += f"Contexto adicional: {context_sql_gen}. "
        respuesta_final_chatbot += "Si necesitas más detalles, por favor aclara tu pregunta."
    # --- FIN: Manejo de resultados y generación de respuesta final ---

    current_pipeline_logger.info(f"DEBUG: [pipeline.py] Respuesta final del chatbot: {respuesta_final_chatbot}")
    return {
        "response_message": respuesta_final_chatbot,
        "data_results": results,
        "sql_query_generated": sql_to_execute,
        "params_used": params_to_execute,
        "data_results_empty": data_results_empty,
        "error_message": error_execution_msg or whitelist_error_msg or error_extract_msg
    }

# --- INICIO: Instanciar SQLGenerator para la sesión ---
sql_generator_instance = SQLGenerator(allowed_tables=[], allowed_columns={})
# --- FIN: Instanciar SQLGenerator ---

import difflib

def sugerir_columna_parecida(columna_inventada: str, columnas_validas: list, umbral: float = 0.7) -> str:
    """
    Dada una columna inventada y una lista de columnas válidas, sugiere la más parecida usando fuzzy matching.
    Si no hay ninguna suficientemente similar, devuelve None.
    Args:
        columna_inventada (str): Nombre de la columna inventada.
        columnas_validas (list): Lista de nombres de columnas válidas.
        umbral (float): Umbral de similitud (0-1) para aceptar la sugerencia.
    Returns:
        str or None: Nombre de la columna sugerida o None si no hay ninguna suficientemente parecida.
    """
    if not columna_inventada or not columnas_validas:
        return None
    matches = difflib.get_close_matches(columna_inventada, columnas_validas, n=1, cutoff=umbral)
    if matches:
        return matches[0]
    return None

import re

# --- INICIO: Robustez para llaves en SQL generada por LLM ---
def limpiar_sql_llaves(sql: str):
    # Reemplaza {{var}} o {var} por ? para SQLite
    sql = re.sub(r"\{\{\s*([\w_]+)\s*\}\}", "?", sql)
    sql = re.sub(r"\{\s*([\w_]+)\s*\}", "?", sql)
    return sql

# Antes de ejecutar cualquier SQL generada por LLM:
if sql_to_execute:
    sql_to_execute = limpiar_sql_llaves(sql_to_execute)
if 'sql_fallback' in locals() and sql_fallback:
    sql_fallback = limpiar_sql_llaves(sql_fallback)
# --- FIN: Robustez para llaves en SQL generada por LLM ---