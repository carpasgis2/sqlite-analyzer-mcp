import json
import logging
import re
import time
from typing import Any, Dict, Optional

import openai
from langchain_core.messages import HumanMessage, SystemMessage

from .rag import DatabaseSchemaRAG # Asumido necesario por componentes llamados o futuros
from .sql_generator import SQLGenerator
from .sql_utils import extract_sql_from_markdown
from .whitelist_validator import WhitelistValidator # MODIFICADO: validate_structured_info_whitelist eliminado

from .db_connector import DBConnector # Para type hinting

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

def extract_info_from_question_llm(
    question: str, 
    db_structure_dict: dict, 
    terms_dict: dict, 
    llm_client, 
    current_logger, 
    max_retries=1 # Añadido max_retries como parámetro
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
    
    rag_instance = DatabaseSchemaRAG(db_structure_dict=db_structure_dict, 
                                     terms_dict=terms_dict,
                                     logger_param=current_logger)
    
    relevant_schema_str, relevant_terms_str = rag_instance.get_relevant_context(
        question,
        top_n_tables=5, # AUMENTADO de 3 a 5
        top_n_columns_per_table=10, # AUMENTADO de 7 a 10
        top_n_terms=10 # AUMENTADO de 8 a 10
    )

    current_logger.info(f"RAG Relevant Schema for LLM: {relevant_schema_str}") # <--- AÑADIDO: Log detallado del esquema RAG
    current_logger.info(f"RAG Relevant Terms for LLM: {relevant_terms_str}") # <--- AÑADIDO: Log detallado de los términos RAG
    current_logger.debug(f"RAG Schema Context: {relevant_schema_str}")
    current_logger.debug(f"RAG Terms Context: {relevant_terms_str}")

    prompt_text = f"""Dada la pregunta del usuario, el diccionario de términos relevantes y la estructura de la base de datos relevante, 
genera una consulta SQL SQLite válida y ejecutable para responder a la pregunta.
Consideraciones importantes para la consulta SQL:
- Utiliza ÚNICAMENTE las tablas y columnas proporcionadas en la ESTRUCTURA DE BASE DE DATOS RELEVANTE. NO inventes tablas o columnas.
- Presta atención a los TIPOS DE DATOS de las columnas al formular las condiciones.
- Si la pregunta implica nombres de pacientes, la tabla PATIENTS contiene PATI_FULL_NAME. NO uses PATI_NAME o PATI_SURNAME a menos que estén explícitamente en la ESTRUCTURA DE BASE DE DATOS RELEVANTE.
- Si necesitas unir tablas, asegúrate de que las condiciones de JOIN sean correctas y utilicen columnas existentes en la ESTRUCTURA DE BASE DE DATOS RELEVANTE.
- El DICCIONARIO DE TÉRMINOS RELEVANTES es tu principal fuente para interpretar qué tablas o columnas son pertinentes para la pregunta.
- Si, DESPUÉS de analizar la ESTRUCTURA DE BASE DE DATOS RELEVANTE y el DICCIONARIO DE TÉRMINOS RELEVANTES proporcionados, consideras que la información es insuficiente para generar una consulta SQL que responda DIRECTAMENTE a la pregunta, explica DETALLADAMENTE qué información específica falta y por qué el contexto proporcionado no es suficiente. NO inventes una consulta si la información no está.
- Devuelve SÓLO la consulta SQL, sin explicaciones adicionales, ni texto introductorio o final, ni markdown, a menos que estés explicando la información faltante según el punto anterior.
- Si la pregunta no puede ser respondida con una consulta SQL (por ejemplo, si es un saludo o una pregunta general sobre tus capacidades), responde amablemente sin generar SQL.

Pregunta del usuario: {question}

Diccionario de términos relevantes:
{relevant_terms_str}

Estructura de base de datos relevante:
{relevant_schema_str}

Respuesta (Consulta SQL SQLite o explicación de información faltante):
"""
    
    current_logger.debug(f"DEBUG: [pipeline.py] Prompt para LLM (extracción info) para pregunta '{question}' será: {prompt_text[:1000]}...") # Loguear solo una parte

    structured_info = {}
    error_message = ""

    for attempt in range(max_retries + 1):
        try:
            current_logger.info(f"Intento {attempt + 1} de {max_retries + 1} para llamar al LLM para extracción de estructura para la pregunta: '{question}'.")
            
            system_message_prompt = "Eres un asistente experto en SQL que ayuda a generar consultas SQLite basadas en la estructura de la base de datos y un diccionario de términos. Tu objetivo es devolver ÚNICAMENTE la consulta SQL o un mensaje de error claro si no puedes generarla."
            
            messages_for_llm = [
                SystemMessage(content=system_message_prompt),
                HumanMessage(content=prompt_text)
            ]

            #(o compatible con Langchain)
            # El modelo (ej. LLM_MODEL_NAME) se configura al instanciar llm_client.
            response_from_llm = llm_client.invoke(messages_for_llm)
            
            structured_info_str = None
            if hasattr(response_from_llm, 'content'):
                structured_info_str = response_from_llm.content
            elif isinstance(response_from_llm, str): # Fallback por si la respuesta es un string directo
                structured_info_str = response_from_llm
            
            if structured_info_str is None:
                current_logger.error(f"Respuesta del LLM no tiene atributo 'content' o es None, y no es un string. Tipo: {type(response_from_llm)}. Contenido: {str(response_from_llm)[:200]}")
                if attempt == max_retries:
                     return {}, f"Formato de respuesta del LLM inesperado tras {max_retries + 1} intentos."
                time.sleep(2) 
                continue

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
                # No es JSON, podría ser SQL directo o una respuesta textual no SQL
                # Si la cadena parece una consulta SQL (heurística simple)
                if ("SELECT".lower() in structured_info_str.lower() or
                    "INSERT".lower() in structured_info_str.lower() or
                    "UPDATE".lower() in structured_info_str.lower() or
                    "DELETE".lower() in structured_info_str.lower()):
                    current_logger.info(f"Respuesta del LLM parece SQL directo (no Markdown, no JSON de error): {structured_info_str}")
                    return {"sql_query": structured_info_str.strip()}, "Consulta SQL (directa) generada por LLM."
                
                current_logger.warning(f"Respuesta del LLM no es SQL en Markdown, ni JSON de error, ni parece SQL directo. Respuesta: {structured_info_str}")
                # Si no es SQL ni un error JSON conocido, y es el último intento, devolver como mensaje.
                if attempt == max_retries:
                    # Considerar si esto debería ser un error o una respuesta no SQL.
                    # Por ahora, si el prompt pide SQL o mensaje amigable, esto podría ser el mensaje.
                    return {"non_sql_response": structured_info_str}, "Respuesta no SQL del LLM."
                # Si no es el último intento, podría ser un fallo transitorio, reintentar.
                error_message = "Respuesta no SQL/JSON inesperada del LLM." # Para el log del reintento
            
            # Si llegamos aquí, es porque se parseó un JSON que no era de error, o algo inesperado.
            # O la respuesta no era SQL directo y no era el último intento.
            # Si no es un error claro para reintentar, y no es SQL, y no es el último intento,
            # es mejor registrar y reintentar.
            current_logger.warning(f"Intento {attempt + 1} no produjo SQL claro. Respuesta: {structured_info_str}. Reintentando si quedan intentos.")
            if attempt == max_retries: # Si es el último intento y no se resolvió
                return {}, f"No se pudo extraer SQL o una respuesta clara del LLM tras {max_retries + 1} intentos. Última respuesta: {structured_info_str}"
            time.sleep(2) # Esperar antes de reintentar

        except openai.APIConnectionError as e: # Suponiendo que Langchain podría lanzar errores de openai si usa ese backend
            error_message = f"Error de conexión con la API del LLM: {e}"
            current_logger.error(error_message, exc_info=True)
            if attempt == max_retries: return {}, error_message
            time.sleep(5) # Espera más larga para problemas de conexión
        except openai.BadRequestError as e: # Errores como "context length exceeded"
            error_message = f"Error en la solicitud al LLM (BadRequestError): {e}"
            current_logger.error(error_message, exc_info=True)
            # Este tipo de error usualmente no se resuelve con reintentos simples si el prompt es el mismo.
            return {}, error_message 
        except openai.APIStatusError as e: # Otros errores de API
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


def chatbot_pipeline_entrypoint( # MODIFICADO: Renombrar la función principal
    user_question: str, 
    db_connector: DBConnector, 
    llm_param: Any, # Puede ser una instancia de ChatOpenAI u otro cliente LLM compatible
    terms_dict_path: str, 
    schema_path: str, # Añadido schema_path
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
        terms_dict_path (str): Ruta al diccionario de términos.
        schema_path (str): Ruta al esquema de la base de datos.
        logger_param (Optional[logging.Logger]): Logger opcional para registrar información.
        max_retries_llm (int): Número máximo de reintentos para la extracción de información con LLM.

    Returns:
        Dict[str, Any]: Un diccionario con la respuesta en lenguaje natural, posibles resultados, y mensajes de error.
    """
    logger.info(f"DEBUG: [pipeline.py] Iniciando chatbot_pipeline_entrypoint con pregunta: '{user_question}'")

    # --- INICIO: Carga de recursos (sin cambios significativos) ---
    try:
        terms_dict = {}
        with open(terms_dict_path, 'r', encoding='utf-8') as terms_file:
            terms_dict = json.load(terms_file)
        logger.info(f"DEBUG: [pipeline.py] Diccionario de términos cargado correctamente desde {terms_dict_path}.")
    except Exception as e_terms:
        logger.error(f"ERROR: [pipeline.py] No se pudo cargar el diccionario de términos desde {terms_dict_path}: {e_terms}")
        return {"error": f"No se pudo cargar el diccionario de términos: {e_terms}", "response_message": "Error crítico: no se pudo cargar el diccionario de términos.", "data_results": [], "sql_query_generated": None, "params_used": [], "data_results_empty": True, "error_message": f"No se pudo cargar el diccionario de términos: {e_terms}"}

    try:
        db_schema_dict = {}
        with open(schema_path, 'r', encoding='utf-8') as schema_file:
            db_schema_dict = json.load(schema_file)
        logger.info(f"DEBUG: [pipeline.py] Esquema de base de datos cargado correctamente desde {schema_path}.")
    except Exception as e_schema:
        logger.error(f"ERROR: [pipeline.py] No se pudo cargar el esquema de la base de datos desde {schema_path}: {e_schema}")
        return {"error": f"No se pudo cargar el esquema de la base de datos: {e_schema}", "response_message": "Error crítico: no se pudo cargar el esquema de la base de datos.", "data_results": [], "sql_query_generated": None, "params_used": [], "data_results_empty": True, "error_message": f"No se pudo cargar el esquema de la base de datos: {e_schema}"}
    
    logger.info(f"DEBUG: [pipeline.py] Esquema de base de datos cargado correctamente desde {schema_path}.")

    # Transformar db_schema_dict al formato esperado por WhitelistValidator
    transformed_schema = {"tables": []}
    if isinstance(db_schema_dict, dict): # Asegurarse que db_schema_dict es un diccionario
        for table_name, table_data in db_schema_dict.items():
            if isinstance(table_data, dict) and "columns" in table_data and isinstance(table_data["columns"], list):
                columns = [{"name": col_name} for col_name in table_data.get("columns", [])]
                transformed_schema["tables"].append({"name": table_name, "columns": columns})
            else:
                logger.warning(f"Formato inesperado para la tabla {table_name} en schema_simple.json. Se omitirá.")
    else:
        logger.error("ERROR: [pipeline.py] db_schema_dict no es un diccionario como se esperaba después de cargar el JSON.")
        # Manejar el error apropiadamente, quizás devolviendo un error o usando un esquema vacío seguro.
        # Por ahora, se procederá con un transformed_schema vacío, lo que probablemente causará fallos en la validación.
        pass


    # Instanciar WhitelistValidator
    whitelist_validator_instance = WhitelistValidator(
        db_schema_dict=transformed_schema, # MODIFICADO: Usar el esquema transformado
        case_sensitive=True  # Considera hacerlo configurable si es necesario
    )
    logger.info("DEBUG: [pipeline.py] WhitelistValidator instanciado.")
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
    is_whitelisted: bool = True # Asumir que es true hasta que se valide
    # --- FIN: Variables para el flujo de SQL y resultados ---

    preprocessed_question = preprocess_question(user_question, terms_dict)
    logger.info(f"DEBUG: [pipeline.py] Pregunta preprocesada: '{preprocessed_question}'")

    structured_info: Dict[str, Any] = {}
    error_extract_msg: str = "" # Mensaje de la extracción, puede ser una nota o un error.
    try:
        structured_info, error_extract_msg = extract_info_from_question_llm(
            preprocessed_question, 
            db_schema_dict, 
            terms_dict, 
            llm_client=llm_param, 
            current_logger=logger, 
            max_retries=max_retries_llm
        )
        logger.info(f"DEBUG: [pipeline.py] Información estructurada extraída: {structured_info}, Mensaje de extracción: '{error_extract_msg}'")
    except Exception as e_extract:
        logger.error(f"DEBUG: [pipeline.py] Excepción durante extract_info_from_question_llm: {str(e_extract)}", exc_info=True)
        error_extract_msg = f"Excepción crítica al intentar extraer información de la pregunta: {str(e_extract)}"
        flow_decision = "extraction_failed_return" # Error grave, no se puede continuar

    # --- INICIO: Decidir si usar SQL extraído, generar SQL, o manejar error de extracción ---
    flow_decision = "generate_new_sql" # Por defecto

    if error_extract_msg:
        logger.warning(f"DEBUG: [pipeline.py] Mensaje de la extracción presente: '{error_extract_msg}'")
        is_direct_sql_note = "Consulta SQL (directa) generada por LLM" in error_extract_msg or \
                             "SQL extraído directamente de la pregunta" in error_extract_msg or \
                             "La consulta SQL parece ser directa" in error_extract_msg
        
        if structured_info and "sql_query" in structured_info and is_direct_sql_note:
            extracted_sql_candidate = structured_info.get("sql_query")
            if extracted_sql_candidate and isinstance(extracted_sql_candidate, str):
                # Intentar extraer SQL de markdown si está presente
                if "```" in extracted_sql_candidate: # Comprobación simple de presencia de markdown
                    logger.info(f"DEBUG: [pipeline.py] SQL directo detectado con posible markdown: '{extracted_sql_candidate[:100]}...'")
                    extracted_sql_candidate = extract_sql_from_markdown(extracted_sql_candidate)
                    logger.info(f"DEBUG: [pipeline.py] SQL directo después de intentar extraer de markdown: '{extracted_sql_candidate}'")
                
                if extracted_sql_candidate and extracted_sql_candidate.strip().upper().startswith(("SELECT", "WITH")):
                    logger.info(f"DEBUG: [pipeline.py] SQL directo detectado y parece válido: '{extracted_sql_candidate}'.")
                    sql_to_execute = extracted_sql_candidate
                    flow_decision = "use_extracted_sql"
                else:
                    logger.error(f"DEBUG: [pipeline.py] SQL directo (post-markdown) no es SELECT/WITH o está vacío. Original structured_info: {structured_info}")
                    error_extract_msg = f"Se detectó un intento de SQL directo, pero la consulta no era válida (post-markdown). Mensaje original: {error_extract_msg}"
                    flow_decision = "extraction_failed_return" # Error grave
            else: # extracted_sql_candidate no es una cadena o es None
                logger.error(f"DEBUG: [pipeline.py] Mensaje de SQL directo, pero 'sql_query' no es una cadena o está ausente. structured_info: {structured_info}")
                error_extract_msg = f"Se detectó un intento de SQL directo, pero la consulta ('sql_query') no es una cadena o está ausente. Mensaje original: {error_extract_msg}"
                flow_decision = "extraction_failed_return" # Error grave
        elif not (structured_info and "sql_query" in structured_info): # Si hay error_extract_msg Y NO hay sql_query en structured_info
            logger.error(f"DEBUG: [pipeline.py] Error definitivo en la extracción (sin SQL en structured_info): {error_extract_msg}.")
            flow_decision = "extraction_failed_return"
        # Si hay error_extract_msg pero también hay sql_query (y no es la nota de SQL directo),
        # se intentará usar ese sql_query si es válido (siguiente bloque).
        # Si no es válido, se pasará a generar.

    if flow_decision == "generate_new_sql" and structured_info and "sql_query" in structured_info:
        extracted_sql_candidate = structured_info.get("sql_query")
        if extracted_sql_candidate and isinstance(extracted_sql_candidate, str):
            # Intentar extraer SQL de markdown si está presente
            if "```" in extracted_sql_candidate: # Comprobación simple de presencia de markdown
                logger.info(f"DEBUG: [pipeline.py] SQL de structured_info con posible markdown: '{extracted_sql_candidate[:100]}...'")
                extracted_sql_candidate = extract_sql_from_markdown(extracted_sql_candidate)
                logger.info(f"DEBUG: [pipeline.py] SQL de structured_info después de intentar extraer de markdown: '{extracted_sql_candidate}'")

            if extracted_sql_candidate and extracted_sql_candidate.strip().upper().startswith(("SELECT", "WITH")):
                logger.info(f"DEBUG: [pipeline.py] SQL extraído de structured_info (post-markdown) parece válido: '{extracted_sql_candidate}'.")
                sql_to_execute = extracted_sql_candidate
                flow_decision = "use_extracted_sql"
            else:
                logger.warning(f"DEBUG: [pipeline.py] structured_info tiene 'sql_query' (post-markdown) pero no es SELECT/WITH válida o está vacía: '{extracted_sql_candidate}'. Se procederá a generar SQL si es necesario.")
                # flow_decision permanece "generate_new_sql"

    if flow_decision == "extraction_failed_return":
        logger.error(f"DEBUG: [pipeline.py] Retornando debido a error en la extracción: {error_extract_msg}")
        respuesta_final_chatbot = f"Error al procesar la pregunta: {error_extract_msg}"
        # El diccionario de retorno se ensambla al final
    elif flow_decision == "use_extracted_sql":
        if sql_to_execute:
            logger.info(f"DEBUG: [pipeline.py] Validando SQL extraído/directo contra whitelist: {sql_to_execute}")
            # Usar la instancia de WhitelistValidator y su nuevo método
            is_whitelisted, whitelist_error_msg, _ = whitelist_validator_instance.validate_sql_string(sql_to_execute)
            if not is_whitelisted:
                logger.warning(f"DEBUG: [pipeline.py] SQL extraído/directo NO permitido por whitelist: {whitelist_error_msg}")
                respuesta_final_chatbot = f"Consulta SQL no permitida: {whitelist_error_msg}"
                sql_to_execute = None # No ejecutar
            else:
                logger.info(f"DEBUG: [pipeline.py] SQL extraído/directo es VÁLIDO y PERMITIDO.")
                params_to_execute = [] 
        else:
            logger.error("ERROR LÓGICO: flow_decision es use_extracted_sql pero sql_to_execute no está definido.")
            respuesta_final_chatbot = "Error lógico interno del pipeline (sql_to_execute ausente)."
            # Esto se tratará como un error que impide la ejecución.
            sql_to_execute = None # Asegurar que no se ejecute nada.

    elif flow_decision == "generate_new_sql":
        logger.info(f"DEBUG: [pipeline.py] Procediendo a generar SQL nuevo.")
    
        try:
            sql_generator_instance = SQLGenerator(db_connector=db_connector, llm=llm_param, logger_instance=logger)
            
            # current_allowed_columns_map ahora viene de la instancia del validador
            # terms_dict_str_for_validation se mantiene, asumiendo que json.dumps(terms_dict) es correcto
            terms_dict_str_for_gen = json.dumps(terms_dict)

            generated_sql_candidate, gen_params, gen_error, gen_context = \
                generate_sql_and_validate_whitelist(
                    structured_info=structured_info, 
                    sql_generator_instance=sql_generator_instance,
                    db_connector_for_sql_gen=db_connector,
                    current_allowed_columns_map=whitelist_validator_instance.allowed_columns_map, # ACTUALIZADO
                    terms_dict_str_for_validation=terms_dict_str_for_gen, # Mantenido
                    question_for_logging=preprocessed_question,
                    logger_param=logger
                )
            
            error_sql_gen = gen_error
            context_sql_gen = gen_context if gen_context else ""

            if generated_sql_candidate and not error_sql_gen:
                logger.info(f"DEBUG: [pipeline.py] SQL generado y validado (o error_sql_gen lo indica): {generated_sql_candidate}")
                sql_to_execute = generated_sql_candidate
                params_to_execute = gen_params if gen_params else []
                is_whitelisted = True # Si llegamos aquí sin error_sql_gen, se asume que pasó la whitelist.
                # whitelist_error_msg debería estar vacío o ser None si no hay error_sql_gen.
            else:
                logger.error(f"DEBUG: [pipeline.py] Fallo en la generación o validación de SQL. Error: {error_sql_gen}")
                # sql_to_execute permanece None. error_sql_gen ya está poblado.
                is_whitelisted = False # No pasó la whitelist o hubo otro error de generación
                whitelist_error_msg = error_sql_gen # Usar el mensaje de error de la generación
        except Exception as e_gen_pipeline:
            logger.error(f"DEBUG: [pipeline.py] Excepción durante el bloque de generación/validación de SQL: {str(e_gen_pipeline)}", exc_info=True)
            error_sql_gen = f"Excepción en pipeline durante generación/validación de SQL: {str(e_gen_pipeline)}"
            sql_to_execute = None
            params_to_execute = []
            is_whitelisted = False
            whitelist_error_msg = error_sql_gen
            context_sql_gen = "" # Resetear contexto en caso de excepción
    # --- FIN: Decisión y procesamiento de SQL ---

    # --- INICIO: Ejecución de SQL y obtención de resultados ---
    if sql_to_execute and is_whitelisted: # Solo ejecutar si hay SQL y está permitido
        try:
            logger.info(f"DEBUG: [pipeline.py] Ejecutando SQL: {sql_to_execute} con params: {params_to_execute}")
            rows, cols = db_connector.execute_query(sql_to_execute, params_to_execute)
            
            logger.debug(f"DEBUG PIPELINE: Filas crudas (rows) desde db_connector: {str(rows)[:200]}") # Loguear solo una parte
            logger.debug(f"DEBUG PIPELINE: Columnas crudas (cols) desde db_connector: {cols}")

            if rows is not None and cols is not None and rows:
                results = [dict(zip(cols, row)) for row in rows]
            else:
                results = []
            
            logger.debug(f"DEBUG PIPELINE: Results procesados (lista de diccionarios): {str(results)[:200]}") # Loguear solo una parte

            if not results:
                data_results_empty = True
                if rows is None or cols is None:
                    respuesta_final_chatbot = "La consulta se ejecutó pero no pude formatear la respuesta correctamente (problema con filas/columnas)."
                else: # rows is not None and cols is not None but rows is empty
                    respuesta_final_chatbot = "La consulta SQL se ejecutó correctamente pero no devolvió ningún dato."
            else:
                data_results_empty = False
                logger.info(f"DEBUG: [pipeline.py] Resultados SQL obtenidos ({len(results)} filas). Generando respuesta en lenguaje natural.")
                
                # Usar error_extract_msg original para la respuesta NL, ya que puede ser una nota útil
                # (ej. "SQL directo proporcionado").
                # context_sql_gen también se pasa.
                _error_extract_msg_for_nl = error_extract_msg if error_extract_msg else ""
                _cols_for_nl = cols if cols is not None else []

                respuesta_final_chatbot = generate_natural_language_response_via_llm(
                    llm=llm_param,
                    original_question=user_question,
                    sql_query=sql_to_execute,
                    results=results,
                    column_names=_cols_for_nl,
                    context=context_sql_gen, # Contexto de la generación de SQL
                    error_extract_msg=_error_extract_msg_for_nl, # Mensaje de la extracción inicial
                    logger_param=logger
                )
                logger.info(f"DEBUG: [pipeline.py] Respuesta en lenguaje natural generada: {respuesta_final_chatbot}")
        except Exception as e_sql:
            logger.error(f"ERROR: [pipeline.py] Excepción durante la ejecución de SQL o procesamiento de resultados: {e_sql}", exc_info=True)
            error_execution_msg = str(e_sql)
            respuesta_final_chatbot = f"Se produjo un error al ejecutar la consulta SQL o procesar sus resultados: {error_execution_msg}"
            results = [] 
            data_results_empty = True
    else: # No hay sql_to_execute o no está permitido por whitelist
        logger.warning(f"DEBUG: [pipeline.py] No hay SQL para ejecutar o no está permitido. SQL: '{sql_to_execute}', Whitelisted: {is_whitelisted}")
        results = []
        data_results_empty = True
        if not is_whitelisted and whitelist_error_msg: # Error de whitelist es prioritario si ocurrió
             respuesta_final_chatbot = f"Consulta SQL no permitida: {whitelist_error_msg}"
        elif error_sql_gen: # Error durante la generación de SQL
            respuesta_final_chatbot = f"No se pudo generar una consulta SQL válida. Detalle: {error_sql_gen}"
        elif flow_decision == "extraction_failed_return" and error_extract_msg: # Error grave de extracción
            respuesta_final_chatbot = f"No se pudo procesar la pregunta. Detalle: {error_extract_msg}"
        elif not sql_to_execute and error_extract_msg: # Si no hay SQL y hubo un mensaje de extracción (que no fue nota de SQL directo manejada)
             # Este caso es para cuando error_extract_msg no fue fatal pero no se encontró/generó SQL
             is_direct_sql_note_handled = "Consulta SQL (directa) generada por LLM" in error_extract_msg or \
                                          "SQL extraído directamente de la pregunta" in error_extract_msg
             if not is_direct_sql_note_handled:
                respuesta_final_chatbot = f"Problema al procesar la pregunta: {error_extract_msg}"
             else: # Si fue una nota de SQL directo pero falló la validación de ese SQL
                respuesta_final_chatbot = "La consulta SQL directa proporcionada no pudo ser procesada."

        if not respuesta_final_chatbot: # Fallback genérico
            respuesta_final_chatbot = "No se pudo determinar o ejecutar una consulta SQL para su pregunta."
    # --- FIN: Ejecución de SQL y obtención de resultados ---

    # --- INICIO: Preparación de la respuesta final del pipeline ----
    if not respuesta_final_chatbot: # Si la generación NL principal no dio resultado o fue omitida
        if not data_results_empty: # Tenemos datos (results no está vacío), pero respuesta_final_chatbot sí lo está.
            logger.info("DEBUG: [pipeline.py] respuesta_final_chatbot está vacía pero hay resultados. Intentando fallback.")
            # Fallback específico si es un resultado de COUNT y la NL principal falló.
            if sql_to_execute and "COUNT(" in sql_to_execute.upper() and results and isinstance(results, list) and len(results) == 1 and isinstance(results[0], dict):
                try:
                    # Intenta extraer el nombre de la columna de conteo y el valor
                    count_col_name = list(results[0].keys())[0]
                    count_value = results[0][count_col_name]
                    respuesta_final_chatbot = f"La consulta de conteo para '{count_col_name}' devolvió: {count_value}."
                    logger.info(f"DEBUG: [pipeline.py] Fallback NL generado para COUNT: {respuesta_final_chatbot}")
                except Exception as e_fallback_nl:
                    logger.warning(f"DEBUG: [pipeline.py] Error generando fallback NL para COUNT ({e_fallback_nl}). Usando mensaje más genérico.")
                    respuesta_final_chatbot = "Se obtuvieron resultados numéricos, pero hubo un problema al formatear la respuesta detallada."
            else:
                # Fallback genérico si hay resultados pero la NL principal está vacía (y no es un COUNT claro)
                respuesta_final_chatbot = "Se han obtenido datos, pero no se pudo generar una descripción en lenguaje natural para ellos."
            # Importante: data_results_empty ya es False en este camino y debe permanecer así.
        elif data_results_empty and sql_to_execute and is_whitelisted and not error_execution_msg:
            # Caso: consulta ejecutada OK, permitida, sin errores de ejecución, pero devolvió 0 filas.
            respuesta_final_chatbot = "La consulta se ejecutó correctamente, pero no se encontraron resultados que coincidan con tu petición."
        else:
            # Otros casos: error de extracción, error de generación SQL, error de whitelist, error de ejecución SQL.
            # La respuesta_final_chatbot ya debería haber sido establecida por esos flujos de error.
            # Si por alguna razón no lo fue, este es un último recurso.
            
            # Construir el mensaje de error a partir de las partes disponibles
            # Esta variable se define más abajo, pero la necesitamos aquí para el log y la respuesta.
            current_error_message_parts_temp = []
            if error_extract_msg and flow_decision == "extraction_failed_return":
                current_error_message_parts_temp.append(f"Extracción: {error_extract_msg}")
            elif error_extract_msg and not ("Consulta SQL (directa)" in error_extract_msg or "SQL extraído directamente" in error_extract_msg) and not sql_to_execute :
                current_error_message_parts_temp.append(f"Extracción: {error_extract_msg}")
            if error_sql_gen:
                current_error_message_parts_temp.append(f"Generación SQL: {error_sql_gen}")
            if error_execution_msg:
                current_error_message_parts_temp.append(f"Ejecución SQL: {error_execution_msg}")
            if not is_whitelisted and whitelist_error_msg:
                 current_error_message_parts_temp.append(f"Whitelist: {whitelist_error_msg}")

            if current_error_message_parts_temp:
                respuesta_final_chatbot = f"No se pudo completar la solicitud. Errores detectados: {'; '.join(current_error_message_parts_temp)}"
            elif not sql_to_execute:
                 respuesta_final_chatbot = "No se pudo determinar o generar una consulta SQL para su pregunta."
            else:
                respuesta_final_chatbot = "No se pudo completar la solicitud debido a un error interno o falta de información."
            logger.warning(f"DEBUG: [pipeline.py] Fallback final para respuesta_final_chatbot: {respuesta_final_chatbot}")

    logger.info(f"DEBUG: [pipeline.py] Respuesta final del chatbot para el usuario: {respuesta_final_chatbot}")

    final_error_message_parts = []
    # Solo añadir error_extract_msg si fue un error real que impidió el flujo o es relevante
    if error_extract_msg and flow_decision == "extraction_failed_return":
        final_error_message_parts.append(f"Extracción: {error_extract_msg}")
    elif error_extract_msg and not ("Consulta SQL (directa)" in error_extract_msg or "SQL extraído directamente" in error_extract_msg) and not sql_to_execute : # Si hubo un error de extracción que no fue una nota y no se pudo generar SQL
        final_error_message_parts.append(f"Extracción: {error_extract_msg}")

    if error_sql_gen:
        final_error_message_parts.append(f"Generación SQL: {error_sql_gen}")
    if error_execution_msg:
        final_error_message_parts.append(f"Ejecución SQL: {error_execution_msg}")
    if not is_whitelisted and whitelist_error_msg:
         final_error_message_parts.append(f"Whitelist: {whitelist_error_msg}")
    
    final_error_message_str = "; ".join(final_error_message_parts) if final_error_message_parts else None

    return_dict = {
        "response_message": respuesta_final_chatbot,
        "data_results": results,
        "sql_query_generated": sql_to_execute, 
        "params_used": params_to_execute,
        "data_results_empty": data_results_empty,
        "error_message": final_error_message_str
    }
    logger.debug(f"DEBUG: [pipeline.py] Diccionario de retorno final: {return_dict}")
    return return_dict
# --- FIN: chatbot_pipeline_entrypoint ---
