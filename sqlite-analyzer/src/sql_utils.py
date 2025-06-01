"""
Utilidades para procesamiento y validación de consultas SQL en el chatbot médico.
Contiene funciones para validar seguridad SQL, corregir consultas y formatear resultados.
"""

import re
import logging
import os
import json
from typing import List, Tuple, Optional, Dict, Any, Set
import difflib
from difflib import get_close_matches
# Dinámico para evitar módulo faltante
try:
    from src.llm_utils import call_llm
except ImportError:
    from llm_utils import call_llm
from json.decoder import JSONDecodeError, JSONDecoder
from pathlib import Path
import sqlparse
from sqlparse.sql import Identifier, IdentifierList, Case
from sqlparse.tokens import Keyword, Name, Punctuation

# Configuración del logger para este módulo
logger = logging.getLogger(__name__)
# Evitar añadir múltiples manejadores si el logger ya está configurado por otra parte del programa
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # Establecer el nivel de logging. Cambiar a logging.DEBUG para más detalles si es necesario.
    logger.setLevel(logging.INFO)

try:
    from fuzzywuzzy import fuzz, process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    logging.warning("fuzzywuzzy no está instalado. La corrección de errores tipográficos será limitada.")

# Fichero de mapeos de términos
TERMS_FILE = os.path.join(os.path.dirname(__file__), "data", "dictionary.json") # Nueva ruta al diccionario generado

def validate_sql_query(sql_query: str, db_connector: Optional[Any] = None) -> bool:
    # El parámetro db_connector es opcional y no se usa en esta validación básica.
    # Se mantiene por compatibilidad si se quisiera extender la validación usando el conector.
    # Verificar que no haya cláusulas ON vacías
    if re.search(r'ON\s+(?:INNER|LEFT|RIGHT|JOIN|WHERE)', sql_query, re.IGNORECASE):
        return False
        
    # Verificar que WHERE no esté vacío al final
    if sql_query.strip().upper().endswith('WHERE'):
        return False
        
    return True

def correct_typos_in_question(question: str, valid_terms: List[str]) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Detecta y corrige errores tipográficos en la pregunta del usuario.
    
    Args:
        question: La pregunta original del usuario
        valid_terms: Lista de términos válidos para comparar
        
    Returns:
        Tuple con (pregunta_corregida, lista_de_correcciones)
    """
    words = re.findall(r'\b\w+\b', question.lower())
    corrections = []
    corrected_question = question
    
    # Ignorar palabras comunes o cortas
    stopwords = {'de', 'la', 'el', 'en', 'los', 'las', 'con', 'por', 'para', 'que', 'como', 'se', 'un', 'una', 'unos'}
    
    for word in words:
        # Ignorar palabras cortas, números o stopwords
        if len(word) <= 3 or word.isdigit() or word in stopwords:
            continue
            
        # Usar fuzzy matching si está disponible
        if FUZZY_AVAILABLE and len(valid_terms) > 0:
            # Obtener las mejores coincidencias con un umbral alto
            matches = process.extractBests(word, valid_terms, score_cutoff=85, limit=1)
            if matches and matches[0][1] > 90 and word != matches[0][0]:
                correction = matches[0][0]
                pattern = r'\b' + re.escape(word) + r'\b'
                corrected_question = re.sub(pattern, correction, corrected_question, flags=re.IGNORECASE)
                corrections.append((word, correction))
                logging.info(f"Corrección de typo: '{word}' -> '{correction}'")
        else:
            # Fallback a difflib si fuzzywuzzy no está disponible
            if len(valid_terms) > 0:
                matches = get_close_matches(word, valid_terms, n=1, cutoff=0.85)
                if matches and word != matches[0]:
                    correction = matches[0]
                    pattern = r'\b' + re.escape(word) + r'\b'
                    corrected_question = re.sub(pattern, correction, corrected_question, flags=re.IGNORECASE)
                    corrections.append((word, correction))
                    logging.info(f"Corrección de typo (difflib): '{word}' -> '{correction}'")
    
    if corrections:
        logging.info(f"Pregunta original: '{question}'")
        logging.info(f"Pregunta corregida: '{corrected_question}'")
    
    return corrected_question, corrections

def validate_where_conditions(text: str) -> List[Dict[str, str]]:
    """
    Extrae automáticamente condiciones WHERE del texto original.
    Ahora siempre devuelve una lista de condiciones en lugar de un booleano.
    
    Args:
        text: Texto de la pregunta o consulta
        
    Returns:
        Lista de condiciones en formato estructurado [{column, operator, value}]
    """
    conditions = []

    # Verificar que el texto tiene contenido válido y filtrar caracteres problemáticos
    if not text or len(text.strip()) < 3:
        return conditions
        
    # Filtrar caracteres especiales y símbolos sueltos que causan problemas
    text = re.sub(r'[\[\]\{\}]', ' ', text)  # Eliminar [], {} que pueden causar problemas
    
    # Ignorar textos demasiado cortos o sin contenido real
    if len(text.strip()) <= 2 or text.strip().isdigit():
        return conditions
    
    # Detectar patrones comunes de IDs de entidades de forma genérica
    entity_patterns = []
    
    # Buscar patrones de tipo "entidad número" (ej: "paciente 1909")
    for entity_match in re.finditer(r'(\w+)\s+(\d+)', text.lower()):
        entity_type = entity_match.group(1)
        entity_id = entity_match.group(2)
        
        # Añadir una condición genérica basada en el tipo de entidad
        # La columna exacta se determinará más adelante en el pipeline
        conditions.append({
            "entity_type": entity_type,
            "entity_id": entity_id,
            "operator": "=",
            "value": entity_id
        })
        
        logging.info(f"Condición automática detectada para {entity_type} con ID {entity_id}")
    
    return conditions

def parameterize_query(query: str) -> Tuple[str, List[Any]]:
    """
    Convierte una consulta SQL a una versión parametrizada segura.
    
    Args:
        query: Consulta SQL con valores literales
        
    Returns:
        Tuple con (consulta_parametrizada, lista_de_valores)
    """
    values = []
    
    # Reemplazar literales de texto (entre comillas)
    string_pattern = r"'([^']*)'|\"([^\"]*)\""
    
    def replace_string(match):
        val = match.group(1) or match.group(2)
        values.append(val)
        return '?'
    
    param_query = re.sub(string_pattern, replace_string, query)
    
    # Reemplazar valores numéricos
    num_pattern = r'\b\d+\.?\d*\b'
    
    def replace_number(match):
        val = float(match.group(0)) if '.' in match.group(0) else int(match.group(0))
        values.append(val)
        return '?'
    
    param_query = re.sub(num_pattern, replace_number, param_query)
    
    logging.info(f"SQL original: {query}")
    logging.info(f"SQL parametrizado: {param_query}")
    logging.info(f"Valores: {values}")
    
    return param_query, values

def fix_count_query(query: str, question: str = "") -> str:
    """
    Corrige consultas para preguntas de tipo "cuántos" asegurándose de usar COUNT.
    
    Args:
        query: La consulta SQL generada
        question: La pregunta original
    
    Returns:
        Consulta SQL corregida
    """
    query_lower = query.lower()
    
    # Si ya es una consulta COUNT, verificar si está bien formada
    if 'count(' in query_lower:
        return query
    
    # Si es una pregunta de "cuántos" pero no usa COUNT, convertirla
    if question and re.search(r'(cuant[oa]s|número|total|cuantos)', question.lower()):
        # Extraer la tabla de la consulta original
        table_match = re.search(r'from\s+(\w+)', query_lower)
        if not table_match:
            logging.error("No se pudo identificar la tabla en la consulta")
            return query
        
        table = table_match.group(1)
        
        # Mantener la cláusula WHERE si existe
        where_clause = ""
        where_match = re.search(r'where\s+(.*?)(?:order by|group by|limit|$)', query_lower, re.DOTALL)
        if where_match:
            where_clause = f"WHERE {where_match.group(1)}"
        
        # Construir la nueva consulta COUNT
        new_query = f"SELECT COUNT(*) AS total FROM {table} {where_clause}".strip()
        logging.info(f"Consulta corregida de SELECT a COUNT: {new_query}")
        return new_query
    
    # Si no se detecta que sea una pregunta de conteo, devolver la consulta original
    return query

def sanitize_sql_identifier(identifier: str) -> str:
    """
    Sanitiza identificadores SQL para prevenir SQL injection.
    
    Args:
        identifier: Nombre de tabla o columna
        
    Returns:
        Identificador sanitizado
    """
    # Eliminar caracteres no permitidos
    sanitized = re.sub(r'[^\w]', '', identifier)
    return sanitized

def format_results(rows: List[tuple], columns: List[str], action: str = "SELECT") -> str:
    """
    Formatea los resultados de la consulta para su presentación.
    
    Args:
        rows: Filas de resultados
        columns: Nombres de columnas
        action: Tipo de acción (SELECT, COUNT, etc.)
        
    Returns:
        Texto formateado
    """
    if not rows or not columns:
        return "No se encontraron resultados para esta consulta."
    
    # Formateo especial para consultas agregadas
    if action.upper() in ["COUNT", "AVG", "MAX", "MIN", "SUM"] or (len(columns) == 1 and any(col.lower().startswith(('count', 'avg', 'max', 'min', 'sum')) for col in columns)):
        value = rows[0][0]
        
        if action.upper() == "COUNT" or "count" in columns[0].lower():
            return f"He encontrado {value} resultados."
        elif action.upper() == "AVG" or "avg" in columns[0].lower():
            return f"El promedio es: {value:.2f}"
        elif action.upper() == "MAX" or "max" in columns[0].lower():
            return f"El valor máximo es: {value}"
        elif action.upper() == "MIN" or "min" in columns[0].lower():
            return f"El valor mínimo es: {value}"
        elif action.upper() == "SUM" or "sum" in columns[0].lower():
            return f"La suma total es: {value}"
    
    # Para consultas normales con pocos resultados
    if len(rows) <= 20:
        # Crear tabla ASCII básica
        header = "\t".join(map(str, columns))
        data_rows = []
        if rows and isinstance(rows[0], dict):
            for row_dict in rows:
                data_rows.append("\t".join([str(row_dict.get(col_name, '')) for col_name in columns]))
        elif rows: # Asumiendo lista de tuplas/listas
            data_rows = ["\t".join(map(str, row_tuple)) for row_tuple in rows]
        
        result = "\n".join([header] + data_rows)
        return f"He encontrado {len(rows)} resultados:\n{result}"
    else:
        # Para muchos resultados, mostrar solo los primeros 5
        header = "\t".join(map(str, columns))
        data_rows = []
        if rows and isinstance(rows[0], dict):
            for row_dict in rows[:5]: # Tomar solo los primeros 5
                data_rows.append("\t".join([str(row_dict.get(col_name, '')) for col_name in columns]))
        elif rows: # Asumiendo lista de tuplas/listas
            data_rows = ["\t".join(map(str, row_tuple)) for row_tuple in rows[:5]] # Tomar solo los primeros 5

        result = "\n".join([header] + data_rows)
        return f"He encontrado {len(rows)} resultados. Mostrando los primeros 5:\n{result}\n...y {len(rows)-5} más."
    

def validate_sql_entities(sql_query: str, db_structure: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Verifica que todas las tablas y columnas de la consulta existan y que los alias de SELECT
    no se usen incorrectamente en WHERE, GROUP BY, HAVING, ORDER BY.
    Devuelve (True, "") si es válida, o (False, "mensaje de error") si no.
    """
    logger = logging.getLogger(__name__)
    sql_query_upper_for_keywords = sql_query.upper()

    # 1. Extraer tablas y sus alias
    table_references = {}  # alias.upper() -> real_table_name.upper() (o marcador para subconsulta)
    actual_tables_in_query = set()  # Store real table names (UPPER) found

    # Regex para FROM y JOIN clauses.
    # Modificado para capturar también subconsultas con alias.
    from_join_clauses_match = re.search(r'\bFROM\s+(.+?)(?=(\bWHERE\b|\bGROUP BY\b|\bORDER BY\b|\bLIMIT\b|\bUNION\b|\bINTERSECT\b|\bEXCEPT\b|;|\Z))', sql_query, re.IGNORECASE | re.DOTALL)
    
    if from_join_clauses_match:
        full_from_join_segment = from_join_clauses_match.group(1)
        
        # Primero, buscar subconsultas con alias en el segmento FROM/JOIN
        # Ejemplo: JOIN (SELECT ... ) AS alias_subconsulta ON ...
        # Ejemplo: FROM (SELECT ... ) alias_subconsulta, otra_tabla ...
        # Este regex es una simplificación y podría no cubrir todos los casos complejos de subconsultas anidadas.
        subquery_alias_pattern = r'\(\s*SELECT.*?\)\s*(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]+)'
        
        # Iterar sobre las coincidencias de subconsultas con alias
        for sub_match in re.finditer(subquery_alias_pattern, full_from_join_segment, re.IGNORECASE | re.DOTALL):
            alias_name = sub_match.group(1)  # Corregido de group(2) a group(1)
            alias_upper = alias_name.upper()
            # Marcar este alias como proveniente de una subconsulta.
            # No tenemos un "nombre real de tabla" para él, pero el alias es válido.
            table_references[alias_upper] = "__SUBQUERY__" 
            logger.info(f"Subconsulta con alias \'{alias_name}\' detectada y registrada.")
            # Eliminar la subconsulta procesada del segmento para evitar que se procese como tabla normal
            # Esto es complicado de hacer correctamente sin un parseador SQL completo.
            # Por ahora, confiamos en que el procesamiento posterior de tablas no se confunda demasiado.
            # Una mejora sería reemplazar la subconsulta con un placeholder o eliminarla con más cuidado.

        # Procesamiento existente de tablas y alias (puede necesitar ajustes si la subconsulta no se eliminó bien)
        join_segments = re.split(r'\bJOIN\b', full_from_join_segment, flags=re.IGNORECASE)
        
        processed_segments = []
        for i, segment in enumerate(join_segments):
            if i == 0: # First segment (FROM part)
                processed_segments.extend(segment.split(','))
            else: # Subsequent segments (JOIN parts)
                # Remove ON clause for simplicity before matching table name and alias
                segment_no_on = re.sub(r'\sON\s+.+', '', segment, flags=re.IGNORECASE | re.DOTALL)
                processed_segments.append(segment_no_on)

        for part in processed_segments:
            part = part.strip()
            if not part: continue
            # Match 'table_name alias' or 'table_name AS alias' or just 'table_name'
            match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]+)(?:\s+AS)?(?:\s+([a-zA-Z_][a-zA-Z0-9_]+))?', part, re.IGNORECASE)
            if match:
                raw_table_name, raw_alias_name = match.group(1), match.group(2)
                raw_table_name_upper = raw_table_name.upper()
                
                resolved_real_name_upper = None
                if raw_table_name_upper in db_structure: # db_structure keys should be upper
                    resolved_real_name_upper = raw_table_name_upper
                else:
                    # Attempt to find with get_close_matches if db_structure keys are not guaranteed upper
                    # Or if the input raw_table_name_upper needs matching against mixed-case keys
                    # For simplicity, assume db_structure keys are UPPERCASE
                    matches = get_close_matches(raw_table_name_upper, db_structure.keys(), n=1, cutoff=0.85)
                    if matches:
                        resolved_real_name_upper = matches[0].upper() # Ensure it's upper
                        logger.info(f"Tabla '{raw_table_name}' resuelta a '{resolved_real_name_upper}' por similitud.")
                    else:
                        logger.warning(f"Tabla desconocida: '{raw_table_name}' en la consulta.")
                        return False, f"Tabla desconocida: '{raw_table_name}'."
                
                actual_tables_in_query.add(resolved_real_name_upper)
                
                if raw_alias_name:
                    table_references[raw_alias_name.upper()] = resolved_real_name_upper
                else: # No explicit alias, table name is its own alias
                    table_references[raw_table_name_upper] = resolved_real_name_upper

    if not actual_tables_in_query and "DUAL" not in sql_query_upper_for_keywords and not sql_query_upper_for_keywords.startswith("VALUES"):
        if re.search(r'\bFROM\b|\bJOIN\b', sql_query, re.IGNORECASE):
            logger.warning(f"No se pudieron identificar tablas válidas en la consulta: '{sql_query}'")
            return False, "No se pudieron identificar tablas válidas en la consulta."

    # 2. Extraer alias de la cláusula SELECT
    select_aliases = set()
    select_clause_match = re.search(r'SELECT\s+(.+?)\s+FROM', sql_query, re.IGNORECASE | re.DOTALL)
    if select_clause_match:
        select_content = select_clause_match.group(1)
        aliases_found = re.findall(r'\sAS\s+([a-zA-Z_][a-zA-Z0-9_]+)\b', select_content, re.IGNORECASE)
        for alias in aliases_found:
            select_aliases.add(alias.upper())
    logger.debug(f"Alias de SELECT encontrados: {select_aliases}")

    # 3. Validar columnas en cláusulas problemáticas (WHERE, GROUP BY, HAVING)
    problematic_clauses_regex = r'(?:\bWHERE\s+(?P<where_clause>.+?)(?=(\bGROUP BY\b|\bORDER BY\b|\bLIMIT\b|\bHAVING\b|\bUNION\b|\bINTERSECT\b|\bEXCEPT\b|;|\Z)))|' \
                                r'(?:\bGROUP BY\s+(?P<groupby_clause>.+?)(?=(\bORDER BY\b|\bLIMIT\b|\bHAVING\b|\bUNION\b|\bINTERSECT\b|\bEXCEPT\b|;|\Z)))|' \
                                r'(?:\bHAVING\s+(?P<having_clause>.+?)(?=(\bORDER BY\b|\bLIMIT\b|\bUNION\b|\bINTERSECT\b|\bEXCEPT\b|;|\Z)))'
    
    sql_keywords_for_ident_check = { # Common keywords/functions to ignore when checking identifiers
        "COUNT", "SUM", "AVG", "MIN", "MAX", "DISTINCT", "CASE", "WHEN", "THEN", "ELSE", "END",
        "AND", "OR", "NOT", "NULL", "IS", "IN", "LIKE", "BETWEEN", "EXISTS", "ALL", "ANY",
        "TRUE", "FALSE", "CAST", "CONVERT", "SUBSTRING", "TRIM", "DATE", "YEAR", "MONTH", "DAY",
        "STRFTIME", "JULIANDAY", "NOW", "CURRENT_DATE", "CURRENT_TIME", "CURRENT_TIMESTAMP"
    }.union(db_structure.keys()) # Add all known table names (UPPER)

    for clause_match in re.finditer(problematic_clauses_regex, sql_query, re.IGNORECASE | re.DOTALL):
        clause_content = ""
        clause_type_name = ""
        if clause_match.group("where_clause"):
            clause_content = clause_match.group("where_clause")
            clause_type_name = "WHERE"
        elif clause_match.group("groupby_clause"):
            clause_content = clause_match.group("groupby_clause")
            clause_type_name = "GROUP BY"
        elif clause_match.group("having_clause"):
            clause_content = clause_match.group("having_clause")
            clause_type_name = "HAVING"
        
        if clause_content and clause_type_name:
            logger.debug(f"Analizando cláusula {clause_type_name}: {clause_content}")
            # Extract potential identifiers (words not part of literals, trying to avoid function names)
            # This regex is simplified: it extracts words.
            clause_identifiers = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', clause_content)
            for ident_str in clause_identifiers:
                ident_upper = ident_str.upper()

                if ident_upper in sql_keywords_for_ident_check or ident_upper in table_references: # Skip keywords, table names/aliases
                    continue
                
                # Check if it's a column of any table in the query
                is_valid_column_of_table = False
                for real_tbl_name_upper in actual_tables_in_query:
                    if real_tbl_name_upper in db_structure and any(col_def['name'].upper() == ident_upper for col_def in db_structure[real_tbl_name_upper].get('columns',[])):
                        is_valid_column_of_table = True
                        break
                
                if not is_valid_column_of_table:
                    if ident_upper in select_aliases:
                        msg = f"El alias de SELECT '{ident_str}' no puede ser usado directamente en la cláusula {clause_type_name}."
                        logger.warning(msg + f" Consulta: {sql_query}")
                        return False, msg
                    else:
                        # Not a direct column, not a SELECT alias, not a keyword/table.
                        # Could be an error, or a function/literal not caught.
                        # For now, we don't flag this as a hard error to avoid false positives with simple regex.
                        # A more sophisticated check would be needed here.
                        logger.debug(f"Identificador '{ident_str}' en {clause_type_name} no es una columna directa ni un alias de SELECT conocido. Se omite la validación estricta para este identificador.")
                        pass

    # 4. Validate qualified columns (table_alias.column or table_name.column)
    qualified_column_matches = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]+)\.([a-zA-Z_][a-zA-Z0-9_]+)\b', sql_query)
    for tbl_ref_name, col_name_str in qualified_column_matches:
        tbl_ref_upper = tbl_ref_name.upper()
        col_name_upper = col_name_str.upper()

        actual_table_name_for_qualified_col = table_references.get(tbl_ref_upper)

        if not actual_table_name_for_qualified_col:
            # tbl_ref_upper is not an alias/table from the main query.
            # It could be an alias defined INSIDE a subquery, or a SELECT alias used incorrectly.
            if tbl_ref_upper in select_aliases: # It's a SELECT alias
                 msg = f"No se puede usar un alias de SELECT \'{tbl_ref_name}\' para cualificar una columna (\'{tbl_ref_name}.{col_name_str}\')."
                 logger.warning(msg + f" Consulta: {sql_query}")
                 return False, msg
            else:
                # Not a SELECT alias and not an alias/table from the main query.
                # It is likely an alias internal to a subquery (e.g., 'ai' in the user's case).
                # To avoid false positives, we don't flag this as an error here.
                # A full validation would require parsing subqueries.
                logger.info(f"Alias \'{tbl_ref_name}\' en \'{tbl_ref_name}.{col_name_str}\' no es de la consulta principal ni un alias de SELECT. Se asume que es un alias interno de subconsulta y se omite su validación estricta aquí.")
                continue # Move to the next qualified column
        
        # If actual_table_name_for_qualified_col IS '__SUBQUERY__' (from previous fix for derived tables in FROM/JOIN)
        if actual_table_name_for_qualified_col == "__SUBQUERY__":
            logger.info(f"Columna cualificada \'{tbl_ref_name}\' en \'{tbl_ref_name}.{col_name_str}\' pertenece a una subconsulta (tabla derivada en FROM/JOIN). Se omite validación de columna contra esquema principal.")
            continue

        # If it's a real table from the main query, validate the column against db_structure
        if actual_table_name_for_qualified_col not in db_structure:
            # This should not happen if table extraction was correct
            msg = f"Error interno: tabla resuelta \'{actual_table_name_for_qualified_col}\' no encontrada en db_structure."
            logger.error(msg + f" Consulta: {sql_query}")
            return False, msg

        table_cols_info = db_structure[actual_table_name_for_qualified_col].get('columns', [])
        if not any(c_info['name'].upper() == col_name_upper for c_info in table_cols_info):
            msg = f"Columna desconocida \'{col_name_str}\' para la tabla/alias \'{tbl_ref_name}\' (resuelta a \'{actual_table_name_for_qualified_col}\')."
            logger.warning(msg + f" Consulta: {sql_query}")
            return False, msg
            
    logger.info(f"Validación de entidades SQL para consulta directa прошла успешно.") # Consider changing "прошла успешно" (passed successfully in Russian)
    return True, ""


def extract_json_from_text(text: str) -> str:
    """
    Extrae un objeto JSON válido de un texto que podría contener texto descriptivo
    
    Args:
        text: Texto que puede contener un objeto JSON
        
    Returns:
        String con JSON válido o un JSON mínimo si no se encuentra
    """
    # Primero, buscar el inicio y fin de cualquier objeto JSON
    try:
        # Buscar el primer '{'
        json_start = text.find('{')
        if json_start >= 0:
            # Intentar analizar todo desde el primer '{'
            candidate = text[json_start:]
            try:
                json.loads(candidate)
                return candidate  # Si el parseo es exitoso, retornar
            except json.JSONDecodeError:
                # Podríamos tener texto después del JSON.
                # Buscar pares balanceados de llaves {} para encontrar el JSON completo
                stack = []
                for i, char in enumerate(candidate):
                    if char == '{':
                        stack.append(i)
                    elif char == '}':
                        if stack:
                            start = stack.pop()
                            # Si la pila está vacía, hemos encontrado el objeto completo
                            if not stack:
                                # Probar si el substring es un JSON válido
                                potential_json = candidate[:i+1]
                                try:
                                    json.loads(potential_json)
                                    return potential_json
                                except:
                                    pass  # Continuar buscando
                        
                # Si no hemos podido encontrar un JSON válido, intentar con regex
                json_pattern = re.search(r'(\{.*\})', text, re.DOTALL)
                if json_pattern:
                    potential_json = json_pattern.group(1)
                    try:
                        json.loads(potential_json)
                        logging.info("JSON extraído correctamente mediante regex")
                        return potential_json
                    except:
                        pass
    except Exception as e:
        logging.error(f"Error al extraer JSON: {e}")
        
    # Si todo falla, devolver un JSON mínimo
    return '{"PATI_PATIENTS": {"columns": [{"name": "PATI_ID", "type": "INTEGER"}]}}'


def extract_sql_from_markdown(markdown_text: str) -> Optional[str]:
    """
    Extrae una consulta SQL de un bloque de código Markdown.

    Args:
        markdown_text: Texto en formato Markdown.

    Returns:
        La consulta SQL extraída o None si no se encuentra.
    """
    # Patrón para buscar bloques de código SQL
    # Admite ```sql ... ``` o ``` ... ``` (asumiendo que es SQL si no se especifica lenguaje)
    pattern = r"""```(?:sql)?\n(.*?)```"""
    match = re.search(pattern, markdown_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def load_terms_mapping(file_path: str = TERMS_FILE) -> Dict[str, Any]:
    """
    Carga el diccionario enriquecido desde .bak o .json, soportando múltiples objetos JSON
    concatenados. Decodifica cada uno con raw_decode y los fusiona en un solo dict.
    Extrae valid_terms, table_mappings, column_mappings, descripciones y sinónimos.
    """
    raw = Path(file_path).read_text(encoding="utf-8")
    decoder = JSONDecoder()
    idx = 0
    enriched: Dict[str, Any] = {}

    # Extraer todos los JSON que aparezcan en el fichero
    while idx < len(raw):
        try:
            obj, next_idx = decoder.raw_decode(raw, idx)
        except JSONDecodeError:
            break
        # Fusionar obj en enriched
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k not in enriched:
                    enriched[k] = v
                else:
                    if isinstance(v, dict) and isinstance(enriched[k], dict):
                        enriched[k].update(v)
                    elif isinstance(v, list) and isinstance(enriched[k], list):
                        enriched[k].extend(v)
                    else:
                        enriched[k] = v
        idx = next_idx

    table_mappings: Dict[str, str] = {}
    column_mappings: Dict[str, str] = {}
    valid_terms = set()
    table_descriptions = {}
    column_descriptions = {}
    table_synonyms = {}
    column_synonyms = {}

    # enriched["tables"] debe contener common_terms y description para tablas
    for table, info in enriched.get("tables", {}).items():
        desc = info.get("description", "")
        synonyms = info.get("common_terms", [])
        table_descriptions[table] = desc
        table_synonyms[table] = synonyms
        for term in synonyms:
            table_mappings[term.lower()] = table
            valid_terms.add(term.lower())

    # enriched["columns"] debe contener common_terms y description para columnas
    for column, info in enriched.get("columns", {}).items():
        desc = info.get("description", "")
        synonyms = info.get("common_terms", [])
        column_descriptions[column] = desc
        column_synonyms[column] = synonyms
        for term in synonyms:
            column_mappings[term.lower()] = info  # Cambiado de 'column' a 'info'
            valid_terms.add(term.lower())

    return {
        "valid_terms": sorted(valid_terms),
        "table_mappings": table_mappings,
        "column_mappings": column_mappings,
        "table_descriptions": table_descriptions,
        "column_descriptions": column_descriptions,
        "table_synonyms": table_synonyms,
        "column_synonyms": column_synonyms
    }

def detect_table_from_question(question: str, terms_dict: Optional[Dict[str, Any]] = None, db_structure: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Detecta tablas relevantes para una pregunta utilizando LLM y heurísticas.
    
    Args:
        question: Pregunta del usuario
        terms_dict: Diccionario de términos (opcional)
        db_structure: Estructura de la base de datos
        
    Returns:
        Nombre de la tabla o None si no se encuentra ninguna
    """
    logging.debug(f"Detectando tabla para pregunta: {question}")
    
    if db_structure is not None and not isinstance(db_structure, dict):
        logging.warning("db_structure no es un diccionario, no se puede usar para detección de tablas")
        return None
    
    if not db_structure:
        logging.warning("No se proporcionó estructura de base de datos a detect_table_from_question")
        return None
        
    # Usar LLM para identificar la tabla más relevante
    system_message = (
        "Como experto en bases de datos médicas, tu tarea es identificar la tabla más adecuada "
        "para responder a una pregunta. Considera el contexto y el significado de los términos médicos.\\n\\n"
        "INSTRUCCIONES:\\n"
        "- Analiza la pregunta y determina qué recurso o entidad es el foco principal.\\n"
        "- Identifica la tabla más relevante de la lista proporcionada.\\n"
        "- Responde SOLAMENTE con el nombre de la tabla, sin explicaciones adicionales. Ejemplo: NOMBRETABLA\\n"
        "INSTRUCCIONES ADICIONALES PARA CONSULTAS DE MEDICAMENTOS:\\n"
        "- Si la pregunta se refiere a medicamentos 'aptos para niños', 'pediátricos' o similares, asegúrate de que la lógica SQL generada incluya condiciones para filtrar por descripciones que contengan 'niños' o 'pediátrico' Y EXCLUYAN 'adultos' en el campo de descripción del medicamento. Por ejemplo: (MEDI_DESCRIPTION_ES NOT LIKE '%adultos%' OR MEDI_DESCRIPTION_ES LIKE '%niños%' OR MEDI_DESCRIPTION_ES LIKE '%pediátrico%')\\n"
        "INSTRUCCIONES ADICIONALES PARA CONSULTAS DE DIAGNÓSTICOS Y DURACIÓN DE HOSPITALIZACIÓN:\\n"
        "- Para obtener la descripción textual OFICIAL de un diagnóstico, debes unir la tabla `EPIS_DIAGNOSTICS` (alias 'd') con `CODR_DIAGNOSTIC_GROUPS` (alias 'cg') usando `d.CDTE_ID = cg.DGGR_ID`. La descripción está en `cg.DGGR_DESCRIPTION_ES`.\\n"
        "- El campo `EPIS_DIAGNOSTICS.DIAG_OTHER_DIAGNOSTIC` es un texto libre y NO debe usarse para la descripción oficial del diagnóstico principal.\\n"
        "- Para diagnósticos principales, filtra siempre usando `d.DIAG_MAIN = 1`.\\n"
        "- Cuando calcules duraciones de hospitalización (por ejemplo, usando `EPIS_EPISODES.EPIS_START_DATE` y `EPIS_EPISODES.EPIS_CLOSED_DATE`), asegúrate de filtrar los episodios donde `EPIS_CLOSED_DATE` no sea nulo (`e.EPIS_CLOSED_DATE IS NOT NULL`).\\n"
        "- ERROR COMÚN A EVITAR: Al calcular promedios de duración, la forma correcta es `AVG(julianday(e.EPIS_CLOSED_DATE) - julianday(e.EPIS_START_DATE))`. NO anides funciones AVG como `AVG(... - AVG(...))`.\\n"
        "- Si necesitas agrupar resultados basados en una expresión CASE (por ejemplo, para categorizar diagnósticos como 'Cardiaca' o 'Neumonía' basados en `cg.DGGR_DESCRIPTION_ES`), DEBES REPETIR la expresión CASE completa en la cláusula `GROUP BY`. NO uses el alias de la columna del SELECT en el GROUP BY para SQLite. Ejemplo:\\n"
        "  SELECT CASE WHEN cg.DGGR_DESCRIPTION_ES LIKE '%cardiaca%' THEN 'Cardiaca' ELSE 'Otro' END AS tipo_diag, COUNT(*) \\n"
        "  FROM EPIS_DIAGNOSTICS d JOIN CODR_DIAGNOSTIC_GROUPS cg ON d.CDTE_ID = cg.DGGR_ID \\n"
        "  GROUP BY CASE WHEN cg.DGGR_DESCRIPTION_ES LIKE '%cardiaca%' THEN 'Cardiaca' ELSE 'Otro' END;\\n"
    )
    
    tables_list_str = "\\n".join([f"- {table_name}: {list(info.get('columns', {}).keys())[:3]}" # Mostrar menos columnas para brevedad
                               for table_name, info in db_structure.items()])
    
    # Añadir información de terms_dict al prompt si está disponible
    term_hints = ""
    if terms_dict and terms_dict.get("table_mappings"):
        # Tomar algunos ejemplos de mapeos de términos a tablas
        hint_mappings = {k: v for i, (k, v) in enumerate(terms_dict["table_mappings"].items()) if i < 5} # Limitar a 5 hints
        if hint_mappings:
            term_hints = "\nConsidera también estos términos y sus tablas asociadas:\n"
            for term, table_name in hint_mappings.items():
                term_hints += f"- El término '{term}' suele referirse a la tabla '{table_name}'.\n"

    user_message = (
        f"Pregunta del usuario: {question}\n\n"
        f"Tablas disponibles en la base de datos (nombre_tabla: [columnas_ejemplo]):\n{tables_list_str}\n"
        f"{term_hints}\n"
        "¿Cuál es la tabla más relevante para responder esta pregunta? Responde solo con el nombre de la tabla."
    )
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    try:
        config = {"temperature": 0.0, "max_tokens": 50} # temperature 0 para más determinismo
        response_text = call_llm(messages, config, step_name="Detección de tabla por LLM")
        
        # Limpieza más robusta de la respuesta
        # Intentar extraer la primera palabra que parezca un nombre de tabla (mayúsculas, guiones bajos)
        match = re.search(r'\b([A-Z0-9_]+)\b', response_text)
        detected_table = None
        if match:
            detected_table = match.group(1)
        else:
            # Si no hay un patrón claro, tomar la respuesta limpiada como antes
            detected_table = response_text.strip().strip('\"\'').split()[0] if response_text else ""


        if detected_table and detected_table in db_structure:
            logging.info(f"Tabla detectada por LLM: {detected_table}")
            return detected_table
        elif detected_table:
            logging.info(f"Tabla candidata por LLM '{detected_table}' no es exacta, intentando corrección.")
            matches = difflib.get_close_matches(detected_table, list(db_structure.keys()), n=1, cutoff=0.7) # cutoff un poco más alto
            if matches:
                logging.info(f"Tabla corregida por LLM: {matches[0]}")
                return matches[0]
            else:
                logging.warning(f"La tabla detectada por LLM '{detected_table}' no existe y no se pudo corregir.")
        else:
            logging.info("LLM no devolvió un nombre de tabla claro.")

    except Exception as e:
        logging.error(f"Error al detectar tabla con LLM: {e}")
    
    logging.info("LLM no pudo determinar la tabla o falló. Intentando fallback...")

    # Fallback mejorado:
    # 1. Usar terms_dict si está disponible
    if terms_dict and terms_dict.get("table_mappings"):
        question_lower = question.lower()
        # Buscar términos exactos del diccionario en la pregunta
        for term, table_name in terms_dict["table_mappings"].items():
            # Usar word boundaries para evitar coincidencias parciales dentro de otras palabras
            if re.search(r'\b' + re.escape(term.lower()) + r'\b', question_lower):
                if table_name in db_structure:
                    logging.info(f"Tabla detectada por fallback (terms_dict): {table_name} (término: {term})")
                    return table_name
                else:
                    logging.warning(f"Término '{term}' mapea a tabla '{table_name}' que no existe en db_structure.")
    
    # 2. Fallback por coincidencia de nombres de tabla completos en la pregunta
    question_lower_for_direct_match = question.lower()
    for table_name_key in db_structure:
        if f'\b{table_name_key.lower()}\b' in question_lower_for_direct_match:
            logging.info(f"Tabla detectada por fallback (nombre completo en pregunta): {table_name_key}")
            return table_name_key

    # 3. Fallback con FuzzyWuzzy o Difflib para palabras individuales
    question_words = set(re.findall(r'\b\w{4,}\b', question.lower())) # Palabras de al menos 4 caracteres
    
    best_match_table = None
    highest_score = 0

    if FUZZY_AVAILABLE:
        for table_name_key in db_structure:
            score = 0
            # Puntuación base por similitud del nombre de la tabla con la pregunta completa
            score += fuzz.partial_ratio(question_lower, table_name_key.lower())
            
            # Bonus por cada palabra clave de la pregunta que esté en el nombre de la tabla
            # o sea muy similar a él.
            table_name_parts = set(table_name_key.lower().split('_'))
            for word in question_words:
                if word in table_name_parts:
                    score += 30 # Bonus fuerte por coincidencia exacta de parte del nombre
                else:
                    # Bonus más pequeño por similitud de palabra con el nombre completo de la tabla
                    word_to_table_similarity = fuzz.ratio(word, table_name_key.lower())
                    if word_to_table_similarity > 70: # Si la palabra es bastante similar al nombre de la tabla
                        score += word_to_table_similarity / 5 # Bonus proporcional

            if score > highest_score:
                highest_score = score
                best_match_table = table_name_key
        
        # Ajustar el umbral según la escala de puntuación revisada
        if best_match_table and highest_score > 90: # Umbral ajustado para fuzzy (puede necesitar calibración)
            logging.info(f"Tabla detectada por fallback (fuzzywuzzy): {best_match_table} (score: {highest_score})")
            return best_match_table
    else:
        # Fallback con difflib si fuzzywuzzy no está disponible (menos granular)
        # Este enfoque es más simple: busca la mejor coincidencia para cada palabra de la pregunta
        # contra la lista de nombres de tabla.
        # Podríamos intentar sumar "confianzas" o contar "votos" si varias palabras apuntan a la misma tabla.
        # Por simplicidad, mantenemos la lógica de devolver la primera buena coincidencia.
        candidate_tables = {}
        for word in question_words:
            matches = difflib.get_close_matches(word, list(db_structure.keys()), n=1, cutoff=0.8) # cutoff alto para palabra individual
            if matches:
                # Contar cuántas palabras apuntan a esta tabla
                candidate_tables[matches[0]] = candidate_tables.get(matches[0], 0) + 1
        
        if candidate_tables:
            # Elegir la tabla con más "votos" (más palabras clave apuntando a ella)
            best_fallback_table = max(candidate_tables, key=candidate_tables.get)
            logging.info(f"Tabla detectada por fallback (difflib - más votada): {best_fallback_table}")
            return best_fallback_table

    logging.warning("Fallback no pudo detectar ninguna tabla relevante.")
    return None

def whitelist_validate_query(query: str, allowed_tables: Set[str], allowed_columns: Dict[str, List[str]]) -> bool:
    """
    Valida que una consulta SQL solo use tablas y columnas permitidas en la whitelist.
    
    Args:
        query: Consulta SQL a validar
        allowed_tables: Conjunto de nombres de tablas permitidas
        allowed_columns: Diccionario con tablas como claves y listas de columnas permitidas como valores
        
    Returns:
        True si la consulta solo usa tablas y columnas permitidas
    """
    query_lower = query.lower()
    
    # Extraer tablas mencionadas en la consulta
    tables_pattern = r'\bfrom\s+([a-z0-9_]+)|join\s+([a-z0-9_]+)'
    found_tables = []
    for match in re.finditer(tables_pattern, query_lower):
        table = match.group(1) or match.group(2)
        if table and table not in found_tables:
            found_tables.append(table)
    
    # Verificar que todas las tablas estén en la whitelist
    for table in found_tables:
        if table not in [t.lower() for t in allowed_tables]:
            logging.error(f"Tabla no permitida en la consulta: {table}")
            return False
    
    # TODO: Si necesitamos validar columnas, tendríamos que extraerlas de la consulta
    # y verificar que estén en allowed_columns para sus respectivas tablas
    
    return True

def _get_schema_tables(schema):
    # Si es lista, es formato clásico de schema_enhanced.json
    if isinstance(schema, list):
        return schema
    # Si es dict, buscar clave 'tables' (como en dictionary.json enriquecido)
    if isinstance(schema, dict):
        # Puede ser dict con 'tables' como dict de tablas
        if "tables" in schema and isinstance(schema["tables"], dict):
            return [
                {"table_name": k, **v}
                for k, v in schema["tables"].items()
            ]
        # O dict con 'tables' como lista
        if "tables" in schema and isinstance(schema["tables"], list):
            return schema["tables"]
        # Nuevo: formato plano schema_simple.json, claves de nivel superior son tablas
        simple_tables = []
        all_plain = True
        for k, v in schema.items():
            if isinstance(v, dict) and 'columns' in v:
                simple_tables.append({"table_name": k, **v})
            else:
                all_plain = False
                break
        if all_plain and simple_tables:
            return simple_tables
    raise ValueError("Formato de esquema no soportado para extracción de tablas.")

def list_tables(schema_path:str) -> list[str]:
    """
    Devuelve la lista de nombres de tablas reales a partir de un archivo de esquema JSON.
    """
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
    tables = _get_schema_tables(schema)
    return [t["table_name"] for t in tables]

def list_columns(schema_path:str, table_name:str) -> list[str]:
    """
    Devuelve la lista de columnas reales de una tabla a partir de un archivo de esquema JSON.
    Soporta tanto formato clásico (lista/anidado) como formato dictionary.json (dict plano de columnas).
    Añade logs detallados para depuración robusta.
    """
    import logging
    logger = logging.getLogger()
    
    # Limpiar el nombre de la tabla de entrada
    original_table_name_for_log = table_name
    table_name = table_name.strip().strip('"\'') # Eliminar espacios y comillas comunes
    if original_table_name_for_log != table_name:
        logger.debug(f"[list_columns] Nombre de tabla limpiado de '{original_table_name_for_log}' a '{table_name}'")
    else:
        logger.debug(f"[list_columns] Buscando columnas para tabla: '{table_name}' en schema: {schema_path}")

    try:
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)
    except Exception as e:
        logger.error(f"[list_columns] Error al cargar el esquema: {e}")
        return []

    # 1. Intentar formato clásico (schema_enhanced.json)
    try:
        tables = _get_schema_tables(schema)
        logger.debug(f"[list_columns] _get_schema_tables encontró {len(tables)} tablas")
        for t in tables:
            if t.get("table_name", "").lower() == table_name.lower():
                cols = t.get("columns", [])
                logger.debug(f"[list_columns] Tabla encontrada en formato clásico: {t['table_name']}, columnas: {cols}")
                if isinstance(cols, dict):
                    logger.info(f"[list_columns] Formato clásico: columnas como dict para {table_name}")
                    return list(cols.keys())
                if isinstance(cols, list) and cols and isinstance(cols[0], dict) and "name" in cols[0]:
                    logger.info(f"[list_columns] Formato clásico: columnas como lista de dicts para {table_name}")
                    return [c["name"] for c in cols]
                if isinstance(cols, list) and cols and isinstance(cols[0], str):
                    logger.info(f"[list_columns] Formato clásico: columnas como lista de strings para {table_name}")
                    return cols
                logger.warning(f"[list_columns] Formato clásico: columnas no reconocidas para {table_name}: {cols}")
    except Exception as e:
        logger.warning(f"[list_columns] Excepción en formato clásico: {e}")

    # 2. Intentar formato dictionary.json (dict plano de columnas)
    if isinstance(schema, dict) and "columns" in schema and isinstance(schema["columns"], dict):
        prefix_upper = table_name.upper() + "."
        prefix_lower = table_name.lower() + "."
        found_keys = list(schema["columns"].keys())
        logger.debug(f"[list_columns] Revisando {len(found_keys)} claves en schema['columns']")
        colnames = [k.split(".",1)[1] for k in found_keys if k.upper().startswith(prefix_upper)]
        if not colnames:
            colnames = [k.split(".",1)[1] for k in found_keys if k.lower().startswith(prefix_lower)]
        logger.info(f"[list_columns] Formato plano: columnas encontradas para {table_name}: {colnames}")
        return colnames
    else:
        logger.warning(f"[list_columns] El esquema no tiene clave 'columns' o no es un dict plano")

    # 3. Fallback: buscar columnas por coincidencia parcial (muy robusto, solo para depuración)
    if isinstance(schema, dict) and "columns" in schema and isinstance(schema["columns"], dict):
        found_keys = list(schema["columns"].keys())
        colnames = [k for k in found_keys if table_name.lower() in k.lower()]
        logger.info(f"[list_columns] Fallback parcial: columnas que contienen '{table_name}': {colnames}")
        return [k.split(".",1)[1] for k in colnames if "." in k]

    logger.error(f"[list_columns] No se encontraron columnas para la tabla '{table_name}' en ningún formato")
    return []

def search_schema(schema_path:str, keyword:str) -> dict:
    """
    Busca tablas y columnas que contengan el keyword (en nombre o descripción).
    Devuelve un dict con coincidencias.
    """
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
    tables = _get_schema_tables(schema)
    result = {"tables":[], "columns":[]}
    keyword = keyword.lower()
    for t in tables:
        if keyword in t["table_name"].lower() or keyword in (t.get("description") or "").lower():
            result["tables"].append(t["table_name"])
        cols = t.get("columns", [])
        # Manejar formato de columnas como lista de strings (schema_simple.json)
        if isinstance(cols, list) and cols and isinstance(cols[0], str):
            for cname in cols:
                if keyword in cname.lower():
                    result["columns"].append({"table": t["table_name"], "column": cname, "description": ""})
            continue  # Ya procesadas las columnas simples
        if isinstance(cols, dict):
            for cname, cinfo in cols.items():
                if keyword in cname.lower() or keyword in (cinfo.get("description") or "").lower():
                    result["columns"].append({"table": t["table_name"], "column": cname, "description": cinfo.get("description","")})
        else:
            for c in cols:
                if keyword in c["name"].lower() or keyword in (c.get("description") or "").lower():
                    result["columns"].append({"table": t["table_name"], "column": c["name"], "description": c.get("description","")})
    return result

def _extract_select_aliases(stmt):
    """
    Extrae un mapeo de alias a sus expresiones originales de la cláusula SELECT.
    Devuelve: dict {alias_string: expression_string}
    """
    aliases_map = {}
    if not stmt or not hasattr(stmt, 'tokens'):
        return aliases_map

    select_elements_token = None
    in_select_clause = False
    
    # Buscar el inicio de la cláusula SELECT y luego los identificadores
    # Esto es una simplificación; sqlparse puede ser complejo para casos borde.
    
    # Primero, encontrar el token SELECT
    select_keyword_idx = -1
    for i, token in enumerate(stmt.tokens):
        if token.is_keyword and token.normalized == 'SELECT':
            select_keyword_idx = i
            break
    
    if select_keyword_idx == -1:
        return aliases_map # No hay SELECT

    # Buscar el primer Identifier o IdentifierList después de SELECT
    # que no esté dentro de subconsultas (esto es una simplificación)
    for i in range(select_keyword_idx + 1, len(stmt.tokens)):
        token = stmt.tokens[i]
        if token.is_whitespace:
            continue
        if isinstance(token, (IdentifierList, Identifier)):
            select_elements_token = token
            break
        # Si encontramos otra palabra clave principal antes de un Identifier/IdentifierList,
        # entonces la lista de selección podría estar vacía o ser algo inesperado.
        if token.is_keyword and token.normalized in ('FROM', 'WHERE', 'GROUP', 'ORDER', 'LIMIT'):
            break
            
    if not select_elements_token:
        return aliases_map

    identifiers = []
    if isinstance(select_elements_token, IdentifierList):
        identifiers = [t for t in select_elements_token.get_identifiers()]
    elif isinstance(select_elements_token, Identifier):
        identifiers = [select_elements_token]

    for ident in identifiers:
        alias = ident.get_alias()
        if alias:
            # Intentar obtener la expresión original antes de 'AS alias' o ' alias'
            # sqlparse a veces incluye 'AS' en el token anterior, a veces no.
            # ident.tokens usually es [expresion, whitespace, AS, whitespace, alias_token]
            # o [expresion, whitespace, alias_token]
            
            expression_tokens = []
            found_alias_or_as = False
            for t_idx in range(len(ident.tokens) -1, -1, -1):
                tok = ident.tokens[t_idx]
                if tok.is_keyword and tok.normalized == 'AS':
                    found_alias_or_as = True
                    # Los tokens anteriores a AS (y el espacio antes de AS) son la expresión
                    expression_tokens = ident.tokens[:t_idx-1] if t_idx > 0 and ident.tokens[t_idx-1].is_whitespace else ident.tokens[:t_idx]
                    break
                # Comprobar si el token es el alias mismo
                # Puede ser un Name o un Identifier simple
                if tok.to_unicode().strip() == alias:
                    found_alias_or_as = True
                    # Los tokens anteriores al alias (y el espacio antes del alias) son la expresión
                    expression_tokens = ident.tokens[:t_idx-1] if t_idx > 0 and ident.tokens[t_idx-1].is_whitespace else ident.tokens[:t_idx]
                    break
            
            if found_alias_or_as and expression_tokens:
                expression_str = "".join(t.to_unicode() for t in expression_tokens).strip()
                aliases_map[alias] = expression_str
            elif not found_alias_or_as: # No 'AS' y el último token no es el alias? Podría ser un alias implícito sin espacio
                 # Si el alias es el último token y no hay 'AS'
                if ident.tokens and ident.tokens[-1].to_unicode().strip() == alias:
                    # Tomar todos los tokens excepto el último (y el espacio anterior si existe)
                    num_tokens_for_alias_part = 1
                    if len(ident.tokens) > 1 and ident.tokens[-2].is_whitespace:
                        num_tokens_for_alias_part = 2
                    
                    expression_tokens = ident.tokens[:-num_tokens_for_alias_part]
                    if expression_tokens:
                        expression_str = "".join(t.to_unicode() for t in expression_tokens).strip()
                        aliases_map[alias] = expression_str
                    else: # Caso: SELECT mi_columna AS alias -> ident.tokens puede ser solo [mi_columna, AS, alias] o [mi_columna, alias]
                          # Si expression_tokens está vacío, significa que la "expresión" es el primer token antes del alias/AS.
                          # Esto es cubierto por get_name() o el inicio de ident.
                        first_part = ident.get_name() # Intenta obtener la parte principal antes del alias
                        if first_part and first_part != alias : # Asegurarse que no es el alias mismo
                             aliases_map[alias] = first_part
                        else:
                             logger.debug(f"No se pudo extraer la expresión para el alias implícito \'{alias}\' de forma simple a partir de \'{ident.to_unicode()}\'. Tokens: {[t.to_unicode() for t in ident.tokens]}")
                else:
                    logger.debug(f"No se pudo determinar la expresión para el alias \'{alias}\' en \'{ident.to_unicode()}\'. Tokens: {[t.to_unicode() for t in ident.tokens]}")

            else: # Fallback o log si la extracción no fue clara
                logger.debug(f"No se pudo extraer la expresión para el alias \'{alias}\' de forma clara a partir de \'{ident.to_unicode()}\'. Tokens: {[t.to_unicode() for t in ident.tokens]}")
                # Como fallback muy simple, si la expresión no se pudo aislar,
                # se podría intentar tomar todo hasta el alias, pero es arriesgado.
                # Por ahora, si no se extrae limpiamente, se omite.

    return aliases_map

def _replace_aliases_in_group_by_elements(group_by_elements, aliases_map):
    """
    Reemplaza alias en una lista de elementos de GROUP BY (que son Identifiers).
    Devuelve una lista de cadenas (expresiones o nombres de columna) y un flag si hubo modificación.
    """
    new_elements_str_list = []
    modified_in_clause = False
    
    for item_token in group_by_elements: 
        element_identifier_str = item_token.to_unicode().strip()

        if element_identifier_str in aliases_map:
            expression = aliases_map[element_identifier_str]
            # Envolver expresiones en paréntesis para seguridad en GROUP BY, especialmente si son complejas.
            # No envolver si ya es un identificador simple o ya está entre paréntesis.
            is_simple_identifier = expression.isidentifier() and not (expression.upper() in ['CASE', 'WHEN', 'THEN', 'ELSE', 'END'])
            is_already_parenthesized = expression.startswith('(') and expression.endswith(')')

            if not is_simple_identifier and not is_already_parenthesized:
                 # Evitar doble paréntesis si la expresión original ya los tenía (ej. (col1 + col2) )
                 # Esto es difícil de determinar perfectamente sin un parseo más profundo de la expresión misma.
                 # Una heurística simple: si es un CASE, no añadir paréntesis extra.
                if expression.upper().startswith("CASE"):
                    new_elements_str_list.append(expression)
                else:
                    new_elements_str_list.append(f"({expression})")
            else:
                new_elements_str_list.append(expression)
            modified_in_clause = True
        else:
            new_elements_str_list.append(element_identifier_str)
            
    return new_elements_str_list, modified_in_clause

def fix_sqlite_group_by_aliases(sql_query):
    """
    Reescribe una consulta SQL para reemplazar alias en cláusulas GROUP BY
    con sus expresiones originales, para compatibilidad con SQLite.
    """
    try:
        parsed_statements = sqlparse.parse(sql_query)
        if not parsed_statements:
            return sql_query # No se pudo parsear
        
        stmt = parsed_statements[0] # Asumir una sola declaración por simplicidad
        
        # Primero, extraer todos los alias de la cláusula SELECT
        aliases_map = _extract_select_aliases(stmt)

        if not aliases_map:
            return sql_query # No hay alias o no se pudieron extraer

        overall_modified = False
        
        reconstructed_tokens = []
        idx = 0
        while idx < len(stmt.tokens):
            token = stmt.tokens[idx]

            if token.is_keyword and token.normalized == 'GROUP BY':
                reconstructed_tokens.append(token) # Añadir "GROUP BY"
                idx += 1 

                # Manejar espacio después de GROUP BY
                if idx < len(stmt.tokens) and stmt.tokens[idx].is_whitespace:
                    reconstructed_tokens.append(stmt.tokens[idx])
                    idx += 1
                
                if idx < len(stmt.tokens):
                    group_by_content_token = stmt.tokens[idx]
                    elements_to_group_by = []

                    # El contenido de GROUP BY puede ser un IdentifierList (múltiples columnas)
                    # o un solo Identifier (una columna/expresión)
                    if isinstance(group_by_content_token, IdentifierList):
                        # get_identifiers() devuelve los elementos individuales
                        elements_to_group_by = list(group_by_content_token.get_identifiers())
                    elif isinstance(group_by_content_token, Identifier): 
                        elements_to_group_by = [group_by_content_token]
                    # Podría haber otros casos, como funciones, etc., que sqlparse trata como Identifier.
                    
                    if elements_to_group_by:
                        new_group_by_strs, modified_this_clause = _replace_aliases_in_group_by_elements(elements_to_group_by, aliases_map)
                        if modified_this_clause:
                            overall_modified = True
                            # Unir los elementos con comas y crear un nuevo token para ellos
                            # Esto es una simplificación; idealmente, se preservarían los tokens originales
                            # de coma y espacios si no hay modificaciones.
                            reconstructed_tokens.append(sqlparse.sql.Token(sqlparse.tokens.Text, ", ".join(new_group_by_strs)))
                        else: 
                            reconstructed_tokens.append(group_by_content_token)
                idx += 1 
                continue 

            # TODO: Implementar manejo para la cláusula HAVING si es necesario.
            # La lógica sería similar: identificar expresiones en HAVING,
            # y si usan un alias, reemplazarlo.

            reconstructed_tokens.append(token)
            idx += 1
        
        if overall_modified:
            new_query = "".join(t.to_unicode() for t in reconstructed_tokens)
            logger.info(f"Consulta SQL modificada para compatibilidad de alias en GROUP BY (SQLite): {new_query}")
            return new_query
        else:
            return sql_query

    except Exception as e:
        logger.error(f"Error al intentar corregir alias en SQL para SQLite: {e}", exc_info=True)
        return sql_query # Devolver original en caso de error

# Se recomienda que la función que valida y sanea el SQL en sql_validator.py
# llame a fix_sqlite_group_by_aliases. Por ejemplo:
#
# En sql_validator.py (o donde se encuentre SQLValidator.validate_and_sanitize_sql):
#
# from . import sql_utils # o la importación correcta
#
# class SQLValidator:
#     def validate_and_sanitize_sql(self, sql_query, allowed_tables_columns_map, db_structure=None):
#         # ... (otro código de validación) ...
#
#         # Aplicar corrección de alias para SQLite ANTES de otras validaciones de entidades si es posible
#         # o justo antes de la ejecución si la validación de entidades necesita la forma original.
#         # Para el problema actual, la corrección debe aplicarse a la consulta que se va a ejecutar.
#         
#         corrected_sql_query = sql_utils.fix_sqlite_group_by_aliases(sql_query)
#
#         # Continuar validaciones con corrected_sql_query
#         # Por ejemplo, la advertencia de sql_utils.validate_sql_entities sobre alias en GROUP BY
#         # ya no debería aparecer si la corrección fue exitosa.
#         # ...
#         # return corrected_sql_query
#
# Nota: La integración exacta depende de la estructura de sql_validator.py.
# La función validate_sql_entities que emite la advertencia también podría ser un lugar
# para invocar esta corrección o usar su lógica de detección de alias.

import re # Asegúrate de que re está importado al principio del archivo

def _rewrite_group_by_alias_for_sqlite(sql_query: str) -> str:
    """
    Reescribe la cláusula GROUP BY si usa un alias de una expresión CASE en el SELECT,
    reemplazando el alias por la expresión CASE completa. Específico para SQLite.
    Devuelve la consulta modificada o la original si no se aplica el patrón.
    """
    try:
        # Patrón mejorado para capturar la expresión CASE y su alias, y el uso del alias en GROUP BY
        # Busca: SELECT ... CASE ... END AS alias ... GROUP BY alias
        # Es sensible a comentarios y otras cláusulas, por lo que se mantiene relativamente simple.
        # Limitación: asume que el alias es una palabra simple.
        pattern = re.compile(
            r"(SELECT\\s+.*?CASE\\s+.*?\\s+END\\s+AS\\s+([a-zA-Z_][a-zA-Z0-9_]*).*?)(GROUP\\s+BY\\s+\\2)",
            re.IGNORECASE | re.DOTALL
        )
        match = pattern.search(sql_query)

        if match:
            select_part_with_case = match.group(1) # Parte del SELECT hasta el alias
            alias_name = match.group(2)
            group_by_clause_to_replace = match.group(3) # "GROUP BY alias_name"

            # Extraer la expresión CASE completa
            case_expression_match = re.search(
                r"CASE\\s+.*?\\s+END",
                select_part_with_case,
                re.IGNORECASE | re.DOTALL
            )
            if case_expression_match:
                case_expression = case_expression_match.group(0)
                
                # Reemplazar "GROUP BY alias" con "GROUP BY [expresión CASE completa]"
                # Se usa un marcador temporal para evitar problemas si el alias aparece en otro lugar.
                placeholder = f"__GROUP_BY_ALIAS_PLACEHOLDER_FOR_{alias_name}__"
                temp_sql = sql_query.replace(group_by_clause_to_replace, placeholder)
                corrected_sql = temp_sql.replace(placeholder, f"GROUP BY {case_expression}")

                if corrected_sql != sql_query:
                    # logger.info(f"Corrección de GROUP BY con alias para SQLite aplicada. Alias: {alias_name}")
                    # print(f"INFO (sql_utils_rewrite): Corrección de GROUP BY con alias para SQLite aplicada. Alias: {alias_name}") # Para debug
                    return corrected_sql
    except Exception as e:
        # logger.error(f"Error al intentar reescribir GROUP BY con alias: {e}")
        # print(f"ERROR (sql_utils_rewrite): Error al intentar reescribir GROUP BY con alias: {e}") # Para debug
        pass # Si falla, devuelve la consulta original
    return sql_query

# ... más abajo, donde se procesa o valida la consulta antes de la ejecución ...
# Ejemplo de cómo podría integrarse (esto es conceptual):
#
# if is_direct_sql_query:
#     # ...
#     prepared_sql = sql_utils.prepare_sql_for_execution(direct_sql_query, db_type="sqlite")
#     results = db_connector.execute_query(prepared_sql, ...)
#     # ...
# --- FIN DE LA SECCIÓN A MODIFICAR ---

# ... resto del archivo sql_utils.py ...

