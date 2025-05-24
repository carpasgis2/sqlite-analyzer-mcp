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
    

def validate_sql_entities(sql_query: str, db_structure: Dict[str, Any]) -> bool:
    """
    Verifica que todas las tablas y columnas de la consulta existan.
    """
    import re, difflib
    # extraer tablas
    tables = set(t for m in re.findall(r'\bFROM\s+([A-Za-z0-9_]+)|\bJOIN\s+([A-Za-z0-9_]+)', sql_query)
                 for t in m if t)
    real_tables = {}
    for tbl in tables:
        if tbl not in db_structure:
            matchs = difflib.get_close_matches(tbl, db_structure.keys(), n=1, cutoff=0.6)
            if not matchs:
                return False
            real_tables[tbl] = matchs[0]
        else:
            real_tables[tbl] = tbl
    # extraer columnas alias.tabla o tabla.col
    cols = re.findall(r'([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)', sql_query)
    for tbl, col in cols:
        rt = real_tables.get(tbl)
        if not rt or col not in {c['name'] for c in db_structure[rt]['columns']}:
            return False
    return True

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
        "para responder a una pregunta. Considera el contexto y el significado de los términos médicos.\n\n"
        "INSTRUCCIONES:\n"
        "- Analiza la pregunta y determina qué recurso o entidad es el foco principal.\n"
        "- Identifica la tabla más relevante de la lista proporcionada.\n"
        "- Responde SOLAMENTE con el nombre de la tabla, sin explicaciones adicionales. Ejemplo: NOMBRETABLA\n"
    )
    
    tables_list_str = "\n".join([f"- {table_name}: {list(info.get('columns', {}).keys())[:3]}" # Mostrar menos columnas para brevedad
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

