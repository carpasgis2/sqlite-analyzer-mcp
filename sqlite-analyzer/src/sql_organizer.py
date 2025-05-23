"""
Módulo para organizar, validar y normalizar estructuras de consultas SQL.
Este módulo ayuda a preparar la información JSON antes de generar consultas SQL,
asegurando que todos los campos tengan el formato correcto.
"""

import logging
import copy
import json
import re
from typing import Dict, Any, List, Union, Optional, Tuple

# Estructura esperada para la información de consulta SQL
SQL_STRUCTURE_TEMPLATE = {
    "original_question": "",     # Pregunta original del usuario
    "query_type": "select",      # Tipo de consulta: select, count, avg, max, min
    "tables": [],                # Lista de tablas para consultar
    "columns": [],               # Lista de columnas para seleccionar
    "conditions": [],            # Lista de condiciones para WHERE
    "joins": [],                 # Lista de joins entre tablas
    "group_by": [],              # Lista de columnas para GROUP BY
    "having": [],                # Lista de condiciones para HAVING
    "order_by": [],              # Lista de ordenamientos
    "limit": None,               # Valor límite para resultados
    "distinct": False            # Si se aplica DISTINCT
}

def parse_join_string(join_str: str) -> dict:
    """
    Convierte un JOIN en formato string (TABLA1.COLUMNA1 = TABLA2.COLUMNA2) al formato
    de diccionario requerido por el sistema.
    
    Args:
        join_str: String con formato "tabla1.columna1 = tabla2.columna2"
        
    Returns:
        Diccionario con formato {table, column, foreign_table, foreign_column} o None si no se puede parsear
    """
    try:
        # Dividir por el signo igual
        parts = join_str.strip().split('=')
        if len(parts) != 2:
            return None
            
        # Limpiar espacios en blanco
        left = parts[0].strip()
        right = parts[1].strip()
        
        # Extraer tabla y columna de cada lado
        if '.' in left and '.' in right:
            left_parts = left.split('.')
            right_parts = right.split('.')
            
            if len(left_parts) == 2 and len(right_parts) == 2:
                return {
                    "table": left_parts[0],
                    "column": left_parts[1],
                    "foreign_table": right_parts[0],
                    "foreign_column": right_parts[1]
                }
    except Exception as e:
        logging.warning(f"Error al parsear JOIN string: {e}")
    
    return None


def normalize_structured_info(info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normaliza la estructura de información para una consulta SQL.
    
    Args:
        info: Diccionario con información para generar SQL
        
    Returns:
        Diccionario normalizado con todos los campos requeridos
    """
    # Crear copia para no modificar original
    normalized = copy.deepcopy(info)
    
    # Asegurarse de que existan todos los campos necesarios
    for key, default_value in SQL_STRUCTURE_TEMPLATE.items():
        if key not in normalized:
            normalized[key] = copy.deepcopy(default_value)
    
    # Corregir tipos de datos - campos que deben ser listas
    list_fields = ["tables", "columns", "conditions", "joins", "group_by", "having", "order_by"]
    for field in list_fields:
        # Si el campo existe pero no es una lista
        if field in normalized:
            if normalized[field] is None:
                normalized[field] = []
            elif isinstance(normalized[field], str):
                # Intentar parsear como JSON si parece un JSON
                try:
                    if normalized[field].strip().startswith(('[', '{')):
                        parsed = json.loads(normalized[field])
                        if isinstance(parsed, list):
                            normalized[field] = parsed
                        else:
                            normalized[field] = [parsed]
                    else:
                        normalized[field] = [normalized[field]]
                except:
                    normalized[field] = [normalized[field]]
                logging.info(f"Campo '{field}' convertido de string a lista: {normalized[field]}")
            elif not isinstance(normalized[field], list):
                logging.info(f"Campo '{field}' tiene tipo incorrecto {type(normalized[field])}, convirtiendo a lista")
                try:
                    # Intentar convertir a lista si es posible
                    normalized[field] = list(normalized[field])
                except:
                    normalized[field] = []
    
    # Normalizar conditions: asegurar que sea una lista de diccionarios
    if "conditions" in normalized and normalized["conditions"]:
        if isinstance(normalized["conditions"], dict):
            normalized["conditions"] = [normalized["conditions"]]
        
        # Validar cada condición individual
        validated_conditions = []
        for condition in normalized["conditions"]:
            if isinstance(condition, str):
                try:
                    # Intentar parsear como JSON
                    condition_dict = json.loads(condition)
                    validated_conditions.append(condition_dict)
                except:
                    # Si falla, intentar crear una condición simple
                    validated_conditions.append({"column": "id", "operator": "=", "value": condition})
            elif isinstance(condition, dict):
                validated_conditions.append(condition)
            # Ignorar otros tipos
        
        normalized["conditions"] = validated_conditions
    
    # Normalizar joins: asegurar que sea una lista de diccionarios válidos
    if "joins" in normalized:
        # Normalizar a lista si es necesario
        if normalized["joins"] is None:
            normalized["joins"] = []
        elif isinstance(normalized["joins"], str):
            normalized["joins"] = [normalized["joins"]] if normalized["joins"].strip() else []
        elif isinstance(normalized["joins"], dict):
            normalized["joins"] = [normalized["joins"]]
        # Convertir cada join de string a diccionario ANTES de cualquier otro procesamiento
        for i in range(len(normalized["joins"])):
            join = normalized["joins"][i]
            if isinstance(join, str):
                # Si el string ya es una cláusula JOIN completa, añadirlo directamente.
                # Comprobación simple: contiene "JOIN" y "ON" (con espacios para evitar falsos positivos en nombres de columna/tabla)
                # y no empieza con '{' (para evitar confundirlo con un JSON string).
                if " JOIN " in join.upper() and \
                   " ON " in join.upper() and \
                   not join.strip().startswith("{"):
                    logging.info(f"JOIN string completo añadido directamente sin parsear: {join}")
                    continue  # Procesar el siguiente join

                # Si no es un JOIN string completo, intentar parsearlo como JSON o estructura simple
                try:
                    # Intentar interpretar como JSON si es un string que parece un objeto o array JSON
                    if join.strip().startswith("{") or join.strip().startswith("["):
                        parsed_dict_from_str = json.loads(join)
                        # Si es un diccionario, añadirlo para procesamiento posterior como dict
                        if isinstance(parsed_dict_from_str, dict):
                            normalized["joins"][i] = parsed_dict_from_str  # Se tratará como dict más adelante
                            logging.debug(f"JOIN string parseado como JSON a dict: {parsed_dict_from_str}")
                            continue
                        # Si es una lista (de dicts o strings), extender normalized["joins"]
                        elif isinstance(parsed_dict_from_str, list):
                            normalized["joins"].extend(parsed_dict_from_str)  # Extender con elementos de la lista
                            logging.debug(f"JOIN string parseado como JSON a list, elementos añadidos: {parsed_dict_from_str}")
                            continue
                    else:
                        # No parece JSON, intentar parse_join_string para estructura simple "T1.c1 = T2.c2" o "JOIN T2 ON T1.c1 = T2.c2"
                        pass  # Continuar para parse_join_string más abajo

                except json.JSONDecodeError:
                    logging.debug(f"JOIN string no es JSON: '{join}'. Se intentará parse_join_string.")
                    # Continuar para parse_join_string

                # Intentar parsear el string a un diccionario estructurado con parse_join_string
                # Esto maneja formatos como "TABLE_A.COLUMN_A = TABLE_B.COLUMN_B" o "JOIN TABLE_B ON TABLE_A.COLUMN_A = TABLE_B.COLUMN_B"
                parsed_join_dict = parse_join_string(join)
                if parsed_join_dict:
                    normalized["joins"][i] = parsed_join_dict
                    logging.info(f"JOIN string '{join}' parseado a dict: {parsed_join_dict}")
                else:
                    # Si parse_join_string falla y no fue reconocido como JOIN completo arriba.
                    logging.warning(f"Detalle de JOIN string no reconocido y omitido: {join}")

            elif isinstance(join, dict):
                # Si ya es un diccionario, verificar que tenga el formato correcto
                if not all(k in join for k in ["table", "column", "foreign_table", "foreign_column"]):
                    normalized_join = normalize_join_format(join)
                    if normalized_join:
                        normalized["joins"][i] = normalized_join
                        logging.info(f"JOIN en formato diccionario normalizado: {normalized_join}")
                    else:
                        logging.warning(f"JOIN en formato inválido y no se pudo normalizar: {join}")

    # Normalizar actions a query_type si existe
    if "actions" in normalized:
        actions = normalized.pop("actions")
        if isinstance(actions, list) and actions:
            action = actions[0].lower() if isinstance(actions[0], str) else "select"
            if action in ["count", "avg", "max", "min", "sum"]:
                normalized["query_type"] = action
                logging.info(f"Acción {action} establecida como query_type")
    
    return normalized

def enhance_structured_info(info: Dict[str, Any], question: str) -> Dict[str, Any]:
    """
    Enriquece la información estructurada con detalles adicionales basados en la pregunta.
    
    Args:
        info: Información estructurada de la consulta
        question: Pregunta original del usuario
        
    Returns:
        Información estructurada enriquecida
    """
    enhanced = copy.deepcopy(info)
    
    # Asegurar que la pregunta original esté guardada
    if "original_question" not in enhanced or not enhanced["original_question"]:
        enhanced["original_question"] = question
        
    # Importación de call_llm movida aquí para asegurar disponibilidad
    try:
        from .llm_utils import call_llm
    except ImportError:
        # Manejar el caso donde la importación relativa falla (ej. ejecución directa del script)
        from llm_utils import call_llm

    # Detectar tipo de consulta si no está especificado
    if "query_type" not in enhanced or enhanced["query_type"] == "select":
        query_type = "select"  # por defecto
        
        # Patrones simples para detectar tipos de consulta
        if any(term in question.lower() for term in ["cuántos", "cuantos", "contar", "count", "total", "número"]):
            query_type = "count"
        elif any(term in question.lower() for term in ["promedio", "media", "avg"]):
            query_type = "avg"
        elif any(term in question.lower() for term in ["máximo", "maximo", "max", "mayor"]):
            query_type = "max"
        elif any(term in question.lower() for term in ["mínimo", "minimo", "min", "menor"]):
            query_type = "min"
            
        enhanced["query_type"] = query_type
    
    # Detectar patrones generales de relaciones entre recursos y pacientes
    resource_patient_pattern = re.search(r'(?:(\w+)s?)\s+(?:del|de la|de)\s+(?:paciente|persona)\s+(\d+)', question.lower())
    
    if resource_patient_pattern:
        resource_term = resource_patient_pattern.group(1)
        patient_id = resource_patient_pattern.group(2)

        # La importación de call_llm ya se hizo arriba

        system_msg = (
        "Como experto en bases de datos médicas, identifica la tabla correcta para un recurso médico.\n"
        "Considera el esquema de la base de datos y el tipo de recurso mencionado."
    )
        user_msg = (
        f"Para el término médico '{resource_term}', ¿qué tabla del siguiente esquema sería la más apropiada?\n\n"
        f"Tablas disponibles: {enhanced.get('tables', [])}\n\n"
        "Responde SOLAMENTE con el nombre de la tabla, sin explicaciones adicionales."
    )

        messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
    
        try:
            config = {"temperature": 0.1, "max_tokens": 50}
            resource_table_response = call_llm(messages, config, step_name="Detección de tabla de recurso")
            resource_table = resource_table_response.strip()
            logging.info(f"LLM identificó la tabla '{resource_table}' para el recurso '{resource_term}'")
            
            # Determinar la tabla de pacientes
            patient_table = "PATI_PATIENTS"  # Tabla por defecto para pacientes
            
            # Verificar si existe table de pacientes en las tablas disponibles
            for table in enhanced.get("tables", []):
                if "PATI" in table:
                    patient_table = table
                    break
            
            # Verificar si ya existe una condición para el ID de paciente
            patient_condition_exists = False
            for condition in enhanced.get("conditions", []):
                if isinstance(condition, dict) and "PATI_ID" in str(condition.get("column", "")) and str(condition.get("value", "")) == patient_id:
                    patient_condition_exists = True
                    break
            
            if not patient_condition_exists:
                enhanced.setdefault("conditions", []).append({
                    "column": f"{patient_table}.PATI_ID",
                    "operator": "=",
                    "value": patient_id
                })
                logging.info(f"Añadida condición para paciente {patient_id} en {patient_table}.PATI_ID")
            
            # Mantener la tabla original como principal
            if enhanced.get("tables") and enhanced["tables"][0] != resource_table:
                # Reorganizar para que la tabla de recursos sea la principal
                enhanced["tables"] = [table for table in enhanced["tables"] if table != resource_table]
                enhanced["tables"].insert(0, resource_table)
                logging.info(f"Tabla {resource_table} establecida como principal para mantener el enfoque en el recurso")
            
        except Exception as e:
            logging.warning(f"Error al procesar llamada al LLM para identificar tabla de recurso: {e}")
            breakpoint() # Detener ejecución para depuración
    
    # Detectar ID de paciente general
    else:
        patient_match = re.search(r'paciente\s+(\d+)', question, flags=re.IGNORECASE)
        if patient_match:
            patient_id = patient_match.group(1)
            current_tables = enhanced.get("tables", [])
            
            if current_tables:
                main_table = current_tables[0]
                
                # Si la tabla principal es diferente a PATI_PATIENTS
                if main_table != "PATI_PATIENTS":
                    # Asegurar que PATI_PATIENTS esté en las tablas
                    if "PATI_PATIENTS" not in current_tables:
                        enhanced.setdefault("tables", []).append("PATI_PATIENTS")
                    
                    # Crear un JOIN entre la tabla principal y PATI_PATIENTS
                    # basado en el ID de paciente
                    join_exists = False
                    for join in enhanced.get("joins", []):
                        if isinstance(join, dict) and "PATI_PATIENTS" in str(join):
                            join_exists = True
                            break
                    
                    if not join_exists:
                        enhanced.setdefault("joins", []).append({
                            "type": "INNER",
                            "table1": main_table,
                            "table2": "PATI_PATIENTS",
                            "on": f"{main_table}.PATI_ID = PATI_PATIENTS.PATI_ID"
                        })
                        logging.info(f"Añadido JOIN entre {main_table} y PATI_PATIENTS")
                    
                    # Añadir la condición para el paciente específico
                    enhanced.setdefault("conditions", []).append({
                        "column": "PATI_PATIENTS.PATI_ID",
                        "operator": "=",
                        "value": patient_id
                    })
                else:
                    # Si la tabla principal ya es PATI_PATIENTS
                    enhanced.setdefault("conditions", []).append({
                        "column": "PATI_PATIENTS.PATI_ID", 
                        "operator": "=", 
                        "value": patient_id
                    })
    
    return enhanced


def normalize_join_format(join_dict):
    """
    Normaliza el formato de JOIN para hacerlo compatible con SQLGenerator.
    
    Args:
        join_dict: Diccionario que representa un JOIN, puede tener diferentes formatos
        
    Returns:
        Diccionario normalizado con las claves requeridas:
        - table: tabla principal
        - column: columna de la tabla principal
        - foreign_table: tabla secundaria
        - foreign_column: columna de la tabla secundaria
    """
    if not isinstance(join_dict, dict):
        logging.warning(f"Formato de JOIN inválido, no es diccionario: {join_dict}")
        return None
    
    # Verificar si ya tiene el formato correcto
    required_fields = ["table", "column", "foreign_table", "foreign_column"]
    if all(field in join_dict for field in required_fields):
        return join_dict
    
    # Intentar normalizar diferentes formatos comunes
    normalized = {}
    
    # Formato 1: {table1: col1, table2: col2}
    if len(join_dict) == 2 and all(isinstance(k, str) for k in join_dict):
        tables = list(join_dict.keys())
        columns = list(join_dict.values())
        normalized["table"] = tables[0]
        normalized["column"] = columns[0]
        normalized["foreign_table"] = tables[1]
        normalized["foreign_column"] = columns[1]
        return normalized
    
    # Formato 2: {left: {table: t1, column: c1}, right: {table: t2, column: c2}}
    if "left" in join_dict and "right" in join_dict:
        left = join_dict.get("left", {})
        right = join_dict.get("right", {})
        
        if isinstance(left, dict) and isinstance(right, dict):
            normalized["table"] = left.get("table", "")
            normalized["column"] = left.get("column", "")
            normalized["foreign_table"] = right.get("table", "")
            normalized["foreign_column"] = right.get("column", "")
            return normalized
    
    # Formato 3: {from_table: t1, from_column: c1, to_table: t2, to_column: c2}
    mapping = {
        "from_table": "table",
        "from_column": "column",
        "to_table": "foreign_table",
        "to_column": "foreign_column"
    }
    
    for old_key, new_key in mapping.items():
        if old_key in join_dict:
            normalized[new_key] = join_dict[old_key]
    
    # Verificar si después del mapeo tenemos los campos necesarios
    if all(field in normalized for field in required_fields):
        return normalized
    
    # Formato 4 - {type: INNER, table1: t1, table2: t2, on: t1.col1 = t2.col2}
    if "type" in join_dict and "table1" in join_dict and "table2" in join_dict and "on" in join_dict:
        try:
            # Extraer las columnas de join de la cláusula ON
            on_clause = join_dict["on"]
            # Dividir la condición por el operador de igualdad
            parts = on_clause.split('=')
            if len(parts) == 2:
                left_part = parts[0].strip()
                right_part = parts[1].strip()
                
                # Extraer tabla y columna de cada parte
                if '.' in left_part and '.' in right_part:
                    table1_col = left_part.split('.')
                    table2_col = right_part.split('.')
                    
                    # Determinar qué parte corresponde a qué tabla
                    if join_dict["table1"] in table1_col[0]:
                        normalized["table"] = join_dict["table1"]
                        normalized["column"] = table1_col[1]
                        normalized["foreign_table"] = join_dict["table2"]
                        normalized["foreign_column"] = table2_col[1]
                    else:
                        normalized["table"] = join_dict["table2"]
                        normalized["column"] = table2_col[1]
                        normalized["foreign_table"] = join_dict["table1"]
                        normalized["foreign_column"] = table1_col[1]
                    
                    return normalized
        except Exception as e:
            logging.warning(f"Error al procesar formato JOIN tipo 'ON': {e}")
    
    # Verificar si pudimos normalizar adecuadamente con cualquiera de los formatos
    if all(field in normalized for field in required_fields):
        return normalized
    
    logging.warning(f"No se pudo normalizar el formato del JOIN: {join_dict}")
    return None


# En sql_organizer.py, en la función que normaliza joins:
def normalize_field_to_list(value, field_name=None):
    """
    Normaliza un campo que debe ser lista, filtrando valores vacíos.
    """
    if value is None:
        return []
    if isinstance(value, str):
        # Si es un string vacío o solo espacios, devolver lista vacía
        if not value.strip():
            return []
        # Si contiene comas, separar por comas
        if "," in value:
            items = [item.strip() for item in value.split(",")]
            return [item for item in items if item]  # Filtrar elementos vacíos
        # String único no vacío
        return [value]
    if isinstance(value, list):
        # Filtrar elementos vacíos o None de la lista
        return [element for element in value if element and not (isinstance(element, str) and not element.strip())]
    # Devolver como lista de un elemento si no es None
    return [value] if value is not None else []

# En la función donde procesas joins
def validate_and_fix_relations(structured_info: Dict[str, Any]) -> Dict[str, Any]:
    """Valida y corrige las relaciones entre tablas basado en condiciones y columnas"""
    result = structured_info.copy()
    
    # 1. Identificar todas las tablas mencionadas en condiciones
    tables_in_conditions = set()
    tables_in_main = set(result.get("tables", []))
    
    # Buscar tablas mencionadas en condiciones
    for condition in result.get("conditions", []):
        if isinstance(condition, dict) and "column" in condition:
            parts = condition.get("column", "").split(".")
            if len(parts) > 1:
                table_name = parts[0]
                tables_in_conditions.add(table_name)
    
    # 2. Identificar tablas en condiciones que no están en las tablas principales
    missing_tables = tables_in_conditions - tables_in_main
    
    # 3. Si hay tablas faltantes, añadirlas y crear joins necesarios
    if missing_tables:
        for table in missing_tables:
            if table not in result.get("tables", []):
                if "tables" not in result:
                    result["tables"] = []
                result["tables"].append(table)
                logging.info(f"Tabla {table} añadida automáticamente porque aparece en condiciones")
        
        # 4. Caso especial: si tenemos PATI_PATIENTS y MEDI_MEDICATIONS, asegurar join correcto
        if "PATI_PATIENTS" in result.get("tables", []) and "MEDI_MEDICATIONS" in result.get("tables", []):
            # Verificar si ya existe un join entre estas tablas
            has_join = False
            if "joins" in result:
                for join in result["joins"]:
                    if isinstance(join, dict):
                        if ("table" in join and "foreign_table" in join and
                            {join["table"], join["foreign_table"]} == {"PATI_PATIENTS", "MEDI_MEDICATIONS"}):
                            has_join = True
                            break
            
            # Si no hay join, añadirlo
            if not has_join:
                if "joins" not in result:
                    result["joins"] = []
                
                # CORRECCIÓN: Asegurar que el join se configura correctamente
                result["joins"].append({
                    "table": "MEDI_MEDICATIONS",
                    "column": "PATI_ID",  # Columna en MEDI_MEDICATIONS
                    "foreign_table": "PATI_PATIENTS",
                    "foreign_column": "PATI_ID"  # Columna en PATI_PATIENTS
                })
                logging.info("JOIN añadido automáticamente entre PATI_PATIENTS y MEDI_MEDICATIONS")
                
                # Si la consulta es sobre medicamentos, asegurar que MEDI_MEDICATIONS sea la tabla principal
                if any("medicament" in condition.lower() for condition in result.get("original_question", "").split()):
                    result["tables"] = [t for t in result["tables"] if t != "MEDI_MEDICATIONS"] + ["MEDI_MEDICATIONS"]
                    logging.info("MEDI_MEDICATIONS movida al final de la lista de tablas para priorizar su selección")
    
    # Verificar si hay condiciones que impliquen PATI_ID pero no hay JOIN adecuado
    for condition in result.get("conditions", []):
        if isinstance(condition, dict) and "column" in condition:
            # Si hay condición sobre PATI_ID y la tabla principal es APPO_APPOINTMENTS
            if "PATI_ID" in condition["column"] and "APPO_APPOINTMENTS" in result.get("tables", []):
                # Verificar si ya existe JOIN entre estas tablas
                join_exists = False
                for join in result.get("joins", []):
                    if isinstance(join, dict):
                        tables = {join.get("table", ""), join.get("foreign_table", "")}
                        if "APPO_APPOINTMENTS" in tables and "PATI_PATIENTS" in tables:
                            join_exists = True
                            break
                
                # Si no existe JOIN y ambas tablas están presentes, crear el JOIN
                if not join_exists and "PATI_PATIENTS" in result.get("tables", []):
                    result.setdefault("joins", []).append({
                        "table": "APPO_APPOINTMENTS",
                        "column": "APPO_PATIENT_ID",  # Ajusta según tu esquema real
                        "foreign_table": "PATI_PATIENTS",
                        "foreign_column": "PATI_ID"
                    })
                    logging.info("Añadido JOIN entre APPO_APPOINTMENTS y PATI_PATIENTS")
    
    # Filtrar joins vacíos o inválidos
    if "joins" in structured_info:
        structured_info["joins"] = normalize_field_to_list(structured_info["joins"], "joins")
        
        # Aplicar la función normalize_join_format a cada join no vacío
        valid_joins = []
        for join in structured_info["joins"]:
            if not join:  # Saltar elementos vacíos
                continue
                
            if isinstance(join, dict):
                normalized_join = normalize_join_format(join)
                if normalized_join:  # Solo incluir joins válidos
                    valid_joins.append(normalized_join)
            else:
                logging.warning(f"Join no válido (no es un diccionario): {join}")
        
        structured_info["joins"] = valid_joins
    
    return result

class SQLGenerator:
    """
    Clase para generar consultas SQL a partir de información estructurada.
    """
    
    def __init__(self, allowed_tables, allowed_columns):
        """
        Inicializa el generador SQL con las tablas y columnas permitidas.
        
        Args:
            allowed_tables: Lista de nombres de tablas permitidas
            allowed_columns: Diccionario de {tabla: [columnas]} permitidas
        """
        self.allowed_tables = allowed_tables
        self.allowed_columns = allowed_columns
        
    def generate_sql(self, structured_info: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """
        Genera una consulta SQL a partir de información estructurada.
        
        Args:
            structured_info: Diccionario con la información estructurada
            
        Returns:
            Tupla de (consulta SQL, parámetros)
        """
        try:
            # Normalizar la información
            info = normalize_structured_info(structured_info)
            
            # Validar tablas
            if not info.get("tables"):
                return "SELECT 'Error: No se especificaron tablas' AS mensaje", []
            
            # Filtrar tablas no permitidas
            tables = [table for table in info["tables"] if table in self.allowed_tables]
            if not tables:
                return "SELECT 'Error: Ninguna tabla válida especificada' AS mensaje", []
            
            # Procesar columnas
            select_cols = info.get("columns", ["*"])
            if not select_cols:
                select_cols = ["*"]
            
            # Construir cláusula SELECT
            if info.get("query_type") != "select":
                query_type = info["query_type"].upper()
                if select_cols == ["*"]:
                    select_clause = f"{query_type}(*)"
                else:
                    select_clause = f"{query_type}({', '.join(select_cols)})"
            else:
                if info.get("distinct", False):
                    select_clause = f"DISTINCT {', '.join(select_cols)}"
                else:
                    select_clause = f"{', '.join(select_cols)}"
            
            # Tabla principal
            main_table = tables[0]
            
            # Consulta básica
            sql = f"SELECT {select_clause} FROM {main_table}"
            
            # Procesar JOINs
            params = []
            for join in info.get("joins", []):
                if not isinstance(join, dict):
                    continue
                    
                join_type = join.get("type", "INNER").upper()
                
                # Determinar las tablas del JOIN
                join_table = join.get("table") or join.get("table2")
                if not join_table:
                    continue
                
                # Obtener la condición ON
                on_clause = join.get("on")
                if not on_clause:
                    on_field1 = join.get("field1") or join.get("column1")
                    on_field2 = join.get("field2") or join.get("column2")
                    if on_field1 and on_field2:
                        on_clause = f"{on_field1} = {on_field2}"
                    else:
                        continue  # No podemos crear el JOIN sin condición ON
                
                sql += f" {join_type} JOIN {join_table} ON {on_clause}"
            
            # Procesar WHERE
            conditions = info.get("conditions", [])
            if conditions:
                where_clauses = []
                for condition in conditions:
                    if not isinstance(condition, dict):
                        continue
                        
                    column = condition.get("column")
                    operator = condition.get("operator", "=")
                    value = condition.get("value")
                    
                    if column and value is not None:
                        if operator.upper() in ("IN", "NOT IN") and isinstance(value, list):
                            placeholders = ", ".join(["?"] * len(value))
                            where_clauses.append(f"{column} {operator} ({placeholders})")
                            params.extend(value)
                        else:
                            where_clauses.append(f"{column} {operator} ?")
                            params.append(value)
                
                if where_clauses:
                    sql += f" WHERE {' AND '.join(where_clauses)}"
            
            # Procesar GROUP BY
            group_by = info.get("group_by", [])
            if group_by:
                sql += f" GROUP BY {', '.join(group_by)}"
                
                # Procesar HAVING si hay GROUP BY
                having = info.get("having", [])
                if having:
                    having_clauses = []
                    for condition in having:
                        if not isinstance(condition, dict):
                            continue
                            
                        column = condition.get("column")
                        operator = condition.get("operator", "=")
                        value = condition.get("value")
                        
                        if column and value is not None:
                            having_clauses.append(f"{column} {operator} ?")
                            params.append(value)
                    
                    if having_clauses:
                        sql += f" HAVING {' AND '.join(having_clauses)}"
            
            # Procesar ORDER BY
            order_by = info.get("order_by", [])
            if order_by:
                order_clauses = []
                for order in order_by:
                    if isinstance(order, str):
                        order_clauses.append(order)
                    elif isinstance(order, dict):
                        column = order.get("column")
                        direction = order.get("direction", "ASC").upper()
                        if column:
                            order_clauses.append(f"{column} {direction}")
                
                if order_clauses:
                    sql += f" ORDER BY {', '.join(order_clauses)}"
            
            # Procesar LIMIT
            limit = info.get("limit")
            if limit is not None:
                try:
                    limit_value = int(limit)
                    sql += f" LIMIT {limit_value}"
                except (ValueError, TypeError):
                    pass  # Ignorar LIMIT inválido
            
            # En generate_sql, modifica la sección que maneja los JOINs:
            sql_query, params = sql, params

            # Detectar tablas en WHERE que no están en FROM
            tables_in_query = set()
            from_match = re.search(r'FROM\s+(.+?)(\s+WHERE|\s+GROUP|\s+ORDER|\s*$)', sql_query, re.IGNORECASE)
            if from_match:
                tables_in_query = {t.strip() for t in from_match.group(1).split(',')}

            # Detectar tablas en condiciones
            tables_in_where = set()
            where_match = re.search(r'WHERE\s+(.*?)(\s+GROUP|\s+ORDER|\s*$)', sql_query, re.IGNORECASE | re.DOTALL)
            if where_match:
                # Buscar patrones como "TABLA.COLUMNA"
                where_tables = re.findall(r'([A-Za-z0-9_]+)\.', where_match.group(1))
                tables_in_where = set(where_tables)

            # Añadir JOINs para tablas en WHERE que no están en FROM
            missing_tables = tables_in_where - tables_in_query
            if missing_tables and from_match:
                first_table = list(tables_in_query)[0] if tables_in_query else structured_info["tables"][0]
                
                # Construir JOINs para las tablas faltantes
                join_clauses = []
                for missing_table in missing_tables:
                    # Intentar encontrar un JOIN en structured_info
                    join_info = next((j for j in structured_info.get("joins", []) 
                                    if (j.get("table") == missing_table or j.get("foreign_table") == missing_table)
                                    and (j.get("table") == first_table or j.get("foreign_table") == first_table)), None)
                    
                    if join_info:
                        # Construir JOIN con ON
                        on_clause = f"{join_info['table']}.{join_info['column']} = {join_info['foreign_table']}.{join_info['foreign_column']}"
                        join_clauses.append(f"JOIN {missing_table} ON {on_clause}")
                
                # Aplicar todos los JOINs a la consulta
                if join_clauses:
                    new_from = f"FROM {first_table} " + " ".join(join_clauses)
                    sql_query = sql_query.replace(from_match.group(0), new_from + " " + (from_match.group(2) or " "))
                    logging.info(f"JOINs añadidos para tablas en WHERE: {missing_tables}")

            return sql_query, params
            
        except Exception as e:
            error_message = f"Error al generar SQL: {str(e)}"
            logging.error(error_message)
            return f"SELECT '{error_message}' AS mensaje", []