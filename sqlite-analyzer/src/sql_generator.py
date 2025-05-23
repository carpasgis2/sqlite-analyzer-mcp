import re
import logging
import sys
import os
import json
from typing import Dict, Any, Tuple, List, Optional, Union

# Ajustar el path para importaciones
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Ahora importamos desde el directorio actual
from sql_validator import whitelist_validate_query

# Intentar importar funciones de .pipeline a nivel de módulo
# Comentadas temporalmente para aislar el problema de importación circular / funciones no encontradas
# try:
#     from .pipeline import (
#         validate_query_structure, 
#         fallback_query,
#         normalize_query_structure, 
#         apply_table_aliases
#     )
#     PIPELINE_FUNCTIONS_LOADED = True
#     logging.info("Funciones de .pipeline cargadas exitosamente en sql_generator.")
# except ImportError as e:
#     PIPELINE_FUNCTIONS_LOADED = False
#     logging.error(f"Error al importar funciones de .pipeline en sql_generator: {e}. Estas funciones no estarán disponibles.")
#     # Definir stubs o placeholders si es necesario para que la clase se cargue
#     def validate_query_structure(sql): return True, ""
#     def fallback_query(tables): return "SELECT 'Error: fallback_query no implementada' AS mensaje", []
#     def normalize_query_structure(info): return info
#     def apply_table_aliases(sql, tables, cols): return sql


class SQLGenerator:
    """Clase para generar consultas SQL a partir de información estructurada"""
    
    def __init__(self, allowed_tables: List[str], allowed_columns: Dict[str, List[str]]):
        """
        Inicializa el generador SQL con las tablas y columnas permitidas
        
        Args:
            allowed_tables: Lista de nombres de tablas permitidas
            allowed_columns: Diccionario que mapea tablas a sus columnas permitidas
        """
        self.allowed_tables = allowed_tables
        self.allowed_columns = allowed_columns
        
    def _ensure_string(self, value: Any) -> str:
        """
        Convierte cualquier valor a string de manera segura
        
        Args:
            value: Valor a convertir
            
        Returns:
            El valor convertido a string
        """
        if isinstance(value, str):
            return value
        elif isinstance(value, (list, tuple)) and len(value) > 0:
            return str(value[0])  # Tomar el primer elemento si es una lista
        else:
            return str(value)
    
    def _normalize_operator(self, operator: Any) -> str:
        """
        Normaliza el operador a un formato válido
        
        Args:
            operator: Operador a normalizar
            
        Returns:
            Operador normalizado
        """
        if not isinstance(operator, str):
            operator = self._ensure_string(operator)
            
        # Normalizar a mayúsculas de forma segura
        operator = operator.upper()
        
        valid_operators = {
            "=": "=", "IGUAL": "=", "EQUALS": "=", "EQ": "=",
            "!=": "!=", "<>": "!=", "DISTINTO": "!=", "DIFERENTE": "!=", "NE": "!=",
            ">": ">", "MAYOR": ">", "GT": ">",
            "<": "<", "MENOR": "<", "LT": "<",
            ">=": ">=", "MAYOR_IGUAL": ">=", "GTE": ">=",
            "<=": "<=", "MENOR_IGUAL": "<=", "LTE": "<=",
            "LIKE": "LIKE", "CONTIENE": "LIKE", "CONTAINS": "LIKE",
            "IN": "IN", "ENTRE": "IN",
            "NOT IN": "NOT IN", "NO_ENTRE": "NOT IN",
            "BETWEEN": "BETWEEN", "RANGO": "BETWEEN"
        }
        
        return valid_operators.get(operator, "=")
    
    def generate_sql(self, structured_info):
        """
        Genera una consulta SQL a partir de información estructurada.
        Corrige:
        - FROM solo con la tabla principal
        - JOINs con ON obligatoria
        - No repite tablas ni usa comas tras JOINs
        - Todas las condiciones tras los JOINs, en un único WHERE
        """
        logging.info("[SQLGenerator] Iniciando generate_sql") # LOG INICIO MÉTODO
        params = []

        logging.debug(f"[SQLGenerator] structured_info recibido: {json.dumps(structured_info, indent=2)}")

        # Validar que exista al menos una tabla
        if not structured_info.get("tables"):
            logging.error("[SQLGenerator] No se especificó ninguna tabla en structured_info.")
            return "SELECT 'Error: No se especificó ninguna tabla' AS mensaje", []

        main_table = structured_info["tables"][0]
        logging.info(f"[SQLGenerator] Tabla principal: {main_table}")
        # TODO: Re-activar validación de tablas permitidas cuando el flujo esté más estable
        # if main_table not in self.allowed_tables:
        #     return f"SELECT 'Error: Tabla {main_table} no permitida' AS mensaje", []

        # Procesar columnas
        select_cols = structured_info.get("columns", ["*"])
        if not select_cols:
            select_cols = ["*"]
        logging.info(f"[SQLGenerator] Columnas a seleccionar: {select_cols}")

        # TODO: Re-activar validación de columnas permitidas
        # ...

        # Construir la cláusula FROM con JOINs
        sql = f"SELECT {', '.join(select_cols)} FROM {main_table}"
        logging.debug(f"[SQLGenerator] SQL inicial: {sql}")

        # --- NUEVA LÓGICA DE JOIN ROBUSTA ---
        joins_to_apply = structured_info.get("joins", [])
        logging.info(f"[SQLGenerator] Procesando {len(joins_to_apply)} JOINs... (originalmente)")

        # Conjunto para rastrear las tablas ya unidas para evitar redundancia
        already_joined_tables = {main_table.upper()} # La tabla principal ya está en el FROM

        # Lista para los strings de JOIN que se añadirán al SQL
        join_clauses_sql = []

        for i, join_item in enumerate(joins_to_apply):
            logging.debug(f"[SQLGenerator] Procesando JOIN #{i+1}: {join_item}")
            join_dict = None
            join_sql_segment = ""

            if isinstance(join_item, str):
                # ... (manejo de join_item como string, no se modifica por ahora)
                # Esta parte es compleja y podría necesitar una revisión separada si se usa.
                # Por ahora, nos centramos en el formato de diccionario para los joins.
                logging.warning(f"[SQLGenerator] JOIN string no procesado en esta revisión: {join_item}")
                continue
            elif isinstance(join_item, dict):
                join_dict = join_item
                logging.debug(f"[SQLGenerator] JOIN es un diccionario: {join_dict}")
            
            if join_dict:
                # Determinar las tablas involucradas en este JOIN específico
                # Priorizar foreign_table si existe, sino table (asumiendo que table es la que se une a la principal/ya unidas)
                table_being_joined = join_dict.get('foreign_table', join_dict.get('table'))
                
                if not table_being_joined:
                    logging.warning(f"[SQLGenerator] JOIN dict no tiene 'table' o 'foreign_table': {join_dict}. Se omite.")
                    continue

                # Normalizar el nombre de la tabla a mayúsculas para la comprobación
                table_being_joined_upper = table_being_joined.upper()

                if table_being_joined_upper in already_joined_tables:
                    logging.info(f"[SQLGenerator] Tabla '{table_being_joined}' ya unida. Omitiendo JOIN redundante: {join_dict}")
                    continue # Saltar este JOIN si la tabla ya fue unida

                # Construir la condición ON
                on_condition_str = join_dict.get('on')
                if not on_condition_str: # Intentar construir desde campos explícitos
                    t1 = join_dict.get('table')
                    c1 = join_dict.get('column')
                    t2 = join_dict.get('foreign_table') # Esta es la tabla que se está uniendo
                    c2 = join_dict.get('foreign_column')
                    if t1 and c1 and t2 and c2:
                        on_condition_str = f"{t1}.{c1} = {t2}.{c2}"
                    else:
                        logging.warning(f"[SQLGenerator] JOIN para tabla '{table_being_joined}' omitido por falta de condición ON explícita o campos para construirla: {join_dict}")
                        continue
                
                join_type = join_dict.get('type', 'INNER').upper() # Default a INNER JOIN
                join_sql_segment = f" {join_type} JOIN {table_being_joined} ON {on_condition_str}"
                
                join_clauses_sql.append(join_sql_segment)
                already_joined_tables.add(table_being_joined_upper) # Marcar como unida
                logging.debug(f"[SQLGenerator] JOIN procesado y añadido: {join_sql_segment}")
            else:
                logging.warning(f"[SQLGenerator] Elemento JOIN no es string ni diccionario procesable: {join_item}")

        # Añadir todas las cláusulas JOIN construidas al SQL principal
        if join_clauses_sql:
            sql += "".join(join_clauses_sql)

        logging.info("[SQLGenerator] Procesamiento de JOINs completado.")
        logging.debug(f"[SQLGenerator] SQL después de JOINs: {sql}")
        # --- FIN LÓGICA JOIN ---

        # Procesar condiciones WHERE
        conditions = structured_info.get("conditions", [])
        logging.info(f"[SQLGenerator] Procesando {len(conditions)} condiciones WHERE...")
        where_clauses = []
        
        if isinstance(conditions, dict): 
            conditions = [conditions]
            logging.debug("[SQLGenerator] Condiciones convertidas de dict a lista.")
        elif isinstance(conditions, str): 
            try:
                parsed_conditions = json.loads(conditions)
                if isinstance(parsed_conditions, (list, dict)):
                    conditions = parsed_conditions if isinstance(parsed_conditions, list) else [parsed_conditions]
                    logging.debug("[SQLGenerator] Condiciones string parseadas como JSON.")
                else: 
                    conditions = [conditions] 
                    logging.debug("[SQLGenerator] Condiciones string envueltas en lista (no JSON).")
            except json.JSONDecodeError: 
                 conditions = [conditions] 
                 logging.debug("[SQLGenerator] Condiciones string envueltas en lista (error JSONDecode).")

        for i, condition_item in enumerate(conditions): 
            logging.debug(f"[SQLGenerator] Procesando condición WHERE #{i+1}: {condition_item}")
            if isinstance(condition_item, dict):
                column = condition_item.get("column", "")
                operator = condition_item.get("operator", "=")
                value = condition_item.get("value", "")
                
                if not column: 
                    logging.warning(f"[SQLGenerator] Condición omitida por falta de columna: {condition_item}")
                    continue

                operator = self._normalize_operator(operator)
                logging.debug(f"[SQLGenerator] Condición dict: col='{column}', op='{operator}', val='{value}'")
                
                # ... (resto de la lógica de operadores IN, BETWEEN, LIKE, etc.)
                if operator in ["IN", "NOT IN"]:
                    if isinstance(value, list) and value:
                        placeholders = ", ".join(["?" for _ in value])
                        where_clauses.append(f"{column} {operator} ({placeholders})")
                        params.extend(value)
                    elif isinstance(value, str) and value: # Permitir string para IN si es un solo valor
                        where_clauses.append(f"{column} {operator} (?)")
                        params.append(value)
                    else:
                        logging.warning(f"[SQLGenerator] Valor no válido para operador {operator} en columna {column}: {value}. Condición omitida.")
                        continue
                elif operator == "BETWEEN":
                    if isinstance(value, list) and len(value) == 2:
                        where_clauses.append(f"{column} BETWEEN ? AND ?")
                        params.extend(value)
                    else:
                        logging.warning(f"[SQLGenerator] Valor no válido para operador BETWEEN en columna {column}: {value}. Condición omitida.")
                        continue
                elif operator == "LIKE":
                    if not isinstance(value, str):
                        try:
                            value = str(value)
                        except:
                            logging.warning(f"[SQLGenerator] No se pudo convertir el valor para LIKE a string en columna {column}: {value}. Condición omitida.")
                            continue
                    where_clauses.append(f"{column} LIKE ?")
                    params.append(value)
                else: 
                    where_clauses.append(f"{column} {operator} ?")
                    params.append(value)
                logging.debug(f"[SQLGenerator] Condición dict procesada. where_clauses: {where_clauses}, params: {params}")
            
            elif isinstance(condition_item, str):
                logging.debug(f"[SQLGenerator] Procesando condición string: '{condition_item}'")
                match = re.match(r"\s*([\w\.]+)\s*(>=|<=|!=|=|>|<|LIKE|NOT LIKE|IN|NOT IN)\s*('([^']+)'|\"([^\"]+)\"|\d+(?:\.\d+)?|\([^\)]+\))\s*", condition_item, re.IGNORECASE)
                
                if match:
                    col, op, val_str = match.groups()
                    op = self._normalize_operator(op.strip().upper())

                    if (val_str.startswith("'") and val_str.endswith("'")) or \
                       (val_str.startswith('"') and val_str.endswith('"')):
                        val_to_param = val_str[1:-1]
                    elif val_str.startswith("(") and val_str.endswith(")") and op in ["IN", "NOT IN"]:
                        logging.warning(f"[SQLGenerator] Condición IN/NOT IN como string con lista literal: '{condition_item}'. Se añade directamente, pero es preferible formato de diccionario.")
                        where_clauses.append(condition_item)
                        continue 
                    else:
                        val_to_param = val_str

                    if op in ["IN", "NOT IN"] and not (val_str.startswith("(") and val_str.endswith(")")):
                        where_clauses.append(f"{col} {op} (?)")
                        params.append(val_to_param)
                    elif op in ["IN", "NOT IN"] and (val_str.startswith("(") and val_str.endswith(")")):
                        pass
                    else:
                        where_clauses.append(f"{col} {op} ?")
                        params.append(val_to_param)
                    
                    logging.debug(f"[SQLGenerator] Condición string parametrizada: {col} {op} ?, con valor: {val_to_param}")

                else: 
                    logging.warning(f"[SQLGenerator] Añadiendo condición de string directamente (potencialmente inseguro o no parametrizado): {condition_item}")
                    where_clauses.append(condition_item)
            else:
                logging.warning(f"[SQLGenerator] Tipo de condición no reconocido: {type(condition_item)}. Omitida.")

        logging.info("[SQLGenerator] Procesamiento de condiciones WHERE completado.")
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
            logging.debug(f"[SQLGenerator] SQL después de WHERE: {sql}")

        # Procesar GROUP BY
        group_by = structured_info.get("group_by", [])
        logging.info(f"[SQLGenerator] Procesando GROUP BY: {group_by}")
        if group_by:
            if isinstance(group_by, str):
                group_by = [g.strip() for g in group_by.split(',') if g.strip()]
            
            valid_group_by = []
            for gb_col in group_by:
                valid_group_by.append(gb_col) 
            
            if valid_group_by:
                sql += f" GROUP BY {', '.join(valid_group_by)}"
                logging.debug(f"[SQLGenerator] SQL después de GROUP BY: {sql}")

        # Procesar ORDER BY
        order_by = structured_info.get("order_by", [])
        logging.info(f"[SQLGenerator] Procesando ORDER BY: {order_by}")
        if order_by:
            if isinstance(order_by, str): 
                order_by_list = []
                for part in order_by.split(','):
                    part = part.strip()
                    col_order = part.split()
                    column = col_order[0]
                    direction = col_order[1].upper() if len(col_order) > 1 and col_order[1].upper() in ["ASC", "DESC"] else "ASC"
                    order_by_list.append({"column": column, "direction": direction})
                order_by = order_by_list
            
            order_clauses = []
            if isinstance(order_by, list): 
                for order_item in order_by: 
                    if isinstance(order_item, dict):
                        column = order_item.get("column", "")
                        direction = order_item.get("direction", "ASC").upper()
                        if column and direction in ["ASC", "DESC"]:
                            order_clauses.append(f"{column} {direction}") 
                    elif isinstance(order_item, str): 
                        col_order = order_item.split()
                        column = col_order[0]
                        direction = col_order[1].upper() if len(col_order) > 1 and col_order[1].upper() in ["ASC", "DESC"] else "ASC"
                        order_clauses.append(f"{column} {direction}")

            if order_clauses:
                sql += f" ORDER BY {', '.join(order_clauses)}"
                logging.debug(f"[SQLGenerator] SQL después de ORDER BY: {sql}")

        # Procesar LIMIT
        limit = structured_info.get("limit")
        logging.info(f"[SQLGenerator] Procesando LIMIT: {limit}")
        if limit:
            try:
                limit_value = int(limit)
                if limit_value > 0: 
                    sql += f" LIMIT {limit_value}"
                    logging.debug(f"[SQLGenerator] SQL después de LIMIT: {sql}")
            except (ValueError, TypeError):
                logging.warning(f"[SQLGenerator] Valor de LIMIT inválido: {limit}. Omitido.")
                pass
        
        logging.info(f"[SQLGenerator] SQL final generado: {sql}")
        logging.info(f"[SQLGenerator] Parámetros finales: {params}")
        logging.info("[SQLGenerator] Fin de generate_sql")

        return sql, params