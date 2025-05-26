import re
import logging # Añadir logging
from typing import List, Dict, Any, Optional, Tuple

# Configurar un logger para este módulo si no existe uno global
logger = logging.getLogger(__name__)
# No añadir handlers aquí si se espera que el logging global los configure.
# Si se ejecuta standalone y no hay config, los logs podrían no mostrarse.


class SQLValidator:
    def __init__(self, allowed_tables: List[str], allowed_columns: Dict[str, List[str]]):
        """
        Inicializa el validador SQL con tablas y columnas permitidas.
        
        Args:
            allowed_tables: Lista de nombres de tablas permitidas
            allowed_columns: Diccionario donde las claves son tablas y los valores son listas de columnas permitidas
        """
        self.allowed_tables = [table.upper() for table in allowed_tables]
        self.allowed_columns = {table.upper(): [col.upper() for col in cols] 
                               for table, cols in allowed_columns.items()}
        self.current_query_select_columns: Optional[List[str]] = None # Nuevo miembro
        logger.debug(f"SQLValidator inicializado. Tablas permitidas: {self.allowed_tables}")
        # Loguear solo una muestra de las columnas permitidas si es muy grande
        log_cols = {k: v[:5] + ['...'] if len(v) > 5 else v for k, v in self.allowed_columns.items()}
        logger.debug(f"SQLValidator inicializado. Columnas permitidas (muestra): {log_cols}")
    
    def validate_query_parts(self, query_parts: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Valida las partes de una consulta SQL.
        
        Args:
            query_parts: Diccionario con las partes de la consulta
            
        Returns:
            Tupla (válido, mensaje_error)
        """
        logger.debug(f"Iniciando validación de query_parts: {query_parts}")
        self.current_query_select_columns = query_parts.get('columns') # Almacenar columnas del SELECT

        # Validar tablas
        tables_in_query_upper = []
        if 'tables' in query_parts and query_parts['tables']:
            if not isinstance(query_parts['tables'], list):
                logger.warning(f"'tables' en query_parts no es una lista: {type(query_parts['tables'])}")
                return False, f"'tables' debe ser una lista, se obtuvo {type(query_parts['tables'])}"
            for table in query_parts['tables']:
                if not isinstance(table, str) or not table.strip():
                    logger.warning(f"Nombre de tabla inválido o vacío encontrado en query_parts: '{table}'")
                    return False, f"Nombre de tabla inválido o vacío: '{table}'"
                table_upper = table.upper()
                if table_upper not in self.allowed_tables:
                    logger.warning(f"Tabla no permitida: {table}. Permitidas: {self.allowed_tables}")
                    return False, f"Tabla no permitida: {table}"
                tables_in_query_upper.append(table_upper)
            logger.debug(f"Tablas en consulta (validadas): {tables_in_query_upper}")
        else:
            logger.debug("No hay 'tables' especificadas en query_parts o está vacía.")


        # Validar columnas
        if 'columns' in query_parts and query_parts['columns']:
            if not isinstance(query_parts['columns'], list):
                logger.warning(f"'columns' en query_parts no es una lista: {type(query_parts['columns'])}")
                return False, f"'columns' debe ser una lista, se obtuvo {type(query_parts['columns'])}"

            if query_parts['columns'] == ['*']:
                logger.debug("Columnas es ['*'], permitido.")
                # Verificar si hay tablas en la consulta; '*' sin tablas no tiene mucho sentido para la validación de esquema.
                if not tables_in_query_upper and not self.allowed_tables: # Si no hay tablas en consulta Y no hay tablas permitidas en general
                    logger.warning("Se usó '*' como columna, pero no hay tablas en la consulta ni tablas permitidas definidas globalmente.")
                    # Esto podría ser un error dependiendo de la política, pero '*' es sintácticamente válido.
                    # Dejamos pasar, asumiendo que el generador SQL lo manejará.
            else:
                for column in query_parts['columns']:
                    if not isinstance(column, str) or not column.strip():
                         logger.warning(f"Nombre de columna inválido o vacío encontrado: '{column}'")
                         return False, f"Nombre de columna inválido o vacío: '{column}'"

                    # Permitir funciones de agregación comunes y otras expresiones que no son solo columnas
                    # Esta regex es una simplificación. Un parser SQL real sería más robusto.
                    if re.match(r'^\s*(COUNT|AVG|SUM|MIN|MAX|DISTINCT)\s*\(.+\)\s*$', column, flags=re.IGNORECASE) or \
                       re.match(r'^\s*CASE\s+WHEN.+END\s*$', column, flags=re.IGNORECASE) or \
                       re.match(r'^\s*\w+\(.*\)\s*$', column): # Función genérica simple
                        logger.debug(f"Columna de agregación/función permitida: {column}")
                        continue
                    
                    if column == '*':
                        logger.warning("El selector '*' debe ser la única 'columna' si se utiliza y no lo fue.")
                        return False, "El selector '*' debe ser la única 'columna' si se utiliza (y no como parte de COUNT(*))."

                    if not self._is_valid_column_in_context(column, tables_in_query_upper):
                        # _is_valid_column_in_context ya loguea el detalle del fallo
                        return False, f"Columna '{column}' no es válida o no se encuentra en el contexto de las tablas de la consulta."
        
        # Validar condiciones
        if 'conditions' in query_parts and query_parts['conditions']:
            current_conditions = query_parts['conditions']
            logger.debug(f"Validando condiciones: {current_conditions} con columnas SELECT: {self.current_query_select_columns}")
            valid, error_msg = self._validate_conditions_recursive(current_conditions, tables_in_query_upper)
            if not valid:
                logger.warning(f"Validación de condiciones fallida: {error_msg}")
                self.current_query_select_columns = None # Limpiar
                return False, error_msg
            logger.debug("Condiciones validadas exitosamente.")
        
        logger.info(f"Validación de query_parts completada exitosamente para: {query_parts}")
        self.current_query_select_columns = None # Limpiar al final
        return True, ""
    
    def _validate_conditions_recursive(self, conditions_obj: Any, tables_in_query_upper: List[str]) -> Tuple[bool, str]:
        """Valida recursivamente una estructura de condiciones."""
        if not conditions_obj: # None, [], {} vacíos
            return True, ""
        
        if isinstance(conditions_obj, dict):
            # Validar una única condición (diccionario)
            # Puede ser una condición simple: {'column': 'X', 'operator': '=', 'value': 'Y'}
            # O un operador lógico: {'AND': [cond1, cond2]} o {'OR': [cond1, cond2]}
            
            # Comprobar si es un operador lógico
            logical_ops = [key for key in ['AND', 'OR'] if key in conditions_obj]
            if logical_ops:
                op = logical_ops[0]
                if len(logical_ops) > 1 or len(conditions_obj) > 1:
                    return False, f"Operador lógico '{op}' debe ser la única clave en un diccionario de condición lógica."
                
                sub_conditions = conditions_obj[op]
                if not isinstance(sub_conditions, list):
                    return False, f"Las sub-condiciones para '{op}' deben ser una lista."
                return self._validate_conditions_recursive(sub_conditions, tables_in_query_upper)

            # Si no es un operador lógico, es una condición de campo
            required_fields = ['column', 'operator'] # 'value' es opcional (ej. IS NULL)
            if not all(field in conditions_obj for field in required_fields):
                return False, f"Faltan campos requeridos ('column', 'operator') en la condición: {conditions_obj}"
            
            column_name_in_cond = conditions_obj['column']
            if not self._is_valid_column_in_context(column_name_in_cond, tables_in_query_upper):
                # Inicio de la nueva lógica para detectar si es un alias de SELECT problemático
                is_problematic_select_alias = False
                if self.current_query_select_columns and isinstance(column_name_in_cond, str) and column_name_in_cond.strip():
                    logger.debug(f"Columna '{column_name_in_cond}' no válida en contexto, verificando si es alias de SELECT de: {self.current_query_select_columns}")
                    for select_entry_str in self.current_query_select_columns:
                        if not isinstance(select_entry_str, str): continue

                        match_as = re.match(r'^(.*?)\\s+AS\\s+([a-zA-Z_][a-zA-Z0-9_]*)$', select_entry_str, re.IGNORECASE)
                        if match_as:
                            alias_name_from_select = match_as.group(2)
                            if alias_name_from_select.upper() == column_name_in_cond.upper():
                                is_problematic_select_alias = True
                                logger.debug(f"Columna '{column_name_in_cond}' coincide con alias de SELECT '{alias_name_from_select}' de la expresión '{select_entry_str}'")
                                break
                
                if is_problematic_select_alias:
                    error_msg = f"La columna '{column_name_in_cond}' en la condición parece ser un alias definido con 'AS' en la cláusula SELECT. Los alias de SELECT no se pueden usar directamente en WHERE/HAVING en el mismo nivel de consulta. Por favor, repita la expresión completa en la condición o use una Expresión Común de Tabla (CTE)."
                    logger.warning(f"_validate_conditions_recursive: {error_msg}")
                    return False, error_msg
                # Fin de la nueva lógica

                logger.warning(f"_validate_conditions_recursive: Columna '{column_name_in_cond}' en condición no es válida (según _is_valid_column_in_context) y no se identificó como alias de SELECT problemático.")
                return False, f"Columna '{column_name_in_cond}' en condición no es válida o no se encuentra en tablas de consulta."

            # Validar operador (lista extendida)
            valid_operators = [
                '=', '!=', '<>', '>', '<', '>=', '<=', 
                'LIKE', 'NOT LIKE', 'ILIKE', 'NOT ILIKE',
                'IN', 'NOT IN', 
                'BETWEEN', 'NOT BETWEEN',
                'IS NULL', 'IS NOT NULL', 'IS', 'IS NOT',
                'SIMILAR TO', 'NOT SIMILAR TO' # PostgreSQL specific, but good to have
            ]
            operator = conditions_obj.get('operator', '').upper() # Asegurar que el operador sea string
            if not isinstance(operator, str) or operator not in valid_operators:
                return False, f"Operador inválido o no string en condición: {operator}"
            
            # Podríamos añadir validación de 'value' aquí si es necesario (ej. para IN, BETWEEN)
            
            return True, ""

        elif isinstance(conditions_obj, list):
            # Validar una lista de condiciones (posiblemente anidadas o con 'AND'/'OR' implícitos)
            for i, item in enumerate(conditions_obj):
                # Si el item es un string como 'AND' o 'OR', es un conector lógico, lo permitimos.
                # Esto es para formatos de lista plana como [cond1, 'AND', cond2]
                if isinstance(item, str) and item.upper() in ['AND', 'OR']:
                    if i == 0 or i == len(conditions_obj) -1 : # No puede estar al principio o al final
                         return False, f"Conector lógico '{item}' en posición inválida en lista de condiciones."
                    continue 
                
                # Si es un dict (condición) o una list (sub-grupo de condiciones)
                is_valid, error = self._validate_conditions_recursive(item, tables_in_query_upper)
                if not is_valid:
                    return False, f"Error en condición/subgrupo {i+1}: {error}"
            return True, ""
        
        elif isinstance(conditions_obj, str):
            # Validar una condición como string (más permisivo, menos seguro)
            logger.debug(f"Validando condición de string: '{conditions_obj}'. Esta validación es limitada.")
            # Extraer posibles nombres de columna de la cadena de condición
            # Regex mejorada para capturar nombres con alias/tabla y nombres simples
            potential_cols_in_str = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*(\s*\.\s*[a-zA-Z_][a-zA-Z0-9_]*)?)\b', conditions_obj)
            
            has_potentially_invalid_col = False
            for match_tuple in potential_cols_in_str:
                col_name_str = match_tuple[0].replace(" ", "") # Eliminar espacios si los hubiera en "alias . col"
                # Evitar validar palabras clave SQL comunes como columnas
                sql_keywords = {'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'NOT', 'NULL', 'LIKE', 'IN', 'BETWEEN', 'IS', 'GROUP', 'ORDER', 'BY', 'LEFT', 'RIGHT', 'JOIN', 'ON', 'AS'}
                if col_name_str.upper() in sql_keywords:
                    continue

                if not self._is_valid_column_in_context(col_name_str, tables_in_query_upper):
                    logger.warning(f"Posible columna inválida '{col_name_str}' encontrada en condición de string '{conditions_obj}'.")
                    # No retornamos False inmediatamente para permitir que otras partes de la condición se validen,
                    # pero marcamos que hay un problema. Si es muy estricto, retornar aquí.
                    has_potentially_invalid_col = True 
            
            if has_potentially_invalid_col:
                 return False, f"Una o más posibles columnas en la condición de string '{conditions_obj}' no son válidas."
            return True, ""
        else:
            return False, f"Formato de condiciones no soportado: {type(conditions_obj)}"

    def _is_valid_column_in_context(self, column_name: str, tables_in_query_upper: List[str]) -> bool:
        """
        Verifica si una columna es válida, considerando el contexto de las tablas en la consulta.
        Loguea detalladamente el motivo del fallo si no es válida.
        """
        if not isinstance(column_name, str) or not column_name.strip():
            logger.warning(f"_is_valid_column_in_context: Nombre de columna inválido (no string o vacío): '{column_name}'")
            return False

        original_column_name = column_name # Para logging
        logger.debug(f"_is_valid_column_in_context: Validando columna '{original_column_name}' con tablas de consulta: {tables_in_query_upper}")

        if '.' in column_name:
            table_part, col_part = column_name.split('.', 1)
            table_part_upper = table_part.upper()
            col_part_upper = col_part.upper()
            logger.debug(f"  Columna con prefijo: Tabla/Alias='{table_part_upper}', Columna='{col_part_upper}'")

            # Escenario 1: table_part es una tabla real conocida y permitida
            if table_part_upper in self.allowed_columns:
                if col_part_upper in self.allowed_columns[table_part_upper]:
                    logger.debug(f"  VÁLIDA: Columna '{col_part_upper}' encontrada en tabla real '{table_part_upper}'.")
                    return True
                else:
                    logger.warning(f"  INVÁLIDA: Columna '{col_part_upper}' NO encontrada en tabla real '{table_part_upper}'. Columnas de '{table_part_upper}': {self.allowed_columns.get(table_part_upper)}")
                    return False
            # Escenario 2: table_part es un alias. Resolver usando tables_in_query_upper.
            else:
                logger.debug(f"  Prefijo '{table_part_upper}' no es una tabla directamente conocida. Tratando como alias.")
                if not tables_in_query_upper:
                    logger.warning(f"  INVÁLIDA: Prefijo '{table_part_upper}' tratado como alias, pero no hay tablas en la consulta (tables_in_query_upper) para resolverlo.")
                    return False
                for actual_table_name_upper in tables_in_query_upper:
                    if actual_table_name_upper in self.allowed_columns and \
                       col_part_upper in self.allowed_columns[actual_table_name_upper]:
                        logger.debug(f"  VÁLIDA: Columna '{col_part_upper}' (con alias '{table_part_upper}') encontrada en tabla de consulta '{actual_table_name_upper}'.")
                        return True
                logger.warning(f"  INVÁLIDA: Columna '{col_part_upper}' (con alias '{table_part_upper}') NO encontrada en ninguna de las tablas de consulta {tables_in_query_upper} que tienen columnas definidas. Columnas permitidas globales (muestra): {{k: v[:3] for k,v in self.allowed_columns.items()}}")
                return False
        else: # Columna sin prefijo
            col_upper = column_name.upper()
            logger.debug(f"  Columna sin prefijo: '{col_upper}'")
            if tables_in_query_upper: # Buscar primero en las tablas de la consulta
                for table_in_q_upper in tables_in_query_upper:
                    if table_in_q_upper in self.allowed_columns and \
                       col_upper in self.allowed_columns[table_in_q_upper]:
                        logger.debug(f"  VÁLIDA: Columna sin prefijo '{col_upper}' encontrada en tabla de consulta '{table_in_q_upper}'.")
                        return True
                logger.debug(f"  Columna sin prefijo '{col_upper}' no encontrada en tablas de consulta explícitas {tables_in_query_upper}. Verificando globalmente...")
            
            # Si no se encontró en las tablas de la consulta, o no había tablas en la consulta,
            # buscar globalmente (puede ser ambiguo, pero el validador solo chequea existencia)
            # Esto es útil si la columna es única en todo el esquema.
            found_globally_in_tables = []
            for table_name_iter, allowed_cols_iter in self.allowed_columns.items():
                if col_upper in allowed_cols_iter:
                    found_globally_in_tables.append(table_name_iter)
            
            if found_globally_in_tables:
                if len(found_globally_in_tables) == 1:
                    logger.debug(f"  VÁLIDA: Columna sin prefijo '{col_upper}' encontrada globalmente en una única tabla '{found_globally_in_tables[0]}'.")
                    return True
                else: # Encontrada en múltiples tablas globalmente
                    # Si ya había tablas en la consulta y no se encontró allí, esto es un problema.
                    # Si no había tablas en la consulta, permitirla aquí es arriesgado por ambigüedad.
                    # Sin embargo, el rol del validador es si *podría* ser válida.
                    logger.debug(f"  ADVERTENCIA/VÁLIDA (potencialmente ambigua): Columna sin prefijo '{col_upper}' encontrada globalmente en múltiples tablas: {found_globally_in_tables}. Se permite, pero se recomienda calificarla.")
                    return True # Es válida en al menos una tabla.
            
            logger.warning(f"  INVÁLIDA: Columna sin prefijo '{col_upper}' NO encontrada en ninguna tabla permitida (ni en tablas de consulta, ni globalmente).")
            return False

# La función whitelist_validate_query permanece igual, solo instancia SQLValidator.
# (Asegúrate de que el logger esté configurado en el script que llama a esto para ver los logs DEBUG)
def whitelist_validate_query(query_parts: Dict[str, Any], allowed_tables: List[str], 
                            allowed_columns: Dict[str, List[str]]) -> Tuple[bool, str]:
    """
    Valida una consulta SQL contra una lista blanca de tablas y columnas.
    
    Args:
        query_parts: Diccionario con las partes de la consulta
        allowed_tables: Lista de nombres de tablas permitidas
        allowed_columns: Diccionario donde las claves son tablas y los valores son listas de columnas permitidas
        
    Returns:
        Tupla (válido, mensaje_error)
    """
    # El logger para SQLValidator se define a nivel de clase/módulo.
    # Si necesitas un nivel de log específico para esta función, configúralo antes de llamar.
    # Ejemplo: logging.getLogger('sql_validator').setLevel(logging.DEBUG)
    validator = SQLValidator(allowed_tables, allowed_columns)
    return validator.validate_query_parts(query_parts)
