import re
from typing import List, Dict, Any, Optional, Tuple

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
    
    def validate_query_parts(self, query_parts: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Valida las partes de una consulta SQL.
        
        Args:
            query_parts: Diccionario con las partes de la consulta
            
        Returns:
            Tupla (válido, mensaje_error)
        """
        # Validar tablas
        if 'tables' in query_parts:
            for table in query_parts['tables']:
                if table.upper() not in self.allowed_tables:
                    return False, f"Tabla no permitida: {table}"
        
        # Validar columnas
        if 'columns' in query_parts:
            # Si query_parts['columns'] es exactamente ['*'], se considera válido para las columnas.
            if query_parts['columns'] == ['*']:
                pass  # Válido, no se necesita más validación de columnas aquí.
            else:
                for column in query_parts['columns']:
                    # Permitir expresiones COUNT(...)
                    if re.match(r'^\s*COUNT\(.+\)\s*$', column, flags=re.IGNORECASE):
                        continue
                    
                    # Si se encuentra '*' pero no es el único elemento (ya que no entramos en el if anterior)
                    if column == '*':
                        return False, "El selector '*' debe ser la única 'columna' si se utiliza."

                    # Manejar columnas con prefijo de tabla
                    if '.' in column:
                        table, col = column.split('.')
                        table_upper = table.upper()
                        # Comprobar que la tabla exista en allowed_columns y luego que la columna exista en esa tabla
                        if (table_upper not in self.allowed_columns or
                                col.upper() not in self.allowed_columns.get(table_upper, [])):
                            return False, f"Columna no permitida: {column}"
                    else:
                        # Columna sin prefijo: verificar si existe en alguna tabla permitida.
                        found = False
                        # Primero, intentar buscar en las tablas explícitamente mencionadas en query_parts (si existen)
                        tables_in_query = query_parts.get('tables', [])
                        if tables_in_query:
                            for table_in_q_str in tables_in_query:
                                table_in_q_upper = table_in_q_str.upper()
                                if table_in_q_upper in self.allowed_columns and \
                                   column.upper() in self.allowed_columns[table_in_q_upper]:
                                    found = True
                                    break
                        
                        # Si no se encontró (o no había tablas en query_parts), buscar en todas las allowed_columns
                        if not found:
                            for table_name_iter, allowed_cols_iter in self.allowed_columns.items():
                                if column.upper() in allowed_cols_iter:
                                    found = True
                                    break
                        
                        if not found:
                            return False, f"Columna no permitida o no encontrada en tablas relevantes: {column}"
        
        # Validar condiciones
        if 'conditions' in query_parts and query_parts['conditions']:
            valid, error_msg = self._validate_conditions(query_parts['conditions'])
            if not valid:
                return False, error_msg
        
        return True, ""
    
    def _validate_conditions(self, conditions):
        """
        Valida las condiciones WHERE.
        Puede recibir un string o un diccionario/lista de condiciones.
        """
        if not conditions:
            return True, ""
        
        # Si conditions es un diccionario, validar sus campos
        if isinstance(conditions, dict):
            # Verificar que el diccionario tenga los campos requeridos
            required_fields = ['column', 'operator']
            missing_fields = [field for field in required_fields if field not in conditions]
            
            if missing_fields:
                return False, f"Faltan campos en la condición: {', '.join(missing_fields)}"
            
            # Verificar que la columna sea válida
            column_name = conditions.get('column', '')
            if column_name and not self._is_valid_column(column_name):
                return False, f"Columna inválida en condición: {column_name}"
                
            # Validar operador
            valid_operators = ['=', '!=', '>', '<', '>=', '<=', 'LIKE', 'IN', 'NOT IN', 'BETWEEN']
            operator = conditions.get('operator', '').upper()
            
            if operator not in valid_operators:
                return False, f"Operador inválido en condición: {operator}"
                
            return True, ""
        
        # Si conditions es una lista, validar cada elemento
        elif isinstance(conditions, list):
            for i, condition in enumerate(conditions):
                if isinstance(condition, dict):
                    is_valid, error = self._validate_conditions(condition)
                    if not is_valid:
                        return False, f"Error en condición {i+1}: {error}"
                else:
                    # Si no es un diccionario, intentar validar como string
                    if not isinstance(condition, str):
                        return False, f"Condición {i+1} con formato inválido: {type(condition)}"
                        
                    is_valid, error = self._validate_condition_string(condition)
                    if not is_valid:
                        return False, f"Error en condición {i+1}: {error}"
                        
            return True, ""
        
        # Si conditions es un string, aplicar validación de texto
        elif isinstance(conditions, str):
            return self._validate_condition_string(conditions)
        
        # Para cualquier otro tipo, retornar error
        else:
            return False, f"Formato de condiciones no soportado: {type(conditions)}"

    def _validate_condition_string(self, condition_str):
        """
        Valida una condición en formato string.
        """
        # Normalizar operadores lógicos
        condition_str = re.sub(r'\bO\b', 'OR', condition_str, flags=re.IGNORECASE)
        condition_str = re.sub(r'\bY\b', 'AND', condition_str, flags=re.IGNORECASE)
        
        # Verificar si se usan columnas válidas
        # Extraer nombres de columnas (simplificado)
        column_pattern = r'(\w+)\s*(=|!=|<>|>|<|>=|<=|LIKE|IN|NOT\s+IN|BETWEEN)'
        potential_columns = re.findall(column_pattern, condition_str, re.IGNORECASE)
        
        for col_match in potential_columns:
            column_name = col_match[0].strip()
            if not self._is_valid_column(column_name):
                return False, f"Columna inválida en condición: {column_name}"
        
        return True, ""

    def _is_valid_column(self, column_name: str) -> bool:
        """
        Verifica si una columna es válida en las tablas permitidas.
        """
        for table, cols in self.allowed_columns.items():
            if column_name.upper() in cols:
                return True
        return False

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
    validator = SQLValidator(allowed_tables, allowed_columns)
    return validator.validate_query_parts(query_parts)
