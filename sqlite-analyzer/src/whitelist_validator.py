import json
import logging
import os # Añadido para operaciones de archivo
# import sqlparse # <--- ELIMINADO, sqlglot se usa en su lugar
from typing import Dict, Set, Optional, Any, Tuple
import sqlglot
import sqlglot.optimizer # Necesario para scope

logger = logging.getLogger(__name__)

class WhitelistValidator:
    def __init__(self, db_schema_dict: Optional[Dict[str, Any]] = None, allowed_tables_path=None, allowed_columns_map=None, case_sensitive: bool = True):
        """
        Initializes the WhitelistValidator.
        Can be initialized with a path to a JSON file defining allowed tables and columns,
        or directly with an allowed_columns_map dictionary.

        Args:
            db_schema_dict (dict, optional): A dictionary representing the database schema.
                                              Expected keys are "tables", each with "name" and optional "columns" (list of {"name": "..."}).
            allowed_tables_path (str, optional): Path to the JSON file.
                                                 The JSON should have a structure like:
                                                 { "allowed_tables": ["table1", "table2"],
                                                   "allowed_columns_map": {"table1": ["colA", "colB"], ...} }
            allowed_columns_map (dict, optional): A dictionary where keys are table names
                                                  and values are lists of allowed column names.
                                                  e.g., {"table1": ["colA", "colB"]}
            case_sensitive (bool, optional): If True (default), table and column names are case-sensitive.
                                             If False, names are treated as case-insensitive (converted to lower).
        """
        self.allowed_columns_map: Dict[str, Set[str]] = {}
        self.allowed_tables: Set[str] = set()
        self.case_sensitive = case_sensitive
        self.logger = logging.getLogger(__name__)
        self.current_query_aliases: Dict[str, str] = {}
        self.current_table_context_for_columns: Dict[str, str] = {}

        self.logger.info(f"WhitelistValidator initializing. Case sensitive: {self.case_sensitive}")

        if db_schema_dict is None and allowed_tables_path is None and allowed_columns_map is None:
            self.logger.warning(f"WhitelistValidator initialized without any whitelist configuration. Case sensitive: {self.case_sensitive}. All checks will likely fail.")
            return

        if allowed_columns_map:
            self.logger.info(f"Initializing whitelist from provided allowed_columns_map.")
            processed_map = {}
            for table, columns in allowed_columns_map.items():
                table_key = self._process_name(table)
                processed_map[table_key] = self._process_column_list(columns)
            self.allowed_columns_map = processed_map
            self.allowed_tables = set(self.allowed_columns_map.keys())
            self.logger.info(f"WhitelistValidator initialized with provided allowed_columns_map. Case sensitive: {self.case_sensitive}. Tables: {self.allowed_tables}")
        elif allowed_tables_path:
            self.logger.info(f"Initializing whitelist from JSON file: {allowed_tables_path}")
            try:
                with open(allowed_tables_path, 'r') as f:
                    config = json.load(f)
                
                raw_allowed_tables = config.get("allowed_tables", [])
                self.allowed_tables = {self._process_name(t) for t in raw_allowed_tables}
                
                raw_columns_map = config.get("allowed_columns_map", {})
                processed_map = {}
                for table, columns in raw_columns_map.items():
                    table_key = self._process_name(table)
                    processed_map[table_key] = self._process_column_list(columns)
                self.allowed_columns_map = processed_map
                
                # Ensure all tables in allowed_columns_map are also in allowed_tables
                for table_key in self.allowed_columns_map.keys(): # keys are already processed
                    self.allowed_tables.add(table_key)
                
                self.logger.info(f"WhitelistValidator initialized from {allowed_tables_path}. Case sensitive: {self.case_sensitive}. Tables: {self.allowed_tables}")
            except FileNotFoundError:
                self.logger.error(f"Whitelist config file not found: {allowed_tables_path}")
            except json.JSONDecodeError:
                self.logger.error(f"Error decoding JSON from whitelist config file: {allowed_tables_path}")
        else:
            self.logger.warning(f"WhitelistValidator initialized without any whitelist configuration. Case sensitive: {self.case_sensitive}. All checks will likely fail.")

        # Si se proporciona un esquema de base de datos, se carga en la inicialización
        if db_schema_dict:
            self.logger.info(f"Processing db_schema_dict for whitelist configuration.")
            if "tables" not in db_schema_dict:
                self.logger.error("WhitelistValidator __init__: 'tables' key is missing in db_schema_dict. No whitelist configuration will be loaded.")
                self.logger.debug(f"db_schema_dict keys: {list(db_schema_dict.keys()) if isinstance(db_schema_dict, dict) else 'Not a dict'}")
                return
                
            if not isinstance(db_schema_dict["tables"], list):
                self.logger.error(f"WhitelistValidator __init__: db_schema_dict['tables'] is not a list (type: {type(db_schema_dict['tables'])}). No whitelist configuration will be loaded.")
                return

            if not db_schema_dict["tables"]:
                self.logger.warning("WhitelistValidator __init__: db_schema_dict['tables'] is an empty list. No tables to load into whitelist.")
                return
                
            self.logger.info(f"WhitelistValidator __init__: Processing {len(db_schema_dict['tables'])} entries from db_schema_dict['tables'].")

            for i, table_info in enumerate(db_schema_dict["tables"]):
                if not isinstance(table_info, dict):
                    self.logger.warning(f"Table entry {i} is not a dictionary, skipping: {str(table_info)[:100]}")
                    continue

                table_name = table_info.get("name")
                if not table_name:
                    self.logger.warning(f"Table entry {i} in schema is missing a 'name': {str(table_info)[:100]}, skipping.")
                    continue
                
                processed_table_name = table_name if self.case_sensitive else table_name.lower()
                self.allowed_tables.add(processed_table_name)
                self.logger.debug(f"Added table to allowed_tables: '{processed_table_name}' (original: '{table_name}')")
                
                allowed_cols_for_table = set()
                if "columns" in table_info and isinstance(table_info["columns"], list):
                    for j, col_info in enumerate(table_info["columns"]):
                        if not isinstance(col_info, dict):
                            self.logger.warning(f"Column entry {j} for table '{table_name}' is not a dictionary, skipping: {str(col_info)[:100]}")
                            continue
                        col_name = col_info.get("name")
                        if col_name:
                            processed_col_name = col_name if self.case_sensitive else col_name.lower()
                            allowed_cols_for_table.add(processed_col_name)
                        else:
                            self.logger.warning(f"Column entry {j} for table '{table_name}' is missing a 'name': {str(col_info)[:100]}, skipping.")
                elif "columns" in table_info:
                     self.logger.warning(f"Table '{table_name}' has a 'columns' entry but it's not a list (type: {type(table_info['columns'])}), no columns will be loaded for this table.")

                if not allowed_cols_for_table:
                    self.logger.info(f"Table '{processed_table_name}' added to whitelist without explicit columns being processed or found. Column checks for this table might be restrictive if not intended.")
                    self.allowed_columns_map[processed_table_name] = set()
                else:
                    self.allowed_columns_map[processed_table_name] = allowed_cols_for_table
                    self.logger.debug(f"Table '{processed_table_name}' added with columns: {allowed_cols_for_table}")
            
            if not self.allowed_tables:
                self.logger.error(
                    "WhitelistValidator __init__ completed, but NO tables were successfully processed and added to the whitelist. "
                    "Ensure db_schema_dict['tables'] is a list of dictionaries, and each dictionary has a 'name' key."
                )
            else:
                self.logger.info(f"WhitelistValidator __init__ completed. {len(self.allowed_tables)} tables loaded. Allowed tables (first 50): {list(self.allowed_tables)[:50]}")
                # self.logger.debug(f"Final allowed_columns_map: {self.allowed_columns_map}") # Can be very verbose

    def _process_name(self, name):
        return name if self.case_sensitive else name.lower()

    def _process_column_list(self, columns: list[str]) -> Set[str]:
        return {self._process_name(col) for col in columns}

    def is_table_allowed(self, table_name: str) -> bool:
        processed_table_name = table_name if self.case_sensitive else table_name.lower()
        if processed_table_name not in self.allowed_tables:
            self.logger.warning(f"Table '{table_name}' (processed as '{processed_table_name}') is not in the allowed list. Allowed list (first 50): {list(self.allowed_tables)[:50]}")
            return False
        return True

    def are_columns_allowed(self, table_name, column_names):
        """Checks if all specified columns for a given table are in the whitelist."""
        table_name_internal = table_name if self.case_sensitive else table_name.lower()

        if not self.is_table_allowed(table_name): # is_table_allowed handles its own casing for table_name
            return False
        
        if table_name_internal not in self.allowed_columns_map:
            self.logger.warning(
                f"Table '{table_name}' (processed as '{table_name_internal}') is allowed, but no specific column whitelist "
                f"is defined in allowed_columns_map. Allowing all columns for this table by default. "
                f"To restrict columns, add an entry for '{table_name_internal}' in allowed_columns_map, "
                f"even with an empty list (to deny all non-'*' columns)."
            )
            return True

        allowed_cols_for_table = set(self.allowed_columns_map.get(table_name_internal, []))

        if not column_names: 
            return True # No specific columns requested, so vacuously true.
            
        for col_original in column_names:
            # For '*', we check against the literal '*' in the allowed list.
            # The '*' in the allowed list would have been cased during init if case_sensitive is False.
            # So, col_internal should be compared with items in allowed_cols_for_table.
            col_internal = col_original if self.case_sensitive else col_original.lower()
            
            if col_internal not in allowed_cols_for_table:
                self.logger.warning(
                    f"Column '{col_original}' (processed as '{col_internal}') in table '{table_name}' "
                    f"(processed as '{table_name_internal}') is not in the allowed list: {allowed_cols_for_table}"
                )
                return False
        return True

    def get_allowed_columns_for_table(self, table_name):
        """Returns the list of allowed columns for a table, or an empty list if table is allowed but has no specific columns, or None if table not allowed."""
        # table_name is assumed to be the raw name, will be processed by is_table_allowed
        if not self.is_table_allowed(table_name): # is_table_allowed handles casing
            return None

        table_name_internal = self._process_name(table_name)
        
        # Columns in self.allowed_columns_map are already processed for case.
        return self.allowed_columns_map.get(table_name_internal, set()) # Return a set

    def is_column_allowed_for_table(self, table_name: str, column_name: str) -> bool:
        # table_name: REAL table name, ALREADY PROCESSED for case.
        # column_name: column name from query, ALREADY PROCESSED for case.

        if table_name not in self.allowed_columns_map:
            self.logger.warning(f"Table '{table_name}' is allowed but not in allowed_columns_map, implying no individual columns are whitelisted unless '*' is used globally for it.")
            # This situation implies an empty set of allowed columns for this table in the map.
            # If '*' is not in this (empty) set, then the column is not allowed.
            # An empty set means only '*' could potentially match if '*' means "all columns".
            # However, our logic for '*' is that it must be *explicitly* in the allowed set.
            return False

        allowed_cols_for_table = self.allowed_columns_map[table_name]

        if column_name in allowed_cols_for_table:
            return True
        
        # Check if a generic '*' is allowed for this table, which would permit any column.
        star_representation = self._process_name("*")
        if star_representation in allowed_cols_for_table:
            self.logger.debug(f"Column '{column_name}' allowed for table '{table_name}' because '*' is whitelisted for this table.")
            return True

        self.logger.warning(
            f"Column '{column_name}' is not in the allowed columns for table '{table_name}'. "
            f"Allowed columns: {allowed_cols_for_table}"
        )
        return False

    def validate_sql_string(self, sql_string: str) -> Tuple[bool, str, Set[str]]:
        self.logger.debug(f"Starting SQL validation for: {sql_string[:200]}...")
        # Reiniciar el estado para la nueva consulta
        self.current_query_aliases = {}
        self.current_table_context_for_columns = {}
        actual_tables_referenced: Set[str] = set()

        try:
            parsed_expressions = sqlglot.parse(sql_string)
            if not parsed_expressions:
                self.logger.error(f"Error al parsear SQL '{sql_string[:100]}...': No se generaron expresiones.")
                return False, "Error al parsear SQL: No se generaron expresiones.", set()
            parsed = parsed_expressions[0]
            self.logger.debug(f"SQL parseado con éxito: {type(parsed)}")
        except sqlglot.errors.ParseError as e:
            self.logger.error(f"Error de parseo de SQL con sqlglot para '{sql_string[:100]}...': {e}")
            return False, f"Error de parseo de SQL: {e}", set()

        # Primero, extraer todos los alias de tabla y de columna SELECT
        self._extract_aliases_and_table_context(parsed)
        
        # Recopilar todas las tablas reales referenciadas después de procesar alias
        for table_node in parsed.find_all(sqlglot.exp.Table):
            real_name = table_node.name
            if real_name:
                actual_tables_referenced.add(self._process_name(real_name))
        
        self.logger.info(f"Validación de SQL: Tablas referenciadas (reales, procesadas): {actual_tables_referenced}, Alias de columna SELECT: {list(self.current_query_aliases.keys())}, Contexto de tabla para columnas: {self.current_table_context_for_columns}")

        if not self.allowed_tables:
            self.logger.error("VALIDATION ABORTED: self.allowed_tables is empty.")
            return False, "Error de configuración interna: la lista de tablas permitidas está vacía.", actual_tables_referenced
        
        for table_name_to_check in actual_tables_referenced:
            if not self.is_table_allowed(table_name_to_check): 
                return False, f"Tabla '{table_name_to_check}' no permitida.", actual_tables_referenced
        
        self.logger.debug("Todas las tablas referenciadas están permitidas.")

        # Validar cláusulas principales
        # SELECT clause
        select_expr = parsed.find(sqlglot.exp.Select)
        if select_expr:
            valid_select, msg_select = self._validate_select_clause(select_expr)
            if not valid_select:
                return False, msg_select, actual_tables_referenced
        
        # JOIN ON conditions
        for join_node in parsed.find_all(sqlglot.exp.Join):
            on_expression = join_node.args.get('on')
            table_being_joined_name = join_node.this.sql() if join_node.this else "desconocida"
            if on_expression:
                self.logger.debug(f"Validando JOIN ON para {table_being_joined_name}: {on_expression.sql()}")
                valid_on, msg_on = self._validate_expression_columns(on_expression, f"JOIN ON para {table_being_joined_name}")
                if not valid_on:
                    return False, msg_on, actual_tables_referenced
        
        # WHERE clause
        where_expr = parsed.find(sqlglot.exp.Where)
        if where_expr:
            valid_where, msg_where = self._validate_where_clause(where_expr)
            if not valid_where:
                return False, msg_where, actual_tables_referenced

        # GROUP BY clause
        group_expr = parsed.find(sqlglot.exp.Group)
        if group_expr:
            valid_group, msg_group = self._validate_groupby_clause(group_expr)
            if not valid_group:
                return False, msg_group, actual_tables_referenced

        # ORDER BY clause
        order_expr = parsed.find(sqlglot.exp.Order)
        if order_expr:
            valid_order, msg_order = self._validate_orderby_clause(order_expr)
            if not valid_order:
                return False, msg_order, actual_tables_referenced
        
        self.logger.info(f"Validación de SQL para '{sql_string[:100]}...' completada exitosamente.")
        return True, "SQL permitido.", actual_tables_referenced

    def _extract_aliases_and_table_context(self, parsed_sql: sqlglot.exp.Expression):
        self.current_query_aliases = {} 
        self.current_table_context_for_columns = {}

        for table_node in parsed_sql.find_all(sqlglot.exp.Table):
            real_table_name = table_node.name
            alias_or_name_used = table_node.alias_or_name
            if real_table_name:
                processed_real_name = self._process_name(real_table_name)
                processed_alias_or_name = self._process_name(alias_or_name_used)
                self.current_table_context_for_columns[processed_alias_or_name] = processed_real_name
                if processed_alias_or_name != processed_real_name:
                    # Podríamos tener un self.current_table_aliases si fuera necesario
                    self.logger.debug(f"Alias de tabla: {processed_alias_or_name} -> {processed_real_name}")

        select_expression = parsed_sql.find(sqlglot.exp.Select)
        if select_expression and hasattr(select_expression, 'expressions'):
            for col_expr_node in select_expression.expressions:
                if isinstance(col_expr_node, sqlglot.exp.Alias):
                    alias_name = self._process_name(col_expr_node.alias)
                    self.current_query_aliases[alias_name] = True # Marcar como alias de columna SELECT
                    self.logger.debug(f"Alias de columna SELECT detectado: {alias_name} (de {col_expr_node.this.sql()})")
        
        self.logger.info(f"Contexto de tabla para columnas: {self.current_table_context_for_columns}")
        self.logger.info(f"Alias de columna SELECT: {list(self.current_query_aliases.keys())}")

    def _validate_select_clause(self, select_expr: sqlglot.exp.Select) -> tuple[bool, str]:
        if not select_expr or not hasattr(select_expr, 'expressions') or not select_expr.expressions:
            return True, "No hay columnas en SELECT para validar."

        for item_expr in select_expr.expressions: # 'expressions' es la lista de elementos en SELECT
            self.logger.debug(f"Validando item SELECT: {item_expr.sql()} (Tipo: {type(item_expr)})")
            # Validar todas las columnas dentro de este item (sea un alias, una función, una columna directa, etc.)
            is_valid, msg = self._validate_expression_columns(item_expr, f"item SELECT '{item_expr.sql()}'")
            if not is_valid:
                return False, msg
        return True, "Cláusula SELECT validada."

    def _validate_where_clause(self, where_expr) -> Tuple[bool, str]:
        """
        Valida todas las columnas de la cláusula WHERE.
        """
        return self._validate_expression_columns(where_expr, "WHERE")

    def _validate_groupby_clause(self, group_expr) -> Tuple[bool, str]:
        """
        Valida todas las columnas de la cláusula GROUP BY.
        """
        return self._validate_expression_columns(group_expr, "GROUP BY")

    def _validate_orderby_clause(self, order_expr) -> Tuple[bool, str]:
        """
        Valida todas las columnas de la cláusula ORDER BY.
        """
        return self._validate_expression_columns(order_expr, "ORDER BY")

    def _validate_expression_columns(self, expression_node: Optional[sqlglot.exp.Expression], context_name: str) -> Tuple[bool, str]:
        """
        Validates all columns found within a single SQL expression node.
        Iterates over all instances of sqlglot.exp.Column in the expression.
        """
        if expression_node is None:
            self.logger.debug(f"Expresión nula en {context_name}, validación omitida.")
            return True, f"Expresión nula en {context_name}, validación omitida."

        # Verificar que expression_node es una instancia de sqlglot.exp.Expression
        if not isinstance(expression_node, sqlglot.exp.Expression):
            self.logger.error(
                f"Error de validación interna: Se esperaba un nodo sqlglot.exp.Expression para '{context_name}', "
                f"pero se recibió un tipo {type(expression_node)}. Valor: {str(expression_node)[:200]}"

            )
            # Esto cubre el caso donde expression_node es una función, causando el AttributeError.
            return False, f"Error interno: el objeto para '{context_name}' ('{str(expression_node)[:50]}...') no es un nodo de expresión SQL válido."

        # Ahora es seguro asumir que expression_node es un sqlglot.exp.Expression y tiene .sql() y .find_all()
        self.logger.debug(f"Validando columnas en la expresión para {context_name}: {expression_node.sql()}")

        try:
            column_expressions = expression_node.find_all(sqlglot.exp.Column)
            if column_expressions is None:
                self.logger.debug(f"expression_node.find_all(sqlglot.exp.Column) devolvió None para {context_name}. Asumiendo que no hay columnas directas para validar en este nodo de expresión de tipo {type(expression_node)}.")
                return True, f"No se encontraron columnas directas para validar en {context_name}."

            # Iterar sobre todas las expresiones de columna encontradas dentro de este nodo
            for col_expr in column_expressions:
                # _validate_single_column se encarga de resolver la tabla y verificar la columna.
                is_valid, msg = self._validate_single_column(col_expr, f"columna en {context_name}")
                if not is_valid:
                    return False, msg # msg de _validate_single_column debería ser específico
            
            # La lógica para sqlglot.exp.Star (ej. SELECT *) se maneja principalmente en _validate_select_clause
            # y para COUNT(*) en _validate_single_column.
            # Aquí nos centramos en instancias explícitas de sqlglot.exp.Column.

        except Exception as e_find_all:
            # Esto no debería ocurrir si la comprobación isinstance pasó y sqlglot es consistente.
            self.logger.error(f"Error inesperado al procesar expresión para {context_name} (tipo: {type(expression_node)}): {e_find_all}. SQL: {expression_node.sql()}")
            return False, f"Error interno al analizar las columnas de '{context_name}'."

        return True, f"Todas las columnas en la expresión para {context_name} son válidas."

    def _get_real_table_name_for_column(self, column_expr: sqlglot.exp.Column) -> Optional[str]:
        """
        Intenta obtener el nombre real de la tabla para una columna, resolviendo alias.
        Usa self.current_table_context_for_columns que se pobló en validate_sql_string.
        El nombre de tabla devuelto ya está procesado (ej. lowercased si !case_sensitive).
        """
        table_alias_or_name_in_query = column_expr.table # Esto es el prefijo de la columna, ej. 'p' en 'p.ACIN_ID' o 'MEDI_ACTIVE_INGREDIENTS'
        
        if not table_alias_or_name_in_query:
            if len(self.current_table_context_for_columns) == 1:
                return next(iter(self.current_table_context_for_columns.values()))
            elif not self.current_table_context_for_columns:
                 self.logger.debug(f"No table context for column '{column_expr.name}', cannot resolve real table name.")
                 return None
            else: # multiple tables in context
                self.logger.debug(f"Column '{column_expr.name}' has no explicit table, attempting to find unique table in context from real table names: {list(set(self.current_table_context_for_columns.values()))}.")
                
                found_in_tables = []
                # column_expr.name es el nombre de la columna tal como aparece en la consulta, ej. "ACIN_DESCRIPTION_ES"
                # Necesitamos procesarlo (ej. minúsculas) para la comprobación con la whitelist.
                processed_col_name_for_check = self._process_name(column_expr.name)

                # Iterar sobre los NOMBRES REALES (ya procesados) de las tablas en el contexto del query actual.
                # self.current_table_context_for_columns.values() da los nombres reales procesados.
                # Usamos set() para evitar duplicados si una tabla se referencia varias veces.
                for real_table_name_in_context in set(self.current_table_context_for_columns.values()):
                    if self.is_column_allowed_for_table(real_table_name_in_context, processed_col_name_for_check):
                        found_in_tables.append(real_table_name_in_context)
                
                if len(found_in_tables) == 1:
                    self.logger.info(f"Column '{column_expr.name}' (processed: '{processed_col_name_for_check}') uniquely found and allowed in table '{found_in_tables[0]}' among context tables.")
                    return found_in_tables[0]
                elif len(found_in_tables) > 1:
                    self.logger.warning(f"Column '{column_expr.name}' (processed: '{processed_col_name_for_check}') is ambiguous, found and allowed in multiple tables in context: {found_in_tables}. Query needs to qualify the column.")
                    return None
                else: # len(found_in_tables) == 0
                    self.logger.warning(f"Column '{column_expr.name}' (processed: '{processed_col_name_for_check}') not found or not allowed in any table in context: {list(set(self.current_table_context_for_columns.values()))}.")
                    return None
    

        processed_table_alias_or_name = self._process_name(table_alias_or_name_in_query)

        real_table_name = self.current_table_context_for_columns.get(processed_table_alias_or_name)
        
        if not real_table_name:
            self.logger.warning(f"Could not resolve real table name for alias/table prefix '{table_alias_or_name_in_query}' (processed: '{processed_table_alias_or_name}') using context {self.current_table_context_for_columns}.")
            # Fallback: if the processed_table_alias_or_name is actually a known real table (e.g. no alias was used)
            # self.allowed_tables contains processed names.
            if processed_table_alias_or_name in self.allowed_tables:
                 self.logger.debug(f"Fallback: Resolved '{processed_table_alias_or_name}' as a real table name.")
                 return processed_table_alias_or_name
            self.logger.error(f"Failed to resolve real table name for '{table_alias_or_name_in_query}'.")
            return None # Explicitly return None if no resolution

        return real_table_name

    def _validate_single_column(self, column_expr: sqlglot.exp.Column, clause_name: str) -> Tuple[bool, str]:
        """
        Validates a single column expression.
        Assumes column_expr is sqlglot.exp.Column.
        """
        col_name_from_query = column_expr.name
        
        if col_name_from_query == '*' and not column_expr.table: # ej. COUNT(*)
            # This is a "bare" star, often an argument to an aggregate.
            # It doesn't refer to a specific column to be read from a table in the same way
            # as "SELECT *" or "SELECT col".
            # We assume this is generally permissible if the function itself is allowed (not checked here).
            # Or, if it's part of SELECT *, it's handled by _validate_expression_columns's Star instance check.
            self.logger.debug(f"Bare '*' (e.g., in COUNT(*)) in {clause_name}, considered valid.")
            return True, f"Bare '*' in {clause_name} allowed."

        processed_col_name = self._process_name(col_name_from_query)
        real_table_name = self._get_real_table_name_for_column(column_expr)

        if not real_table_name:
            if processed_col_name in self.current_query_aliases:
                self.logger.debug(f"Columna '{col_name_from_query}' (procesada: '{processed_col_name}') en {clause_name} es un alias SELECT. Permitido.")
                return True, f"Alias SELECT '{processed_col_name}' permitido en {clause_name}."
            else:
                self.logger.warning(f"En {clause_name}: No se pudo determinar tabla para '{col_name_from_query}' y no es alias SELECT. SQL: {column_expr.sql()}")
                return False, f"En {clause_name}: No se pudo determinar tabla para '{col_name_from_query}' y no es alias SELECT."
        
        if not self.is_column_allowed_for_table(real_table_name, processed_col_name):
            return False, f"En {clause_name}: Columna '{processed_col_name}' no permitida para tabla '{real_table_name}'."    
        
        self.logger.debug(f"Columna '{processed_col_name}' permitida para tabla '{real_table_name}' en {clause_name}.")
        return True, f"Columna '{processed_col_name}' permitida para tabla '{real_table_name}'." # Corregido el f-string

    def _process_name(self, name: str) -> str:
        """
        Processes a name based on the case sensitivity setting.
        If case_sensitive is False, converts the name to lowercase.
        """
        return name if self.case_sensitive else name.lower()
    
    def _process_column_list(self, columns: list[str]) -> Set[str]:
        """ Processes a list of column names based on the case sensitivity setting.     
        If case_sensitive is False, converts each column name to lowercase.
        Returns a set of processed column names.
        """
        return {self._process_name(col) for col in columns} # Asegurar que esta línea no tenga indentación extra.