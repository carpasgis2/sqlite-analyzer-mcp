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
    
    def __init__(self, allowed_tables: List[str], allowed_columns: Dict[str, List[str]], enhanced_schema_path: Optional[str] = None):
        """
        Inicializa el generador SQL con las tablas y columnas permitidas y opcionalmente un esquema mejorado.
        
        Args:
            allowed_tables: Lista de nombres de tablas permitidas
            allowed_columns: Diccionario que mapea tablas a sus columnas permitidas
            enhanced_schema_path: Ruta opcional al archivo schema_enhanced.json
        """
        self.allowed_tables = allowed_tables
        self.allowed_columns = allowed_columns
        self.enhanced_schema = None
        if enhanced_schema_path:
            self._load_enhanced_schema(enhanced_schema_path)
        
    def _load_enhanced_schema(self, schema_path: str):
        """Carga el esquema mejorado desde un archivo JSON."""
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                self.enhanced_schema = json.load(f)
            logging.info(f"[SQLGenerator] Esquema mejorado cargado exitosamente desde {schema_path}")
        except FileNotFoundError:
            logging.error(f"[SQLGenerator] Archivo de esquema mejorado no encontrado en {schema_path}. La normalización de columnas no estará disponible.")
            self.enhanced_schema = None
        except json.JSONDecodeError:
            logging.error(f"[SQLGenerator] Error al decodificar el archivo JSON del esquema mejorado en {schema_path}. La normalización de columnas no estará disponible.")
            self.enhanced_schema = None
        except Exception as e:
            logging.error(f"[SQLGenerator] Ocurrió un error inesperado al cargar el esquema mejorado desde {schema_path}: {e}")
            self.enhanced_schema = None

    def _find_actual_column_name(self, table_name: str, column_name_variant: str) -> str:
        """
        Intenta encontrar el nombre real de una columna en el esquema mejorado,
        probando variaciones como añadir/quitar el sufijo '_ES'.
        Devuelve None si no se encuentra o no hay esquema.
        """
        if not self.enhanced_schema or not table_name or not column_name_variant:
            return None

        table_name_upper = table_name.upper()
        column_name_variant_upper = column_name_variant.upper()

        schema_table_name_key = None
        for t_name_key in self.enhanced_schema.get("tables", {{}}).keys():
            if t_name_key.upper() == table_name_upper:
                schema_table_name_key = t_name_key
                break
        if not schema_table_name_key:
            logging.debug(f"[SQLGenerator] Tabla '{table_name}' no encontrada en el esquema mejorado para normalizar columna '{column_name_variant}'.")
            return None

        table_schema = self.enhanced_schema["tables"].get(schema_table_name_key, {{}})
        actual_columns_in_schema = {{col_info.get("name", "").upper(): col_info.get("name", "") 
                                    for col_info in table_schema.get("columns", []) if col_info.get("name")}}

        if column_name_variant_upper in actual_columns_in_schema:
            return actual_columns_in_schema[column_name_variant_upper]

        variations_to_try = []
        if column_name_variant_upper.endswith("_ES"):
            variations_to_try.append(column_name_variant_upper[:-3])
        else:
            variations_to_try.append(column_name_variant_upper + "_ES")
        if column_name_variant_upper.endswith("_DESCRIPTION_ES"):
            variations_to_try.append(column_name_variant_upper.replace("_DESCRIPTION_ES", "_DESCRIPTION"))
        elif column_name_variant_upper.endswith("_DESCRIPTION"):
            variations_to_try.append(column_name_variant_upper.replace("_DESCRIPTION", "_DESCRIPTION_ES"))

        for variation_upper in variations_to_try:
            if variation_upper in actual_columns_in_schema:
                logging.info(f"[SQLGenerator] Columna '{column_name_variant}' normalizada a '{actual_columns_in_schema[variation_upper]}' para la tabla '{table_name}'.")
                return actual_columns_in_schema[variation_upper]

        logging.warning(f"[SQLGenerator] No se pudo encontrar una coincidencia para la columna '{column_name_variant}' en la tabla '{table_name}' usando el esquema mejorado. Se omitirá la condición.")
        return None

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
    
    def _normalize_on_condition_string(self, on_condition_str: str) -> str:
        """
        Normaliza los nombres de columna dentro de una cadena de condición ON.
        Ejemplo: "TABLE1.COL_VAR = TABLE2.OTHER_COL_VAR" -> "TABLE1.COL_ACTUAL = TABLE2.OTHER_COL_ACTUAL"
        Actualmente maneja condiciones simples T1.C1 = T2.C2.
        """
        if not self.enhanced_schema or '=' not in on_condition_str:
            return on_condition_str

        # Usar regex para capturar T1.C1 = T2.C2 de forma más robusta
        match = re.fullmatch(r"\s*([\w_]+)\.([\w_]+)\s*=\s*([\w_]+)\.([\w_]+)\s*", on_condition_str)
        if not match:
            logging.debug(f"[SQLGenerator] Condición ON '{on_condition_str}' no coincide con el patrón T1.C1 = T2.C2 para normalización.")
            return on_condition_str

        t1, c1_variant, t2, c2_variant = match.groups()

        actual_c1 = self._find_actual_column_name(t1, c1_variant)
        actual_c2 = self._find_actual_column_name(t2, c2_variant)

        normalized_condition = f"{t1}.{actual_c1} = {t2}.{actual_c2}"
        if normalized_condition != on_condition_str:
            logging.info(f"[SQLGenerator] Condición ON normalizada: '{on_condition_str}' -> '{normalized_condition}'")
        return normalized_condition

    def generate_sql(self, structured_info: Dict[str, Any], db_structure: Optional[Dict[str, Any]] = None, relations_map: Optional[Dict[str, Any]] = None) -> Tuple[str, List[Any]]:
        """
        Genera una consulta SQL a partir de información estructurada.
        Args:
            structured_info: Diccionario con la información estructurada de la consulta.
            db_structure: (Opcional) Diccionario con la estructura de la base de datos.
            relations_map: (Opcional) Diccionario con el mapa de relaciones entre tablas.

        Returns:
            Una tupla con la consulta SQL generada y una lista de parámetros.
        """
        logging.info("[SQLGenerator] Iniciando generate_sql") # LOG INICIO MÉTODO
        params = []

        logging.debug(f"[SQLGenerator] structured_info recibido: {json.dumps(structured_info, indent=2)}")

        raw_tables_list = structured_info.get("tables")
        if not raw_tables_list or not isinstance(raw_tables_list, list) or not raw_tables_list[0]:
            logging.error("[SQLGenerator] No se especificó ninguna tabla válida en structured_info['tables'].")
            return "SELECT 'Error: No se especificó ninguna tabla válida' AS mensaje", []

        # Validación robusta: abortar si alguna tabla no existe en el esquema permitido
        if db_structure is not None:
            tablas_invalidas = [t for t in raw_tables_list if t not in db_structure]
            if tablas_invalidas:
                logging.error(f"No existen en el esquema las siguientes tablas requeridas: {tablas_invalidas}")
                return f"SELECT 'Error: No se puede responder porque faltan tablas en la base de datos: {', '.join(tablas_invalidas)}' AS mensaje", []

        # Estrategia para elegir la tabla principal:
        candidate_main_tables_ordered = [] # Lista ordenada de candidatos por prioridad
        
        # Prioridad 1: Tablas en columnas SELECT
        select_cols_info = structured_info.get("columns", [])
        if isinstance(select_cols_info, list):
            for col_ref in select_cols_info:
                if isinstance(col_ref, str) and '.' in col_ref:
                    candidate_main_tables_ordered.append(col_ref.split('.')[0].upper())

        # Prioridad 2: Tablas en condiciones WHERE
        conditions_info = structured_info.get("conditions", [])
        if isinstance(conditions_info, list):
            for cond in conditions_info:
                # CAMBIO: Usar "column" en lugar de "field" para que coincida con la estructura de 'conditions'
                if isinstance(cond, dict) and "column" in cond and isinstance(cond["column"], str) and '.' in cond["column"]:
                    candidate_main_tables_ordered.append(cond["column"].split('.')[0].upper())
        
        # Prioridad 3: Tablas en JOINs del LLM (t1 primero, luego t2)
        llm_joins_input = structured_info.get("joins", [])
        if isinstance(llm_joins_input, list):
            for join_item in llm_joins_input:
                if isinstance(join_item, dict):
                    t1 = join_item.get('table', join_item.get('table1'))
                    t2 = join_item.get('foreign_table', join_item.get('table2'))
                    if t1 and isinstance(t1, str): candidate_main_tables_ordered.append(t1.upper())
                    if t2 and isinstance(t2, str): candidate_main_tables_ordered.append(t2.upper())
        
        # Añadir todas las tablas de raw_tables_list como candidatos de menor prioridad
        for tbl in raw_tables_list:
            if isinstance(tbl, str):
                candidate_main_tables_ordered.append(tbl.upper())
        
        main_table = None
        raw_tables_list_upper_set = {tbl.upper() for tbl in raw_tables_list if isinstance(tbl, str)}

        if not raw_tables_list_upper_set:
             logging.error("[SQLGenerator] El conjunto de tablas válidas (raw_tables_list_upper_set) está vacío.")
             return "SELECT 'Error: No hay tablas válidas para procesar' AS mensaje", []

        for cand_upper in candidate_main_tables_ordered:
            if cand_upper in raw_tables_list_upper_set:
                # Encontrar el nombre original con la capitalización correcta
                try:
                    original_cand_idx = [tbl.upper() for tbl in raw_tables_list].index(cand_upper)
                    main_table = raw_tables_list[original_cand_idx]
                    logging.info(f"[SQLGenerator] Tabla principal elegida por heurística: {main_table} (Candidato: {cand_upper})")
                    break
                except ValueError: # pragma: no cover
                    # Esto no debería ocurrir si cand_upper está en raw_tables_list_upper_set
                    logging.warning(f"[SQLGenerator] Candidato {cand_upper} encontrado pero no se pudo obtener su nombre original.")
                    continue 
        
        if not main_table:
            # Fallback si ningún candidato priorizado es válido o si raw_tables_list_upper_set estaba vacío inicialmente
            # (aunque el chequeo anterior de raw_tables_list_upper_set debería prevenir esto)
            first_valid_raw_table = next((tbl for tbl in raw_tables_list if isinstance(tbl, str)), None)
            if first_valid_raw_table:
                main_table = first_valid_raw_table
                logging.info(f"[SQLGenerator] Tabla principal elegida por fallback (primera de la lista raw): {main_table}")
            else: # pragma: no cover
                 # Esto no debería ser alcanzable si raw_tables_list tiene al menos una cadena válida.
                logging.error("[SQLGenerator] No se pudo determinar una tabla principal válida.")
                return "SELECT 'Error: No se pudo determinar la tabla principal' AS mensaje", []


        logging.info(f"[SQLGenerator] Tabla principal FINAL: {main_table}")

        select_cols_input = structured_info.get("columns", ["*"])
        if not select_cols_input:
            select_cols_input = ["*"]
        
        normalized_select_cols = []
        if isinstance(select_cols_input, list):
            for col_ref in select_cols_input: # CORREGIDO: Usar select_cols_input
                if isinstance(col_ref, str):
                    if col_ref == "*":
                        normalized_select_cols.append("*")
                        continue
                    
                    table_prefix = main_table 
                    col_name_part = col_ref
                    if '.' in col_ref:
                        parts = col_ref.split('.', 1)
                        table_prefix = parts[0]
                        col_name_part = parts[1]
                    
                    actual_col_name = self._find_actual_column_name(table_prefix, col_name_part)
                    
                    if '.' in col_ref:
                        normalized_select_cols.append(f"{table_prefix}.{actual_col_name}")
                    elif actual_col_name != col_name_part: 
                        normalized_select_cols.append(f"{main_table}.{actual_col_name}")
                    else: 
                        normalized_select_cols.append(actual_col_name)
                else:
                    normalized_select_cols.append(str(col_ref)) 
        else:
            normalized_select_cols = ["*"] 

        logging.info(f"[SQLGenerator] Columnas a seleccionar (normalizadas): {normalized_select_cols}")

        sql = f"SELECT {', '.join(normalized_select_cols)} FROM {main_table}"
        logging.debug(f"[SQLGenerator] SQL inicial: {sql}")

        already_joined_tables = {main_table.upper()}
        join_clauses_sql = []
        
        llm_joins_input = structured_info.get("joins", []) # Asegurar que llm_joins_input está definido aquí
        # Procesar JOINs explícitos del LLM de forma iterativa
        if isinstance(llm_joins_input, list) and llm_joins_input:
            current_llm_joins_to_process = list(llm_joins_input)
            max_passes = len(current_llm_joins_to_process) + 2 # Un poco más de margen

            for pass_num in range(max_passes):
                if not current_llm_joins_to_process:
                    logging.info(f"[SQLGenerator] Todos los {len(llm_joins_input)} JOINs del LLM procesados o descartados en {pass_num} pasadas.")
                    break

                joins_added_in_this_pass_count = 0
                next_round_pending_joins = []
                logging.debug(f"[SQLGenerator] LLM JOINs - Pase {pass_num + 1}: Procesando {len(current_llm_joins_to_process)} JOINs pendientes.")

                for i, join_item in enumerate(current_llm_joins_to_process):
                    logging.debug(f"[SQLGenerator] LLM JOINs - Pase {pass_num + 1}, Item {i+1}: {join_item}")
                    if not isinstance(join_item, dict):
                        logging.warning(f"[SQLGenerator] Elemento JOIN del LLM no es un diccionario: {join_item}. Se omite.")
                        continue

                    join_type = join_item.get('join_type', join_item.get('type', 'INNER')).upper()
                    llm_table1 = join_item.get('table', join_item.get('table1'))
                    llm_table2 = join_item.get('foreign_table', join_item.get('table2'))
                    on_condition = join_item.get('on')

                    if not (llm_table1 and isinstance(llm_table1, str) and \
                              llm_table2 and isinstance(llm_table2, str) and \
                              on_condition and isinstance(on_condition, str)):
                        
                        if not on_condition and llm_table1 and llm_table2:
                            col1_variant = join_item.get('column')
                            col2_variant = join_item.get('foreign_column')
                            if col1_variant and col2_variant and isinstance(col1_variant, str) and isinstance(col2_variant, str):
                                # Normalizar columnas antes de construir la condición ON
                                actual_col1 = self._find_actual_column_name(llm_table1, col1_variant)
                                actual_col2 = self._find_actual_column_name(llm_table2, col2_variant)
                                on_condition = f"{llm_table1}.{actual_col1} = {llm_table2}.{actual_col2}"
                                logging.debug(f"[SQLGenerator] ON condition construida y normalizada para JOIN LLM: {on_condition}")
                            else:
                                logging.warning(f"[SQLGenerator] JOIN LLM #{i+1} omitido: faltan tablas, on_condition o columnas para construirla. Item: {join_item}")
                                next_round_pending_joins.append(join_item) 
                                continue
                        else:
                            logging.warning(f"[SQLGenerator] JOIN LLM #{i+1} omitido: información de tabla o condición ON inválida/faltante. Item: {join_item}")
                            continue
                    else:
                        # Normalizar la condición ON existente si fue provista directamente
                        on_condition = self._normalize_on_condition_string(on_condition)
                    
                    llm_table1_upper = llm_table1.upper()
                    llm_table2_upper = llm_table2.upper()

                    can_process_this_join = False
                    table_to_add_to_sql_clause_str = None # Nombre original de la tabla a añadir
                    table_to_add_to_already_joined_set_upper = None

                    if llm_table1_upper in already_joined_tables and llm_table2_upper not in already_joined_tables:
                        can_process_this_join = True
                        table_to_add_to_sql_clause_str = llm_table2 # Usar el nombre con capitalización original
                        table_to_add_to_already_joined_set_upper = llm_table2_upper
                    elif llm_table2_upper in already_joined_tables and llm_table1_upper not in already_joined_tables:
                        can_process_this_join = True
                        table_to_add_to_sql_clause_str = llm_table1 # Usar el nombre con capitalización original
                        table_to_add_to_already_joined_set_upper = llm_table1_upper
                    elif llm_table1_upper in already_joined_tables and llm_table2_upper in already_joined_tables:
                        logging.debug(f"[SQLGenerator] JOIN LLM {join_item} conecta tablas ({llm_table1}, {llm_table2}) ya en el grafo. Se omite la adición de cláusula JOIN duplicada, pero se considera procesado.")
                        # No se añade cláusula, pero se considera "procesado" para que no quede pendiente.
                        joins_added_in_this_pass_count +=1 # Contabilizar como manejado para la lógica del bucle
                        continue 

                    if can_process_this_join and table_to_add_to_sql_clause_str and table_to_add_to_already_joined_set_upper:
                        # TODO: Validación de columnas en on_condition usando db_structure si está disponible
                        # Por ahora, se asume que el LLM proporciona on_conditions válidas.
                        
                        on_condition_normalized = self._normalize_on_condition_string(on_condition)

                        join_sql_segment = f" {join_type} JOIN {table_to_add_to_sql_clause_str} ON {on_condition_normalized}"
                        join_clauses_sql.append(join_sql_segment)
                        already_joined_tables.add(table_to_add_to_already_joined_set_upper)
                        logging.info(f"[SQLGenerator] JOIN LLM procesado y añadido: {join_sql_segment}. Tablas unidas ahora: {already_joined_tables}")
                        joins_added_in_this_pass_count += 1
                    else:
                        logging.debug(f"[SQLGenerator] JOIN LLM {join_item} no se puede procesar en esta pasada (una tabla no conectada o ambas ya conectadas sin ser el caso anterior). Se pasa a la siguiente ronda.")
                        next_round_pending_joins.append(join_item)
                
                current_llm_joins_to_process = next_round_pending_joins
                
                if not current_llm_joins_to_process: # Todos los JOINs del LLM se procesaron o descartaron
                    logging.info(f"[SQLGenerator] Todos los JOINs del LLM manejados en la pasada {pass_num + 1}.")
                    break 
                
                if joins_added_in_this_pass_count == 0:
                    logging.warning(f"[SQLGenerator] No se añadieron nuevos JOINs del LLM en la pasada {pass_num + 1}, pero {len(current_llm_joins_to_process)} JOINs siguen pendientes. Deteniendo el procesamiento de JOINs del LLM.")
                    break
            
            if current_llm_joins_to_process:
                 logging.error(f"[SQLGenerator] {len(current_llm_joins_to_process)} JOINs del LLM no pudieron ser integrados al grafo final después de {max_passes} pasadas: {current_llm_joins_to_process}")

        # Consolidar todas las tablas que necesitan estar en la consulta para la inferencia de JOINs

        # Consolidar todas las tablas que necesitan estar en la consulta para la inferencia de JOINs
        all_tables_from_structured_info = {tbl.upper() for tbl in raw_tables_list if isinstance(tbl, str)} # Usar la lista original de tablas
        
        tables_from_select_upper = set()
        # Usar las columnas originales de structured_info para determinar las tablas mencionadas en SELECT
        original_select_columns = structured_info.get("columns", [])
        if isinstance(original_select_columns, list):
            for col_ref in original_select_columns:
                if isinstance(col_ref, str) and '.' in col_ref:
                    # Extraer solo el nombre de la tabla antes del primer punto
                    table_name_candidate = col_ref.split('.')[0].upper()
                    # Validar que no sea parte de una función, ej. "COUNT(TABLE.COL)"
                    # Esta es una heurística simple; un parseo SQL completo sería más robusto.
                    if '(' not in table_name_candidate:
                        tables_from_select_upper.add(table_name_candidate)
        
        tables_from_where_upper = set()

        if isinstance(conditions_info, list): # Reusar conditions_info
            for cond in conditions_info:
                if isinstance(cond, dict) and "column" in cond and isinstance(cond["column"], str) and '.' in cond["column"]:
                    tables_from_where_upper.add(cond["column"].split('.')[0].upper())

        required_tables_for_query_inference = set(all_tables_from_structured_info) # Empezar con todas las tablas mencionadas
        required_tables_for_query_inference.update(tables_from_select_upper)
        required_tables_for_query_inference.update(tables_from_where_upper)
        # Asegurar que las tablas de los JOINs del LLM (incluso los no procesados) estén consideradas si estaban en la lista original de tablas
        # already_joined_tables ya contiene las tablas de los JOINs del LLM que SÍ se procesaron.
        # El objetivo de la inferencia es unir las de required_tables_for_query_inference que aún no estén en already_joined_tables.

        logging.info(f"[SQLGenerator] Tablas requeridas para inferencia de JOINs (después de JOINs del LLM): {required_tables_for_query_inference}")
        logging.info(f"[SQLGenerator] Tablas ya unidas (después de JOINs del LLM y antes de inferencia): {already_joined_tables}")
        
        # --- INICIO SECCIÓN DE INFERENCIA DE JOINS (adaptada) ---
        if relations_map and db_structure:
            max_join_inference_passes = len(required_tables_for_query_inference) + 1
            for pass_num_inf in range(max_join_inference_passes):
                if already_joined_tables.issuperset(required_tables_for_query_inference):
                    logging.info(f"[SQLGenerator] Inferencia de JOINs: Todas las {len(required_tables_for_query_inference)} tablas requeridas ya están unidas.")
                    break

                new_join_made_in_inference_pass = False
                tables_to_try_to_reach_in_inference = list(required_tables_for_query_inference - already_joined_tables)
                
                if not tables_to_try_to_reach_in_inference:
                    logging.debug("[SQLGenerator] Inferencia de JOINs: No quedan tablas por alcanzar.")
                    break
                
                logging.debug(f"[SQLGenerator] Inferencia de JOINs (Pase {pass_num_inf+1}): Intentando alcanzar {tables_to_try_to_reach_in_inference} desde {already_joined_tables}")

                for table_to_reach_upper in tables_to_try_to_reach_in_inference:
                    found_path_for_current_target = False
                    for joined_table_upper in list(already_joined_tables): # Iterar sobre copia
                        # Asumimos que relations_map usa claves en MAYÚSCULAS y nombres de tabla en MAYÚSCULAS dentro de las relaciones
                        possible_relations = relations_map.get(joined_table_upper, []) 
                        
                        for rel in possible_relations:
                            if not isinstance(rel, dict): # pragma: no cover
                                logging.warning(f"[SQLGenerator] Elemento de relación no es un dict: {rel} para tabla {joined_table_upper}")
                                continue

                            rel_table1_upper = rel.get("table", "").upper()
                            rel_table2_upper = rel.get("foreign_table", "").upper()
                            on_condition_inferred_raw = None # Condición ON sin normalizar
                            table_to_add_via_inferred_join_upper = None
                            original_table_name_to_join_inferred = None

                            # Nombres de columna originales de la relación
                            rel_col1 = rel.get('column')
                            rel_col2 = rel.get('foreign_column')

                            if not rel_col1 or not rel_col2:
                                logging.warning(f"[SQLGenerator] Relación inválida, faltan nombres de columna: {rel}")
                                continue

                            if rel_table1_upper == joined_table_upper and rel_table2_upper == table_to_reach_upper:
                                # No normalizar aquí todavía, _normalize_on_condition_string lo hará
                                on_condition_inferred_raw = f"{rel.get('table')}.{rel_col1} = {rel.get('foreign_table')}.{rel_col2}"
                                table_to_add_via_inferred_join_upper = table_to_reach_upper
                                original_table_name_to_join_inferred = rel.get('foreign_table')
                            elif rel_table2_upper == joined_table_upper and rel_table1_upper == table_to_reach_upper:
                                # No normalizar aquí todavía, _normalize_on_condition_string lo hará
                                on_condition_inferred_raw = f"{rel.get('foreign_table')}.{rel_col2} = {rel.get('table')}.{rel_col1}"
                                table_to_add_via_inferred_join_upper = table_to_reach_upper
                                original_table_name_to_join_inferred = rel.get('table')
                            
                            if on_condition_inferred_raw and table_to_add_via_inferred_join_upper and \
                               table_to_add_via_inferred_join_upper not in already_joined_tables and \
                               original_table_name_to_join_inferred:
                                
                                # Normalizar la condición ON inferida
                                on_condition_inferred_normalized = self._normalize_on_condition_string(on_condition_inferred_raw)
                                
                                join_type_inferred = rel.get("join_type", "INNER").upper()
                                join_sql_segment = f" {join_type_inferred} JOIN {original_table_name_to_join_inferred} ON {on_condition_inferred_normalized}"
                                join_clauses_sql.append(join_sql_segment)
                                already_joined_tables.add(table_to_add_via_inferred_join_upper)
                                new_join_made_in_inference_pass = True
                                found_path_for_current_target = True
                                logging.info(f"[SQLGenerator] JOIN INFERIDO añadido: {join_sql_segment} (Original: {on_condition_inferred_raw})")
                                break 
                        
                        if found_path_for_current_target:
                            break 
                
                if not new_join_made_in_inference_pass and not already_joined_tables.issuperset(required_tables_for_query_inference):
                    missing_tables = required_tables_for_query_inference - already_joined_tables
                    logging.warning(f"[SQLGenerator] Inferencia de JOINs (Pase {pass_num_inf+1}): No se pudieron inferir más JOINs, pero aún faltan: {missing_tables}.")
                    break 
            
        # Después de todos los intentos de JOIN (LLM e inferidos), verificar si todas las tablas requeridas están conectadas.
        final_missing_tables = required_tables_for_query_inference - already_joined_tables
        if final_missing_tables:
            error_detail = f"Faltan tablas: {', '.join(sorted(list(final_missing_tables)))}" # sorted para consistencia en logs/mensajes
            
            # Determinar la causa del fallo
            if not (relations_map and db_structure) and required_tables_for_query_inference != already_joined_tables:
                # No se pudo intentar la inferencia porque faltaba relations_map/db_structure, y se necesitaban JOINs.
                log_message = (f"[SQLGenerator] No se proporcionó 'relations_map' y/o 'db_structure'. "
                               f"{error_detail}. La inferencia de JOINs no se pudo realizar.")
                error_query_message = (f"Error: Falta mapa de relaciones y/o estructura DB y no se pudieron unir las tablas. "
                                       f"{error_detail}")
            else:
                # La inferencia se intentó (relations_map y db_structure estaban disponibles) pero falló en conectar todo.
                log_message = f"[SQLGenerator] ERROR CRÍTICO de INFERENCIA: Después de todos los pases, {error_detail}."
                error_query_message = f"Error: No se pudieron conectar todas las tablas requeridas para la consulta. {error_detail}"
            
            logging.error(log_message)
            # Escapar comillas simples en el mensaje de error para que sea una cadena SQL válida
            safe_error_query_message = error_query_message.replace("'", "''")
            return f"SELECT '{safe_error_query_message}' AS mensaje_error_sql_generator", []

        # --- FIN SECCIÓN DE INFERENCIA DE JOINS ---

        if join_clauses_sql:
            sql += "".join(join_clauses_sql)

        logging.info("[SQLGenerator] Procesamiento de JOINs (LLM + inferidos) completado.")
        logging.debug(f"[SQLGenerator] SQL después de JOINs: {sql}")
        # --- FIN LÓGICA JOIN MEJORADA ---

        # Procesar condiciones WHERE
        conditions = structured_info.get("conditions", [])
        logging.info(f"[SQLGenerator] Procesando {len(conditions)} condiciones WHERE...")
        where_clauses = []

        # Añadir la condición del ID de paciente si está presente
        patient_id_value = structured_info.get("patient_id")
        # Columna por defecto o la detectada. Asegurarse que tiene el nombre de la tabla si es ambiguo.
        # Esto podría necesitar más lógica si patient_id_column no incluye la tabla y es ambiguo.
        patient_id_column_name = structured_info.get("patient_id_column", "PATI_ID") 

        # Heurística simple para prefijar con tabla principal si no tiene ya un punto
        if '.' not in patient_id_column_name and main_table:
            patient_id_column_qualified = f"{main_table}.{patient_id_column_name}"
        else:
            patient_id_column_qualified = patient_id_column_name

        if patient_id_value is not None:
            has_existing_patient_id_condition = False
            if isinstance(conditions, list):
                for cond in conditions:
                    if isinstance(cond, dict):
                        # Comprobar si la columna de la condición (con o sin prefijo de tabla) 
                        # coincide con la columna de ID de paciente (con o sin prefijo de tabla)
                        cond_col = cond.get("column", "")
                        if cond_col == patient_id_column_name or cond_col == patient_id_column_qualified:
                            has_existing_patient_id_condition = True
                            logging.info(f"[SQLGenerator] Condición para {patient_id_column_qualified} (o {patient_id_column_name}) ya existe en 'conditions'. No se añadirá patient_id por separado.")
                            break
            
            if not has_existing_patient_id_condition:
                logging.info(f"[SQLGenerator] Añadiendo condición para patient_id: {patient_id_column_qualified} = {patient_id_value}")
                where_clauses.append(f"{patient_id_column_qualified} = ?")
                params.append(patient_id_value)

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
                column_variant = condition_item.get("column", "")
                operator = condition_item.get("operator", "=")
                value = condition_item.get("value", "")
                is_subquery = condition_item.get("is_subquery", False)
                subquery_details = condition_item.get("subquery_details")
                table_for_column = main_table # Por defecto
                actual_column_name = column_variant
                if '.' in column_variant:
                    parts = column_variant.split('.',1)
                    table_for_column = parts[0]
                    column_name_only = parts[1]
                    actual_column_name = self._find_actual_column_name(table_for_column, column_name_only)
                    if actual_column_name:
                        actual_column_name = f"{table_for_column}.{actual_column_name}"
                elif column_variant:
                    actual_column_name_candidate = self._find_actual_column_name(main_table, column_variant)
                    if actual_column_name_candidate:
                        if actual_column_name_candidate != column_variant:
                            actual_column_name = f"{main_table}.{actual_column_name_candidate}"
                        else:
                            actual_column_name = actual_column_name_candidate
                    else:
                        actual_column_name = None

                if not actual_column_name and not is_subquery: 
                    logging.warning(f"[SQLGenerator] Condición omitida por falta de columna (después de validación en esquema): {condition_item}")
                    continue

                operator = self._normalize_operator(operator)
                logging.debug(f"[SQLGenerator] Condición dict: col='{actual_column_name}' (original: '{column_variant}'), op='{operator}', val='{value}', subquery={is_subquery}")
                
                if is_subquery and subquery_details:
                    sub_select_variant = subquery_details.get("select_column")
                    sub_from = subquery_details.get("from_table")
                    sub_where_col_variant = subquery_details.get("where_column")
                    sub_where_op = self._normalize_operator(subquery_details.get("where_operator", "LIKE"))
                    sub_where_val = subquery_details.get("where_value")

                    # Normalizar columnas de la subconsulta
                    actual_sub_select = sub_select_variant
                    if sub_from and sub_select_variant and '.' not in sub_select_variant: # Asumir que pertenece a sub_from si no está calificada
                        actual_sub_select = self._find_actual_column_name(sub_from, sub_select_variant)
                    elif sub_select_variant and '.' in sub_select_variant:
                        sub_sel_table, sub_sel_col = sub_select_variant.split('.',1)
                        actual_sub_select = f"{sub_sel_table}.{self._find_actual_column_name(sub_sel_table, sub_sel_col)}"
                    
                    actual_sub_where_col = sub_where_col_variant
                    if sub_from and sub_where_col_variant and '.' not in sub_where_col_variant:
                        actual_sub_where_col = self._find_actual_column_name(sub_from, sub_where_col_variant)
                    elif sub_where_col_variant and '.' in sub_where_col_variant:
                        sub_wh_table, sub_wh_col = sub_where_col_variant.split('.',1)
                        actual_sub_where_col = f"{sub_wh_table}.{self._find_actual_column_name(sub_wh_table, sub_wh_col)}"

                    if not all([actual_sub_select, sub_from, actual_sub_where_col, sub_where_val]):
                        logging.warning(f"[SQLGenerator] Subconsulta mal formada (después de normalización), omitida: {subquery_details}")
                        continue
                    
                    processed_sub_where_val = sub_where_val
                    if sub_where_op == "LIKE" and not ('%' in sub_where_val or '_' in sub_where_val):
                        processed_sub_where_val = f"%{sub_where_val}%"


                    where_clauses.append(f"{actual_column_name} {operator} (SELECT {actual_sub_select} FROM {sub_from} WHERE {actual_sub_where_col} {sub_where_op} ?)")
                    params.append(processed_sub_where_val)
                    logging.debug(f"[SQLGenerator] Subconsulta añadida: {actual_column_name} {operator} (SELECT {actual_sub_select} FROM {sub_from} WHERE {actual_sub_where_col} {sub_where_op} {processed_sub_where_val})")

                elif operator in ["IN", "NOT IN"]:
                    if isinstance(value, list) and value:
                        placeholders = ", ".join(["?" for _ in value])
                        where_clauses.append(f"{actual_column_name} {operator} ({placeholders})")
                        params.extend(value)
                    elif isinstance(value, str) and value: 
                        where_clauses.append(f"{actual_column_name} {operator} (?)")
                        params.append(value)
                    else:
                        logging.warning(f"[SQLGenerator] Valor no válido para operador {operator} en columna {actual_column_name}: {value}. Condición omitida.")
                        continue
                elif operator == "BETWEEN":
                    # ... (código existente para BETWEEN, sin cambios)
                    if isinstance(value, list) and len(value) == 2:
                        where_clauses.append(f"{actual_column_name} BETWEEN ? AND ?")
                        params.extend(value)
                    else:
                        logging.warning(f"[SQLGenerator] Valor no válido para operador BETWEEN en columna {actual_column_name}: {value}. Condición omitida.")
                        continue
                elif operator == "LIKE":
                    # ... (código existente para LIKE, sin cambios)
                    if not isinstance(value, str):
                        try:
                            value = str(value)
                        except:
                            logging.warning(f"[SQLGenerator] No se pudo convertir el valor para LIKE a string en columna {actual_column_name}: {value}. Condición omitida.")
                            continue
                    # Asegurarse de que el valor para LIKE tenga comodines si es necesario
                    # Esta es una heurística, el LLM debería idealmente proveerlos.
                    processed_value = value
                    if not ('%' in value or '_' in value):
                        processed_value = f"%{value}%" # Añadir comodines si no están
                    
                    where_clauses.append(f"{actual_column_name} {operator} ?")
                    params.append(processed_value)
                else: 
                    # ... (código existente para otros operadores, sin cambios)
                    where_clauses.append(f"{actual_column_name} {operator} ?")
                    params.append(value)
                logging.debug(f"[SQLGenerator] Condición dict procesada. where_clauses: {where_clauses}, params: {params}")
            
            elif isinstance(condition_item, str):
                # ... (código existente para condiciones string, sin cambios)
                logging.debug(f"[SQLGenerator] Procesando condición string: '{condition_item}'")
                # Simplificación: Asumimos que las condiciones string ya están bien formadas o son simples.
                # Una mejora futura podría ser parsear estas strings de forma más robusta.
                # Por ahora, si es una string, la añadimos tal cual si no contiene '?'
                # Si contiene '?', asumimos que los parámetros ya están en `params` (esto es arriesgado)
                if '?' not in condition_item:
                    where_clauses.append(condition_item) # Añadir directamente si no hay placeholders
                else:
                    # Esto es problemático si la string viene con '?' pero los params no están alineados.
                    # Se mantiene por retrocompatibilidad pero se debería evitar.
                    logging.warning(f"[SQLGenerator] Condición string con '?' encontrada: '{condition_item}'. Se asume que los parámetros están correctamente gestionados externamente.")
                    where_clauses.append(condition_item)

        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
            logging.debug(f"[SQLGenerator] SQL con WHERE: {sql}")
        else:
            logging.info("[SQLGenerator] No se generaron cláusulas WHERE.")

        # Aplicar GROUP BY si está presente
        group_by_cols = structured_info.get("group_by")
        if group_by_cols:
            if isinstance(group_by_cols, str):
                group_by_cols = [group_by_cols]
            if isinstance(group_by_cols, list) and all(isinstance(col, str) for col in group_by_cols):
                sql += f" GROUP BY {", ".join(group_by_cols)}"
                logging.info(f"[SQLGenerator] Cláusula GROUP BY añadida: {group_by_cols}")
            else:
                logging.warning(f"[SQLGenerator] 'group_by' tiene un formato inválido y será ignorado: {group_by_cols}")

        # Aplicar ORDER BY si está presente
        order_by_info = structured_info.get("order_by")
        if order_by_info:
            order_clauses = []
            if isinstance(order_by_info, dict): # Formato: {"column": "COL_NAME", "direction": "ASC|DESC"}
                col = order_by_info.get("column")
                direction = order_by_info.get("direction", "ASC").upper()
                if col and direction in ["ASC", "DESC"]:
                    order_clauses.append(f"{col} {direction}")
            elif isinstance(order_by_info, list): # Formato: [{"column": "COL1", "direction": "ASC"}, "COL2"] (COL2 usará ASC por defecto)
                for item in order_by_info:
                    if isinstance(item, dict):
                        col = item.get("column")
                        direction = item.get("direction", "ASC").upper()
                        if col and direction in ["ASC", "DESC"]:
                            order_clauses.append(f"{col} {direction}")
                    elif isinstance(item, str):
                        order_clauses.append(f"{item} ASC") # Por defecto ASC si es solo string
            elif isinstance(order_by_info, str): # Formato: "COL_NAME ASC" o "COL_NAME"
                parts = order_by_info.strip().split()
                col = parts[0]
                direction = "ASC"
                if len(parts) > 1 and parts[1].upper() in ["ASC", "DESC"]:
                    direction = parts[1].upper()
                order_clauses.append(f"{col} {direction}")
            
            if order_clauses:
                sql += f" ORDER BY {", ".join(order_clauses)}"
                logging.info(f"[SQLGenerator] Cláusula ORDER BY añadida: {order_clauses}")
            else:
                logging.warning(f"[SQLGenerator] 'order_by' tiene un formato inválido y será ignorado: {order_by_info}")

        # Aplicar LIMIT si está presente
        limit = structured_info.get("limit")
        if limit is not None:
            try:
                limit_val = int(limit)
                if limit_val > 0:
                    sql += f" LIMIT {limit_val}"
                    logging.info(f"[SQLGenerator] Cláusula LIMIT añadida: {limit_val}")
            except ValueError:
                logging.warning(f"[SQLGenerator] 'limit' tiene un valor inválido y será ignorado: {limit}")

        logging.info(f"[SQLGenerator] SQL final generado: {sql}")
        logging.debug(f"[SQLGenerator] Parámetros finales: {params}")

        # Validación final (si está disponible y configurada)
        # if PIPELINE_FUNCTIONS_LOADED:
        #     is_valid, error_msg = validate_query_structure(sql) # Asumiendo que esta función existe y es importable
        #     if not is_valid:
        #         logging.error(f"[SQLGenerator] La consulta generada no pasó la validación estructural: {error_msg}")
        #         # Podríamos devolver un error o intentar un fallback aquí
        #         # return fallback_query(structured_info.get("tables", [])), [] # Ejemplo de fallback
        # else:
        #     logging.warning("[SQLGenerator] Funciones de validación de .pipeline no cargadas. Saltando validación estructural.")
        
        # Whitelist validation (siempre se intenta)
        # try:
        #     whitelist_validate_query(sql, self.allowed_tables, self.allowed_columns)
        #     logging.info("[SQLGenerator] La consulta pasó la validación de la lista blanca.")
        # except ValueError as e:
        #     logging.error(f"[SQLGenerator] Error de validación de la lista blanca: {e}")
        #     # Considerar devolver un error o una consulta de fallback segura
        #     # return "SELECT 'Error: Consulta no permitida por la lista blanca' AS mensaje", []

        return sql, params