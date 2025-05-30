import json
import logging
import os # Añadido para operaciones de archivo

logger = logging.getLogger(__name__)

class WhitelistValidator:
    def __init__(self, allowed_tables_path=None, allowed_columns_map=None, case_sensitive=True):
        """
        Initializes the WhitelistValidator.
        Can be initialized with a path to a JSON file defining allowed tables and columns,
        or directly with an allowed_columns_map dictionary.

        Args:
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
        self.case_sensitive = case_sensitive
        self.allowed_tables = set()
        self.allowed_columns_map = {}

        def _process_name(name):
            return name if self.case_sensitive else name.lower()

        def _process_column_list(columns):
            return columns if self.case_sensitive else [_process_name(col) for col in columns]

        if allowed_columns_map:
            processed_map = {}
            for table, columns in allowed_columns_map.items():
                table_key = _process_name(table)
                processed_map[table_key] = _process_column_list(columns)
            self.allowed_columns_map = processed_map
            self.allowed_tables = set(self.allowed_columns_map.keys())
            logger.info(f"WhitelistValidator initialized with provided allowed_columns_map. Case sensitive: {self.case_sensitive}. Tables: {self.allowed_tables}")
        elif allowed_tables_path:
            try:
                with open(allowed_tables_path, 'r') as f:
                    config = json.load(f)
                
                raw_allowed_tables = config.get("allowed_tables", [])
                self.allowed_tables = {_process_name(t) for t in raw_allowed_tables}
                
                raw_columns_map = config.get("allowed_columns_map", {})
                processed_map = {}
                for table, columns in raw_columns_map.items():
                    table_key = _process_name(table)
                    processed_map[table_key] = _process_column_list(columns)
                self.allowed_columns_map = processed_map
                
                # Ensure all tables in allowed_columns_map are also in allowed_tables
                for table_key in self.allowed_columns_map.keys(): # keys are already processed
                    self.allowed_tables.add(table_key)
                
                logger.info(f"WhitelistValidator initialized from {allowed_tables_path}. Case sensitive: {self.case_sensitive}. Tables: {self.allowed_tables}")
            except FileNotFoundError:
                logger.error(f"Whitelist config file not found: {allowed_tables_path}")
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from whitelist config file: {allowed_tables_path}")
        else:
            logger.warning(f"WhitelistValidator initialized without any whitelist configuration. Case sensitive: {self.case_sensitive}. All checks will likely fail.")

    def is_table_allowed(self, table_name):
        """Checks if a table is in the whitelist."""
        table_name_internal = table_name if self.case_sensitive else table_name.lower()
        allowed = table_name_internal in self.allowed_tables
        if not allowed:
            logger.warning(f"Table '{table_name}' (processed as '{table_name_internal}') is not in the allowed list: {self.allowed_tables}")
        return allowed

    def are_columns_allowed(self, table_name, column_names):
        """Checks if all specified columns for a given table are in the whitelist."""
        table_name_internal = table_name if self.case_sensitive else table_name.lower()

        if not self.is_table_allowed(table_name): # is_table_allowed handles its own casing for table_name
            return False
        
        if table_name_internal not in self.allowed_columns_map:
            logger.warning(
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
                logger.warning(
                    f"Column '{col_original}' (processed as '{col_internal}') in table '{table_name}' "
                    f"(processed as '{table_name_internal}') is not in the allowed list: {allowed_cols_for_table}"
                )
                return False
        return True

    def get_allowed_columns_for_table(self, table_name):
        """Returns the list of allowed columns for a table, or an empty list if table is allowed but has no specific columns, or None if table not allowed."""
        if not self.is_table_allowed(table_name): # is_table_allowed handles casing
            return None

        table_name_internal = table_name if self.case_sensitive else table_name.lower()
        
        # Columns in self.allowed_columns_map are already processed for case.
        # .get() returns the default [] if table_name_internal is not a key.
        return self.allowed_columns_map.get(table_name_internal, [])

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    
    dummy_config_path = "dummy_whitelist_config.json"
    # Added "*" to PATIENTS for testing the new explicit '*' rule
    # Added mixed case table "MixedCaseTable" for case-insensitivity testing
    dummy_config_content = {
        "allowed_tables": ["PATIENTS", "ALLERGIES", "OBSERVATIONS", "MixedCaseTable"],
        "allowed_columns_map": {
            "PATIENTS": ["PATI_ID", "PATI_NAME", "PATI_BIRTH_DATE", "*"], 
            "ALLERGIES": ["ALLERGY_ID", "PATI_ID", "ALLERGY_TYPE"], # No ALLERGY_DESCRIPTION, no "*"
            "OBSERVATIONS": ["OBS_ID", "PATI_ID", "OBS_TEXT", "OBS_DATE"],
            "MixedCaseTable": ["MixID", "MixData"]
        }
    }
    
    try:
        with open(dummy_config_path, 'w') as f:
            json.dump(dummy_config_content, f, indent=4)
        logger.info(f"Archivo de configuración temporal creado: {dummy_config_path}")

        # --- Test Case-Sensitive Validator (default) ---
        validator_cs = WhitelistValidator(allowed_tables_path=dummy_config_path) # case_sensitive=True by default
        print("\\n--- Pruebas con validador SENSIBLE a mayúsculas/minúsculas (por defecto) ---")
        print(f"Is PATIENTS allowed? {validator_cs.is_table_allowed('PATIENTS')}") # True
        print(f"Is patients allowed? {validator_cs.is_table_allowed('patients')}") # False
        print(f"Are PATIENTS columns [PATI_ID, PATI_BIRTH_DATE] allowed? {validator_cs.are_columns_allowed('PATIENTS', ['PATI_ID', 'PATI_BIRTH_DATE'])}") # True
        print(f"Are PATIENTS columns [pati_id, pati_birth_date] allowed? {validator_cs.are_columns_allowed('PATIENTS', ['pati_id', 'pati_birth_date'])}") # False
        print(f"Are PATIENTS columns ['*'] allowed? {validator_cs.are_columns_allowed('PATIENTS', ['*'])}") # True, "*" is in list
        print(f"Are ALLERGIES columns ['*'] allowed? {validator_cs.are_columns_allowed('ALLERGIES', ['*'])}") # False, "*" not in list
        print(f"Are ALLERGIES columns [ALLERGY_ID, ALLERGY_DESCRIPTION] allowed? {validator_cs.are_columns_allowed('ALLERGIES', ['ALLERGY_ID', 'ALLERGY_DESCRIPTION'])}") # False, ALLERGY_DESCRIPTION not in list
        print(f"Get allowed columns for PATIENTS: {validator_cs.get_allowed_columns_for_table('PATIENTS')}")
        print(f"Get allowed columns for patients: {validator_cs.get_allowed_columns_for_table('patients')}") # None


        # --- Test Case-Insensitive Validator ---
        validator_ci = WhitelistValidator(allowed_tables_path=dummy_config_path, case_sensitive=False)
        print("\\n--- Pruebas con validador INSENSIBLE a mayúsculas/minúsculas ---")
        print(f"Is PATIENTS allowed? {validator_ci.is_table_allowed('PATIENTS')}") # True
        print(f"Is patients allowed? {validator_ci.is_table_allowed('patients')}") # True
        print(f"Is PaTiEnTs allowed? {validator_ci.is_table_allowed('PaTiEnTs')}") # True
        print(f"Are PATIENTS columns [PATI_ID, PATI_BIRTH_DATE] allowed? {validator_ci.are_columns_allowed('PATIENTS', ['PATI_ID', 'PATI_BIRTH_DATE'])}") # True
        print(f"Are patients columns [pati_id, pati_birth_date] allowed? {validator_ci.are_columns_allowed('patients', ['pati_id', 'pati_birth_date'])}") # True
        print(f"Are PaTiEnTs columns [PaTi_Id, PATI_birth_date] allowed? {validator_ci.are_columns_allowed('PaTiEnTs', ['PaTi_Id', 'PATI_birth_date'])}") # True
        
        print(f"Is MixedCaseTable allowed? {validator_ci.is_table_allowed('MixedCaseTable')}") # True
        print(f"Is mixedcasetable allowed? {validator_ci.is_table_allowed('mixedcasetable')}") # True
        print(f"Are MixedCaseTable columns [MixID, MixData] allowed? {validator_ci.are_columns_allowed('MixedCaseTable', ['MixID', 'MixData'])}") # True
        print(f"Are mixedcasetable columns [mixid, mixdata] allowed? {validator_ci.are_columns_allowed('mixedcasetable', ['mixid', 'mixdata'])}") # True

        print(f"Are PATIENTS columns ['*'] allowed (CI)? {validator_ci.are_columns_allowed('PATIENTS', ['*'])}") # True
        print(f"Are patients columns ['*'] allowed (CI)? {validator_ci.are_columns_allowed('patients', ['*'])}") # True
        print(f"Are ALLERGIES columns ['*'] allowed (CI)? {validator_ci.are_columns_allowed('ALLERGIES', ['*'])}") # False
        print(f"Are allergies columns ['*'] allowed (CI)? {validator_ci.are_columns_allowed('allergies', ['*'])}") # False
        print(f"Are ALLERGIES columns [ALLERGY_ID, ALLERGY_DESCRIPTION] allowed (CI)? {validator_ci.are_columns_allowed('ALLERGIES', ['ALLERGY_ID', 'ALLERGY_DESCRIPTION'])}") # False
        print(f"Are allergies columns [allergy_id, allergy_description] allowed (CI)? {validator_ci.are_columns_allowed('allergies', ['allergy_id', 'allergy_description'])}") # False
        
        print(f"Get allowed columns for PATIENTS (CI): {validator_ci.get_allowed_columns_for_table('PATIENTS')}")
        print(f"Get allowed columns for patients (CI): {validator_ci.get_allowed_columns_for_table('patients')}")
        # Test table defined only in "allowed_tables" (if any, not in this dummy config)
        # Test table allowed but not in allowed_columns_map (covered by default behavior of are_columns_allowed)

        # Ejemplo de inicialización directa con un mapa (para comparación)
        validator_map_ci = WhitelistValidator(allowed_columns_map={
            "CUSTOM_TABLE": ["CT_ID", "CT_DATA", "*"],
            "Another_Table": ["AT_ID", "at_info"]
        }, case_sensitive=False)
        print("\\n--- Pruebas con validador INSENSIBLE a mayúsculas/minúsculas, inicializado desde mapa ---")
        print(f"Is CUSTOM_TABLE allowed? {validator_map_ci.is_table_allowed('CUSTOM_TABLE')}") # True
        print(f"Is custom_table allowed? {validator_map_ci.is_table_allowed('custom_table')}") # True
        print(f"Are CUSTOM_TABLE columns [ct_id, CT_DATA] allowed? {validator_map_ci.are_columns_allowed('custom_table', ['ct_id', 'CT_DATA'])}") # True
        print(f"Are CUSTOM_TABLE columns ['*'] allowed? {validator_map_ci.are_columns_allowed('CUSTOM_TABLE', ['*'])}") # True
        print(f"Is Another_Table allowed? {validator_map_ci.is_table_allowed('Another_Table')}") # True
        print(f"Is another_table allowed? {validator_map_ci.is_table_allowed('another_table')}") # True
        print(f"Are Another_Table columns [AT_ID, at_info] allowed? {validator_map_ci.are_columns_allowed('Another_Table', ['AT_ID', 'at_info'])}") # True
        print(f"Are Another_Table columns [at_id, AT_INFO] allowed? {validator_map_ci.are_columns_allowed('another_table', ['at_id', 'AT_INFO'])}") # True
        print(f"Are Another_Table columns ['*'] allowed? {validator_map_ci.are_columns_allowed('Another_Table', ['*'])}") # False

    except Exception as e:
        logger.error(f"Error durante las pruebas de WhitelistValidator: {e}", exc_info=True)
    finally:
        if os.path.exists(dummy_config_path):
            os.remove(dummy_config_path)
            logger.info(f"Archivo de configuración temporal eliminado: {dummy_config_path}")

def validate_structured_info_whitelist(structured_info, allowed_columns_map, terms_dictionary, logger_param=None):
    logger_instance = logger_param if logger_param else logging.getLogger(__name__)
    
    if not isinstance(structured_info, dict):
        logger_instance.warning("validate_structured_info_whitelist: structured_info no es un diccionario o es None.")
        return False, "La información estructurada proporcionada no es válida (no es un diccionario)."

    # Asumimos que allowed_columns_map ya tiene las claves de tabla y nombres de columna en el caso correcto (MAYÚSCULAS)
    # y que WhitelistValidator debe operar en modo sensible a mayúsculas/minúsculas.
    # El valor por defecto de case_sensitive en WhitelistValidator es True.
    validator = WhitelistValidator(allowed_columns_map=allowed_columns_map)

    # 1. Validar 'from_table'
    from_table_info = structured_info.get("from_table")
    if isinstance(from_table_info, dict):
        from_table_name = from_table_info.get("name")
        if from_table_name:
            if not validator.is_table_allowed(from_table_name):
                msg = f"Tabla '{from_table_name}' en la cláusula FROM no está permitida."
                logger_instance.warning(f"Whitelist validation failed: {msg}")
                return False, msg
        # else: # No 'name' key in from_table_info dict. Could be a subquery, etc.
            # logger_instance.debug("validate_structured_info_whitelist: 'from_table' no tiene 'name'. Se omite validación de tabla FROM.")
    elif isinstance(from_table_info, str): # Caso simple donde from_table es solo el nombre
        if not validator.is_table_allowed(from_table_info):
            msg = f"Tabla '{from_table_info}' en la cláusula FROM no está permitida."
            logger_instance.warning(f"Whitelist validation failed: {msg}")
            return False, msg
    elif from_table_info: # from_table existe pero no es dict ni str
        logger_instance.warning(f"validate_structured_info_whitelist: 'from_table' tiene un formato inesperado: {type(from_table_info)}.")
        return False, "Formato inesperado para 'from_table' en la información estructurada."

    # 2. Validar 'joins'
    joins_info = structured_info.get("joins")
    if isinstance(joins_info, list):
        for i, join_item in enumerate(joins_info):
            if isinstance(join_item, dict):
                target_table_info = join_item.get("target_table")
                target_table_name = None
                if isinstance(target_table_info, dict):
                    target_table_name = target_table_info.get("name")
                elif isinstance(target_table_info, str): # Si target_table es solo el nombre
                    target_table_name = target_table_info
                
                if target_table_name:
                    if not validator.is_table_allowed(target_table_name):
                        msg = f"Tabla '{target_table_name}' en JOIN #{i} no está permitida."
                        logger_instance.warning(f"Whitelist validation failed: {msg}")
                        return False, msg
                else:
                    logger_instance.warning(f"validate_structured_info_whitelist: JOIN #{i} 'target_table' no tiene 'name' o formato no reconocido.")
                    # Considerar si esto debe ser un fallo o una advertencia. Por ahora, si no hay nombre, no se puede validar.
            else:
                logger_instance.warning(f"validate_structured_info_whitelist: Elemento JOIN #{i} no es un diccionario.")
                return False, f"Formato inesperado para el elemento JOIN #{i}."
    elif joins_info:
        logger_instance.warning(f"validate_structured_info_whitelist: 'joins' tiene un formato inesperado: {type(joins_info)}.")
        return False, "Formato inesperado para 'joins' en la información estructurada."

    # 3. Validar 'select_columns'
    select_columns_info = structured_info.get("select_columns")
    if isinstance(select_columns_info, list):
        for i, col_item in enumerate(select_columns_info):
            if isinstance(col_item, dict):
                table_name = col_item.get("table")
                column_name = col_item.get("column")
                
                if not table_name and column_name == "*": # Manejar SELECT * sin tabla específica (raro aquí, más común en SQL directo)
                    logger_instance.debug(f"SELECT #{i} es '*' sin tabla específica. Se permite por ahora en validación estructurada.")
                    continue

                if not table_name and column_name and not col_item.get("is_aggregate_or_expression"):
                    # Si no hay tabla pero hay columna, y no está marcada como expresión, es ambiguo.
                    # Podría ser una función global, o un error en la extracción.
                    # logger_instance.warning(f"SELECT #{i}: Columna '{column_name}' sin tabla asociada y no marcada como expresión. Se omite validación.")
                    continue

                if table_name: # Solo validar si hay un nombre de tabla
                    if not validator.is_table_allowed(table_name):
                        msg = f"Tabla '{table_name}' en SELECT #{i} ('{column_name or '*'}') no está permitida."
                        logger_instance.warning(f"Whitelist validation failed: {msg}")
                        return False, msg
                    
                    if column_name: # Solo validar columna si hay un nombre de columna
                        if not validator.are_columns_allowed(table_name, [column_name]):
                            msg = f"Columna '{column_name}' de la tabla '{table_name}' en SELECT #{i} no está permitida."
                            logger_instance.warning(f"Whitelist validation failed: {msg}")
                            return False, msg
                    # else: No column_name, podría ser table.* que es manejado por is_table_allowed si '*' no está en columnas.
                        # O podría ser un error en la info estructurada.
                        # logger_instance.debug(f"SELECT #{i} para tabla '{table_name}' no especifica columna. Se asume table.* o similar.")
            else: # col_item no es un dict
                logger_instance.warning(f"validate_structured_info_whitelist: Elemento SELECT #{i} no es un diccionario.")
                return False, f"Formato inesperado para el elemento SELECT #{i}."
    elif select_columns_info:
        logger_instance.warning(f"validate_structured_info_whitelist: 'select_columns' tiene un formato inesperado: {type(select_columns_info)}.")
        return False, "Formato inesperado para 'select_columns' en la información estructurada."

    # Aquí se podrían añadir validaciones para otras cláusulas como 'where_clause', 'group_by', etc.
    # Esto requeriría parsear las columnas de las expresiones en esas cláusulas.

    logger_instance.info("validate_structured_info_whitelist: Validación de información estructurada completada exitosamente.")
    return True, "Validación de whitelist de información estructurada exitosa."
