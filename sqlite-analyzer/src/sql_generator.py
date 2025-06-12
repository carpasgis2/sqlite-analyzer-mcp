import re
import logging
import sys
import os
import json
from typing import Dict, Any, Tuple, List, Optional, Union, Set # Added Set

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


# Definición de constantes para nombres de tablas y columnas clave
TABLE_EPISODES = "EPIS_EPISODES"
TABLE_EPIS_DIAGNOSTICS = "EPIS_DIAGNOSTICS"
TABLE_CODR_DIAGNOSTIC_GROUPS = "CODR_DIAGNOSTIC_GROUPS"
COLUMN_DGGR_ID = "DGGR_ID" # En CODR_DIAGNOSTIC_GROUPS, usado para unirse desde EPIS_DIAGNOSTICS.CDTE_ID
COLUMN_CDTE_ID_DIAG = "CDTE_ID" # En EPIS_DIAGNOSTICS
COLUMN_DGGR_DESCRIPTION_ES = "DGGR_DESCRIPTION_ES"
COLUMN_EPIS_START_DATE = "EPIS_START_DATE" # Añadida definición de constante
COLUMN_EPIS_CLOSED_DATE = "EPIS_CLOSED_DATE" # Añadida definición de constante

TABLE_EPIS_PROCEDURES = "EPIS_PROCEDURES"
TABLE_PROC_PROCEDURES = "PROC_PROCEDURES"
TABLE_PROC_PROCEDURE_TYPES = "PROC_PROCEDURE_TYPES"
COLUMN_PRTY_DESCRIPTION_ES = "PRTY_DESCRIPTION_ES" # Añadida definición de constante (en PROC_PROCEDURE_TYPES)

TABLE_MEDI_MEDICATIONS = "MEDI_MEDICATIONS"
TABLE_MEDI_ACTIVE_INGREDIENTS = "MEDI_ACTIVE_INGREDIENTS"
TABLE_MEDI_PHARMA_THERAPEUTIC_GROUPS = "MEDI_PHARMA_THERAPEUTIC_GROUPS"
COLUMN_PHTG_ID = "PHTG_ID"  # Asunción: PK en MEDI_PHARMA_THERAPEUTIC_GROUPS, FK en MEDI_MEDICATIONS.
COLUMN_PHTG_DESCRIPTION_ES = "PHTG_DESCRIPTION_ES" # Asunción: Columna de descripción en MEDI_PHARMA_THERAPEUTIC_GROUPS.


# Nuevas constantes para ingredientes activos y su enlace con medicamentos
TABLE_MEDI_MEDICATION_COMPONENTS = "MEDI_MEDICATION_COMPONENTS" # ¡ASUNCIÓN! Nombre de la tabla de enlace.
COLUMN_MEDI_ID = "MEDI_ID" # Asumo que es el ID en MEDI_MEDICATIONS
COLUMN_ACIN_ID = "ACIN_ID" # Asumo que es el ID en MEDI_ACTIVE_INGREDIENTS
COLUMN_ACIN_DESCRIPTION_ES = "ACIN_DESCRIPTION_ES" # Ya existente
COLUMN_MMC_MEDI_ID = "MEDI_ID" # ¡ASUNCIÓN! Columna FK a MEDI_ID en MEDI_MEDICATION_COMPONENTS
COLUMN_MMC_ACIN_ID = "ACIN_ID" # ¡ASUNCIÓN! Columna FK a ACIN_ID en MEDI_MEDICATION_COMPONENTS


DIAS_INGRESO_PSEUDO_COL = "dias_ingreso"
DIAGNOSTICO_TERMINO_PSEUDO_FIELD = "diagnostico_termino"
TIPO_PROCEDIMIENTO_TERMINO_PSEUDO_FIELD = "tipo_procedimiento_termino"
DIAS_INGRESO_CONDICION_PSEUDO_FIELD = "dias_ingreso_condicion"
GRUPO_TERAPEUTICO_TERMINO_PSEUDO_FIELD = "grupo_terapeutico_termino" # Nuevo pseudo-field

# Nuevo pseudo-campo para ingredientes activos
NO_CONTIENE_INGREDIENTE_ACTIVO_PSEUDO_FIELD = "no_contiene_ingrediente_activo"


class SQLGenerator:
    """Clase para generar consultas SQL a partir de información estructurada"""
    
    def __init__(self, allowed_tables: List[str], allowed_columns: Dict[str, List[str]], enhanced_schema_path: Optional[str] = None, relationships_str: Optional[str] = None, dictionary_str: Optional[str] = None, logger: Optional[logging.Logger] = None): # MODIFICADO: Añadido logger
        """
        Inicializa el generador SQL con las tablas y columnas permitidas y opcionalmente un esquema mejorado.
        
        Args:
            allowed_tables: Lista de nombres de tablas permitidas
            allowed_columns: Diccionario que mapea tablas a sus columnas permitidas
            enhanced_schema_path: Ruta opcional al archivo schema_enhanced.json
            relationships_str: JSON string con relaciones entre tablas
            dictionary_str: JSON string con diccionario de términos
            logger: Instancia de logger opcional
        """
        self.allowed_tables = allowed_tables
        self.allowed_columns = allowed_columns
        self.enhanced_schema = None
        # Almacenar los strings de relaciones y diccionario si se proporcionan
        self.relationships_str = relationships_str
        self.dictionary_str = dictionary_str
        self.all_known_tables_original_case = {} # Mapa de nombre de tabla en mayúsculas a su nombre original
        self.column_aliases_map = {} # Mapa de alias de columna a su expresión original
        self.warnings = []
        self.table_aliases_in_query = {} # Mapa de alias de tabla a nombre real de tabla
        
        # MODIFICADO: Usar el logger proporcionado o crear uno nuevo si no se proporciona
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            if not self.logger.handlers: # Evitar añadir múltiples handlers si ya está configurado
                self.logger.addHandler(logging.StreamHandler())
                self.logger.setLevel(logging.INFO)

        if enhanced_schema_path:
            self._load_enhanced_schema(enhanced_schema_path)
        
    def _load_enhanced_schema(self, schema_path: str):
        """Carga el esquema mejorado desde un archivo JSON."""
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                self.enhanced_schema = json.load(f)
            self.logger.info(f"[SQLGenerator] Esquema mejorado cargado exitosamente desde {schema_path}") # MODIFICADO: Usar self.logger
        except FileNotFoundError:
            self.logger.error(f"[SQLGenerator] Archivo de esquema mejorado no encontrado en {schema_path}. La normalización de columnas no estará disponible.") # MODIFICADO: Usar self.logger
            self.enhanced_schema = None
        except json.JSONDecodeError:
            self.logger.error(f"[SQLGenerator] Error al decodificar el archivo JSON del esquema mejorado en {schema_path}. La normalización de columnas no estará disponible.") # MODIFICADO: Usar self.logger
            self.enhanced_schema = None
        except Exception as e:
            self.logger.error(f"[SQLGenerator] Ocurrió un error inesperado al cargar el esquema mejorado desde {schema_path}: {e}") # MODIFICADO: Usar self.logger
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
        for t_name_key in self.enhanced_schema.get("tables", {}).keys():
            if t_name_key.upper() == table_name_upper:
                schema_table_name_key = t_name_key
                break
        if not schema_table_name_key:
            self.logger.debug(f"[SQLGenerator] Tabla '{table_name}' no encontrada en el esquema mejorado para normalizar columna '{column_name_variant}'.") # MODIFICADO: Usar self.logger
            return None

        table_schema = self.enhanced_schema["tables"].get(schema_table_name_key, {})
        actual_columns_in_schema = {col_info.get("name", "").upper(): col_info.get("name", "") 
                                    for col_info in table_schema.get("columns", []) if col_info.get("name")}

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
                self.logger.info(f"[SQLGenerator] Columna '{column_name_variant}' normalizada a '{actual_columns_in_schema[variation_upper]}' para la tabla '{table_name}'.") # MODIFICADO: Usar self.logger
                return actual_columns_in_schema[variation_upper]

        self.logger.warning(f"[SQLGenerator] No se pudo encontrar una coincidencia para la columna '{column_name_variant}' en la tabla '{table_name}' usando el esquema mejorado. Se omitirá la condición.") # MODIFICADO: Usar self.logger
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
        match = re.fullmatch(r"\\s*([\\w_]+)\\.([\\w_]+)\\s*=\\s*([\\w_]+)\\.([\\w_]+)\\s*", on_condition_str)
        if not match:
            self.logger.debug(f"[SQLGenerator] Condición ON '{on_condition_str}' no coincide con el patrón T1.C1 = T2.C2 para normalización.") # MODIFICADO: Usar self.logger
            return on_condition_str

        t1, c1_variant, t2, c2_variant = match.groups()

        actual_c1 = self._find_actual_column_name(t1, c1_variant)
        actual_c2 = self._find_actual_column_name(t2, c2_variant)

        normalized_condition = f"{t1}.{actual_c1} = {t2}.{actual_c2}"
        if normalized_condition != on_condition_str:
            self.logger.info(f"[SQLGenerator] Condición ON normalizada: '{on_condition_str}' -> '{normalized_condition}'") # MODIFICADO: Usar self.logger
        return normalized_condition

    def _get_join_details_from_relations_map(self, table1_upper: str, table2_upper: str, relations_map: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Intenta encontrar una relación directa entre table1 y table2 en relations_map."""
        if not relations_map:
            return None

        # Buscar relación donde table1_upper es 'table' y table2_upper es 'foreign_table'
        for rel in relations_map.get(table1_upper, []):
            if isinstance(rel, dict) and rel.get("foreign_table", "").upper() == table2_upper:
                if rel.get("column") and rel.get("foreign_column"):
                    return {"from_table": table1_upper, "from_col": rel["column"], 
                            "to_table": table2_upper, "to_col": rel["foreign_column"]}
        
        # Buscar relación donde table2_upper es 'table' y table1_upper es 'foreign_table' (relación inversa)
        for rel in relations_map.get(table2_upper, []):
            if isinstance(rel, dict) and rel.get("foreign_table", "").upper() == table1_upper:
                if rel.get("column") and rel.get("foreign_column"):
                    # Invertir para que 'from' sea table1_upper
                    return {"from_table": table1_upper, "from_col": rel["foreign_column"],
                            "to_table": table2_upper, "to_col": rel["column"]}
        
        self.logger.debug(f"[SQLGenerator] No se encontró relación directa entre {table1_upper} y {table2_upper} en _get_join_details_from_relations_map.") # MODIFICADO: Usar self.logger
        return None

    def _add_specific_join_if_needed(self,
                                 from_table_upper: str,
                                 to_table_upper: str,
                                 already_joined_tables: Set[str],
                                 join_clauses_sql: List[str],
                                 relations_map: Dict[str, Any],
                                 all_known_tables_original_case: Dict[str, str],
                                 preferred_join_type: str = "INNER"):
        if to_table_upper in already_joined_tables:
            return True

        if from_table_upper not in already_joined_tables:
            # Esto podría pasar si la cadena de JOINs se rompe o la tabla 'from' no es la principal.
            # Para _ensure_path, la tabla 'from' ya debería estar unida.
            self.logger.warning(f"[SQLGenerator] Intento de unir {to_table_upper} desde {from_table_upper}, pero {from_table_upper} no está en already_joined_tables.") # MODIFICADO: Usar self.logger
            # No se puede unir si la tabla de origen no está.
            # Podríamos intentar añadir from_table_upper si es la tabla principal, pero eso se complica.
            # Por ahora, si from_table_upper no está, no podemos continuar este camino específico.
            # Esto es una salvaguarda; la lógica de _ensure_path debe manejar el orden.
            if from_table_upper != TABLE_EPISODES.upper(): # Permitir que EPIS_EPISODES sea la primera
                 return False


        original_to_table_name = all_known_tables_original_case.get(to_table_upper)
        original_from_table_name = all_known_tables_original_case.get(from_table_upper)

        if not original_to_table_name:
            self.logger.error(f"[SQLGenerator] No se encontró el nombre original para la tabla a unir: {to_table_upper}") # MODIFICADO: Usar self.logger
            return False
        if not original_from_table_name: # Esto no debería pasar si from_table_upper está en already_joined_tables (excepto main_table)
            self.logger.error(f"[SQLGenerator] No se encontró el nombre original para la tabla desde: {from_table_upper}") # MODIFICADO: Usar self.logger
            return False


        join_details = self._get_join_details_from_relations_map(from_table_upper, to_table_upper, relations_map)
        
        if not join_details:
            self.logger.warning(f"[SQLGenerator] No se pudo determinar la condición ON entre {original_from_table_name} y {original_to_table_name} usando relations_map.") # MODIFICADO: Usar self.logger
            return False

        # Usar los nombres originales de las tablas para la condición ON
        # Asegurarse que join_details tiene los nombres de tabla correctos para _normalize_on_condition_string
        # _normalize_on_condition_string espera "TABLE.COLUMN = TABLE.COLUMN"
        
        # Obtener los nombres originales para las tablas en join_details
        jd_from_table_orig = all_known_tables_original_case.get(join_details['from_table'].upper())
        jd_to_table_orig = all_known_tables_original_case.get(join_details['to_table'].upper())

        if not jd_from_table_orig or not jd_to_table_orig:
            self.logger.error(f"No se pudieron obtener nombres originales para tablas en join_details: {join_details}") # MODIFICADO: Usar self.logger
            return False

        raw_on_condition = f"{jd_from_table_orig}.{join_details['from_col']} = {jd_to_table_orig}.{join_details['to_col']}"
        normalized_on_condition = self._normalize_on_condition_string(raw_on_condition)

        join_sql_segment = f" {preferred_join_type} JOIN {original_to_table_name} ON {normalized_on_condition}"
        join_clauses_sql.append(join_sql_segment)
        already_joined_tables.add(to_table_upper)
        self.logger.info(f"[SQLGenerator] JOIN específico añadido: {join_sql_segment}") # MODIFICADO: Usar self.logger
        return True

    def _ensure_diagnosis_path(self,
                           already_joined_tables: Set[str],
                           join_clauses_sql: List[str],
                           relations_map: Dict[str, Any],
                           all_known_tables_original_case: Dict[str, str]):
        if TABLE_CODR_DIAGNOSTIC_GROUPS.upper() in already_joined_tables:
            return all_known_tables_original_case.get(TABLE_CODR_DIAGNOSTIC_GROUPS.upper())

        if not self._add_specific_join_if_needed(TABLE_EPISODES.upper(), TABLE_EPIS_DIAGNOSTICS.upper(),
                                            already_joined_tables, join_clauses_sql, relations_map, all_known_tables_original_case):
            self.logger.warning(f"Fallo al unir {TABLE_EPISODES} con {TABLE_EPIS_DIAGNOSTICS} para la ruta de diagnóstico.") # MODIFICADO: Usar self.logger
            return None
    
        if not self._add_specific_join_if_needed(TABLE_EPIS_DIAGNOSTICS.upper(), TABLE_CODR_DIAGNOSTIC_GROUPS.upper(),
                                            already_joined_tables, join_clauses_sql, relations_map, all_known_tables_original_case):
            self.logger.warning(f"Fallo al unir {TABLE_EPIS_DIAGNOSTICS} con {TABLE_CODR_DIAGNOSTIC_GROUPS} para la ruta de diagnóstico.") # MODIFICADO: Usar self.logger
            return None
            
        return all_known_tables_original_case.get(TABLE_CODR_DIAGNOSTIC_GROUPS.upper())

    def _ensure_procedure_path(self,
                           already_joined_tables: Set[str],
                           join_clauses_sql: List[str],
                           relations_map: Dict[str, Any],
                           all_known_tables_original_case: Dict[str, str]):
        if TABLE_PROC_PROCEDURE_TYPES.upper() in already_joined_tables:
            return all_known_tables_original_case.get(TABLE_PROC_PROCEDURE_TYPES.upper())

        if not self._add_specific_join_if_needed(TABLE_EPISODES.upper(), TABLE_EPIS_PROCEDURES.upper(),
                                            already_joined_tables, join_clauses_sql, relations_map, all_known_tables_original_case):
            self.logger.warning(f"Fallo al unir {TABLE_EPISODES} con {TABLE_EPIS_PROCEDURES} para la ruta de procedimiento.") # MODIFICADO: Usar self.logger
            return None
    
        if not self._add_specific_join_if_needed(TABLE_EPIS_PROCEDURES.upper(), TABLE_PROC_PROCEDURES.upper(),
                                            already_joined_tables, join_clauses_sql, relations_map, all_known_tables_original_case):
            self.logger.warning(f"Fallo al unir {TABLE_EPIS_PROCEDURES} con {TABLE_PROC_PROCEDURES} para la ruta de procedimiento.") # MODIFICADO: Usar self.logger
            return None

        if not self._add_specific_join_if_needed(TABLE_PROC_PROCEDURES.upper(), TABLE_PROC_PROCEDURE_TYPES.upper(),
                                            already_joined_tables, join_clauses_sql, relations_map, all_known_tables_original_case):
            self.logger.warning(f"Fallo al unir {TABLE_PROC_PROCEDURES} con {TABLE_PROC_PROCEDURE_TYPES} para la ruta de procedimiento.") # MODIFICADO: Usar self.logger
            return None
            
        return all_known_tables_original_case.get(TABLE_PROC_PROCEDURE_TYPES.upper())

    def _ensure_therapeutic_group_path(self,
                                   already_joined_tables: Set[str],
                                   join_clauses_sql: List[str],
                                   relations_map: Dict[str, Any],
                                   all_known_tables_original_case: Dict[str, str],
                                   current_main_table_name: str): # current_main_table_name is the actual main table of the query
        # Path: MEDI_MEDICATIONS -> MEDI_PHARMA_THERAPEUTIC_GROUPS
        phtg_table_upper = TABLE_MEDI_PHARMA_THERAPEUTIC_GROUPS.upper()
        medi_table_upper = TABLE_MEDI_MEDICATIONS.upper()

        if phtg_table_upper in already_joined_tables:
            return all_known_tables_original_case.get(phtg_table_upper)

        # Ensure MEDI_MEDICATIONS is joined or is the current_main_table_name
        if medi_table_upper not in already_joined_tables:
            if current_main_table_name.upper() == medi_table_upper:
                # This should not happen if main_table is added to already_joined_tables at start
                self.logger.error(f"{TABLE_MEDI_MEDICATIONS} is main table but not in already_joined_tables. This is a bug.") # MODIFICADO: Usar self.logger
                # We can try to add it, but it's symptomatic of a deeper issue.
            else:
                # If MEDI_MEDICATIONS is not the main table and not joined, we might need to join it first.
                # This specific helper assumes MEDI_MEDICATIONS is the 'from' side of this particular join.
                # A more generic pathfinder would be needed if MEDI_MEDICATIONS itself needs to be reached from current_main_table_name.
                # For now, if MEDI_MEDICATIONS is not available, this path cannot be built directly by this helper.
                # We could try to join it TO current_main_table_name if a relation exists,
                # or assume it must be joined via other means if current_main_table_name is not MEDI_MEDICATIONS.
                # This part is complex. Let's assume for this helper that MEDI_MEDICATIONS must be present.
                self.logger.warning(f"Cannot ensure therapeutic group path because {TABLE_MEDI_MEDICATIONS} is not the main table ({current_main_table_name}) and not in already_joined_tables.") # MODIFICADO: Usar self.logger
                # Attempt to join MEDI_MEDICATIONS to the current_main_table_name if a relation exists.
                # This is a basic attempt; a full pathfinding algorithm would be more robust.
                if not self._add_specific_join_if_needed(
                    current_main_table_name.upper(), 
                    medi_table_upper,
                    already_joined_tables, 
                    join_clauses_sql, 
                    relations_map, 
                    all_known_tables_original_case):
                    self.logger.warning(f"Failed to join {current_main_table_name} to {TABLE_MEDI_MEDICATIONS} as a prerequisite for therapeutic group path.") # MODIFICADO: Usar self.logger
                    return None
                # If the above join succeeded, medi_table_upper is now in already_joined_tables.
        
        # Now, medi_table_upper (MEDI_MEDICATIONS) should be in already_joined_tables.
        # Proceed to join MEDI_MEDICATIONS to MEDI_PHARMA_THERAPEUTIC_GROUPS
        if not self._add_specific_join_if_needed(
            medi_table_upper, # From MEDI_MEDICATIONS
            phtg_table_upper, # To MEDI_PHARMA_THERAPEUTIC_GROUPS
            already_joined_tables, 
            join_clauses_sql, 
            relations_map, 
            all_known_tables_original_case
        ):
            self.logger.warning(f"Fallo al unir {TABLE_MEDI_MEDICATIONS} con {TABLE_MEDI_PHARMA_THERAPEUTIC_GROUPS} para la ruta de grupo terapéutico.") # MODIFICADO: Usar self.logger
            return None
            
        return all_known_tables_original_case.get(phtg_table_upper)

    def generate_sql(self, question_data: Dict[str, Any], disable_llm_enhancement: bool = False, db_connector: Optional[Any] = None, max_execution_time: Optional[int] = None) -> Tuple[str, List[Any]]:
        """
        Genera una consulta SQL a partir de información estructurada.
        Args:
            structured_info: Diccionario con la información estructurada de la consulta.
            db_structure: (Opcional) Diccionario con la estructura de la base de datos.
            relations_map: (Opcional) Diccionario con el mapa de relaciones entre tablas.

        Returns:
            Una tupla con la consulta SQL generada y una lista de parámetros.
        """
        self.logger.info("[SQLGenerator] Iniciando generate_sql") # MODIFICADO: Usar self.logger # LOG INICIO MÉTODO
        # Mapear parámetros de pipeline a variables internas
        structured_info = question_data  # Estructura procesada de la consulta
        # Determinar estructura de BD a usar
        if self.enhanced_schema is not None:
            db_structure = self.enhanced_schema
        elif db_connector is not None:
            try:
                db_structure = db_connector.get_database_structure()
            except Exception as e:
                self.logger.error(f"[SQLGenerator] Error al obtener estructura de BD desde db_connector: {e}") # MODIFICADO: Usar self.logger
                db_structure = None
        else:
            db_structure = None
        # Determinar mapa de relaciones
        relations_map = {}
        if self.relationships_str:
            try:
                relations_map = json.loads(self.relationships_str)
            except json.JSONDecodeError:
                self.logger.error("[SQLGenerator] Error al decodificar relaciones JSON; usando mapa vacío.") # MODIFICADO: Usar self.logger
        
        params = []
        # Preparar mapeo de nombres de tablas (mayúsculas -> nombre original)
        all_known_tables_original_case = {}
        if db_structure:
            all_known_tables_original_case = {name.upper(): name for name in db_structure.keys()}
        else:
            self.logger.warning("[SQLGenerator] db_structure no proporcionado. Algunas funcionalidades pueden estar limitadas.") # MODIFICADO: Usar self.logger
         

        self.logger.debug(f"[SQLGenerator] structured_info recibido: {json.dumps(structured_info, indent=2)}") # MODIFICADO: Usar self.logger
         
        raw_tables_list = structured_info.get("tables")
        if not raw_tables_list or not isinstance(raw_tables_list, list) or not raw_tables_list[0]:
            self.logger.error("[SQLGenerator] No se especificó ninguna tabla válida en structured_info['tables'].") # MODIFICADO: Usar self.logger
            return ("SELECT 'Error: No se especificó ninguna tabla válida' AS mensaje", [])

        # Validación robusta: abortar si alguna tabla no existe en el esquema permitido
        if db_structure is not None:
            tablas_invalidas = [t for t in raw_tables_list if t not in db_structure]
            if tablas_invalidas:
                self.logger.error(f"No existen en el esquema las siguientes tablas requeridas: {tablas_invalidas}") # MODIFICADO: Usar self.logger
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
             self.logger.error("[SQLGenerator] El conjunto de tablas válidas (raw_tables_list_upper_set) está vacío.")
             return "SELECT 'Error: No hay tablas válidas para procesar' AS mensaje", []

        for cand_upper in candidate_main_tables_ordered:
            if cand_upper in raw_tables_list_upper_set:
                # Encontrar el nombre original con la capitalización correcta
                try:
                    original_cand_idx = [tbl.upper() for tbl in raw_tables_list].index(cand_upper)
                    main_table = raw_tables_list[original_cand_idx]
                    self.logger.info(f"[SQLGenerator] Tabla principal elegida por heurística: {main_table} (Candidato: {cand_upper})")
                    break
                except ValueError: # pragma: no cover
                    # Esto no debería ocurrir si cand_upper está en raw_tables_list_upper_set
                    self.logger.warning(f"[SQLGenerator] Candidato {cand_upper} encontrado pero no se pudo obtener su nombre original.")
                    continue 
        if not main_table:
            # Fallback si ningún candidato priorizado es válido o si raw_tables_list_upper_set estaba vacío
            # (aunque el chequeo anterior de raw_tables_list_upper_set debería prevenir esto)
            first_valid_raw_table = next((tbl for tbl in raw_tables_list if isinstance(tbl, str)), None)
            if first_valid_raw_table:
                main_table = first_valid_raw_table
                self.logger.info(f"[SQLGenerator] Tabla principal elegida por fallback (primera de la lista raw): {main_table}")
            else: # pragma: no cover
                 # Esto no debería ser alcanzable si raw_tables_list tiene al menos una cadena válida.
                self.logger.error("[SQLGenerator] No se pudo determinar una tabla principal válida.")
                return "SELECT 'Error: No se pudo determinar la tabla principal' AS mensaje", []


        self.logger.info(f"[SQLGenerator] Tabla principal FINAL: {main_table}")

        select_cols_input = structured_info.get("columns", ["*"])
        if not select_cols_input:
            select_cols_input = ["*"]
        
        # --- INICIO: USO DE JOINs DINÁMICOS ---
        # Si structured_info['joins'] está presente y es una lista, usarla para los JOINs
        join_clauses_sql = []
        already_joined_tables = {main_table.upper()}
        if isinstance(structured_info.get('joins'), list) and structured_info['joins']:
            for join_def in structured_info['joins']:
                t1 = join_def.get('table')
                t2 = join_def.get('foreign_table')
                c1 = join_def.get('column')
                c2 = join_def.get('foreign_column')
                if t1 and t2 and c1 and c2:
                    join_sql = f" INNER JOIN {t2} ON {t1}.{c1} = {t2}.{c2}"
                    join_clauses_sql.append(join_sql)
                    already_joined_tables.add(t2.upper())
        # Si no hay joins explícitos, intentar inferirlos como antes (lógica previa)
        else:
            # Path: EPISODES -> EPIS_DIAGNOSTICS -> CODR_DIAGNOSTIC_GROUPS
            if TABLE_CODR_DIAGNOSTIC_GROUPS.upper() in already_joined_tables:
                self.logger.info(f"[SQLGenerator] {TABLE_CODR_DIAGNOSTIC_GROUPS} ya está unida.")
            else:
                if not self._add_specific_join_if_needed(TABLE_EPISODES.upper(), TABLE_EPIS_DIAGNOSTICS.upper(),
                                                        already_joined_tables, join_clauses_sql, relations_map, all_known_tables_original_case):
                    self.logger.warning(f"Fallo al unir {TABLE_EPISODES} con {TABLE_EPIS_DIAGNOSTICS} para la ruta de diagnóstico.") # MODIFICADO: Usar self.logger
                
                if not self._add_specific_join_if_needed(TABLE_EPIS_DIAGNOSTICS.upper(), TABLE_CODR_DIAGNOSTIC_GROUPS.upper(),
                                                        already_joined_tables, join_clauses_sql, relations_map, all_known_tables_original_case):
                    self.logger.warning(f"Fallo al unir {TABLE_EPIS_DIAGNOSTICS} con {TABLE_CODR_DIAGNOSTIC_GROUPS} para la ruta de diagnóstico.") # MODIFICADO: Usar self.logger
            
            # Path: EPISODES -> EPIS_PROCEDURES -> PROC_PROCEDURES -> PROC_PROCEDURE_TYPES
            if TABLE_PROC_PROCEDURE_TYPES.upper() in already_joined_tables:
                self.logger.info(f"[SQLGenerator] {TABLE_PROC_PROCEDURE_TYPES} ya está unida.")
            else:
                if not self._add_specific_join_if_needed(TABLE_EPISODES.upper(), TABLE_EPIS_PROCEDURES.upper(),
                                                        already_joined_tables, join_clauses_sql, relations_map, all_known_tables_original_case):
                    self.logger.warning(f"Fallo al unir {TABLE_EPISODES} con {TABLE_EPIS_PROCEDURES} para la ruta de procedimiento.") # MODIFICADO: Usar self.logger
                
                if not self._add_specific_join_if_needed(TABLE_EPIS_PROCEDURES.upper(), TABLE_PROC_PROCEDURES.upper(),
                                                        already_joined_tables, join_clauses_sql, relations_map, all_known_tables_original_case):
                    self.logger.warning(f"Fallo al unir {TABLE_EPIS_PROCEDURES} con {TABLE_PROC_PROCEDURES} para la ruta de procedimiento.") # MODIFICADO: Usar self.logger
                
                if not self._add_specific_join_if_needed(TABLE_PROC_PROCEDURES.upper(), TABLE_PROC_PROCEDURE_TYPES.upper(),
                                                        already_joined_tables, join_clauses_sql, relations_map, all_known_tables_original_case):
                    self.logger.warning(f"Fallo al unir {TABLE_PROC_PROCEDURES} con {TABLE_PROC_PROCEDURE_TYPES} para la ruta de procedimiento.") # MODIFICADO: Usar self.logger
            
            # Path: MEDI_MEDICATIONS -> MEDI_PHARMA_THERAPEUTIC_GROUPS
            if TABLE_MEDI_PHARMA_THERAPEUTIC_GROUPS.upper() in already_joined_tables:
                self.logger.info(f"[SQLGenerator] {TABLE_MEDI_PHARMA_THERAPEUTIC_GROUPS} ya está unida.")
            else:
                if not self._add_specific_join_if_needed(TABLE_MEDI_MEDICATIONS.upper(), TABLE_MEDI_PHARMA_THERAPEUTIC_GROUPS.upper(),
                                                        already_joined_tables, join_clauses_sql, relations_map, all_known_tables_original_case):
                    self.logger.warning(f"Fallo al unir {TABLE_MEDI_MEDICATIONS} con {TABLE_MEDI_PHARMA_THERAPEUTIC_GROUPS} para la ruta de grupo terapéutico.") # MODIFICADO: Usar self.logger
        # --- FIN: USO DE JOINs DINÁMICOS ---

        processed_select_cols = []
        if isinstance(select_cols_input, list):
            for col_ref_item in select_cols_input:
                col_ref = col_ref_item.get("name", col_ref_item) if isinstance(col_ref_item, dict) else col_ref_item
                col_alias = col_ref_item.get("alias") if isinstance(col_ref_item, dict) else None

                actual_col_name_to_add = None
                is_pseudo_column = False

                if col_ref == "diagnostico_descripcion":
                    is_pseudo_column = True
                    codr_diag_table_name_orig = self._ensure_diagnosis_path(already_joined_tables, join_clauses_sql, relations_map, all_known_tables_original_case)
                    if codr_diag_table_name_orig:
                        actual_col_name_to_add = f"{codr_diag_table_name_orig}.{COLUMN_DGGR_DESCRIPTION_ES}"
                        if not col_alias: col_alias = "diagnostico_descripcion" # Alias por defecto
                    else:
                        self.logger.warning("[SQLGenerator] No se pudo añadir la ruta de diagnóstico para la columna SELECT 'diagnostico_descripcion'.")
                        actual_col_name_to_add = f"'Error: diagnostico_descripcion no disponible'"
                        if not col_alias: col_alias = "diagnostico_descripcion_error"
                
                elif col_ref == "tipo_procedimiento_descripcion":
                    is_pseudo_column = True
                    proc_types_table_name_orig = self._ensure_procedure_path(already_joined_tables, join_clauses_sql, relations_map, all_known_tables_original_case)
                    if proc_types_table_name_orig:
                        actual_col_name_to_add = f"{proc_types_table_name_orig}.{COLUMN_PRTY_DESCRIPTION_ES}" # Uso de constante
                        if not col_alias: col_alias = "tipo_procedimiento_descripcion"
                    else:
                        self.logger.warning("[SQLGenerator] No se pudo añadir la ruta de procedimiento para la columna SELECT 'tipo_procedimiento_descripcion'.")
                        actual_col_name_to_add = f"'Error: tipo_procedimiento_descripcion no disponible'"
                        if not col_alias: col_alias = "tipo_procedimiento_descripcion_error"

                elif col_ref == "dias_ingreso":
                    is_pseudo_column = True
                    episodes_table_original_name = all_known_tables_original_case.get(TABLE_EPISODES.upper())
                    # Asegurar que EPIS_EPISODES está unida o es la tabla principal
                    if episodes_table_original_name and (TABLE_EPISODES.upper() in already_joined_tables or \
                                                        self._add_specific_join_if_needed(main_table.upper(), TABLE_EPISODES.upper(), 
                                                                                         already_joined_tables, join_clauses_sql, 
                                                                                         relations_map, all_known_tables_original_case)):
                         actual_col_name_to_add = f"ROUND(julianday({episodes_table_original_name}.{COLUMN_EPIS_CLOSED_DATE}) - julianday({episodes_table_original_name}.{COLUMN_EPIS_START_DATE}), 1)"
                         if not col_alias: col_alias = "dias_ingreso"
                    else:
                        self.logger.warning(f"[SQLGenerator] La tabla {TABLE_EPISODES} no está disponible para calcular 'dias_ingreso'.")
                        actual_col_name_to_add = f"'Error: dias_ingreso no disponible'"
                        if not col_alias: col_alias = "dias_ingreso_error"
                
                if is_pseudo_column:
                    if actual_col_name_to_add:
                        processed_select_cols.append(f"{actual_col_name_to_add} AS {self._sanitize_alias(col_alias)}" if col_alias else actual_col_name_to_add)
                else:
                    # INICIO: Lógica original para columnas normales (adaptada de líneas 100-120 aprox.)
                    # Esta parte debe integrarse cuidadosamente con la lógica existente de _find_actual_column_name
                    # y normalización de prefijos de tabla.
                    if isinstance(col_ref, str):
                        if col_ref == "*":
                            # Si es "*", y main_table es la única tabla unida hasta ahora (o no hay joins aún), usar "*"
                            # Si hay múltiples tablas, es mejor prefijar con main_table.* o listar explícitamente.
                            # Por simplicidad, si es "*", lo dejamos tal cual, asumiendo que el contexto lo resolverá.
                            # O, si solo hay una tabla en `already_joined_tables` (la main_table), entonces "*" está bien.
                            # Si hay más, `main_table.*` podría ser más seguro, o expandir a todas las columnas de todas las tablas.
                            # La lógica original usaba `main_table` como prefijo si `col_ref` no tenía punto.
                            # Esto es un área que puede necesitar más refinamiento basado en el comportamiento deseado para "*".
                            if len(already_joined_tables) <= 1 and main_table:
                                 processed_select_cols.append(f"{main_table}.*") # O solo "*"
                            else: # Si hay múltiples tablas, "*" es ambiguo sin más contexto.
                                  # Podríamos listar todas las columnas de la tabla principal.
                                  # O simplemente pasar "*", pero puede ser problemático.
                                  # Por ahora, si es "*", lo pasamos, pero registramos una advertencia si hay múltiples tablas.
                                if len(already_joined_tables) > 1:
                                    self.logger.debug(f"[SQLGenerator] Columna SELECT '*' usada con múltiples tablas ({already_joined_tables}) unidas. Puede ser ambiguo.")
                                processed_select_cols.append("*")

                        elif '.' in col_ref:
                            parts = col_ref.split('.', 1)
                            table_prefix_original_case = self._find_actual_table_name_case(parts[0], all_known_tables_original_case)
                            col_name_part = parts[1]
                            actual_col_name = self._find_actual_column_name(table_prefix_original_case if table_prefix_original_case else parts[0], col_name_part, db_structure)
                            
                            # Si la tabla prefijo no está unida aún, intentar unirla.
                            # Esto es una heurística y puede ser agresiva.
                            table_prefix_upper = parts[0].upper()
                            if table_prefix_upper not in already_joined_tables and table_prefix_upper in all_known_tables_original_case:
                                self.logger.debug(f"Tabla {parts[0]} en SELECT no unida, intentando JOIN inferido desde {main_table.upper()}")
                                self._add_specific_join_if_needed(main_table.upper(), table_prefix_upper, 
                                                                  already_joined_tables, join_clauses_sql, 
                                                                  relations_map, all_known_tables_original_case)

                            processed_select_cols.append(f"{table_prefix_original_case if table_prefix_original_case else parts[0]}.{actual_col_name if actual_col_name else col_name_part}")
                        else: # No hay prefijo de tabla
                            # Intentar encontrar en la tabla principal primero
                            actual_col_name = self._find_actual_column_name(main_table, col_ref, db_structure)
                            if actual_col_name:
                                processed_select_cols.append(f"{main_table}.{actual_col_name}")
                            else:
                                # Si no está en la tabla principal, buscar en otras tablas ya unidas
                                found_in_other_table = False
                                for joined_tbl_upper in already_joined_tables:
                                    if joined_tbl_upper == main_table.upper(): continue # Ya chequeado
                                    joined_tbl_original_case = all_known_tables_original_case.get(joined_tbl_upper)
                                    if joined_tbl_original_case:
                                        actual_col_name_other = self._find_actual_column_name(joined_tbl_original_case, col_ref, db_structure)
                                        if actual_col_name_other:
                                            processed_select_cols.append(f"{joined_tbl_original_case}.{actual_col_name_other}")
                                            found_in_other_table = True
                                            break
                                if not found_in_other_table:
                                    # Si no se encuentra en ninguna tabla unida, añadir tal cual (puede ser una función SQL o un literal)
                                    # O podría ser un error si se esperaba una columna.
                                    self.logger.debug(f"Columna SELECT '{col_ref}' sin prefijo no encontrada en tablas unidas. Añadiendo tal cual.")
                                    processed_select_cols.append(str(col_ref)) # Podría ser una función como COUNT(*)
                    else: # No es string (ej. si el LLM da algo raro)
                        processed_select_cols.append(str(col_ref))
                    # FIN: Lógica original para columnas normales
        else: # select_cols_input no es una lista (ej. string)
            processed_select_cols = [str(select_cols_input)] if select_cols_input else ["*"]


        if not processed_select_cols: # Fallback final si todo falla
            self.logger.warning("[SQLGenerator] processed_select_cols está vacía, usando '*' por defecto.")
            processed_select_cols = ["*"]
            
        self.logger.info(f"[SQLGenerator] Columnas a seleccionar (procesadas): {processed_select_cols}")

        sql = f"SELECT {', '.join(processed_select_cols)} FROM {main_table}"
        self.logger.debug(f"[SQLGenerator] SQL inicial: {sql}")

        # already_joined_tables = {main_table.upper()} # Ya inicializado arriba
        # join_clauses_sql = [] # Ya inicializado arriba
        # Procesar JOINs explícitos del LLM de forma iterativa
        if isinstance(llm_joins_input, list) and llm_joins_input:
            current_llm_joins_to_process = list(llm_joins_input)
            max_passes = len(current_llm_joins_to_process) + 2 # Un poco más de margen

            for pass_num in range(max_passes):
                if not current_llm_joins_to_process:
                    self.logger.info(f"[SQLGenerator] Todos los {len(llm_joins_input)} JOINs del LLM procesados o descartados en {pass_num} pasadas.")
                    break

                joins_added_in_this_pass_count = 0
                next_round_pending_joins = []
                self.logger.debug(f"[SQLGenerator] LLM JOINs - Pase {pass_num + 1}: Procesando {len(current_llm_joins_to_process)} JOINs pendientes.")

                for i, join_item in enumerate(current_llm_joins_to_process):
                    self.logger.debug(f"[SQLGenerator] LLM JOINs - Pase {pass_num + 1}, Item {i+1}: {join_item}")
                    if not isinstance(join_item, dict):
                        self.logger.warning(f"[SQLGenerator] Elemento JOIN del LLM no es un diccionario: {join_item}. Se omite.")
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
                                self.logger.debug(f"[SQLGenerator] ON condition construida y normalizada para JOIN LLM: {on_condition}")
                            else:
                                self.logger.warning(f"[SQLGenerator] JOIN LLM #{i+1} omitido: faltan tablas, on_condition o columnas para construirla. Item: {join_item}")
                                next_round_pending_joins.append(join_item) 
                                continue
                        else:
                            self.logger.warning(f"[SQLGenerator] JOIN LLM #{i+1} omitido: información de tabla o condición ON inválida/faltante. Item: {join_item}")
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
                        self.logger.debug(f"[SQLGenerator] JOIN LLM {join_item} conecta tablas ({llm_table1}, {llm_table2}) ya en el grafo. Se omite la adición de cláusula JOIN duplicada, pero se considera procesado.")
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
                        self.logger.info(f"[SQLGenerator] JOIN LLM procesado y añadido: {join_sql_segment}. Tablas unidas ahora: {already_joined_tables}")
                        joins_added_in_this_pass_count += 1
                    else:
                        self.logger.debug(f"[SQLGenerator] JOIN LLM {join_item} no se puede procesar en esta pasada (una tabla no conectada o ambas ya conectadas sin ser el caso anterior). Se pasa a la siguiente ronda.")
                        next_round_pending_joins.append(join_item)
                
                current_llm_joins_to_process = next_round_pending_joins
                
                if not current_llm_joins_to_process: # Todos los JOINs del LLM se procesaron o descartaron
                    self.logger.info(f"[SQLGenerator] Todos los JOINs del LLM manejados en la pasada {pass_num + 1}.")
                    break 
                
                if joins_added_in_this_pass_count == 0:
                    self.logger.warning(f"[SQLGenerator] No se añadieron nuevos JOINs del LLM en la pasada {pass_num + 1}, pero {len(current_llm_joins_to_process)} JOINs siguen pendientes. Deteniendo el procesamiento de JOINs del LLM.")
                    break
            
            if current_llm_joins_to_process:
                 self.logger.error(f"[SQLGenerator] {len(current_llm_joins_to_process)} JOINs del LLM no pudieron ser integrados al grafo final después de {max_passes} pasadas: {current_llm_joins_to_process}")

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

        self.logger.info(f"[SQLGenerator] Tablas requeridas para inferencia de JOINs (después de JOINs del LLM): {required_tables_for_query_inference}")
        self.logger.info(f"[SQLGenerator] Tablas ya unidas (después de JOINs del LLM y antes de inferencia): {already_joined_tables}")
        
        # --- INICIO SECCIÓN DE INFERENCIA DE JOINS (adaptada) ---
        if relations_map and db_structure:
            max_join_inference_passes = len(required_tables_for_query_inference) + 1
            for pass_num_inf in range(max_join_inference_passes):
                if already_joined_tables.issuperset(required_tables_for_query_inference):
                    self.logger.info(f"[SQLGenerator] Inferencia de JOINs: Todas las {len(required_tables_for_query_inference)} tablas requeridas ya están unidas.")
                    break

                new_join_made_in_inference_pass = False
                tables_to_try_to_reach_in_inference = list(required_tables_for_query_inference - already_joined_tables)
                
                if not tables_to_try_to_reach_in_inference:
                    self.logger.debug("[SQLGenerator] Inferencia de JOINs: No quedan tablas por alcanzar.")
                    break
                
                self.logger.debug(f"[SQLGenerator] Inferencia de JOINs (Pase {pass_num_inf+1}): Intentando alcanzar {tables_to_try_to_reach_in_inference} desde {already_joined_tables}")

                for table_to_reach_upper in tables_to_try_to_reach_in_inference:
                    found_path_for_current_target = False
                    for joined_table_upper in list(already_joined_tables): # Iterar sobre copia
                        # Asumimos que relations_map usa claves en MAYÚSCULAS y nombres de tabla en MAYÚSCULAS dentro de las relaciones
                        possible_relations = relations_map.get(joined_table_upper, []) 
                        
                        for rel in possible_relations:
                            if not isinstance(rel, dict): # pragma: no cover
                                self.logger.warning(f"[SQLGenerator] Elemento de relación no es un dict: {rel} para tabla {joined_table_upper}")
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
                                self.logger.warning(f"[SQLGenerator] Relación inválida, faltan nombres de columna: {rel}")
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
                                self.logger.info(f"[SQLGenerator] JOIN INFERIDO añadido: {join_sql_segment} (Original: {on_condition_inferred_raw})")
                                break 
                        
                        if found_path_for_current_target:
                            break 
                
                if not new_join_made_in_inference_pass and not already_joined_tables.issuperset(required_tables_for_query_inference):
                    missing_tables = required_tables_for_query_inference - already_joined_tables
                    self.logger.warning(f"[SQLGenerator] Inferencia de JOINs (Pase {pass_num_inf+1}): No se pudieron inferir más JOINs, pero aún faltan: {missing_tables}.")
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
            
            self.logger.error(log_message)
            # Escapar comillas simples en el mensaje de error para que sea una cadena SQL válida
            safe_error_query_message = error_query_message.replace("'", "''")
            return f"SELECT '{safe_error_query_message}' AS mensaje_error_sql_generator", []

        # --- FIN SECCIÓN DE INFERENCIA DE JOINS ---

        if join_clauses_sql:
            sql += "".join(join_clauses_sql)

        self.logger.info("[SQLGenerator] Procesamiento de JOINs (LLM + inferidos) completado.")
        self.logger.debug(f"[SQLGenerator] SQL después de JOINs: {sql}")
        # --- FIN LÓGICA JOIN MEJORADA ---

        # --- INICIO: CONDICIONES WHERE ---
        where_clauses = []
        # Si diagnosis_variants está presente, construir condiciones OR para diagnóstico
        diagnosis_variants = structured_info.get('diagnosis_variants')
        if diagnosis_variants:
            # Asumimos que diagnosis_variants ya contiene todos los sinónimos y variantes relevantes
            table_prefix = 'ED' if 'ED' in all_known_tables_original_case else main_table
            all_conds = []
            for variant in diagnosis_variants:
                # Construir condición LIKE para cada variante
                all_conds.append(f"{table_prefix}.DIAG_DESCRIPTION_ES LIKE '%{variant}%'")
            where_clauses.append('(' + ' OR '.join(all_conds) + ')')
        # --- FIN: CONDICIONES WHERE ---

        # Procesar condiciones WHERE
        conditions = structured_info.get("conditions", [])
        self.logger.info(f"[SQLGenerator] Procesando {len(conditions)} condiciones WHERE...")
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
                            self.logger.info(f"[SQLGenerator] Condición para {patient_id_column_qualified} (o {patient_id_column_name}) ya existe en 'conditions'. No se añadirá patient_id por separado.")
                            break
            
            if not has_existing_patient_id_condition:
                self.logger.info(f"[SQLGenerator] Añadiendo condición para patient_id: {patient_id_column_qualified} = {patient_id_value}")
                where_clauses.append(f"{patient_id_column_qualified} = ?")
                params.append(patient_id_value)

        if isinstance(conditions, dict): 
            conditions = [conditions]
            self.logger.debug("[SQLGenerator] Condiciones convertidas de dict a lista.")
        elif isinstance(conditions, str): 
            try:
                parsed_conditions = json.loads(conditions)
                if isinstance(parsed_conditions, (list, dict)):
                    conditions = parsed_conditions if isinstance(parsed_conditions, list) else [parsed_conditions]
                    self.logger.debug("[SQLGenerator] Condiciones string parseadas como JSON.")
                else: 
                    conditions = [conditions] 
                    self.logger.debug("[SQLGenerator] Condiciones string envueltas en lista (no JSON).")
            except json.JSONDecodeError: 
                 conditions = [conditions] 
                 self.logger.debug("[SQLGenerator] Condiciones string envueltas en lista (error JSONDecode).")

        for i, condition_item in enumerate(conditions): 
            self.logger.debug(f"[SQLGenerator] Procesando condición WHERE #{i+1}: {condition_item}")
            if isinstance(condition_item, dict):
                column_variant = condition_item.get("column", "")
                operator = condition_item.get("operator", "=")
                value = condition_item.get("value", "")
                is_subquery = condition_item.get("is_subquery", False)
                subquery_details = condition_item.get("subquery_details")
                
                # Determinar la tabla para la columna y normalizar el nombre de la columna
                table_for_column = main_table # Por defecto
                actual_column_name_for_condition = column_variant # Mantener original si es pseudo-campo

                if column_variant not in [DIAGNOSTICO_TERMINO_PSEUDO_FIELD, 
                                          TIPO_PROCEDIMIENTO_TERMINO_PSEUDO_FIELD, 
                                          DIAS_INGRESO_CONDICION_PSEUDO_FIELD,
                                          GRUPO_TERAPEUTICO_TERMINO_PSEUDO_FIELD,
                                          NO_CONTIENE_INGREDIENTE_ACTIVO_PSEUDO_FIELD]:
                    if '.' in column_variant:
                        parts = column_variant.split('.',1)
                        table_for_column = self._find_actual_table_name_case(parts[0], all_known_tables_original_case) or parts[0]
                        column_name_only = parts[1]
                        # Usar _find_actual_column_name para normalizar la parte de la columna
                        normalized_col_part = self._find_actual_column_name(table_for_column, column_name_only, db_structure)
                        if normalized_col_part:
                            actual_column_name_for_condition = f"{table_for_column}.{normalized_col_part}"
                        else: # Fallback si no se normaliza
                            actual_column_name_for_condition = f"{table_for_column}.{column_name_only}"
                            self.logger.warning(f"[SQLGenerator] No se pudo normalizar la columna {column_name_only} para la tabla {table_for_column} en condición WHERE. Usando: {actual_column_name_for_condition}")
                    elif column_variant: # Columna sin prefijo de tabla
                        # Intentar normalizar con la tabla principal
                        normalized_col_part = self._find_actual_column_name(main_table, column_variant, db_structure)
                        if normalized_col_part:
                            actual_column_name_for_condition = f"{main_table}.{normalized_col_part}"
                        else: # Fallback si no se normaliza (podría ser un alias o función)
                            actual_column_name_for_condition = column_variant 
                            # Si es un alias y estamos en SQLite, _get_where_clause (para HAVING) lo reemplazará.
                            # Para WHERE, los alias no son válidos directamente.
                            # El manejo de alias en WHERE es más complejo que en HAVING.
                            # Por ahora, asumimos que si no se normaliza, es una columna real o una expresión válida.
                            self.logger.debug(f"[SQLGenerator] Columna '{column_variant}' en WHERE no normalizada contra {main_table}. Usando tal cual.")
                
                # Logging antes de la lógica de pseudo-campos
                self.logger.debug(f"[SQLGenerator] Condición dict (pre-pseudo): col_variant='{column_variant}', actual_col_name_for_condition='{actual_column_name_for_condition}', op='{operator}', val='{value}'")


                if column_variant == DIAGNOSTICO_TERMINO_PSEUDO_FIELD:
                    # ... (existing logic for DIAGNOSTICO_TERMINO_PSEUDO_FIELD)
                    self.logger.debug(f"[SQLGenerator] Procesando pseudo-field: {DIAGNOSTICO_TERMINO_PSEUDO_FIELD}")
                    codr_diag_table_name_orig = self._ensure_diagnosis_path(already_joined_tables, join_clauses_sql, relations_map, all_known_tables_original_case)
                    if codr_diag_table_name_orig:
                        processed_value = value
                        # Asumir LIKE para búsqueda de términos
                        if not ('%' in str(value) or '_' in str(value)):
                            processed_value = f"%{self._ensure_string(value)}%"
                        
                        where_clauses.append(f"{codr_diag_table_name_orig}.{COLUMN_DGGR_DESCRIPTION_ES} LIKE ?")
                        params.append(processed_value)
                        self.logger.info(f"[SQLGenerator] Condición añadida para {DIAGNOSTICO_TERMINO_PSEUDO_FIELD}: {codr_diag_table_name_orig}.{COLUMN_DGGR_DESCRIPTION_ES} LIKE {processed_value}")
                    else:
                        self.logger.warning(f"[SQLGenerator] No se pudo añadir la ruta de diagnóstico para la condición {DIAGNOSTICO_TERMINO_PSEUDO_FIELD}. Condición omitida.")
                    continue

                elif column_variant == TIPO_PROCEDIMIENTO_TERMINO_PSEUDO_FIELD:
                    # ... (existing logic for TIPO_PROCEDIMIENTO_TERMINO_PSEUDO_FIELD)
                    self.logger.debug(f"[SQLGenerator] Procesando pseudo-field: {TIPO_PROCEDIMIENTO_TERMINO_PSEUDO_FIELD}")
                    proc_types_table_name_orig = self._ensure_procedure_path(already_joined_tables, join_clauses_sql, relations_map, all_known_tables_original_case)
                    if proc_types_table_name_orig:
                        processed_value = value
                        if not ('%' in str(value) or '_' in str(value)):
                            processed_value = f"%{self._ensure_string(value)}%"

                        where_clauses.append(f"{proc_types_table_name_orig}.{COLUMN_PRTY_DESCRIPTION_ES} LIKE ?")
                        params.append(processed_value)
                        self.logger.info(f"[SQLGenerator] Condición añadida para {TIPO_PROCEDIMIENTO_TERMINO_PSEUDO_FIELD}: {proc_types_table_name_orig}.{COLUMN_PRTY_DESCRIPTION_ES} LIKE {processed_value}")
                    else:
                        self.logger.warning(f"[SQLGenerator] No se pudo añadir la ruta de procedimiento para la condición {TIPO_PROCEDIMIENTO_TERMINO_PSEUDO_FIELD}. Condición omitida.")
                    continue
                
                elif column_variant == DIAS_INGRESO_CONDICION_PSEUDO_FIELD:
                    # ... (existing logic for DIAS_INGRESO_CONDICION_PSEUDO_FIELD)
                    self.logger.debug(f"[SQLGenerator] Procesando pseudo-field: {DIAS_INGRESO_CONDICION_PSEUDO_FIELD}")
                    episodes_table_original_name = all_known_tables_original_case.get(TABLE_EPISODES.upper())
                    if episodes_table_original_name and (TABLE_EPISODES.upper() in already_joined_tables or \
                                                        self._add_specific_join_if_needed(main_table.upper(), TABLE_EPISODES.upper(),
                                                                                         already_joined_tables, join_clauses_sql,
                                                                                         relations_map, all_known_tables_original_case)):
                        # Construir la expresión para días de ingreso
                        dias_ingreso_expr = f"ROUND(julianday({episodes_table_original_name}.{COLUMN_EPIS_CLOSED_DATE}) - julianday({episodes_table_original_name}.{COLUMN_EPIS_START_DATE}), 1)"
                        
                        # Normalizar operador y valor
                        normalized_op = self._normalize_operator(operator)
                        # El valor debe ser numérico para comparación
                        try:
                            numeric_value = float(value)
                            where_clauses.append(f"{dias_ingreso_expr} {normalized_op} ?")
                            params.append(numeric_value)
                            self.logger.info(f"[SQLGenerator] Condición añadida para {DIAS_INGRESO_CONDICION_PSEUDO_FIELD}: {dias_ingreso_expr} {normalized_op} {numeric_value}")
                        except ValueError:
                            self.logger.warning(f"[SQLGenerator] Valor no numérico para {DIAS_INGRESO_CONDICION_PSEUDO_FIELD}: '{value}'. Condición omitida.")
                    else:
                        self.logger.warning(f"[SQLGenerator] La tabla {TABLE_EPISODES} no está disponible para la condición '{DIAS_INGRESO_CONDICION_PSEUDO_FIELD}'. Condición omitida.")
                    continue

                elif column_variant == GRUPO_TERAPEUTICO_TERMINO_PSEUDO_FIELD:
                    self.logger.debug(f"[SQLGenerator] Procesando pseudo-field: {GRUPO_TERAPEUTICO_TERMINO_PSEUDO_FIELD}")
                    phtg_table_name_orig = self._ensure_therapeutic_group_path(already_joined_tables, join_clauses_sql, relations_map, all_known_tables_original_case, main_table)
                    if phtg_table_name_orig:
                        processed_value = self._ensure_string(value)
                        # Usar LIKE para búsqueda de términos por defecto
                        if not ('%' in processed_value or '_' in processed_value):
                            processed_value = f"%{processed_value}%"
                        
                        where_clauses.append(f"{phtg_table_name_orig}.{COLUMN_PHTG_DESCRIPTION_ES} LIKE ?")
                        params.append(processed_value)
                        self.logger.info(f"[SQLGenerator] Condición añadida para {GRUPO_TERAPEUTICO_TERMINO_PSEUDO_FIELD}: {phtg_table_name_orig}.{COLUMN_PHTG_DESCRIPTION_ES} LIKE '{processed_value}'")
                    else:
                        self.logger.warning(f"[SQLGenerator] No se pudo añadir la ruta para {GRUPO_TERAPEUTICO_TERMINO_PSEUDO_FIELD}. Condición omitida.")
                    continue

                elif column_variant == NO_CONTIENE_INGREDIENTE_ACTIVO_PSEUDO_FIELD:
                    self.logger.debug(f"[SQLGenerator] Procesando pseudo-field: {NO_CONTIENE_INGREDIENTE_ACTIVO_PSEUDO_FIELD}")
                    ingredient_name_to_exclude = self._ensure_string(value)

                    main_medi_table_original_case = all_known_tables_original_case.get(TABLE_MEDI_MEDICATIONS.upper())
                    mmc_table_orig = all_known_tables_original_case.get(TABLE_MEDI_MEDICATION_COMPONENTS.upper())
                    mai_table_orig = all_known_tables_original_case.get(TABLE_MEDI_ACTIVE_INGREDIENTS.upper())

                    if not main_medi_table_original_case:
                        self.logger.error(f"[SQLGenerator] Nombre original de {TABLE_MEDI_MEDICATIONS} no encontrado. No se puede aplicar {NO_CONTIENE_INGREDIENTE_ACTIVO_PSEUDO_FIELD}.")
                        continue
                    if not mmc_table_orig or not mai_table_orig:
                        self.logger.error(f"[SQLGenerator] Nombres originales de tablas para subconsulta de ingrediente no encontrados. MMC: {mmc_table_orig}, MAI: {mai_table_orig}")
                        continue

                    subquery = (
                        f"SELECT DISTINCT {mmc_table_orig}.{COLUMN_MMC_MEDI_ID} "
                        f"FROM {mmc_table_orig} "
                        f"JOIN {mai_table_orig} ON {mmc_table_orig}.{COLUMN_MMC_ACIN_ID} = {mai_table_orig}.{COLUMN_ACIN_ID} "
                        f"WHERE {mai_table_orig}.{COLUMN_ACIN_DESCRIPTION_ES} LIKE ?"
                    )
                    
                    where_clauses.append(f"{main_medi_table_original_case}.{COLUMN_MEDI_ID} NOT IN ({subquery})")
                    params.append(f"%{ingredient_name_to_exclude}%")
                    self.logger.info(f"[SQLGenerator] Condición NOT IN añadida para {NO_CONTIENE_INGREDIENTE_ACTIVO_PSEUDO_FIELD} '{ingredient_name_to_exclude}'")
                    continue
                
                # Si no es un pseudo-campo conocido, procesar como condición normal
                # La variable actual_column_name_for_condition ya fue preparada antes de los pseudo-campos.
                if not actual_column_name_for_condition and not is_subquery: # Revisar si la columna es válida
                    self.logger.warning(f"[SQLGenerator] Condición omitida por falta de nombre de columna válido (después de normalización y pseudo-campos): {condition_item}")
                    continue

                # Normalizar operador después de manejar pseudo_campos que podrían no usarlo directamente
                operator = self._normalize_operator(operator) # Ya se hizo antes, pero asegurar por si acaso
                self.logger.debug(f"[SQLGenerator] Condición dict (post-pseudo): col='{actual_column_name_for_condition}', op='{operator}', val='{value}', subquery={is_subquery}")

                if is_subquery and subquery_details:
                    # ... (existing subquery logic, ensure actual_column_name_for_condition is used for the outer part)
                    sub_select_variant = subquery_details.get("select_column")
                    sub_from = subquery_details.get("from_table")
                    sub_where_col_variant = subquery_details.get("where_column")
                    sub_where_op = self._normalize_operator(subquery_details.get("where_operator", "LIKE"))
                    sub_where_val = subquery_details.get("where_value")

                    # Normalizar columnas de la subconsulta
                    actual_sub_select = sub_select_variant
                    if sub_from and sub_select_variant and '.' not in sub_select_variant: # Asumir que pertenece a sub_from si no está calificada
                        actual_sub_select_candidate = self._find_actual_column_name(sub_from, sub_select_variant, db_structure)
                        actual_sub_select = f"{sub_from}.{actual_sub_select_candidate}" if actual_sub_select_candidate else f"{sub_from}.{sub_select_variant}"
                    elif sub_select_variant and '.' in sub_select_variant:
                        sub_sel_table, sub_sel_col = sub_select_variant.split('.',1)
                        actual_sub_select_candidate = self._find_actual_column_name(sub_sel_table, sub_sel_col, db_structure)
                        actual_sub_select = f"{sub_sel_table}.{actual_sub_select_candidate}" if actual_sub_select_candidate else f"{sub_sel_table}.{sub_sel_col}"
                    
                    actual_sub_where_col = sub_where_col_variant
                    if sub_from and sub_where_col_variant and '.' not in sub_where_col_variant:
                        actual_sub_where_col_candidate = self._find_actual_column_name(sub_from, sub_where_col_variant, db_structure)
                        actual_sub_where_col = f"{sub_from}.{actual_sub_where_col_candidate}" if actual_sub_where_col_candidate else f"{sub_from}.{sub_where_col_variant}"
                    elif sub_where_col_variant and '.' in sub_where_col_variant:
                        sub_wh_table, sub_wh_col = sub_where_col_variant.split('.',1)
                        actual_sub_where_col_candidate = self._find_actual_column_name(sub_wh_table, sub_wh_col, db_structure)
                        actual_sub_where_col = f"{sub_wh_table}.{actual_sub_where_col_candidate}" if actual_sub_where_col_candidate else f"{sub_wh_table}.{sub_wh_col}"

                    if not all([actual_sub_select, sub_from, actual_sub_where_col, sub_where_val]):
                        self.logger.warning(f"[SQLGenerator] Subconsulta mal formada (después de normalización), omitida: {subquery_details}")
                        continue
                    
                    processed_sub_where_val = sub_where_val
                    if sub_where_op == "LIKE" and not ('%' in str(sub_where_val) or '_' in str(sub_where_val)):
                        processed_sub_where_val = f"%{str(sub_where_val)}%"


                    where_clauses.append(f"{actual_column_name_for_condition} {operator} (SELECT {actual_sub_select} FROM {sub_from} WHERE {actual_sub_where_col} {sub_where_op} ?)")
                    params.append(processed_sub_where_val)
                    self.logger.debug(f"[SQLGenerator] Subconsulta añadida: {actual_column_name_for_condition} {operator} (SELECT {actual_sub_select} FROM {sub_from} WHERE {actual_sub_where_col} {sub_where_op} {processed_sub_where_val})")


                elif operator in ["IN", "NOT IN"]:
                    # ... (existing IN/NOT IN logic, ensure actual_column_name_for_condition is used)
                    if isinstance(value, list) and value:
                        placeholders = ", ".join(["?" for _ in value])
                        where_clauses.append(f"{actual_column_name_for_condition} {operator} ({placeholders})")
                        params.extend(value)
                    elif isinstance(value, str) and value: 
                        where_clauses.append(f"{actual_column_name_for_condition} {operator} (?)")
                        params.append(value)
                    else:
                        self.logger.warning(f"[SQLGenerator] Valor no válido para operador {operator} en columna {actual_column_name_for_condition}: {value}. Condición omitida.")
                        continue
                elif operator == "BETWEEN":
                    # ... (existing BETWEEN logic, ensure actual_column_name_for_condition is used)
                    if isinstance(value, list) and len(value) == 2:
                        where_clauses.append(f"{actual_column_name_for_condition} BETWEEN ? AND ?")
                        params.extend(value)
                    else:
                        self.logger.warning(f"[SQLGenerator] Valor no válido para operador BETWEEN en columna {actual_column_name_for_condition}: {value}. Condición omitida.")
                        continue
                elif operator == "LIKE":
                    # ... (existing LIKE logic, ensure actual_column_name_for_condition is used)
                    if not isinstance(value, str):
                        try:
                            value = str(value)
                        except: # pragma: no cover
                            self.logger.warning(f"[SQLGenerator] No se pudo convertir el valor para LIKE a string en columna {actual_column_name_for_condition}: {value}. Condición omitida.")
                            continue
                    processed_value = value
                    if not ('%' in value or '_' in value):
                        processed_value = f"%{value}%"
                    
                    where_clauses.append(f"{actual_column_name_for_condition} {operator} ?")
                    params.append(processed_value)
                else: 
                    # ... (existing logic for other operators, ensure actual_column_name_for_condition is used)
                    where_clauses.append(f"{actual_column_name_for_condition} {operator} ?")
                    params.append(value)

        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
            self.logger.debug(f"[SQLGenerator] SQL con WHERE: {sql}")
        else:
            self.logger.info("[SQLGenerator] No se generaron cláusulas WHERE.")

        # Aplicar GROUP BY si está presente
        group_by_cols = structured_info.get("group_by")
        if group_by_cols:
            if isinstance(group_by_cols, str):
                group_by_cols = [group_by_cols]
            if isinstance(group_by_cols, list) and all(isinstance(col, str) for col in group_by_cols):
                sql += f" GROUP BY {', '.join(group_by_cols)}"
                self.logger.info(f"[SQLGenerator] Cláusula GROUP BY añadida: {group_by_cols}")
            else:
                self.logger.warning(f"[SQLGenerator] 'group_by' tiene un formato inválido y será ignorado: {group_by_cols}")

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
                sql += f" ORDER BY {', '.join(order_clauses)}"
                self.logger.info(f"[SQLGenerator] Cláusula ORDER BY añadida: {order_clauses}")
            else:
                self.logger.warning(f"[SQLGenerator] 'order_by' tiene un formato inválido y será ignorado: {order_by_info}")

        # Aplicar LIMIT si está presente
        limit = structured_info.get("limit")
        if limit is not None:
            try:
                limit_val = int(limit)
                if limit_val > 0:
                    sql += f" LIMIT {limit_val}"
                    self.logger.info(f"[SQLGenerator] Cláusula LIMIT añadida: {limit_val}")
            except ValueError:
                self.logger.warning(f"[SQLGenerator] 'limit' tiene un valor inválido y será ignorado: {limit}")

        self.logger.info(f"[SQLGenerator] SQL final generado: {sql}")
        self.logger.debug(f"[SQLGenerator] Parámetros finales: {params}")

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

    def _build_where_clause(self, conditions: List[Dict[str, Any]], current_aliases: Dict[str, str]) -> str:
        """
        Construye la cláusula WHERE de la consulta SQL a partir de la información estructurada.
        
        Args:
            conditions: Lista de diccionarios con la información de las condiciones WHERE.
            current_aliases: Diccionario con los alias actuales de las tablas en la consulta.

        Returns:
            Cadena con la cláusula WHERE construida.
        """
        where_conditions = []

        for condition in conditions:
            self.logger.debug(f"[SQLGenerator] Procesando condición para WHERE: {condition}")
            condition_type = condition.get("type", "simple").lower()

            if condition_type == "simple":
                # Condiciones simples como "columna = valor"
                column_name = condition.get("column")
                operator = condition.get("operator", "=")
                value = condition.get("value")

                if column_name is not None and value is not None:
                    # Normalizar el operador
                    operator = self._normalize_operator(operator)

                    # Formatear el valor para la consulta
                    formatted_value = self.db_config.format_value_for_query(value)

                    # Asegurarse de que la columna tenga el alias correcto si es necesario
                    column_alias = self._get_alias_for_table(column_name, current_aliases)

                    where_conditions.append(f"{column_alias} {operator} {formatted_value}")

            elif condition_type == "subquery":
                # Esto maneja subconsultas como "columna IN (SELECT ...)"
                subquery_dict = condition.get("subquery")
                subquery_sql_str = self._generate_sql_recursive(subquery_dict, is_subquery=True, outer_aliases=current_aliases)
                
                # Asegurarse de que la columna para comparación esté correctamente aliased
                outer_column_alias = self._get_alias_for_table(condition.get("outer_table", condition.get("table")), current_aliases)
                where_conditions.append(f"{outer_column_alias}.{condition['column']} {condition['operator']} ({subquery_sql_str})")

            elif condition_type == "exists":
                # Manejo de subconsultas EXISTS
                subquery_details = condition.get("subquery_details", {})
                correlation_rules = condition.get("correlation_conditions", [])

                if not subquery_details or not correlation_rules:
                    # Fallback a un EXISTS más simple si faltan detalles (registrar una advertencia o error)
                    where_conditions.append(f"-- MALFORMED EXISTS: {condition.get('original_text', 'Unknown condition')} --")
                    continue

                sub_from_table_info = subquery_details["from_table"]
                sub_from_clause = f"FROM {sub_from_table_info['name']} {sub_from_table_info['alias']}"

                sub_join_parts = []
                for join_info in subquery_details.get("joins", []):
                    sub_join_parts.append(
                        f"{join_info.get('type', 'INNER')} JOIN {join_info['target_table']['name']} {join_info['target_table']['alias']} "
                        f"ON {join_info['on_condition']}"
                    )

                # Correlacionar con las reglas de correlación
                correlation_conditions = []
                for rule in correlation_rules:
                    if isinstance(rule, dict):
                        outer_col = rule.get("outer_column")
                        inner_col = rule.get("inner_column")
                        if outer_col and inner_col:
                            # Normalizar columnas
                            actual_outer_col = self._find_actual_column_name(rule.get("outer_table", ""), outer_col)
                            actual_inner_col = self._find_actual_column_name(rule.get("inner_table", ""), inner_col)
                            

                            if actual_outer_col and actual_inner_col:
                                correlation_conditions.append(f"{actual_outer_col} = {actual_inner_col}")
                            else:
                                self.logger.warning(f"[SQLGenerator] No se pudieron normalizar ambas columnas para la regla de correlación: {rule}. Se omitirá esta regla.")
                    else:
                        self.logger.warning(f"[SQLGenerator] Regla de correlación no válida (no es dict): {rule}. Se omitirá.")

                # Unir condiciones de correlación con AND
                if correlation_conditions:
                    correlation_clause = " AND ".join(correlation_conditions)
                    where_conditions.append(f"EXISTS (SELECT 1 {sub_from_clause} {' '.join(sub_join_parts)} WHERE {correlation_clause})")
                    self.logger.info(f"[SQLGenerator] Condición EXISTS añadida con subconsulta correlacionada.")
                else:
                    # Si no hay condiciones de correlación, usar EXISTS simple
                    where_conditions.append(f"EXISTS (SELECT 1 {sub_from_clause} {' '.join(sub_join_parts)})")
                    self.logger.info(f"[SQLGenerator] Condición EXISTS añadida con subconsulta simple.")

        return " AND ".join(where_conditions) if where_conditions else ""