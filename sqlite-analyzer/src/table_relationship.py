import os
import json
import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict

# Constante para el archivo de esquema
SCHEMA_ENHANCER_FILE = "schema_enhanced.json"

def ensure_schema_file_exists(schema_path: str = SCHEMA_ENHANCER_FILE) -> None:
    """Crea un archivo de esquema básico si no existe"""
    if not os.path.exists(schema_path):
        logging.warning(f"No se encontró el archivo de esquema en: {schema_path}. Creando uno nuevo.")
        
        # Estructura base para el esquema
        base_schema = {
            "schema_knowledge": {
                "tables": {},
                "columns": {},
                "relationships": {},
                "synonyms": {},
                "examples": {}
            },
            "table_corpus": [],
            "table_names": [],
            "column_corpus": {},
            "column_names": {},
            "schema": {}
        }
        
        try:
            # Crear el directorio si no existe
            os.makedirs(os.path.dirname(os.path.abspath(schema_path)), exist_ok=True)
            
            # Escribir el archivo base
            with open(schema_path, 'w', encoding='utf-8') as f:
                json.dump(base_schema, f, ensure_ascii=False, indent=2)
            
            logging.info(f"Archivo de esquema creado exitosamente en: {schema_path}")
        except Exception as e:
            logging.error(f"Error al crear el archivo de esquema: {e}")

def generate_table_relationships_map(db_structure: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Genera un mapa completo de relaciones entre tablas analizando 
    la estructura de la base de datos.
    
    Args:
        db_structure: Diccionario con la estructura de la base de datos
        
    Returns:
        Diccionario con relaciones entre tablas
    """
    logging.info("Generando mapa de relaciones entre tablas...")
    relationship_map = {}
    
    # Identificar tabla de pacientes
    patient_table = None
    for table in db_structure:
        if "PATI" in table and ("PATIENT" in table or "PATIENTS" in table):
            patient_table = table
            break
    
    if not patient_table:
        logging.warning("No se encontró la tabla de pacientes")
        return relationship_map
    
    # Obtener la columna de ID en la tabla de pacientes
    patient_id_column = None
    for col in db_structure.get(patient_table, {}).get("columns", []):
        if "ID" in col.get("name", "") and "PATI" in col.get("name", ""):
            patient_id_column = col.get("name", "")
            break
    
    if not patient_id_column:
        patient_id_column = "PATI_ID"  # valor por defecto si no se encuentra
        
    # Analizar cada tabla para encontrar relaciones con la tabla de pacientes
    for table_name in db_structure:
        if table_name == patient_table:
            continue
            
        # Analizar columnas de esta tabla
        table_cols = [c.get("name", "") for c in db_structure.get(table_name, {}).get("columns", [])]
        
        # Buscar conexión directa con pacientes (columna que contiene PATI e ID)
        patient_fk = None
        for col in table_cols:
            if ("PATI" in col and "ID" in col) or "PATIENT_ID" in col:
                patient_fk = col
                relationship_map[table_name] = {
                    "type": "direct",
                    "patient_table": patient_table,
                    "patient_key": patient_id_column,
                    "table_key": patient_fk
                }
                logging.info(f"Relación directa encontrada: {table_name}.{patient_fk} -> {patient_table}.{patient_id_column}")
                break
    
    # Para tablas sin relación directa, buscar tablas intermedias
    tables_without_relation = [t for t in db_structure if t != patient_table and t not in relationship_map]
    
    for table_name in tables_without_relation:
        # Buscar tablas que tengan relaciones con esta tabla
        for potential_join_table in db_structure:
            if potential_join_table == table_name or potential_join_table == patient_table:
                continue
                
            # Si la tabla potencial ya tiene relación directa con pacientes
            if potential_join_table in relationship_map and relationship_map[potential_join_table]["type"] == "direct":
                # Buscar si hay relación entre la tabla actual y la tabla intermedia
                main_table_cols = [c.get("name", "") for c in db_structure.get(table_name, {}).get("columns", [])]
                join_table_cols = [c.get("name", "") for c in db_structure.get(potential_join_table, {}).get("columns", [])]
                
                # Buscar columnas que puedan servir como FK
                for main_col in main_table_cols:
                    if "ID" in main_col:
                        for join_col in join_table_cols:
                            # Buscar coincidencia de nombre sin el ID
                            main_name = main_col.replace("ID", "").replace("_", "")
                            join_name = join_col.replace("ID", "").replace("_", "")
                            
                            if (main_name in join_name or join_name in main_name) and len(main_name) > 2:
                                # Probable coincidencia encontrada
                                relationship_map[table_name] = {
                                    "type": "indirect",
                                    "patient_table": patient_table,
                                    "patient_key": patient_id_column,
                                    "intermediate_table": potential_join_table,
                                    "intermediate_patient_key": relationship_map[potential_join_table]["table_key"],
                                    "table_key": main_col,
                                    "intermediate_table_key": join_col
                                }
                                logging.info(f"Relación indirecta encontrada: {table_name}.{main_col} -> "
                                           f"{potential_join_table}.{join_col} -> "
                                           f"{patient_table}.{patient_id_column}")
                                break
                    
                    # Si ya encontramos relación, no seguir buscando
                    if table_name in relationship_map:
                        break

    return relationship_map

def save_table_relationships(relationships, file_path="table_relationships.json"):
    """
    Guarda las relaciones entre tablas en un archivo JSON.
    
    Args:
        relationships: Diccionario con las relaciones entre tablas
        file_path: Ruta del archivo donde guardar las relaciones
    """
    try:
        # Convertir claves de tupla a string para poder serializar a JSON
        serializable_relationships = {}
        for key, value in relationships.items():
            if isinstance(key, tuple):
                serializable_relationships[f"{key[0]}:{key[1]}"] = value
            else:
                serializable_relationships[str(key)] = value
        
        # Abrir en modo 'w' para sobrescribir completamente (no añadir al final)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_relationships, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Mapa de relaciones guardado con {len(relationships)} relaciones")
        return True
    except Exception as e:
        logging.error(f"Error al guardar mapa de relaciones: {e}")
        return False
    

def load_table_relationships(file_path="table_relationships.json"):
    """
    Carga las relaciones entre tablas desde un archivo JSON.
    
    Args:
        file_path: Ruta del archivo con las relaciones
        
    Returns:
        Diccionario con las relaciones entre tablas
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            serialized_relationships = json.load(f)
        
        # Convertir claves de string a tuplas
        relationships = {}
        for key, value in serialized_relationships.items():
            if ":" in key:
                table1, table2 = key.split(":")
                relationships[(table1, table2)] = value
            else:
                relationships[key] = value
                
        logging.info(f"Mapa de relaciones cargado con {len(relationships)} relaciones")
        return relationships
    except Exception as e:
        logging.warning(f"No se pudo cargar el mapa de relaciones: {e}")
        return {}
    

def infer_patient_relationship(table_name_or_structure, db_structure=None):
    """
    Intenta inferir la relación entre una tabla y la tabla de pacientes.
    
    Esta función puede ser llamada de dos formas:
    1. infer_patient_relationship(table_name, db_structure) - para una tabla específica
    2. infer_patient_relationship(db_structure) - para analizar todas las tablas
    
    Args:
        table_name_or_structure: Nombre de la tabla o estructura completa de la BD
        db_structure: Estructura de la base de datos (opcional si el primer arg es la estructura)
        
    Returns:
        Diccionario con información de la relación inferida o dict de tablas con sus relaciones
    """
    # Detectar si se llamó con un solo argumento (solo db_structure)
    if db_structure is None and isinstance(table_name_or_structure, dict):
        # Modo de procesamiento por lotes - analizar todas las tablas
        db_structure = table_name_or_structure
        all_relations = {}
        
        # Encontrar la tabla de pacientes primero
        patient_table = None
        for table in db_structure:
            if "PATI" in table and ("PATIENT" in table or "PATIENTS" in table):
                patient_table = table
                break
        
        if not patient_table:
            logging.warning("No se pudo encontrar la tabla de pacientes")
            return {}
            
        # Analizar cada tabla que no sea la de pacientes
        for table in db_structure:
            if table != patient_table:
                # Llamar recursivamente a esta misma función para cada tabla
                relation = infer_patient_relationship(table, db_structure)
                if relation:  # Solo añadir si se encontró relación
                    all_relations[(table, patient_table)] = relation.get("join_info", {})
                    
        return all_relations
    
    # Modo normal - analizar una sola tabla
    table_name = table_name_or_structure
    
    # El resto de la función original queda igual
    # Encontrar la tabla de pacientes
    patient_table = None
    for table in db_structure:
        if "PATI" in table and ("PATIENT" in table or "PATIENTS" in table):
            patient_table = table
            break
    
    if not patient_table:
        logging.warning("No se pudo encontrar la tabla de pacientes")
        return {}
    
    # Obtener columnas para ambas tablas
    table_cols = [c.get("name", "") for c in db_structure.get(table_name, {}).get("columns", [])]
    patient_cols = [c.get("name", "") for c in db_structure.get(patient_table, {}).get("columns", [])]
    
    # Buscar la columna ID en la tabla de pacientes
    patient_id_col = "PATI_ID"  # valor por defecto
    for col in patient_cols:
        if "ID" in col and "PATI" in col:
            patient_id_col = col
            break
    
    # Estrategia 1: Buscar columnas similares a ID de paciente en la tabla
    for col in table_cols:
        if ("PATI" in col and "ID" in col) or col == "PATIENT_ID":
            # Crear un formato de JOIN compatible con SQLGenerator
            join_info = {
                "table": table_name,
                "column": col,
                "foreign_table": patient_table,
                "foreign_column": patient_id_col
            }
            
            return {
                "type": "direct",
                "patient_table": patient_table,
                "patient_key": patient_id_col,
                "table_key": col,
                "join_info": join_info
            }
    # Estrategia 2: Buscar coincidencia por nomenclatura
    for col in table_cols:
        if "ID" in col or "FK" in col:
            # Verificar si el nombre de la columna sugiere relación con pacientes
            if "PAT" in col or "PACIENTE" in col:
                # Crear un formato de JOIN compatible con SQLGenerator
                join_info = {
                    "table": table_name,
                    "column": col,
                    "foreign_table": patient_table,
                    "foreign_column": patient_id_col
                }
                
                return {
                    "type": "direct",
                    "patient_table": patient_table,
                    "patient_key": patient_id_col,
                    "table_key": col,
                    "join_info": join_info  # Formato compatible con SQLGenerator
                }
    
    # Estrategia 3: Buscar tablas intermedias que conecten con ambas
    for int_table in db_structure:
        if int_table == table_name or int_table == patient_table:
            continue
            
        int_cols = [c.get("name", "") for c in db_structure.get(int_table, {}).get("columns", [])]
        
        # Buscar columnas que relacionen con la tabla principal
        main_fk = None
        for col in int_cols:
            if table_name.replace("_", "")[:4] in col.replace("_", ""):
                main_fk = col
                break
        
        # Buscar columnas que relacionen con la tabla de pacientes
        patient_fk = None
        for col in int_cols:
            if "PATI" in col and "ID" in col:
                patient_fk = col
                break
        
        if main_fk and patient_fk:
            # Encontrar la columna a la que hace referencia main_fk en la tabla principal
            main_pk = "ID"  # valor por defecto
            for col in table_cols:
                if "ID" in col and col.replace("ID", "") in main_fk:
                    main_pk = col
                    break
            
            # Crear formato de JOIN compatible
            join_info_1 = {
                "table": table_name,
                "column": main_pk,
                "foreign_table": int_table,
                "foreign_column": main_fk
            }
            
            join_info_2 = {
                "table": int_table,
                "column": patient_fk,
                "foreign_table": patient_table,
                "foreign_column": patient_id_col
            }
                    
            return {
                "type": "indirect",
                "patient_table": patient_table,
                "patient_key": patient_id_col,
                "intermediate_table": int_table,
                "intermediate_patient_key": patient_fk,
                "table_key": main_pk,
                "intermediate_table_key": main_fk,
                "join_info": [join_info_1, join_info_2]  # Lista de JOINs compatible con SQLGenerator
            }
    
    # No se encontró relación
    logging.warning(f"No se pudo inferir relación para {table_name} con {patient_table}")
    return {}

def ensure_relationships_map(db_structure: Dict[str, Dict], existing_relationships: Optional[Dict[str, List[Dict]]] = None) -> Dict[str, any]:
    """
    Asegura que el mapa de relaciones esté completo, infiriendo relaciones faltantes
    y registrando advertencias para tablas centrales si no se pueden conectar.
    
    Args:
        db_structure: Estructura de la base de datos
        force_rebuild: Si es True, regenera el mapa aunque ya exista
    """
    table_names = list(db_structure.keys())
    relationships_map = defaultdict(list)
    
    # Añadir relaciones explícitas existentes primero (ej. de un archivo JSON)
    if existing_relationships:
        for table1, relations in existing_relationships.items():
            for rel in relations:
                table2 = rel.get("related_table")
                if table1 in table_names and table2 in table_names:
                    relationships_map[table1].append({
                        "related_table": table2,
                        "join_condition": rel.get("join_condition"),
                        "type": rel.get("type", "inferred") # Marcar como 'explicit' o 'inferred'
                    })

    # Lista de tablas consideradas centrales para las cuales se emitirán advertencias si no se pueden conectar
    # central_tables_for_warnings = ["PATI_PATIENTS", "EPIS_EPISODES", "ACCI_PATIENT_CONDITIONS"]
    central_tables_for_warnings = ["PATI_PATIENTS", "EPIS_EPISODES"] # ACCI_PATIENT_CONDITIONS eliminada
    
    # Identificar la tabla de pacientes (necesaria para algunas lógicas de inferencia)
    patient_table = None
    for t_name in table_names:
        if "PATI_PATIENTS" in t_name.upper(): # Asumiendo un nombre común
            patient_table = t_name
            break
    if not patient_table:
        logging.warning("Tabla de pacientes no identificada, algunas inferencias de relación pueden ser limitadas.")

    # Inferir relaciones basadas en FKs de db_structure si no están en existing_relationships
    for table_name, table_info in db_structure.items():
        if 'foreign_keys' in table_info:
            for fk_info in table_info['foreign_keys']:
                related_table = fk_info.get('referenced_table')
                if related_table in table_names:
                    # Comprobar si esta relación ya existe para evitar duplicados
                    exists = False
                    for rel_entry in relationships_map.get(table_name, []):
                        if rel_entry.get("related_table") == related_table and \
                           rel_entry.get("join_condition") == f"{table_name}.{fk_info.get('foreign_key_column')} = {related_table}.{fk_info.get('referenced_column')}":
                            exists = True
                            break
                    if not exists:
                        relationships_map[table_name].append({
                            "related_table": related_table,
                            "join_condition": f"{table_name}.{fk_info.get('foreign_key_column')} = {related_table}.{fk_info.get('referenced_column')}",
                            "type": "foreign_key"
                        })
                        # Añadir también la relación inversa para facilitar búsquedas bidireccionales
                        relationships_map[related_table].append({
                            "related_table": table_name,
                            "join_condition": f"{related_table}.{fk_info.get('referenced_column')} = {table_name}.{fk_info.get('foreign_key_column')}",
                            "type": "foreign_key_reverse"
                        })
    
    # Lógica para construir un grafo y encontrar caminos (simplificada o adaptada de tu código)
    # Esta parte necesitaría las funciones build_graph y find_path o una implementación similar.
    # Por ahora, nos centraremos en la advertencia.
    # Ejemplo de cómo podría ser la lógica de advertencia (requiere un grafo y una función de búsqueda de ruta):
    
    # graph = build_graph_from_relationships(relationships_map, table_names) # Necesitarías esta función

    # for central_table in central_tables_for_warnings:
    #     if central_table not in table_names:
    #         logging.warning(f"Tabla central '{central_table}' no encontrada en db_structure.")
    #         continue
            
    #     # Comprobar conectividad a otras tablas importantes (ej. patient_table si es diferente)
    #     # Esto es un placeholder, la lógica real de conectividad es más compleja
    #     connected_to_patient = False
    #     if patient_table and central_table != patient_table:
    #         # path = find_path(graph, central_table, patient_table) # Necesitarías esta función
    #         # if path:
    #         #    connected_to_patient = True
    #         # else:
    #         #    logging.warning(f"No se pudo inferir relación para {central_table} con {patient_table}")
    #         pass # Placeholder para la lógica de búsqueda de ruta

        # Advertir si una tabla central no tiene relaciones directas o inferidas (excepto con ella misma)
        # if not any(rel.get("related_table") != central_table for rel in relationships_map.get(central_table, [])):
        #    logging.warning(f"No se encontraron relaciones salientes para la tabla central '{central_table}'.")

    return dict(relationships_map)

def normalize_join_format(join_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normaliza el formato de JOIN para hacerlo compatible con SQLGenerator.
    
    Args:
        join_data: Diccionario con información de JOIN en cualquier formato
        
    Returns:
        Diccionario con formato compatible con SQLGenerator
    """
    # Si ya tiene el formato esperado por SQLGenerator (table, column, foreign_table, foreign_column)
    if all(k in join_data for k in ["table", "column", "foreign_table", "foreign_column"]):
        return join_data
    
    # Si tiene formato tipo "type", "table1", "table2", "on"
    if all(k in join_data for k in ["type", "table1", "table2", "on"]):
        # Extraer nombres de columnas del "on"
        on_parts = join_data["on"].split("=")
        if len(on_parts) != 2:
            logging.warning(f"No se pudo parsear condición ON: {join_data['on']}")
            return {}
            
        t1_column = on_parts[0].strip().split(".")[-1].strip()
        t2_column = on_parts[1].strip().split(".")[-1].strip()
        
        return {
            "table": join_data["table1"],
            "column": t1_column,
            "foreign_table": join_data["table2"],
            "foreign_column": t2_column
        }
    
    # Si no tiene formato reconocido, devolver vacío
    logging.warning(f"Formato de JOIN no reconocido: {join_data}")
    return {}

def discover_table_relationships(db_structure):
    """
    Descubre automáticamente relaciones entre tablas analizando nombres de columnas.
    No requiere hardcodear ninguna relación específica.
    
    Args:
        db_structure: Diccionario con la estructura de la base de datos
        
    Returns:
        Diccionario con las relaciones detectadas entre tablas
    """
    relationships = {}
    
    # Identificar todas las columnas potencialmente relacionables (que terminan en _ID)
    for table_name, table_info in db_structure.items():
        for column in table_info.get('columns', []):
            col_name = column.get('name', '')
            
            # Buscar columnas que parezcan ser foreign keys
            if col_name.endswith('_ID') and col_name != 'ID':
                # Extraer el prefijo de la tabla a la que podría referirse
                prefix = col_name.replace('_ID', '')
                
                # Buscar tablas que coincidan con este prefijo
                for potential_table in db_structure.keys():
                    # Verificar si el prefijo coincide con el nombre de otra tabla
                    if (potential_table.endswith(prefix) or potential_table.startswith(prefix) or
                        prefix in potential_table.split('_')):
                        # Buscar la columna ID correspondiente en la tabla potencial
                        pk_col = next((c.get('name') for c in db_structure[potential_table].get('columns', []) 
                                    if c.get('name') == 'ID' or c.get('name') == f'{prefix}_ID' 
                                    or c.get('name', '').endswith('_ID') and c.get('primary_key', False)), None)
                        
                        if pk_col:
                            # Registrar la relación
                            if table_name not in relationships:
                                relationships[table_name] = []
                            
                            relationships[table_name].append({
                                'table': table_name,
                                'column': col_name,
                                'foreign_table': potential_table,
                                'foreign_column': pk_col,
                                'confidence': 'high' if col_name == f'{prefix}_ID' else 'medium'
                            })
    
    return relationships