from collections import deque
import logging
from typing import Dict, List, Tuple, Any, Optional, Set

def build_relationship_graph(db_connector, table_relationships: Optional[Dict[Tuple[str, str], Dict[str, str]]] = None, db_structure: Optional[Dict[str, Any]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Construye un grafo de relaciones entre tablas.
    Prioriza table_relationships si se proporciona, de lo contrario, usa claves foráneas de la BD.
    
    Args:
        db_connector: Conector a la base de datos.
        table_relationships: Mapa de relaciones predefinidas (opcional).
        db_structure: Estructura de la base de datos (opcional, para contexto).
        
    Returns:
        Diccionario donde las claves son nombres de tablas y los valores son listas de 
        diccionarios con información de relaciones salientes.
    """
    graph = {}
    
    # Inicializar el grafo con todas las tablas disponibles de db_structure o del conector
    current_db_structure = db_structure or db_connector.get_database_structure()
    for table_name in current_db_structure:
        graph[table_name] = []

    # Si se proporcionan table_relationships, usarlas para construir el grafo
    if table_relationships:
        logging.info("Construyendo grafo de relaciones desde table_relationships proporcionado (formato JSON esperado).")
        # El formato esperado de table_relationships es:
        # {
        #   "TABLE_A": [
        #     {"column": "COL_A1", "foreign_table": "TABLE_B", "foreign_column": "COL_B1"},
        #     ...
        #   ], ...
        # }
        processed_relations_count = 0
        for source_table_name, relations_list in table_relationships.items():
            if source_table_name not in graph:
                graph[source_table_name] = []
                logging.debug(f"Tabla '{source_table_name}' añadida al grafo desde table_relationships.")

            if not isinstance(relations_list, list):
                logging.warning(f"Se esperaba una lista de relaciones para la tabla '{source_table_name}', pero se encontró {type(relations_list)}. Se omite.")
                continue

            for rel_info in relations_list:
                if not isinstance(rel_info, dict):
                    logging.warning(f"Se esperaba un diccionario de información de relación para '{source_table_name}', pero se encontró {type(rel_info)}. Se omite: {rel_info}")
                    continue

                foreign_table_name = rel_info.get("foreign_table")
                from_col = rel_info.get("column")
                to_col = rel_info.get("foreign_column")
                rel_type = rel_info.get("type", "FK_JSON") # Tipo para indicar que vino del JSON

                if not all([foreign_table_name, from_col, to_col]):
                    logging.warning(f"Información de relación incompleta para '{source_table_name}'. Se omite: {rel_info}")
                    continue

                if foreign_table_name not in graph:
                    graph[foreign_table_name] = []
                    logging.debug(f"Tabla foránea '{foreign_table_name}' (referenciada por '{source_table_name}') añadida al grafo.")

                # Añadir relación source_table_name -> foreign_table_name
                graph[source_table_name].append({
                    "other": foreign_table_name,
                    "from_col": from_col,
                    "to_col": to_col,
                    "type": rel_type
                })
                
                # Añadir relación inversa foreign_table_name -> source_table_name
                graph[foreign_table_name].append({
                    "other": source_table_name,
                    "from_col": to_col, # Columna en la tabla referenciada (foreign_table_name)
                    "to_col": from_col, # Columna en la tabla actual (source_table_name)
                    "type": rel_type
                })
                processed_relations_count +=1
        
        logging.info(f"Grafo construido desde table_relationships (formato JSON) con {len(graph)} nodos y {processed_relations_count} relaciones bidireccionales procesadas.")
        return graph

    # Si no se proporcionan table_relationships, usar claves foráneas de la BD
    logging.info("Construyendo grafo de relaciones desde claves foráneas de la BD (fallback).")
    for table in list(graph.keys()): # Iterar sobre una copia de las claves si el grafo se modifica
        try:
            # Obtener las claves foráneas de la tabla usando PRAGMA
            # Asegúrate que db_connector.execute_sql devuelve una lista de diccionarios
            fk_list_raw = db_connector.execute_sql(f"PRAGMA foreign_key_list({table});")
            
            # Convertir fk_list si es necesario (ej. si execute_sql devuelve tuplas)
            fk_list = []
            if fk_list_raw and isinstance(fk_list_raw, list):
                if all(isinstance(item, dict) for item in fk_list_raw):
                    fk_list = fk_list_raw
                elif all(isinstance(item, tuple) for item in fk_list_raw):
                    # Asumir un orden de columnas para PRAGMA foreign_key_list:
                    # id, seq, table, from, to, on_update, on_delete, match
                    fk_columns = ['id', 'seq', 'table', 'from', 'to', 'on_update', 'on_delete', 'match']
                    for row_tuple in fk_list_raw:
                        if len(row_tuple) == len(fk_columns):
                           fk_list.append(dict(zip(fk_columns, row_tuple)))
                        else:
                            logging.warning(f"Fila de clave foránea con longitud inesperada para {table}: {row_tuple}")
                else:
                    logging.warning(f"Formato inesperado para fk_list de la tabla {table}: {fk_list_raw}")

            for fk in fk_list:
                # Añadir relación saliente
                # Asegurarse que fk es un diccionario y tiene las claves esperadas
                if isinstance(fk, dict) and "table" in fk and "from" in fk and "to" in fk:
                    # El valor de fk["table"] es la tabla referenciada (foreign_table)
                    foreign_table_name = fk["table"]
                    
                    # Asegurarse que la tabla referenciada existe en el grafo
                    if foreign_table_name not in graph:
                        graph[foreign_table_name] = []
                        
                    graph[table].append({
                        "other": foreign_table_name,
                        "from_col": fk["from"],
                        "to_col": fk["to"],
                        "type": "FK_DB" # Indicar que esta relación viene de la BD
                    })
                    
                    # Añadir relación inversa para facilitar la búsqueda bidireccional
                    graph[foreign_table_name].append({
                        "other": table,
                        "from_col": fk["to"], # Columna en la tabla referenciada
                        "to_col": fk["from"], # Columna en la tabla actual
                        "type": "FK_DB"
                    })
                else:
                    logging.warning(f"Entrada de clave foránea malformada o incompleta para la tabla {table}: {fk}")
        except Exception as e:
            logging.warning(f"Error al obtener claves foráneas para tabla {table}: {e}")
    
    logging.info(f"Grafo de relaciones construido con {len(graph)} tablas desde la BD.")
    return graph

def find_join_path(graph: Dict[str, List[Dict[str, Any]]], 
                  start: str, 
                  end: str, 
                  max_depth: int = 3) -> Optional[List[str]]:
    """
    Encuentra el camino más corto entre dos tablas usando BFS.
    
    Args:
        graph: Grafo de relaciones entre tablas
        start: Tabla de inicio
        end: Tabla destino
        max_depth: Profundidad máxima de búsqueda
        
    Returns:
        Lista de nombres de tablas que forman el camino o None si no existe
    """
    if start == end:
        return [start]
    
    if start not in graph or end not in graph:
        logging.warning(f"Tabla {start if start not in graph else end} no encontrada en el grafo")
        return None
    
    # Utilizar BFS para encontrar el camino más corto
    queue = deque([(start, [start])])  # (nodo actual, camino hasta ahora)
    visited = {start}
    
    while queue:
        node, path = queue.popleft()
        
        # Comprobación robusta: si el nodo no existe en el grafo, warning y salir
        if node not in graph:
            logging.warning(f"Nodo {node} no está en el grafo de relaciones. Camino parcial: {path}")
            return None
        
        # Limitar la profundidad del camino para evitar búsquedas excesivamente largas
        if len(path) > max_depth:
            continue
        
        # Explorar vecinos
        for edge in graph[node]:
            neighbor = edge["other"]
            
            if neighbor == end:
                # Encontramos el destino
                return path + [neighbor]
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    # No se encontró camino
    logging.info(f"No se encontró camino entre {start} y {end} con profundidad máxima {max_depth}")
    return None

def get_join_definition(graph: Dict[str, List[Dict[str, Any]]], 
                       source: str, 
                       target: str) -> Optional[Dict[str, str]]:
    """
    Obtiene la definición de JOIN entre dos tablas adyacentes.
    
    Args:
        graph: Grafo de relaciones
        source: Tabla origen
        target: Tabla destino
        
    Returns:
        Diccionario con la definición del JOIN o None si no hay conexión directa
    """
    if source not in graph:
        return None
    
    # Buscar una arista directa entre las tablas
    for edge in graph[source]:
        if edge["other"] == target:
            return {
                "table": source,
                "column": edge["from_col"],
                "foreign_table": target,
                "foreign_column": edge["to_col"]
            }
    
    return None

def generate_join_path(relationship_graph, path, config=None):
    """
    Genera una lista de definiciones de JOIN para un camino entre tablas
    
    Args:
        relationship_graph: Grafo de relaciones entre tablas
        path: Lista de tablas que forman el camino
        config: Configuración para LLM y otras opciones (opcional)
        
    Returns:
        Lista de definiciones de JOIN en formato diccionario
    """
    joins = []
    
    # Configuración por defecto si no se proporciona
    if config is None:
        config = {}
    
    for i in range(len(path)-1):
        source_table = path[i]
        target_table = path[i+1]
        
        # Encontrar la relación específica entre estas dos tablas en el grafo
        relation = get_join_definition(relationship_graph, source_table, target_table)
        
        if relation:
            # Si existe una relación definida, usarla
            joins.append(relation)
        else:
            # No hay relación explícita, consultar al LLM para inferirla si está habilitado
            if config.get('infer_joins_with_llm', False):
                # Usar configuración específica para el LLM
                llm_config = {
                    "temperature": config.get('temperature', 0.1),
                    "max_tokens": config.get('max_tokens', 150),
                    "llm_api_key": config.get('llm_api_key', ''),
                    "llm_model": config.get('llm_model', 'deepseek-chat')
                }
                
                join_info = infer_join_relation_with_llm(source_table, target_table, relationship_graph, llm_config)
                joins.append(join_info)
            else:
                # Usar inferencia basada en convenciones si LLM está deshabilitado
                join_info = infer_join_by_convention(source_table, target_table, relationship_graph)
                joins.append(join_info)
                
    return joins

def infer_join_relation_with_llm(source_table, target_table, relationship_graph, llm_config=None):
    """
    Utiliza el LLM para inferir la relación entre dos tablas
    
    Args:
        source_table: Tabla de origen
        target_table: Tabla de destino
        relationship_graph: Grafo de relaciones para contexto
        config: Configuración para el LLM (opcional)
        
    Returns:
        Diccionario con la definición del JOIN
    """
    # Obtener columnas de ambas tablas
    source_columns = get_columns_for_table(relationship_graph, source_table)
    target_columns = get_columns_for_table(relationship_graph, target_table)
    
    # Construir prompt para el LLM
    system_msg = (
        "Eres un experto en bases de datos. Analiza las tablas y columnas proporcionadas "
        "para determinar la relación correcta entre ellas. Responde SOLO con un JSON que incluya "
        "tabla origen, columna origen, tabla destino y columna destino para la relación. "
        "Utiliza EXACTAMENTE los mismos nombres de tablas y columnas proporcionados, respetando mayúsculas y minúsculas."
    )
    
    user_msg = (
        f"Necesito determinar la mejor relación para JOIN entre estas dos tablas:\n\n"
        f"Tabla origen: {source_table}\n"
        f"Columnas: {source_columns}\n\n"
        f"Tabla destino: {target_table}\n"
        f"Columnas: {target_columns}\n\n"
        f"Devuelve SOLO un JSON con esta estructura exacta (sin explicaciones adicionales):\n"
        f"{{\"table\": \"[TABLA_ORIGEN]\", \"column\": \"[COLUMNA_ORIGEN]\", "
        f"\"foreign_table\": \"[TABLA_DESTINO]\", \"foreign_column\": \"[COLUMNA_DESTINO]\"}}"
    )
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
    
    # Llamada al LLM
    try:
        from llm_utils import call_llm
        config_to_use = llm_config or {"temperature": 0.1, "max_tokens": 150}
        
        response = call_llm(messages, config_to_use, "JOIN Inference")
        
        # Extraer y validar el JSON
        import re
        import json
        
        json_match = re.search(r'{.*}', response)
        if json_match:
            join_info = json.loads(json_match.group(0))
            # Validar que el JSON tiene todos los campos necesarios
            if all(k in join_info for k in ["table", "column", "foreign_table", "foreign_column"]):
                # Normalizar los nombres de tablas y columnas para asegurar que coinciden con el esquema
                normalized_join_info = normalize_join_info(
                    join_info, 
                    source_table, 
                    target_table, 
                    source_columns, 
                    target_columns
                )
                return normalized_join_info
    except Exception as e:
        logging.warning(f"Error al inferir JOIN con LLM: {e}")
    
    # Fallback: inferencia basada en convenciones de nombres
    join_info = infer_join_by_convention(source_table, target_table, source_columns, target_columns)
    return join_info

def normalize_join_info(join_info, source_table, target_table, source_columns, target_columns):
    """
    Normaliza los nombres de tablas y columnas para que coincidan exactamente con el esquema.
    
    Args:
        join_info: Diccionario con información de JOIN devuelto por el LLM
        source_table: Nombre real de la tabla de origen
        target_table: Nombre real de la tabla de destino
        source_columns: Columnas reales de la tabla de origen
        target_columns: Columnas reales de la tabla de destino
        
    Returns:
        Diccionario con nombres normalizados
    """
    normalized = join_info.copy()
    
    # Normalizar nombres de tablas
    if normalized["table"].lower() == source_table.lower():
        normalized["table"] = source_table
    
    if normalized["foreign_table"].lower() == target_table.lower():
        normalized["foreign_table"] = target_table
    
    # Normalizar nombres de columnas en tabla origen
    column = normalized["column"]
    best_match = find_best_column_match(column, source_columns)
    if best_match:
        normalized["column"] = best_match
    
    # Normalizar nombres de columnas en tabla destino
    foreign_column = normalized["foreign_column"]
    best_match = find_best_column_match(foreign_column, target_columns)
    if best_match:
        normalized["foreign_column"] = best_match
    
    return normalized

def find_best_column_match(column_name, available_columns):
    """
    Encuentra la mejor coincidencia para un nombre de columna entre las columnas disponibles.
    
    Args:
        column_name: Nombre de columna a buscar
        available_columns: Lista de nombres de columnas disponibles
        
    Returns:
        Nombre de columna coincidente o None si no hay coincidencias
    """
    # Coincidencia exacta
    for col in available_columns:
        if col == column_name:
            return col
    
    # Coincidencia case-insensitive
    for col in available_columns:
        if col.lower() == column_name.lower():
            return col
    
    # Coincidencia parcial
    lower_name = column_name.lower()
    for col in available_columns:
        if lower_name in col.lower() or col.lower() in lower_name:
            return col
    
    return None

def infer_join_by_convention(source_table, target_table, source_columns, target_columns):
    """
    Infiere la relación entre tablas basándose en convenciones de nombres comunes
    
    Args:
        source_table: Tabla de origen
        target_table: Tabla de destino
        source_columns: Lista de columnas de la tabla origen
        target_columns: Lista de columnas de la tabla destino
        
    Returns:
        Diccionario con la definición del JOIN
    """
    # Buscar columnas que parezcan claves primarias/foráneas
    source_pk = None
    target_pk = None
    source_id = f"{source_table.lower()}_id"
    target_id = f"{target_table.lower()}_id"
    
    # Buscar IDs en columnas de origen
    for col in source_columns:
        col_lower = col.lower()
        if col_lower.endswith("_id") or col_lower == "id":
            source_pk = col
            break
    
    # Buscar en columnas de destino posibles FK que referencian a la tabla origen
    for col in target_columns:
        col_lower = col.lower()
        if col_lower == source_id:
            target_pk = col
            break
        # Buscar también columnas ID genéricas si no hay match específico
        elif (col_lower.endswith("_id") or col_lower == "id") and not target_pk:
            target_pk = col
    
    if source_pk and target_pk:
        return {
            "table": source_table,
            "column": source_pk,
            "foreign_table": target_table,
            "foreign_column": target_pk
        }
    
    # Si no encontramos una relación clara, usar convención común "id"
    return {
        "table": source_table,
        "column": "id",
        "foreign_table": target_table,
        "foreign_column": f"{source_table.lower()}_id"
    }

def get_columns_for_table(relationship_graph, table_name):
    """
    Extrae las columnas de una tabla a partir del grafo de relaciones
    
    Args:
        relationship_graph: Grafo de relaciones entre tablas
        table_name: Nombre de la tabla
        
    Returns:
        Lista de nombres de columnas de la tabla
    """
    columns = set()
    
    # Extraer columnas de las relaciones existentes
    if table_name in relationship_graph:
        for relation in relationship_graph[table_name]:
            if "from_col" in relation:
                columns.add(relation["from_col"])
            if "to_col" in relation:
                columns.add(relation["to_col"])
    
    # Si no se encontraron columnas, devolver lista vacía
    return list(filter(None, columns))

def direct_join_exists(joins: List[Dict[str, Any]], table1: str, table2: str) -> bool:
    """
    Verifica si ya existe un JOIN directo entre dos tablas.
    
    Args:
        joins: Lista de JOINs existentes
        table1: Primera tabla
        table2: Segunda tabla
        
    Returns:
        True si ya existe un JOIN entre las tablas
    """
    for join in joins:
        if isinstance(join, dict) and "table" in join and "foreign_table" in join:
            if {join["table"], join["foreign_table"]} == {table1, table2}:
                return True
    return False

def cache_relationship_graph(func):
    """Decorador para cachear el grafo de relaciones por conector de BD"""
    cache = {}
    
    def wrapper(db_connector, *args, **kwargs):
        conn_id = id(db_connector)
        if conn_id not in cache:
            cache[conn_id] = func(db_connector, *args, **kwargs)
        return cache[conn_id]
    
    return wrapper

# Aplicar el decorador de caché a build_relationship_graph
# build_relationship_graph = cache_relationship_graph(build_relationship_graph) # Comentado para resolver TypeError