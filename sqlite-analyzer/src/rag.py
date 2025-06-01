import sqlite3
import numpy as np
import json
import os
import logging
import time
import re
# import requests # No se usa actualmente, se puede quitar si no se planea para llamadas LLM internas en RAG
from typing import Dict, List, Tuple, Any, Optional, Set
# from dotenv import load_dotenv # No se usa directamente aquí
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from tqdm import tqdm # tqdm se usaba en initialize_schema_knowledge, que es para el flujo de BD

class DatabaseSchemaRAG:
    """
    Sistema RAG (Retrieval-Augmented Generation) para consultas a bases de datos.
    Permite seleccionar dinámicamente partes relevantes del esquema y diccionario de términos.
    """
    def __init__(self, 
                 db_structure_dict: Optional[Dict] = None, 
                 terms_dict: Optional[Dict] = None,
                 embeddings_model_name: Optional[str] = None, # Para futura integración
                 conn: Optional[sqlite3.Connection] = None, 
                 cache_file: str = "schema_rag_cache.json",
                 logger_param: Optional[logging.Logger] = None):
        
        self.logger = logger_param if logger_param else logging.getLogger(__name__)
        if not logger_param and not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.conn = conn
        self.cache_file = cache_file
        self.embeddings_model_name = embeddings_model_name

        self.vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        
        self.table_corpus: List[str] = []
        self.table_embeddings: Any = None # Se espera que sea una matriz dispersa de scikit-learn
        self.table_names: List[str] = []
        
        self.column_corpus_map: Dict[str, List[str]] = {}
        self.column_embeddings_map: Dict[str, Any] = {}
        self.column_names_map: Dict[str, List[str]] = {}
        # self.column_vectorizers_map: Dict[str, TfidfVectorizer] = {} # Opción para gestionar vectorizadores de columnas

        self.terms_corpus: List[str] = []
        self.terms_embeddings: Any = None
        self.terms_keys: List[str] = []
        self.terms_vectorizer: Optional[TfidfVectorizer] = None


        self.schema_knowledge: Dict[str, Any] = {
            'tables': {},
            'columns': {}, 
            # 'terms': {}, # Originalmente para la estructura directa de terms_dict
            'terms_flat': {}, # Para el mapa aplanado de término: descripción
            'terms_original_structure': {}, # Para conservar la estructura original si es necesario
            'relationships': {} # Mantenido por si se carga de caché antiguo o se usa flujo de BD
        }

        if db_structure_dict and terms_dict:
            self.logger.info("Inicializando DatabaseSchemaRAG desde db_structure_dict y terms_dict.")
            self._initialize_knowledge_from_dicts(db_structure_dict, terms_dict)
        elif conn:
            self.logger.info("Inicializando DatabaseSchemaRAG desde conexión a BD y/o caché.")
            if not self.load_cache():
                self.logger.info(f"No se pudo cargar desde caché {self.cache_file} o no existe. Inicializando desde BD si está disponible.")
                # El método original initialize_schema_knowledge usaba tqdm y LLMs para descripciones.
                # Esto es un proceso largo. Para el flujo actual, nos centramos en la inicialización desde dicts.
                # Si se requiere inicialización desde BD, ese método necesitaría ser invocado explícitamente
                # y posiblemente adaptado. Por ahora, si no hay dicts y el caché falla, estará vacío.
                # self.initialize_schema_knowledge() # Comentado para evitar ejecución larga no deseada
                pass # Queda vacío si no hay dicts y el caché falla/no existe
            elif self.table_corpus or any(self.column_corpus_map.values()) or self.terms_corpus:
                 self._rebuild_embeddings()
        else:
            self.logger.warning("DatabaseSchemaRAG inicializado sin fuente de datos (ni dicts, ni conexión a BD, ni caché válido).")

    def _initialize_knowledge_from_dicts(self, db_structure: Dict, terms: Dict):
        self.logger.debug(f"Procesando db_structure: {json.dumps(db_structure, indent=2, ensure_ascii=False)[:200]}...")
        self.logger.debug(f"Procesando terms_dict: {json.dumps(terms, indent=2, ensure_ascii=False)[:200]}...")

        tables_to_process = []
        # Comprobar el formato de db_structure
        if 'tables' in db_structure and isinstance(db_structure['tables'], list):
            self.logger.info("Detectado formato db_structure con clave 'tables' como lista.")
            # Formato esperado: {'tables': [{'name': 'T1', ...}, {'name': 'T2', ...}]}
            for table_info_wrapper in db_structure['tables']:
                if isinstance(table_info_wrapper, dict) and 'name' in table_info_wrapper:
                    tables_to_process.append((table_info_wrapper['name'], table_info_wrapper))
                else:
                    self.logger.warning(f"Elemento en db_structure['tables'] no es un diccionario de tabla válido: {table_info_wrapper}")

        elif isinstance(db_structure, dict) and not any(key == 'tables' for key in db_structure.keys()):
            # Formato plano: {'TABLE_NAME_1': {'columns': [], ...}, 'TABLE_NAME_2': {...}}
            # Esto es lo que esperamos de schema_simple.json
            self.logger.info("Detectado formato db_structure plano (diccionario de tablas).")
            for table_name, table_data in db_structure.items():
                if isinstance(table_data, dict):
                    current_table_info = table_data.copy() # Evitar modificar el original
                    current_table_info['name'] = table_name # Asegurar que 'name' está para el procesamiento posterior
                    tables_to_process.append((table_name, current_table_info))
                else:
                    self.logger.warning(f"Valor para la tabla '{table_name}' no es un diccionario: {table_data}")
        else:
            self.logger.error(f"Formato db_structure_dict no reconocido o inesperado. Contenido: {json.dumps(db_structure, indent=2, ensure_ascii=False)[:500]}...")
            # No se procesarán tablas si el formato no es reconocido.

        if not tables_to_process:
            self.logger.warning("No se encontraron tablas para procesar en db_structure.")

        for table_name_key, table_info_dict in tables_to_process:
            table_name = table_info_dict.get('name', table_name_key)
            if not table_name:
                self.logger.warning(f"Tabla sin nombre encontrada (clave: {table_name_key}): {table_info_dict}")
                continue
            
            self.table_names.append(table_name)
            table_description = table_info_dict.get('description', f"Tabla {table_name}")
            display_name = table_info_dict.get('displayName', self._simplify_name(table_name))

            self.schema_knowledge['tables'][table_name] = {
                'name': table_name, 'display_name': display_name,
                'description': table_description, 'columns_count': len(table_info_dict.get('columns', [])),
                'raw_info': table_info_dict
            }
            
            self.column_names_map[table_name] = []
            self.schema_knowledge['columns'][table_name] = {}

            columns_list = table_info_dict.get('columns', [])
            if isinstance(columns_list, list):
                for item_in_columns_list in columns_list:
                    col_name = None
                    col_description = ""
                    col_type = "UNKNOWN"
                    col_raw_info = {}
                    
                    if isinstance(item_in_columns_list, dict):
                        col_name = item_in_columns_list.get('name')
                        col_description = item_in_columns_list.get('description', f"Columna {col_name} de {table_name}")
                        col_type = item_in_columns_list.get('type', 'UNKNOWN')
                        col_raw_info = item_in_columns_list
                    elif isinstance(item_in_columns_list, str):
                        col_name = item_in_columns_list
                        col_description = f"Columna {col_name} de la tabla {table_name}"
                        col_raw_info = {'name': col_name, 'type': col_type, 'description': col_description}
                    else:
                        self.logger.warning(f"Elemento inesperado en la lista de columnas para la tabla {table_name}: {item_in_columns_list}")
                        continue

                    if not col_name:
                        self.logger.warning(f"Columna sin nombre (o tipo no procesable) en tabla {table_name}: {item_in_columns_list}")
                        continue

                    self.column_names_map[table_name].append(col_name)
                    self.schema_knowledge['columns'][table_name][col_name] = {
                        'name': col_name,
                        'display_name': self._simplify_name(col_name),
                        'type': col_type,
                        'description': col_description,
                        'raw_info': col_raw_info
                    }
            else:
                self.logger.warning(f"La clave 'columns' para la tabla '{table_name}' no es una lista o no existe. Columnas no procesadas.")

        # --- MODIFIED: Lógica para Procesamiento de Términos ---
        self.logger.debug(f"Procesando terms_dict original: {json.dumps(terms, indent=2, ensure_ascii=False)[:500]}...")
        self.schema_knowledge['terms_original_structure'] = terms 
        
        flat_terms_map: Dict[str, str] = {}
        
        def _flatten_terms_recursive(current_item: Any, current_path: List[str]):
            if isinstance(current_item, dict):
                for key, value in current_item.items():
                    new_path = current_path + [key]
                    _flatten_terms_recursive(value, new_path)
            elif isinstance(current_item, str): # Llegamos a un valor de cadena (descripción)
                # Usar el último elemento de la ruta como la clave del término si es descriptivo,
                # o una combinación, o manejarlo según la estructura esperada de 'dictionary.json'.
                # Para 'dictionary.json', las claves de nivel superior son categorías,
                # y las claves internas son los términos con sus descripciones como valores.
                if len(current_path) > 0: # Asegurar que hay un camino
                    # Ejemplo: si path es ['PATIENT_RELATED_TERMS', 'PATI_FULL_NAME'] y value es "Nombre completo del paciente"
                    # Queremos que 'PATI_FULL_NAME' sea el término y value la descripción.
                    term_key = current_path[-1] # El término es la última parte de la ruta
                    
                    # Evitar sobrescribir si la misma clave de término aparece en múltiples lugares,
                    # a menos que se decida una estrategia (ej. concatenar, tomar la primera).
                    # Por ahora, la última encontrada sobrescribirá.
                    if term_key not in flat_terms_map:
                         flat_terms_map[term_key] = current_item
                    else:
                        self.logger.debug(f"Término duplicado encontrado al aplanar: '{term_key}'. Se mantiene la primera descripción encontrada.")
            # Se pueden manejar listas u otros tipos si es necesario
        
        if isinstance(terms, dict):
            _flatten_terms_recursive(terms, [])
            self.terms_keys = list(flat_terms_map.keys())
            self.schema_knowledge['terms_flat'] = flat_terms_map
            self.logger.info(f"Diccionario de términos aplanado. Número de términos individuales: {len(self.terms_keys)}.")
            if self.terms_keys:
                self.logger.debug(f"Algunas claves de términos aplanados: {self.terms_keys[:10]}")
                # Log de ejemplo de un término y su descripción
                # if 'PATI_FULL_NAME' in flat_terms_map:
                #    self.logger.debug(f"Ejemplo término aplanado: PATI_FULL_NAME -> {flat_terms_map['PATI_FULL_NAME']}")
        else:
            self.logger.warning("El diccionario de términos (terms_dict) no es un diccionario o está vacío. No se procesarán términos para RAG.")
            self.terms_keys = []
            self.schema_knowledge['terms_flat'] = {}
        
        self._create_corpus_and_embeddings_from_knowledge()
        self.logger.info(f"Conocimiento desde dicts: {len(self.table_names)} tablas, {sum(len(c) for c in self.column_names_map.values())} columnas procesadas, {len(self.terms_keys)} términos individuales.")

    def _create_corpus_and_embeddings_from_knowledge(self):
        self.logger.info("Creando corpus y embeddings desde self.schema_knowledge.")
        self.table_corpus = []
        _table_names_ordered = []
        for table_name in self.table_names:
            if table_name in self.schema_knowledge['tables']:
                meta = self.schema_knowledge['tables'][table_name]
                self.table_corpus.append(f"{table_name} {meta.get('display_name','')} {meta.get('description','')}")
                _table_names_ordered.append(table_name)
        self.table_names = _table_names_ordered

        if self.table_corpus:
            try:
                self.table_embeddings = self.vectorizer.fit_transform(self.table_corpus)
                self.logger.info(f"Embeddings de tablas creados. Forma: {self.table_embeddings.shape}")
            except Exception as e: self.logger.error(f"Error creando embeddings de tablas: {e}")

        self.column_corpus_map = {}
        self.column_embeddings_map = {}
        # self.column_vectorizers_map = {} # Si se decide almacenar los vectorizadores

        for table_name, col_names_list in self.column_names_map.items():
            _col_corpus, _col_names_ordered = [], []
            if table_name in self.schema_knowledge['columns']:
                for col_name in col_names_list:
                    if col_name in self.schema_knowledge['columns'][table_name]:
                        meta = self.schema_knowledge['columns'][table_name][col_name]
                        _col_corpus.append(f"{col_name} {meta.get('display_name','')} {meta.get('description','')} {meta.get('type','')}")
                        _col_names_ordered.append(col_name)
            
            self.column_corpus_map[table_name] = _col_corpus
            self.column_names_map[table_name] = _col_names_ordered # Reordenar por si acaso

            if _col_corpus:
                try:
                    # Usar un vectorizador dedicado para las columnas de esta tabla
                    # Esto es crucial para que `transform` funcione correctamente en `get_relevant_context`
                    # Si no se almacenan, se recrearán allí. Aquí solo creamos embeddings.
                    temp_col_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
                    self.column_embeddings_map[table_name] = temp_col_vectorizer.fit_transform(_col_corpus)
                    # self.column_vectorizers_map[table_name] = temp_col_vectorizer # Almacenar si se elige esta estrategia
                    self.logger.info(f"Embeddings de columnas para '{table_name}' creados. Forma: {self.column_embeddings_map[table_name].shape}")
                except Exception as e: self.logger.error(f"Error creando embeddings de columnas para {table_name}: {e}")

        # --- Lógica de Términos Modificada ---
        self.terms_corpus = []
        _terms_keys_ordered = [] # Para mantener el orden consistente con el corpus
        
        # Usar el mapa aplanado de términos
        flat_terms_dict = self.schema_knowledge.get('terms_flat', {})
        if flat_terms_dict and self.terms_keys: # self.terms_keys ya debería estar poblado desde _initialize_knowledge_from_dicts
            temp_corpus = []
            for term_key in self.terms_keys: # Iterar sobre las claves ya definidas y ordenadas (si es necesario)
                description = flat_terms_dict.get(term_key, "")
                if description: # Solo añadir si hay descripción
                    temp_corpus.append(f"{term_key} {description}")
                    _terms_keys_ordered.append(term_key) # Mantener el orden de los términos que realmente van al corpus
                else:
                    self.logger.debug(f"Término '{term_key}' omitido del corpus RAG por no tener descripción en terms_flat.")
            
            self.terms_corpus = temp_corpus
            self.terms_keys = _terms_keys_ordered # Actualizar self.terms_keys para que coincida con el corpus
        else:
            self.logger.info("No hay términos aplanados (terms_flat) o claves de términos para crear corpus de términos.")
            self.terms_corpus = []
            self.terms_keys = []

        if self.terms_corpus:
            try:
                # Asegurarse de que terms_vectorizer se inicializa si no lo está
                if self.terms_vectorizer is None:
                    self.terms_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
                
                self.terms_embeddings = self.terms_vectorizer.fit_transform(self.terms_corpus)
                self.logger.info(f"Embeddings de términos creados/actualizados. Forma: {self.terms_embeddings.shape}")
            except Exception as e:
                self.logger.error(f"Error creando embeddings de términos: {e}", exc_info=True)
        else:
            self.logger.info("Corpus de términos vacío, no se crearon embeddings de términos.")
            self.terms_embeddings = None # Asegurar que está None si no hay corpus

        self.logger.info("Corpus y embeddings creados/actualizados desde self.schema_knowledge (incluyendo términos aplanados).")

    def _rebuild_embeddings(self):
        self.logger.info("Reconstruyendo embeddings desde corpus almacenados...")
        if self.table_corpus: # No es necesario chequear self.vectorizer, se crea en __init__
            try:
                self.table_embeddings = self.vectorizer.fit_transform(self.table_corpus)
                self.logger.info(f"Embeddings de tablas reconstruidos. Forma: {self.table_embeddings.shape if self.table_embeddings is not None else 'None'}")
            except Exception as e: self.logger.error(f"Error reconstruyendo embeddings de tablas: {e}")
        
        # self.column_vectorizers_map = {} # Reiniciar si se almacenan
        for table, corpus in self.column_corpus_map.items():
            if corpus:
                try:
                    temp_col_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
                    self.column_embeddings_map[table] = temp_col_vectorizer.fit_transform(corpus)
                    # self.column_vectorizers_map[table] = temp_col_vectorizer # Almacenar
                    self.logger.info(f"Embeddings de columnas para '{table}' reconstruidos. Forma: {self.column_embeddings_map[table].shape if self.column_embeddings_map.get(table) is not None else 'None'}")
                except Exception as e: self.logger.error(f"Error reconstruyendo embeddings de columnas para {table}: {e}")

        if self.terms_corpus:
            try:
                self.terms_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
                self.terms_embeddings = self.terms_vectorizer.fit_transform(self.terms_corpus)
                self.logger.info(f"Embeddings de términos reconstruidos. Forma: {self.terms_embeddings.shape if self.terms_embeddings is not None else 'None'}")
            except Exception as e: self.logger.error(f"Error reconstruyendo embeddings de términos: {e}")
    
    def load_cache(self):
        self.logger.info(f"Intentando cargar RAG cache desde: {self.cache_file}")
        if not os.path.exists(self.cache_file):
            self.logger.info(f"Archivo de caché {self.cache_file} no encontrado.")
            return False
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            self.schema_knowledge = cached_data.get('schema_knowledge', self.schema_knowledge)
            self.table_corpus = cached_data.get('table_corpus', [])
            self.table_names = cached_data.get('table_names', [])
            self.column_corpus_map = cached_data.get('column_corpus_map', {})
            self.column_names_map = cached_data.get('column_names_map', {})
            self.terms_corpus = cached_data.get('terms_corpus', [])
            self.terms_keys = cached_data.get('terms_keys', [])
            
            if self.table_corpus or any(self.column_corpus_map.values()) or self.terms_corpus:
                self._rebuild_embeddings()
            
            self.logger.info(f"RAG cache cargado: {len(self.table_names)} tablas, {len(self.terms_keys)} términos.")
            return True
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"Error al cargar caché RAG desde {self.cache_file}: {e}")
        return False

    def save_cache(self):
        self.logger.info(f"Guardando RAG cache en: {self.cache_file}")
        try:
            cache_data = {
                'schema_knowledge': self.schema_knowledge,
                'table_corpus': self.table_corpus, 'table_names': self.table_names,
                'column_corpus_map': self.column_corpus_map, 'column_names_map': self.column_names_map,
                'terms_corpus': self.terms_corpus, 'terms_keys': self.terms_keys
            }
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"RAG cache guardado en {self.cache_file}")
            return True
        except IOError as e:
            self.logger.error(f"Error al guardar caché RAG en {self.cache_file}: {e}")
        return False

    def get_relevant_context(self, question: str, top_n_tables: int = 5, top_n_columns_per_table: int = 10, top_n_terms: int = 10) -> Tuple[str, str]:
        self.logger.info(f"Obteniendo contexto para pregunta: \"{question[:50]}...\" (top_n_tables={top_n_tables}, top_n_cols={top_n_columns_per_table}, top_n_terms={top_n_terms})")
        relevant_tables_data = []
        relevant_terms_data = {}

        # 1. Encontrar tablas relevantes
        if self.table_embeddings is not None and self.table_corpus and self.table_names:
            try:
                question_embedding = self.vectorizer.transform([question])
                table_similarities = cosine_similarity(question_embedding, self.table_embeddings).flatten()
                num_t = min(top_n_tables, len(self.table_names))
                relevant_table_indices = table_similarities.argsort()[-num_t:][::-1]
                
                self.logger.debug(f"Índices de tablas relevantes: {relevant_table_indices}")

                for i in relevant_table_indices:
                    if i >= len(self.table_names): continue # Salvaguarda
                    table_name = self.table_names[i]
                    table_info = self.schema_knowledge['tables'].get(table_name)
                    if not table_info: continue

                    current_table_data = {"name": table_name, "description": table_info.get('description',''), "columns": []}

                    # 2. Para cada tabla relevante, encontrar columnas relevantes
                    if table_name in self.column_embeddings_map and \
                       self.column_embeddings_map[table_name] is not None and \
                       table_name in self.column_corpus_map and self.column_corpus_map[table_name] and \
                       table_name in self.column_names_map and self.column_names_map[table_name]:
                        
                        # Recrear y ajustar vectorizador para columnas de esta tabla (solución temporal)
                        # Una mejor solución es almacenar/gestionar estos vectorizadores.
                        temp_column_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
                        temp_column_vectorizer.fit(self.column_corpus_map[table_name])
                        
                        question_col_embedding = temp_column_vectorizer.transform([question])
                        col_sim = cosine_similarity(question_col_embedding, self.column_embeddings_map[table_name]).flatten()
                        
                        num_c = min(top_n_columns_per_table, len(self.column_names_map[table_name]))
                        relevant_col_indices = col_sim.argsort()[-num_c:][::-1]
                        self.logger.debug(f"Tabla '{table_name}', índices de cols relevantes: {relevant_col_indices}")

                        for j in relevant_col_indices:
                            if j >= len(self.column_names_map[table_name]): continue # Salvaguarda
                            col_name = self.column_names_map[table_name][j]
                            col_info = self.schema_knowledge['columns'].get(table_name, {}).get(col_name)
                            if col_info:
                                current_table_data["columns"].append({
                                    "name": col_name, "type": col_info.get('type',''),
                                    "description": col_info.get('description','')
                                })
                    else: # Fallback: incluir todas las columnas de la tabla relevante si no hay embeddings/corpus de columnas
                        self.logger.debug(f"No hay embeddings/corpus de columnas para '{table_name}'. Incluyendo todas sus columnas.")
                        if table_name in self.schema_knowledge['columns']:
                            for col_name_fb, col_info_fb in self.schema_knowledge['columns'][table_name].items():
                                current_table_data["columns"].append({
                                    "name": col_name_fb, "type": col_info_fb.get('type',''),
                                    "description": col_info_fb.get('description','')
                                })
                    relevant_tables_data.append(current_table_data)
            except Exception as e_table_col:
                self.logger.error(f"Error buscando tablas/columnas relevantes: {e_table_col}", exc_info=True)
        else: self.logger.warning("No hay embeddings/corpus de tablas para búsqueda de relevancia.")

        # 3. Encontrar términos relevantes
        if self.terms_embeddings is not None and self.terms_corpus and self.terms_keys and self.terms_vectorizer:
            try:
                question_terms_embedding = self.terms_vectorizer.transform([question])
                terms_similarities = cosine_similarity(question_terms_embedding, self.terms_embeddings).flatten()
                
                # Asegurar que top_n_terms no excede el número de términos disponibles
                num_k = min(top_n_terms, len(self.terms_keys))
                if num_k > 0:
                    relevant_terms_indices = terms_similarities.argsort()[-num_k:][::-1]
                    self.logger.debug(f"Índices de términos relevantes (aplanados): {relevant_terms_indices}")

                    flat_terms_map = self.schema_knowledge.get('terms_flat', {})
                    # Limpiar relevant_terms_data antes de poblarlo
                    relevant_terms_data = {} 
                    for k_idx in relevant_terms_indices:
                        if k_idx >= len(self.terms_keys): 
                            self.logger.warning(f"Índice de término {k_idx} fuera de rango para terms_keys (longitud {len(self.terms_keys)}). Omitiendo.")
                            continue 
                        term_key = self.terms_keys[k_idx] # Clave de término individual del mapa aplanado
                        description = flat_terms_map.get(term_key, '')
                        if description: # Solo añadir si hay descripción
                             relevant_terms_data[term_key] = description
                        else:
                            self.logger.debug(f"Término '{term_key}' (índice {k_idx}) omitido del contexto RAG por no tener descripción en flat_terms_map.")
                else:
                    self.logger.info("No se seleccionarán términos relevantes (num_k=0 o no hay términos).")

            except Exception as e_terms:
                self.logger.error(f"Error buscando términos relevantes (aplanados): {e_terms}", exc_info=True)
        else:
            self.logger.warning("No hay embeddings/corpus de términos o vectorizador para búsqueda de relevancia de términos.")

        relevant_schema_for_prompt = {"tables": relevant_tables_data}
        self.logger.info(f"Contexto RAG final: {len(relevant_tables_data)} tablas, {len(relevant_terms_data)} términos individuales.")
        # Aumentar la longitud del log para el esquema
        self.logger.debug(f"Schema RAG (JSON): {json.dumps(relevant_schema_for_prompt, ensure_ascii=False, indent=2)[:500]}...")

        # Nuevos logs para depurar relevant_terms_data
        self.logger.debug(f"DEBUG: Tipo de relevant_terms_data antes de dumps: {type(relevant_terms_data)}")
        self.logger.debug(f"DEBUG: Longitud de relevant_terms_data antes de dumps: {len(relevant_terms_data)}")
        self.logger.debug(f"DEBUG: Contenido de relevant_terms_data (dict, primeros 500 chars): {str(relevant_terms_data)[:500]}")
        
        # Aumentar la longitud del log existente para los términos
        self.logger.debug(f"Términos RAG (JSON): {json.dumps(relevant_terms_data, ensure_ascii=False, indent=2)[:500]}...")

        return json.dumps(relevant_schema_for_prompt, ensure_ascii=False), json.dumps(relevant_terms_data, ensure_ascii=False)

    def _simplify_name(self, name: str) -> str:
        # Esta función es un placeholder, la original es más compleja y usa traducciones.
        # Se mantiene para compatibilidad si se llama internamente.
        simplified = re.sub(r'^[A-Z]{3,4}_', '', name)
        simplified = simplified.lower().replace('_', ' ')
        return simplified.strip()

    # Los métodos initialize_schema_knowledge, _generate_*, _fallback_*, _extract_relationships, _create_search_corpus
    # que dependen de una conexión a BD y/o llamadas a LLM para generar descripciones
    # se omiten en esta actualización ya que el flujo principal se centra en la inicialización desde dicts.
    # Si se necesitaran, deberían ser revisados y posiblemente adaptados.
    # Por ejemplo, initialize_schema_knowledge era muy extenso.

# Ejemplo de uso (opcional, para pruebas)
if __name__ == '__main__':
    # Configurar logger básico para pruebas
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger_main = logging.getLogger(__name__)

    # Datos de ejemplo
    sample_db_structure = {
        "database_name": "HospitalDB",
        "tables": [
            {
                "name": "PATI_PATIENTS",
                "description": "Almacena información sobre los pacientes del hospital.",
                "columns": [
                    {"name": "PATI_ID", "type": "INTEGER", "description": "Identificador único del paciente."},
                    {"name": "PATI_FULL_NAME", "type": "TEXT", "description": "Nombre completo del paciente."},
                    {"name": "PATI_BIRTH_DATE", "type": "DATE", "description": "Fecha de nacimiento del paciente."}
                ]
            },
            {
                "name": "APPO_APPOINTMENTS",
                "description": "Registra las citas médicas programadas.",
                "columns": [
                    {"name": "APPO_ID", "type": "INTEGER", "description": "Identificador único de la cita."},
                    {"name": "APPO_PATI_ID", "type": "INTEGER", "description": "ID del paciente asociado a la cita (FK a PATI_PATIENTS)."},
                    {"name": "APPO_DATE", "type": "DATETIME", "description": "Fecha y hora de la cita."},
                    {"name": "APPO_REASON", "type": "TEXT", "description": "Motivo de la cita."}
                ]
            }
        ]
    }
    sample_terms_dict = {
        "paciente": "Persona que recibe atención médica.",
        "cita": "Reserva de hora para una consulta médica.",
        "nombre completo": "Nombre y apellidos de una persona.",
        "fecha de nacimiento": "Día en que nació una persona."
    }

    logger_main.info("Creando instancia de DatabaseSchemaRAG con datos de ejemplo...")
    rag_system = DatabaseSchemaRAG(db_structure_dict=sample_db_structure, 
                                   terms_dict=sample_terms_dict, 
                                   logger_param=logger_main)

    test_question = "Quiero ver el nombre de los pacientes y la razón de sus citas"
    logger_main.info(f"Obteniendo contexto relevante para la pregunta: \"{test_question}\"")
    
    relevant_schema, relevant_terms = rag_system.get_relevant_context(test_question)
    
    logger_main.info("--- Esquema Relevante ---")
    logger_main.info(json.dumps(json.loads(relevant_schema), indent=2, ensure_ascii=False)) # Cargar y volcar para pretty print
    logger_main.info("--- Términos Relevantes ---")
    logger_main.info(json.dumps(json.loads(relevant_terms), indent=2, ensure_ascii=False))

    test_question_2 = "fecha de nacimiento de un paciente"
    logger_main.info(f"Obteniendo contexto relevante para la pregunta: \"{test_question_2}\"")
    relevant_schema_2, relevant_terms_2 = rag_system.get_relevant_context(test_question_2)
    logger_main.info("--- Esquema Relevante 2 ---")
    logger_main.info(json.dumps(json.loads(relevant_schema_2), indent=2, ensure_ascii=False))
    logger_main.info("--- Términos Relevantes 2 ---")
    logger_main.info(json.dumps(json.loads(relevant_terms_2), indent=2, ensure_ascii=False))

    # Prueba de guardado y carga de caché
    cache_file_test = "test_rag_cache.json"
    rag_system_for_cache = DatabaseSchemaRAG(db_structure_dict=sample_db_structure, 
                                             terms_dict=sample_terms_dict, 
                                             cache_file=cache_file_test,
                                             logger_param=logger_main)
    rag_system_for_cache.save_cache()
    
    rag_system_loaded = DatabaseSchemaRAG(cache_file=cache_file_test, logger_param=logger_main)
    logger_main.info(f"Sistema cargado desde caché. Tablas: {len(rag_system_loaded.table_names)}, Términos: {len(rag_system_loaded.terms_keys)}")
    relevant_schema_3, relevant_terms_3 = rag_system_loaded.get_relevant_context(test_question)
    logger_main.info("--- Esquema Relevante 3 (desde caché) ---")
    logger_main.info(json.dumps(json.loads(relevant_schema_3), indent=2, ensure_ascii=False))
    logger_main.info("--- Términos Relevantes 3 (desde caché) ---")
    logger_main.info(json.dumps(json.loads(relevant_terms_3), indent=2, ensure_ascii=False))
    if os.path.exists(cache_file_test):
        os.remove(cache_file_test)
        logger_main.info(f"Archivo de caché de prueba {cache_file_test} eliminado.")