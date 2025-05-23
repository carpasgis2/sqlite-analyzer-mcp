"""
Mejoras para el sistema RAG que usa embeddings y vectores para recuperación semántica
avanzada de información del esquema de base de datos.
"""

import os
import logging
import json
import numpy as np
import re
import sqlite3
from typing import List, Dict, Any, Tuple, Optional, Set, Union

# Verificar si tenemos disponibles las bibliotecas de vectorización
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS no está instalado. Se usará fallback a búsqueda por coincidencia.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers no está instalado. Se usará fallback a búsqueda por coincidencia.")

# Modelo de embeddings a utilizar si está disponible
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

class EnhancedSchemaRAG:
    """
    Sistema RAG mejorado con embeddings y vectorización para recuperar información 
    del esquema de la base de datos.
    """
    
    def __init__(self, db_connection: sqlite3.Connection = None, cache_path: str = "schema_rag_cache.json"):
        """
        Inicializa el sistema RAG con la conexión a la base de datos.
        
        Args:
            db_connection: Conexión SQLite a la base de datos
            cache_path: Ruta al archivo de caché para guardar/cargar información del esquema
        """
        self.db_connection = db_connection
        self.cache_path = cache_path
        self.schema_info = {
            "tables": {},
            "relationships": [],
            "table_descriptions": {},
            "term_mappings": {}
        }
        
        # Vectorización
        self.embedder = None
        self.table_embeddings = None
        self.column_embeddings = None
        self.table_names = []
        self.column_names = []
        self.index_tables = None
        self.index_columns = None
        
        # Inicializar el modelo de embeddings si está disponible
        self._initialize_embeddings()
        
        # Cargar o generar la información del esquema
        self._load_or_generate_schema()
    
    def _initialize_embeddings(self):
        """Inicializa el modelo de embeddings si las bibliotecas están disponibles."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedder = SentenceTransformer(EMBEDDING_MODEL)
                logging.info(f"Modelo de embeddings inicializado: {EMBEDDING_MODEL}")
            except Exception as e:
                logging.error(f"Error al inicializar el modelo de embeddings: {e}")
                self.embedder = None
    
    def _load_or_generate_schema(self):
        """Carga la información del esquema desde caché o la genera desde la BD."""
        loaded = False
        
        # Intentar cargar desde caché primero
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if "schema" in data:
                    self.schema_info = data["schema"]
                    # Generar vectores para los elementos cargados
                    self._generate_embeddings()
                    loaded = True
                    logging.info(f"Caché RAG cargada: {len(self.schema_info.get('tables', {}))} tablas")
            except Exception as e:
                logging.error(f"Error al cargar caché RAG: {e}")
        
        # Si no se pudo cargar desde caché y tenemos conexión a BD, generarla
        if not loaded and self.db_connection:
            try:
                self._extract_schema_from_db()
                self._generate_embeddings()
                self._save_schema_cache()
                logging.info(f"Esquema generado desde BD: {len(self.schema_info.get('tables', {}))} tablas")
            except Exception as e:
                logging.error(f"Error al generar esquema desde BD: {e}")
    
    def _extract_schema_from_db(self):
        """Extrae el esquema completo de la base de datos."""
        if not self.db_connection:
            logging.error("No hay conexión a base de datos para extraer esquema")
            return
        
        cursor = self.db_connection.cursor()
        
        # 1. Extraer tablas
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        # 2. Para cada tabla, extraer sus columnas
        for table in tables:
            cursor.execute(f"PRAGMA table_info('{table}')")
            columns = cursor.fetchall()
            
            # Almacenar información de la tabla
            self.schema_info["tables"][table] = {
                "columns": {},
                "description": f"Tabla {table} con datos sobre {table.lower()}",
                "aliases": [table.lower()]
            }
            
            # Procesar cada columna
            for col in columns:
                col_name = col[1]
                col_type = col[2]
                is_pk = col[5] == 1
                
                # Guardar información de la columna
                self.schema_info["tables"][table]["columns"][col_name] = {
                    "type": col_type,
                    "is_primary_key": is_pk,
                    "description": f"Columna {col_name} de tipo {col_type}" + (" (clave primaria)" if is_pk else "")
                }
                
                # Añadir aliases para columnas comunes
                if col_name.lower().endswith('_id'):
                    base_name = col_name[:-3]  # quitar "_id"
                    self.schema_info["tables"][table]["columns"][col_name]["aliases"] = [base_name]
        
        # 3. Extraer relaciones entre tablas (claves foráneas)
        for table in tables:
            cursor.execute(f"PRAGMA foreign_key_list('{table}')")
            fks = cursor.fetchall()
            
            for fk in fks:
                referenced_table = fk[2]
                from_col = fk[3]
                to_col = fk[4]
                
                # Añadir relación
                relation = {
                    "from_table": table,
                    "from_column": from_col,
                    "to_table": referenced_table,
                    "to_column": to_col
                }
                self.schema_info["relationships"].append(relation)
                
                # Actualizar descripción para reflejar la relación
                if "references" not in self.schema_info["tables"][table]["columns"][from_col]:
                    self.schema_info["tables"][table]["columns"][from_col]["references"] = []
                
                self.schema_info["tables"][table]["columns"][from_col]["references"].append({
                    "table": referenced_table,
                    "column": to_col
                })
                
                # Actualizar descripción de la columna
                current_desc = self.schema_info["tables"][table]["columns"][from_col]["description"]
                relation_desc = f" - Referencia a {referenced_table}.{to_col}"
                if relation_desc not in current_desc:
                    self.schema_info["tables"][table]["columns"][from_col]["description"] += relation_desc
    
    def _generate_embeddings(self):
        """Genera embeddings para tablas y columnas si el embedder está disponible."""
        if not self.embedder or not FAISS_AVAILABLE:
            logging.warning("No se pueden generar embeddings (falta embedder o FAISS)")
            return
        
        # Preparar textos y nombres para tablas
        table_texts = []
        self.table_names = []
        
        for table_name, table_info in self.schema_info.get("tables", {}).items():
            # Incluir nombre y descripción para el embedding
            description = table_info.get("description", "")
            text = f"{table_name}: {description}"
            
            # Añadir aliases si existen
            if "aliases" in table_info:
                text += f" Aliases: {', '.join(table_info['aliases'])}"
                
            table_texts.append(text)
            self.table_names.append(table_name)
        
        # Generar embeddings para tablas
        if table_texts:
            self.table_embeddings = self.embedder.encode(table_texts)
            
            # Crear índice FAISS para tablas
            dimension = self.table_embeddings.shape[1]
            self.index_tables = faiss.IndexFlatIP(dimension)  # Índice de producto interno (similitud coseno)
            self.index_tables.add(np.array(self.table_embeddings).astype('float32'))
            logging.info(f"Índice vectorial creado para {len(self.table_names)} tablas")
        
        # Preparar textos y nombres para columnas
        column_texts = []
        self.column_names = []
        
        for table_name, table_info in self.schema_info.get("tables", {}).items():
            for col_name, col_info in table_info.get("columns", {}).items():
                # Incluir nombre completo y descripción para el embedding
                full_col_name = f"{table_name}.{col_name}"
                description = col_info.get("description", "")
                text = f"{full_col_name}: {description}"
                
                # Añadir aliases si existen
                if "aliases" in col_info:
                    text += f" Aliases: {', '.join(col_info['aliases'])}"
                
                column_texts.append(text)
                self.column_names.append(full_col_name)
        
        # Generar embeddings para columnas
        if column_texts:
            self.column_embeddings = self.embedder.encode(column_texts)
            
            # Crear índice FAISS para columnas
            dimension = self.column_embeddings.shape[1]
            self.index_columns = faiss.IndexFlatIP(dimension)
            self.index_columns.add(np.array(self.column_embeddings).astype('float32'))
            logging.info(f"Índice vectorial creado para {len(self.column_names)} columnas")
    
    def _save_schema_cache(self):
        """Guarda la información del esquema en un archivo de caché."""
        try:
            # Verificar si ya existe un archivo de caché
            existing_data = {}
            if os.path.exists(self.cache_path):
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            
            # Actualizar o añadir la información del esquema
            existing_data["schema"] = self.schema_info
            
            # Guardar el archivo actualizado
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
                
            logging.info(f"Caché RAG guardada en {self.cache_path}")
        except Exception as e:
            logging.error(f"Error al guardar caché RAG: {e}")
    
    def find_matching_tables(self, query: str, top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Encuentra las tablas más relevantes para una consulta dada.
        
        Args:
            query: Consulta en lenguaje natural
            top_n: Número máximo de resultados a devolver
            
        Returns:
            Lista de tuplas (nombre_tabla, puntuación)
        """
        if (not hasattr(self, 'table_embeddings') or 
            self.table_embeddings is None or 
            len(self.table_embeddings) == 0 or 
            not hasattr(self, 'index_tables') or 
            self.index_tables is None):
            return self._fallback_table_search(query, top_n)
        
        try:
            # Generar embedding para la consulta
            query_embedding = self.embedder.encode([query])
            
            # Buscar las tablas más similares
            scores, indices = self.index_tables.search(
                np.array(query_embedding).astype('float32'), top_n
            )
            
            # Construir resultados
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx < len(self.table_names):
                    results.append((self.table_names[idx], float(score)))
            
            return results
        except Exception as e:
            logging.error(f"Error en búsqueda vectorial de tablas: {e}")
            return self._fallback_table_search(query, top_n)
    
    def find_matching_columns(self, query: str, tables: List[str] = None, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Encuentra las columnas más relevantes para una consulta dada.
        
        Args:
            query: Consulta en lenguaje natural
            tables: Lista opcional de tablas para filtrar resultados
            top_n: Número máximo de resultados a devolver
            
        Returns:
            Lista de tuplas (nombre_columna, puntuación)
        """
        if not self.column_embeddings or not self.index_columns:
            return self._fallback_column_search(query, tables, top_n)
        
        try:
            # Generar embedding para la consulta
            query_embedding = self.embedder.encode([query])
            
            # Buscar las columnas más similares
            scores, indices = self.index_columns.search(
                np.array(query_embedding).astype('float32'), top_n * 2  # Buscar más y filtrar después
            )
            
            # Construir resultados, filtrando por tablas si es necesario
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx < len(self.column_names):
                    col_full_name = self.column_names[idx]
                    table_name = col_full_name.split('.')[0]
                    
                    # Filtrar por tablas si se especifican
                    if tables and table_name not in tables:
                        continue
                    
                    results.append((col_full_name, float(score)))
                    
                    # Limitar al número solicitado después de filtrar
                    if len(results) >= top_n:
                        break
            
            return results
        except Exception as e:
            logging.error(f"Error en búsqueda vectorial de columnas: {e}")
            return self._fallback_column_search(query, tables, top_n)
    
    def _fallback_table_search(self, query: str, top_n: int) -> List[Tuple[str, float]]:
        """Método de fallback para buscar tablas cuando no hay vectorización."""
        results = []
        query_lower = query.lower()
        
        # Buscar coincidencia por palabras clave
        for table_name, table_info in self.schema_info.get("tables", {}).items():
            score = 0.0
            
            # Coincidencia exacta del nombre de tabla
            if table_name.lower() in query_lower:
                score += 0.8
            
            # Coincidencia en la descripción
            description = table_info.get("description", "").lower()
            for word in re.findall(r'\b\w+\b', query_lower):
                if len(word) > 3 and word in description:
                    score += 0.1
            
            # Coincidencia en aliases
            for alias in table_info.get("aliases", []):
                if alias.lower() in query_lower:
                    score += 0.5
            
            if score > 0:
                results.append((table_name, score))
        
        # Ordenar por puntuación y limitar resultados
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]
    
    def _fallback_column_search(self, query: str, tables: List[str], top_n: int) -> List[Tuple[str, float]]:
        """Método de fallback para buscar columnas cuando no hay vectorización."""
        results = []
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        # Recorrer todas las tablas y columnas
        for table_name, table_info in self.schema_info.get("tables", {}).items():
            # Filtrar por tablas si es necesario
            if tables and table_name not in tables:
                continue
                
            for col_name, col_info in table_info.get("columns", {}).items():
                score = 0.0
                full_col_name = f"{table_name}.{col_name}"
                
                # Coincidencia exacta en nombre de columna
                if col_name.lower() in query_lower:
                    score += 0.7
                
                # Coincidencia en la descripción
                description = col_info.get("description", "").lower()
                desc_words = set(re.findall(r'\b\w+\b', description))
                common_words = query_words.intersection(desc_words)
                score += len(common_words) * 0.1
                
                # Coincidencia en aliases
                for alias in col_info.get("aliases", []):
                    if alias.lower() in query_lower:
                        score += 0.4
                
                if score > 0:
                    results.append((full_col_name, score))
        
        # Ordenar por puntuación y limitar resultados
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]
    
    def get_schema_context(self, tables: List[str] = None) -> str:
        """
        Obtiene el contexto del esquema para las tablas especificadas.
        
        Args:
            tables: Lista de tablas para incluir en el contexto (None = todas)
            
        Returns:
            String con la información del esquema en formato legible
        """
        context = []
        
        # Si no se especifican tablas, encontrar las más relevantes
        if not tables and self.schema_info.get("tables"):
            tables = list(self.schema_info["tables"].keys())[:5]  # Limitar a 5 por defecto
        
        # Si no hay tablas, devolver mensaje informativo
        if not tables:
            return "No hay información de esquema disponible."
        
        # Generar contexto para cada tabla
        for table in tables:
            if table in self.schema_info.get("tables", {}):
                table_info = self.schema_info["tables"][table]
                
                # Descripción de la tabla
                table_desc = table_info.get("description", f"Tabla {table}")
                context.append(f"Tabla {table}: {table_desc}")
                
                # Columnas principales
                cols = []
                for col_name, col_info in table_info.get("columns", {}).items():
                    col_type = col_info.get("type", "")
                    is_pk = col_info.get("is_primary_key", False)
                    col_desc = f"{col_name} ({col_type})"
                    if is_pk:
                        col_desc += " [PK]"
                    cols.append(col_desc)
                
                if cols:
                    context.append("Columnas: " + ", ".join(cols))
        
        # Añadir información sobre relaciones
        related_tables = set(tables)
        relations = []
        
        for rel in self.schema_info.get("relationships", []):
            if rel["from_table"] in tables or rel["to_table"] in tables:
                relations.append(f"{rel['from_table']}.{rel['from_column']} -> {rel['to_table']}.{rel['to_column']}")
                # Añadir tablas relacionadas al conjunto
                related_tables.add(rel["from_table"])
                related_tables.add(rel["to_table"])
        
        if relations:
            context.append("Relaciones: " + "; ".join(relations))
        
        # Si hay tablas relacionadas no incluidas en el contexto original, añadirlas brevemente
        extra_tables = related_tables - set(tables)
        if extra_tables:
            extra_context = []
            for table in extra_tables:
                if table in self.schema_info.get("tables", {}):
                    table_desc = self.schema_info["tables"][table].get("description", f"Tabla {table}")
                    extra_context.append(f"Tabla {table}: {table_desc}")
            
            if extra_context:
                context.append("Tablas relacionadas: " + "; ".join(extra_context))
        
        return "\n".join(context)
    
    def enhance_query_info(self, query: str, structured_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mejora la información estructurada de una consulta con conocimiento del RAG.
        
        Args:
            query: Consulta en lenguaje natural
            structured_info: Información estructurada extraída de la consulta
            
        Returns:
            Información estructurada mejorada
        """
        enhanced_info = structured_info.copy()
        
        # Si no hay tablas detectadas, intentar encontrarlas
        if not enhanced_info.get('tables'):
            matching_tables = self.find_matching_tables(query, top_n=2)
            if matching_tables:
                enhanced_info['tables'] = [table for table, _ in matching_tables]
                logging.info(f"RAG encontró tablas relevantes: {matching_tables}")
        
        # Si hay tablas pero no hay columnas, sugerir columnas relevantes
        if enhanced_info.get('tables') and not enhanced_info.get('columns'):
            # Buscar columnas relevantes para la consulta
            matching_columns = self.find_matching_columns(
                query, tables=enhanced_info['tables'], top_n=3
            )
            
            if matching_columns:
                # Extraer solo los nombres de columna (sin tabla)
                enhanced_info['columns'] = [col_name.split('.')[1] for col_name, _ in matching_columns]
                logging.info(f"RAG sugirió columnas: {enhanced_info['columns']}")
        
        # Añadir información del esquema como contexto
        enhanced_info['schema_context'] = self.get_schema_context(enhanced_info.get('tables', []))
        
        return enhanced_info

def initialize_enhanced_rag(db_structure: Dict[str, Any], 
                          use_cache: bool = False,  # Cambia a False para forzar reconstrucción
                          force_rebuild: bool = True,  # Añade este parámetro
                          limit_tables: Optional[int] = None  # Opcional: para limitar si son muchas
                         ) -> EnhancedSchemaRAG:
    """
    Inicializa el sistema RAG mejorado para el esquema de base de datos
    """
    logging.info("Inicializando sistema RAG mejorado para esquema de base de datos...")
    
    # Si force_rebuild es True, ignorar caché existente
    if force_rebuild and os.path.exists("schema_rag_cache.json"):
        logging.info("Forzando reconstrucción del índice RAG, ignorando caché")
        os.rename("schema_rag_cache.json", "schema_rag_cache.json.bak")  # Hacer backup
