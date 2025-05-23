import sqlite3
import numpy as np
import json
import os
import logging
import time
import re
import requests
from typing import Dict, List, Tuple, Any, Optional, Set
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm  # Importar tqdm para barra de progreso

class DatabaseSchemaRAG:
    """
    Sistema RAG (Retrieval-Augmented Generation) para consultas a bases de datos.
    Permite traducir automáticamente términos de lenguaje natural a elementos de SQL.
    """
    def __init__(self, conn: sqlite3.Connection = None, cache_file: str = "schema_rag_cache.json"):
        """
        Inicializa el sistema RAG para esquema de base de datos.
        
        Args:
            conn: Conexión SQLite a la base de datos
            cache_file: Archivo para almacenar/cargar el cache de descripciones
        """
        self.conn = conn
        self.cache_file = cache_file
        
        # Vectorizador para búsqueda semántica
        self.vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        
        # Almacenamiento de embeddings
        self.table_corpus = []
        self.table_embeddings = None
        self.table_names = []
        
        self.column_corpus = {}
        self.column_embeddings = {}
        self.column_names = {}
        
        # Conocimiento del esquema
        self.schema_knowledge = {
            'tables': {},        # Información y descripciones de tablas
            'columns': {},       # Información y descripciones de columnas
            'relationships': {}, # Relaciones entre tablas
            'synonyms': {},      # Diccionario de sinónimos
            'examples': {}       # Ejemplos de uso para referencia
        }
        
        # Cargar conocimiento desde caché si existe
        self.load_cache()
        
        # Si tenemos conexión a la BD, actualizar el conocimiento
        if conn:
            self.initialize_schema_knowledge()
    
    def load_cache(self):
        """Carga el conocimiento del esquema desde el archivo de caché"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                self.schema_knowledge = cached_data.get('schema_knowledge', self.schema_knowledge)
                self.table_corpus = cached_data.get('table_corpus', [])
                self.table_names = cached_data.get('table_names', [])
                
                for table, corpus in cached_data.get('column_corpus', {}).items():
                    self.column_corpus[table] = corpus
                    self.column_names[table] = cached_data.get('column_names', {}).get(table, [])
                
                # Reconstruir embeddings si tenemos corpus
                if self.table_corpus:
                    self._rebuild_embeddings()
                
                logging.info(f"RAG cache cargado: {len(self.table_names)} tablas")
                return True
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Error al cargar caché RAG: {e}")
        
        return False
    
    def save_cache(self):
        """Guarda el conocimiento del esquema en el archivo de caché"""
        try:
            cache_data = {
                'schema_knowledge': self.schema_knowledge,
                'table_corpus': self.table_corpus,
                'table_names': self.table_names,
                'column_corpus': self.column_corpus,
                'column_names': self.column_names
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            logging.info(f"RAG cache guardado en {self.cache_file}")
            return True
        except IOError as e:
            logging.error(f"Error al guardar caché RAG: {e}")
            return False
    
    def _rebuild_embeddings(self):
        """Reconstruye los embeddings a partir de los corpus almacenados"""
        # Reconstruir embeddings de tablas
        if self.table_corpus:
            self.table_embeddings = self.vectorizer.fit_transform(self.table_corpus)
        
        # Reconstruir embeddings de columnas para cada tabla
        for table, corpus in self.column_corpus.items():
            if corpus:
                table_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
                self.column_embeddings[table] = table_vectorizer.fit_transform(corpus)
    
    def initialize_schema_knowledge(self):
        """Inicializa el conocimiento del esquema extrayendo información de la BD"""
        try:
            # 1. Extraer lista de tablas
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            self.table_names = tables
            
            logging.info(f"Extrayendo información de {len(tables)} tablas...")
            
            # 2. Para cada tabla, extraer definiciones y generar descripciones
            # Usar tqdm para mostrar progreso
            for i, table in enumerate(tqdm(tables, desc="Procesando tablas", ncols=100)):
                # Mostrar progreso periódicamente en logs
                if i % 10 == 0 or i == len(tables) - 1:
                    logging.debug(f"Progreso: {i+1}/{len(tables)} tablas ({(i+1)/len(tables)*100:.1f}%)")
                
                # Comprobar si ya tenemos esta tabla en caché
                if table in self.schema_knowledge['tables']:
                    logging.debug(f"Tabla {table} ya en caché, omitiendo...")
                    continue
                
                # Obtener columnas y tipos
                cursor.execute(f"PRAGMA table_info({table})")
                columns_info = cursor.fetchall()
                
                # Mostrar información de la tabla actual
                logging.debug(f"Procesando tabla: {table} con {len(columns_info)} columnas")
                
                # Extraer información semántica con LLM para la tabla
                table_description = self._generate_table_description(table, columns_info)
                
                # Guardar en conocimiento del esquema
                self.schema_knowledge['tables'][table] = {
                    'name': table,
                    'display_name': self._simplify_name(table),
                    'columns_count': len(columns_info),
                    'description': table_description,
                    'common_uses': self._generate_table_use_cases(table, columns_info)
                }
                
                # Inicializar lista de columnas para esta tabla
                self.column_names[table] = []
                
                # Procesar cada columna
                for col_info in columns_info:
                    col_name = col_info[1]
                    col_type = col_info[2]
                    is_pk = col_info[5] == 1
                    
                    self.column_names[table].append(col_name)
                    
                    # Generar descripción para la columna usando LLM
                    col_description = self._generate_column_description(table, col_name, col_type, is_pk)
                    
                    # Generar sinónimos para la columna
                    synonyms = self._generate_column_synonyms(table, col_name)
                    
                    # Guardar metadata de columna
                    if table not in self.schema_knowledge['columns']:
                        self.schema_knowledge['columns'][table] = {}
                    
                    self.schema_knowledge['columns'][table][col_name] = {
                        'name': col_name,
                        'display_name': self._simplify_name(col_name),
                        'type': col_type,
                        'is_primary_key': is_pk,
                        'description': col_description,
                        'synonyms': synonyms
                    }
            
            # 3. Extraer relaciones entre tablas
            logging.info("Extrayendo relaciones entre tablas...")
            self._extract_relationships()
            
            # 4. Crear corpus para búsqueda de similitud
            logging.info("Creando corpus para búsqueda semántica...")
            self._create_search_corpus()
            
            # 5. Guardar cache
            self.save_cache()
            
            logging.info(f"Conocimiento de esquema inicializado con {len(tables)} tablas")
            return True
        
        except sqlite3.Error as e:
            logging.error(f"Error al extraer esquema de BD: {e}")
            return False
    
    def _create_search_corpus(self):
        """Crea corpus de texto para búsqueda de similitud"""
        # Crear corpus para tablas
        self.table_corpus = []
        
        for table in self.table_names:
            if table in self.schema_knowledge['tables']:
                metadata = self.schema_knowledge['tables'][table]
                
                # Combinar nombre, nombre simplificado y descripción para mejor búsqueda
                table_text = f"{table} {metadata['display_name']} {metadata['description']}"
                
                # Añadir usos comunes si existen
                if 'common_uses' in metadata:
                    table_text += f" {metadata['common_uses']}"
                
                self.table_corpus.append(table_text)
        
        # Crear embeddings de tablas
        if self.table_corpus:
            self.table_embeddings = self.vectorizer.fit_transform(self.table_corpus)
        
        # Crear corpus para columnas de cada tabla
        for table in self.table_names:
            if table not in self.schema_knowledge['columns']:
                continue
            
            self.column_corpus[table] = []
            
            for col_name in self.column_names[table]:
                if col_name in self.schema_knowledge['columns'][table]:
                    col_metadata = self.schema_knowledge['columns'][table][col_name]
                    
                    # Combinar nombre, nombre simplificado, descripción y sinónimos
                    col_text = f"{col_name} {col_metadata['display_name']} {col_metadata['description']}"
                    
                    # Añadir sinónimos
                    if 'synonyms' in col_metadata:
                        col_text += " " + " ".join(col_metadata['synonyms'])
                    
                    self.column_corpus[table].append(col_text)
            
            # Crear embeddings para columnas de esta tabla
            if self.column_corpus[table]:
                table_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
                self.column_embeddings[table] = table_vectorizer.fit_transform(self.column_corpus[table])
    
    def _simplify_name(self, name: str) -> str:
        """Simplifica un nombre técnico a un término más natural y lo traduce al español si es necesario"""
        # Quitar prefijos de tabla/columna (ej: PATI_, APPO_)
        simplified = re.sub(r'^[A-Z]{3,4}_', '', name)
        
        # Convertir snake_case a palabras separadas
        simplified = simplified.lower().replace('_', ' ')
        
        # Traducir términos comunes del inglés al español
        translations = {
            'vehicle types': 'tipos de vehículos',
            'patient': 'paciente',
            'patients': 'pacientes',
            'appointment': 'cita',
            'appointments': 'citas',
            'doctor': 'médico',
            'doctors': 'médicos',
            'name': 'nombre',
            'date': 'fecha',
            'time': 'hora',
            'status': 'estado',
            'type': 'tipo',
            'types': 'tipos',
            'id': 'identificador',
            'description': 'descripción',
            'address': 'dirección',
            'phone': 'teléfono',
            'email': 'correo',
        }
        
        # Aplicar traducciones
        for eng, esp in translations.items():
            if simplified == eng or simplified.startswith(eng + " ") or simplified.endswith(" " + eng) or f" {eng} " in f" {simplified} ":
                simplified = simplified.replace(eng, esp)
        
        return simplified
    
    def _extract_relationships(self):
        """Extrae relaciones entre tablas basadas en claves foráneas"""
        try:
            if not self.conn:
                return
                
            cursor = self.conn.cursor()
            
            for table in self.table_names:
                # Consultar claves foráneas
                cursor.execute(f"PRAGMA foreign_key_list({table})")
                fks = cursor.fetchall()
                
                # Guardar relaciones
                if table not in self.schema_knowledge['relationships']:
                    self.schema_knowledge['relationships'][table] = []
                
                for fk in fks:
                    ref_table = fk[2]  # Tabla referenciada
                    from_col = fk[3]   # Columna en esta tabla
                    to_col = fk[4]     # Columna en tabla referenciada
                    
                    # Generar descripción de la relación
                    relationship = {
                        'from_table': table,
                        'to_table': ref_table,
                        'from_column': from_col,
                        'to_column': to_col,
                        'description': f"La tabla {self._simplify_name(table)} está relacionada con {self._simplify_name(ref_table)} mediante {from_col} -> {to_col}"
                    }
                    
                    self.schema_knowledge['relationships'][table].append(relationship)
        
        except sqlite3.Error as e:
            logging.error(f"Error al extraer relaciones: {e}")
    
    def _generate_table_description(self, table_name: str, columns_info: List) -> str:
        """
        Genera una descripción en lenguaje natural para una tabla usando LLM.
        Si falla, genera una descripción básica.
        """
        try:
            # Preparar información para enviar al LLM
            column_details = []
            for col in columns_info:
                col_name = col[1]
                col_type = col[2]
                is_pk = "PRIMARY KEY" if col[5] == 1 else ""
                is_nullable = "NULL" if col[3] == 0 else "NOT NULL"
                column_details.append(f"{col_name} ({col_type} {is_pk} {is_nullable})")
            
            # Llamar al LLM para generar descripción
            simple_name = self._simplify_name(table_name)
            prompt = f"""
            Genera una descripción clara y concisa en español para la tabla de base de datos '{table_name}' con nombre simplificado '{simple_name}'.
            La tabla tiene estas columnas:
            {', '.join(column_details)}
            
            Describe el propósito de esta tabla en un sistema médico. La descripción debe ser breve (máximo 2 frases)
            y explicar qué tipo de datos almacena esta tabla.
            """
            
            description = self._call_llm_for_description(prompt)
            if description:
                return description
            
        except Exception as e:
            logging.error(f"Error al generar descripción para tabla {table_name}: {e}")
        
        # Si falla, generar descripción básica por reglas
        return self._fallback_table_description(table_name, columns_info)
    
    def _fallback_table_description(self, table_name: str, columns_info: List) -> str:
        """Genera una descripción básica para una tabla basada en reglas"""
        simple_name = self._simplify_name(table_name)
        
        # Detectar el propósito por prefijo
        purpose = ""
        if "PATI" in table_name or "PATIENT" in table_name:
            purpose = "almacena información sobre pacientes"
        elif "APPO" in table_name or "APPOINTMENT" in table_name:
            purpose = "registra citas médicas"
        elif "ONCO" in table_name or "CANCER" in table_name:
            purpose = "contiene datos sobre casos de cáncer"
        elif "EPIS" in table_name or "EPISODE" in table_name:
            purpose = "registra episodios médicos"
        elif "PARA" in table_name:
            purpose = "contiene parámetros o catálogos del sistema"
        elif "HIST" in table_name:
            purpose = "almacena historial médico"
        elif "DOC" in table_name:
            purpose = "guarda documentos o archivos"
        else:
            purpose = "almacena registros relacionados con el sistema médico"
        
        return f"La tabla {simple_name} {purpose}."
    
    def _generate_column_description(self, table: str, column: str, data_type: str, is_pk: bool = False) -> str:
        """
        Genera descripción en lenguaje natural para una columna usando LLM.
        Si falla, genera una descripción básica.
        """
        try:
            # Simplificar nombres
            simple_table = self._simplify_name(table)
            simple_column = self._simplify_name(column)
            
            # Preparar prompt para LLM
            prompt = f"""
            Genera una descripción concisa en español para la columna '{column}' de la tabla '{table}' 
            con nombre simplificado '{simple_column}' en la tabla '{simple_table}'.
            
            La columna es de tipo {data_type}.
            {' Es clave primaria de la tabla.' if is_pk else ''}
            
            Describe en una frase corta qué información almacena esta columna.
            """
            
            description = self._call_llm_for_description(prompt)
            if description:
                return description
                
        except Exception as e:
            logging.error(f"Error al generar descripción para columna {table}.{column}: {e}")
        
        # Si falla, generar descripción básica
        return self._fallback_column_description(table, column, data_type, is_pk)
    
    def _fallback_column_description(self, table: str, column: str, data_type: str, is_pk: bool) -> str:
        """Genera descripción básica para una columna basada en reglas"""
        # Simplificar nombres
        simple_column = self._simplify_name(column)
        
        # Detectar tipo de columna según nombre y tipo de datos
        column_role = ""
        if is_pk:
            column_role = "identificador único de cada registro"
        elif column.endswith("_ID"):
            if table.split("_")[0] in column:
                column_role = "identificador único del registro"
            else:
                # Es probablemente una clave foránea
                ref_table = re.sub(r'_ID$', '', column)
                column_role = f"referencia a la tabla {ref_table}"
        elif "NAME" in column or "NOMBRE" in column:
            column_role = "nombre o título"
        elif "DESC" in column:
            column_role = "descripción textual"
        elif "DATE" in column or "FECHA" in column:
            column_role = "fecha"
        elif "TIME" in column or "HORA" in column:
            column_role = "hora o momento"
        elif "STATUS" in column or "ESTADO" in column:
            column_role = "estado o situación"
        elif "TYPE" in column or "TIPO" in column:
            column_role = "tipo o categoría"
        else:
            # Inferir por tipo de datos
            if "INT" in data_type:
                column_role = "valor numérico"
            elif "CHAR" in data_type or "TEXT" in data_type:
                column_role = "valor textual"
            elif "DATE" in data_type:
                column_role = "fecha"
            elif "BOOL" in data_type:
                column_role = "indicador sí/no"
            else:
                column_role = "atributo"
        
        return f"Almacena el {column_role} en formato {data_type}."
    
    def _generate_table_use_cases(self, table: str, columns_info: List) -> str:
        """Genera casos de uso comunes para una tabla"""
        try:
            # Preparar prompt para LLM
            simple_table = self._simplify_name(table)
            
            # Extraer nombres de columnas para enviar al LLM
            column_names = [col[1] for col in columns_info]
            
            prompt = f"""
            Genera 2-3 ejemplos de consultas en lenguaje natural que alguien podría hacer sobre la tabla '{table}' 
            (nombre simplificado: '{simple_table}') con columnas: {', '.join(column_names)}.
            
            Las consultas deben ser preguntas que un usuario podría hacer en español sobre esta tabla.
            Sé muy breve, solo lista las preguntas separadas por punto y coma.
            """
            
            use_cases = self._call_llm_for_description(prompt)
            if use_cases:
                return use_cases
                
        except Exception as e:
            logging.error(f"Error al generar casos de uso para tabla {table}: {e}")
        
        # Fallback básico
        return f"Consultas sobre {self._simplify_name(table)}"
    
    def _generate_column_synonyms(self, table: str, column: str) -> List[str]:
        """Genera sinónimos para una columna usando LLM"""
        try:
            # Simplificar nombres
            simple_table = self._simplify_name(table)
            simple_column = self._simplify_name(column)
            
            # Preparar prompt para LLM
            prompt = f"""
            Genera 3-5 sinónimos o términos alternativos en español que un usuario podría usar para referirse 
            a la columna '{column}' (nombre simplificado: '{simple_column}') de la tabla '{table}' 
            (nombre simplificado: '{simple_table}').
            
            Devuelve solo los términos separados por comas, sin explicaciones.
            """
            
            synonyms_text = self._call_llm_for_description(prompt)
            if synonyms_text:
                # Limpiar y convertir a lista
                synonyms = [s.strip() for s in synonyms_text.split(',')]
                return [s for s in synonyms if s]  # Filtrar vacíos
            
        except Exception as e:
            logging.error(f"Error al generar sinónimos para {table}.{column}: {e}")
        
        # Fallback: crear sinónimos básicos
        return self._fallback_column_synonyms(column)
    
    def _fallback_column_synonyms(self, column: str) -> List[str]:
        """Genera sinónimos básicos para una columna basado en reglas"""
        synonyms = []
        simple = self._simplify_name(column)
        synonyms.append(simple)
        
        # Añadir variantes
        if "_ID" in column:
            synonyms.append(column.replace("_ID", ""))
            synonyms.append("id " + simple.replace("id", "").strip())
            
        if "NAME" in column:
            synonyms.append(column.replace("NAME", ""))
            synonyms.append("nombre")
            
        if "DATE" in column:
            synonyms.append("fecha")
            
        if "STATUS" in column:
            synonyms.append("estado")
        
        return list(set(synonyms))  # Eliminar duplicados
    
    def _guess_referenced_table(self, column_name: str) -> str:
        """Intenta adivinar la tabla referenciada por una columna que parece FK"""
        if not column_name.endswith('_ID'):
            return ""
            
        # Extraer el prefijo antes de _ID
        prefix = column_name[:-3]
        
        # Buscar tabla que comience con ese prefijo
        for table in self.table_names:
            if table.startswith(prefix):
                return table
                
        return ""
    
    def _call_llm_for_description(self, prompt: str) -> str:
        """
        Llama al LLM para generar una descripción.
        Utiliza las credenciales de Deepseek desde variables de entorno.
        """
        # Asegurar que las variables de entorno están cargadas
        load_dotenv()
        
        # Obtener credenciales
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        api_url = os.environ.get("DEEPSEEK_API_URL", 
                  "https://api.deepseek.com/v1/chat/completions")
        model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        
        if not api_key:
            logging.error("No se encontró API key para LLM en variables de entorno")
            return ""
        
        # Preparar request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Eres un experto en bases de datos que genera descripciones concisas y claras."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 150
        }
        
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=15)
            response.raise_for_status()
            
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                # Limpiar la respuesta
                return content.strip()
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error al llamar al LLM: {e}")
        except Exception as e:
            logging.error(f"Error inesperado con LLM: {e}")
            
        return ""
            
    def find_matching_table(self, query: str, top_n: int = 1) -> List[Tuple[str, float]]:
        """
        Encuentra las tablas más similares a la consulta en lenguaje natural.
        
        Args:
            query: Consulta en lenguaje natural
            top_n: Número de resultados a devolver
            
        Returns:
            Lista de tuplas (nombre_tabla, puntuación_similitud)
        """
        if not self.table_embeddings is not None or not self.table_corpus:
            logging.warning("No hay embeddings de tablas disponibles")
            return []
            
        # Vectorizar la consulta
        query_vector = self.vectorizer.transform([query])
        
        # Calcular similitud con todas las tablas
        similarities = cosine_similarity(query_vector, self.table_embeddings).flatten()
        
        # Obtener los índices de las mejores coincidencias
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        # Crear lista de resultados (tabla, puntuación)
        results = [(self.table_names[i], float(similarities[i])) for i in top_indices]
        
        return results
    
    def find_matching_columns(self, query: str, table: str, top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Encuentra las columnas más similares a la consulta para una tabla específica.
        
        Args:
            query: Consulta en lenguaje natural
            table: Nombre de la tabla donde buscar columnas
            top_n: Número de resultados a devolver
            
        Returns:
            Lista de tuplas (nombre_columna, puntuación_similitud)
        """
        if table not in self.column_embeddings or table not in self.column_corpus:
            logging.warning(f"No hay embeddings de columnas para tabla {table}")
            return []
            
        # Crear vectorizador específico para esta tabla
        table_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        table_vectorizer.fit(self.column_corpus[table])
        
        # Vectorizar la consulta
        query_vector = table_vectorizer.transform([query])
        
        # Calcular similitud con todas las columnas
        similarities = cosine_similarity(query_vector, self.column_embeddings[table]).flatten()
        
        # Obtener los índices de las mejores coincidencias
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        # Crear lista de resultados - convertimos explícitamente a float para evitar arrays numpy
        results = [(self.column_names[table][i], float(similarities[i])) 
                    for i in top_indices if i < len(self.column_names[table])]
        
        return results
    
    def enhance_sql_info(self, keywords: Dict[str, Any], question: str) -> Dict[str, Any]:
        """
        Mejora la información extraída para generar SQL usando RAG.
        
        Args:
            keywords: Diccionario con la información estructurada extraída
            question: Pregunta original en lenguaje natural
            
        Returns:
            Diccionario actualizado con información mejorada
        """
        result = keywords.copy()
        
        # 1. Si no hay tablas o las tablas no existen, encontrar las más relevantes
        if not result.get('tables') or not any(t in self.table_names for t in result.get('tables', [])):
            matching_tables = self.find_matching_table(question, top_n=2)
            if matching_tables:
                logging.info(f"RAG encontró tablas relevantes: {matching_tables}")
                result['tables'] = [t[0] for t in matching_tables]
        
        # 2. Si hay tablas válidas pero no hay columnas, encontrar columnas relevantes
        if result.get('tables') and not result.get('columns'):
            main_table = result['tables'][0]
            if main_table in self.table_names:
                matching_columns = self.find_matching_columns(question, main_table, top_n=3)
                if matching_columns:
                    logging.info(f"RAG encontró columnas relevantes: {matching_columns}")
                    result['columns'] = [c[0] for c in matching_columns]
        
        # 3. Si hay condiciones pero los nombres de columnas no coinciden, intentar mapearlos
        if result.get('conditions'):
            updated_conditions = []
            
            for condition in result['conditions']:
                if isinstance(condition, str):
                    # Texto de condición: buscar términos que podrían ser columnas
                    words = re.findall(r'\b\w+\b', condition)
                    for word in words:
                        for table in result.get('tables', []):
                            if table in self.table_names:
                                # Buscar si esta palabra es similar a alguna columna
                                col_matches = self.find_matching_columns(word, table, top_n=1)
                                if col_matches and col_matches[0][1] > 0.6:  # Umbral de confianza
                                    mapped_col = col_matches[0][0]
                                    if mapped_col != word:
                                        logging.info(f"RAG mapeando columna '{word}' a '{mapped_col}'")
                                        condition = condition.replace(word, mapped_col)
                    
                    updated_conditions.append(condition)
                
                elif isinstance(condition, dict) and 'column' in condition:
                    # Diccionario de condición: mapear el nombre de columna
                    col_name = condition['column']
                    for table in result.get('tables', []):
                        if table in self.table_names:
                            col_matches = self.find_matching_columns(col_name, table, top_n=1)
                            if col_matches and col_matches[0][1] > 0.6:
                                mapped_col = col_matches[0][0]
                                if mapped_col != col_name:
                                    logging.info(f"RAG mapeando columna '{col_name}' a '{mapped_col}'")
                                    condition = condition.copy()  # Evitar modificar el original
                                    condition['column'] = mapped_col
                    
                    updated_conditions.append(condition)
                
                else:
                    updated_conditions.append(condition)
            
            result['conditions'] = updated_conditions
        
        # Añadir información contextual de RAG
        self._add_rag_context(result, question)
        
        return result
    
    def _add_rag_context(self, result: Dict[str, Any], question: str):
        """Añade información contextual de RAG para ayudar en la generación de SQL"""
        if 'rag_context' not in result:
            result['rag_context'] = {}
        
        # Añadir descripciones de tablas
        if result.get('tables'):
            table_descriptions = {}
            for table in result['tables']:
                if table in self.schema_knowledge['tables']:
                    table_info = self.schema_knowledge['tables'][table]
                    table_descriptions[table] = {
                        'description': table_info.get('description', ''),
                        'display_name': table_info.get('display_name', '')
                    }
            
            result['rag_context']['table_descriptions'] = table_descriptions
        
        # Añadir descripciones de columnas si hay tablas
        if result.get('tables'):
            column_descriptions = {}
            for table in result['tables']:
                if table in self.schema_knowledge['columns']:
                    column_descriptions[table] = {}
                    for col, info in self.schema_knowledge['columns'][table].items():
                        column_descriptions[table][col] = {
                            'description': info.get('description', ''),
                            'display_name': info.get('display_name', '')
                        }
            
            result['rag_context']['column_descriptions'] = column_descriptions
                
        # Añadir relaciones relevantes
        if result.get('tables') and len(result['tables']) > 1:
            relationships = {}
            for table in result['tables']:
                if table in self.schema_knowledge['relationships']:
                    relations = []
                    for rel in self.schema_knowledge['relationships'][table]:
                        # Usar .get() para acceder a 'to_table' de forma segura
                        to_table = rel.get('to_table', '')
                        if to_table and to_table in result['tables']:
                            relations.append(rel)
                    
                    if relations:
                        relationships[table] = relations
            
            if relationships:
                result['rag_context']['relationships'] = relationships
    
    def get_schema_context(self, tables: List[str] = None) -> str:
        """
        Genera un contexto de esquema para incluir en prompts de LLM.
        
        Args:
            tables: Lista de tablas para incluir (None = todas)
            
        Returns:
            Texto con información de esquema para incluir en prompts
        """
        context = "Información del esquema de base de datos:\n"
        
        # Si no se especifican tablas, usar todas las disponibles (limitadas a 5)
        if not tables:
            tables = self.table_names[:5]
        
        # Incluir información de cada tabla
        for table in tables:
            if table in self.schema_knowledge['tables']:
                table_info = self.schema_knowledge['tables'][table]
                context += f"\n- Tabla {table} ({table_info.get('display_name', '')}): {table_info.get('description', '')}\n"
                
                # Incluir columnas importantes
                if table in self.schema_knowledge['columns']:
                    cols = list(self.schema_knowledge['columns'][table].items())
                    # Limitar a 8 columnas para no sobrecargar el contexto
                    for col_name, col_info in cols[:8]:
                        context += f"  * {col_name}: {col_info.get('description', '')}\n"
                    
                    if len(cols) > 8:
                        context += f"  * ... y {len(cols) - 8} columnas más\n"
        
        # Incluir algunas relaciones
        added_relations = 0
        context += "\nRelaciones importantes:\n"
        for table in tables:
            if table in self.schema_knowledge['relationships']:
                for rel in self.schema_knowledge['relationships'][table][:2]:  # Limitar a 2 por tabla
                    if rel['to_table'] in tables:
                        context += f"- {rel['description']}\n"
                        added_relations += 1
                        if added_relations >= 5:  # Máximo 5 relaciones
                            break
            if added_relations >= 5:
                break
        
        return context.strip()

# Función para inicializar el RAG desde el pipeline principal
def initialize_database_rag(conn: sqlite3.Connection) -> DatabaseSchemaRAG:
    """
    Inicializa el sistema RAG para la base de datos especificada.
    
    Args:
        conn: Conexión a la base de datos
        
    Returns:
        Instancia de DatabaseSchemaRAG
    """
    start_time = time.time()
    logging.info("Inicializando sistema RAG de esquema de base de datos...")
    
    # Crear instancia de RAG
    rag = DatabaseSchemaRAG(conn)
    
    # Verificar si se inicializó correctamente
    if not rag.table_names:
        logging.warning("No se pudo inicializar RAG con tablas. Verificando conexión...")
        try:
            # Verificar conexión
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 5")
            tables = cursor.fetchall()
            if tables:
                logging.info(f"Conexión OK, tablas detectadas: {tables}")
                # Forzar inicialización
                rag.conn = conn
                rag.initialize_schema_knowledge()
            else:
                logging.error("La base de datos parece estar vacía")
        except sqlite3.Error as e:
            logging.error(f"Error en conexión a la BD: {e}")
    
    logging.info(f"RAG inicializado en {time.time() - start_time:.2f}s")
    return rag

if __name__ == "__main__":
    # Script para probar el RAG de forma independiente
    from dotenv import load_dotenv
    import sqlite3
    
    # Configurar logging para mostrar también DEBUG
    logging.basicConfig(level=logging.DEBUG, 
                       format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Cargar variables de entorno
    load_dotenv()
    
    # Conectar a la base de datos
    try:
        db_path = "sqlite-analyzer/src/db/database.sqlite3.db"  # Ajustar según corresponda
        conn = sqlite3.connect(db_path)
        
        # Crear y probar RAG
        rag = DatabaseSchemaRAG(conn)
        
        # Mostrar las tablas detectadas
        print(f"Tablas detectadas: {rag.table_names}")
        
        # Prueba de búsqueda
        test_query = "pacientes que tienen citas programadas"
        matching_tables = rag.find_matching_table(test_query, top_n=2)
        print(f"\nPara la consulta '{test_query}':")
        print(f"Tablas más relevantes: {matching_tables}")
        
        if matching_tables:
            main_table = matching_tables[0][0]
            columns = rag.find_matching_columns(test_query, main_table, top_n=3)
            print(f"Columnas relevantes en {main_table}: {columns}")
        
        # Mostrar ejemplo de contexto para LLM
        print("\nEjemplo de contexto para LLM:")
        print(rag.get_schema_context())
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"Error de base de datos: {e}")
    except Exception as e:
        print(f"Error: {e}")