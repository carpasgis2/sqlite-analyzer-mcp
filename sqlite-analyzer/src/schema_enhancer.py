import json
import os
import logging
import re
import sqlite3
import sys
from datetime import datetime  # Añadido para generar timestamps
from typing import Dict, Any, List, Optional, Set, Tuple
from tqdm import tqdm  # Añadir esta línea para importar tqdm

# Agregar path para importaciones desde el directorio actual
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from llm_utils import call_llm, call_llm_with_fallbacks  # Importamos la utilidad para llamar al LLM
except ImportError:
    logging.warning("No se pudo importar call_llm. Asegúrate de que está disponible.")
    def call_llm(messages, **kwargs):
        return "No se pudo llamar al LLM. Descripción no disponible."
    def call_llm_with_fallbacks(messages, config, step_name=""):
        return "No se pudo llamar al LLM. Descripción no disponible."

# Definir ruta absoluta al archivo de esquema
DATA_DIR = os.path.join(current_dir, "data")
SCHEMA_ENHANCER_FILE = os.path.join(DATA_DIR, "schema_enhanced.json")

class SchemaEnhancer:
    """Clase para mejorar y gestionar el esquema de la base de datos"""
    
    def __init__(self, schema_path: str = None):
        """
        Inicializa el enhancer de esquema.
        
        Args:
            schema_path: Ruta al archivo de esquema (opcional)
        """
        # Si se proporciona una ruta específica, usarla; de lo contrario usar la ruta constante
        self.schema_path = schema_path if schema_path else SCHEMA_ENHANCER_FILE
        
        # Verificar si el archivo especificado existe
        if schema_path and not os.path.exists(schema_path):
            logging.warning(f"El archivo de esquema no existe: {schema_path}")
            # Si no existe, usar la ruta estándar
            self.schema_path = SCHEMA_ENHANCER_FILE
        
        # Asegurar que el directorio existe
        os.makedirs(os.path.dirname(self.schema_path), exist_ok=True)
        
        self.schema_file = self.schema_path  # Usar la misma ruta para lectura/escritura
        self.loaded = False
        self.schema_data = {}
        self.load_schema()
        
    def generate_db_schema_json(self, db_path, output_path: Optional[str] = None, force_regenerate: bool = False) -> str:
        """
        Genera un JSON con la estructura completa de la base de datos SQLite,
        incluyendo descripciones generadas por un LLM.
        
        Args:
            db_path: Ruta a la base de datos SQLite o estructura de BD ya extraída
            output_path: Ruta donde guardar el JSON (opcional)
            force_regenerate: Si es True, regenera el esquema incluso si ya existe
            
        Returns:
            Ruta al archivo JSON generado
        """
        # Si no se especifica una ruta de salida, usar la ruta por defecto
        if output_path is None:
            output_path = os.path.join(DATA_DIR, "schema_enhanced.json")
        
        # Verificar si el archivo ya existe y si no se fuerza la regeneración
        if os.path.exists(output_path) and not force_regenerate:
            logging.info(f"El esquema ya existe en: {output_path}. Usando archivo existente. Para regenerar, usa --force.")
            return output_path
        
        # Asegurar que el directorio existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        logging.info(f"El esquema se guardará en: {output_path}")
        
        # El resto del método continúa igual...
        
        # Usar tqdm para mostrar el progreso general
        with tqdm(total=5, desc="Generando esquema", unit="paso") as pbar:
            # Paso 1: Extraer estructura de la BD
            pbar.set_description("Extrayendo estructura de BD")
            if isinstance(db_path, str):
                logging.info(f"Generando esquema para la BD: {db_path}")
                db_structure = self._extract_db_structure(db_path)
            else:
                logging.info("Usando estructura de BD proporcionada")
                db_structure = db_path
            pbar.update(1)
            
            # Paso 2: Enriquecer con descripciones
            pbar.set_description("Generando descripciones con LLM")
            enriched_structure = self._enrich_with_llm_descriptions(db_structure)
            pbar.update(1)

            # Prepare tables_data for schema_knowledge, starting from enriched_structure.tables
            # We will add 'columns_list' to this data.
            # Make a copy to avoid modifying enriched_structure directly if it's used elsewhere unmodified
            # (e.g., by _generate_table_corpus).
            current_schema_knowledge_tables = {}
            if "tables" in enriched_structure:
                for table_name, table_info_enriched in enriched_structure["tables"].items():
                    current_schema_knowledge_tables[table_name] = table_info_enriched.copy() # Shallow copy of the table's dict

            # Add a list of actual column names (strings) to each table's data for schema_knowledge
            for table_name, table_data_for_knowledge in current_schema_knowledge_tables.items():
                if table_name in db_structure and "columns" in db_structure[table_name]:
                    # Extract list of column name strings from db_structure
                    list_of_column_names = [col["name"] for col in db_structure[table_name]["columns"]]
                    # Add this list to the table's data
                    table_data_for_knowledge["columns_list"] = list_of_column_names
                else:
                    # If table from enriched data isn't in db_structure or has no columns there,
                    # add an empty list and log a warning.
                    table_data_for_knowledge["columns_list"] = []
                    logging.warning(
                        f"Table '{table_name}' from enriched data not in db_structure or lacks columns; "
                        f"'columns_list' will be empty for schema_knowledge section."
                    )

            # Paso 3: Crear estructura completa
            pbar.set_description("Generando estructura completa")
            schema = {
                "schema_knowledge": {
                    "tables": current_schema_knowledge_tables, # Use the modified copy which includes 'columns_list'
                    "columns": enriched_structure.get("columns", {}),
                    "relationships": self._extract_relationships(db_structure),
                    "synonyms": {},
                    "examples": {}
                },
                "table_corpus": self._generate_table_corpus(enriched_structure), # Uses original enriched_structure
                "table_names": list(db_structure.keys()),
                "column_corpus": self._generate_column_corpus(enriched_structure), # Uses original enriched_structure
                "column_names": {table: [col["name"] for col in info.get("columns", [])] 
                                for table, info in db_structure.items()},
                "schema": db_structure
            }
            pbar.update(1)
            
            # Paso 4: Validar esquema para RAG
            pbar.set_description("Validando esquema para RAG")
            is_valid, errors = self._validate_schema_for_rag(schema)
            if not is_valid:
                logging.warning(f"El esquema generado tiene {len(errors)} problemas que pueden afectar al RAG")
            pbar.update(1)
            
            # Paso 5: Guardar el esquema
            pbar.set_description("Guardando esquema en archivo")
            try:
                # Crear el directorio si no existe
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                
                # Guardar el archivo JSON
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(schema, f, ensure_ascii=False, indent=2)
                    
                logging.info(f"Esquema guardado exitosamente en: {output_path}")
            except Exception as e:
                logging.error(f"Error al guardar el esquema: {e}")
                raise
            pbar.update(1)
        
        return output_path

    def _extract_db_structure(self, db_path: str) -> Dict[str, Any]:
        """
        Extrae la estructura completa de la base de datos SQLite.
        
        Args:
            db_path: Ruta a la base de datos
            
        Returns:
            Diccionario con la estructura de tablas y columnas
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"No se encontró la base de datos: {db_path}")
        
        logging.info(f"Conectando a la base de datos: {db_path}")
        structure = {}
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Obtener todas las tablas
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            for table in tables:
                # Obtener estructura de cada tabla
                cursor.execute(f"PRAGMA table_info({table});")
                columns_info = cursor.fetchall()
                
                columns = []
                for col_info in columns_info:
                    # col_info: (cid, name, type, notnull, dflt_value, pk)
                    col = {
                        "name": col_info[1],
                        "type": col_info[2],
                        "nullable": not col_info[3],
                        "default": col_info[4],
                        "primary_key": bool(col_info[5])
                    }
                    columns.append(col)
                
                # Obtener índices
                cursor.execute(f"PRAGMA index_list({table});")
                indices = cursor.fetchall()
                
                table_structure = {
                    "columns": columns,
                    "indices": [{"name": idx[1], "unique": bool(idx[2])} for idx in indices]
                }
                
                # Obtener claves foráneas
                cursor.execute(f"PRAGMA foreign_key_list({table});")
                fkeys = cursor.fetchall()
                
                if fkeys:
                    foreign_keys = []
                    for fk in fkeys:
                        # fk: (id, seq, table, from, to, on_update, on_delete, match)
                        foreign_keys.append({
                            "referenced_table": fk[2],
                            "column": fk[3],
                            "referenced_column": fk[4],
                            "on_update": fk[5],
                            "on_delete": fk[6]
                        })
                    table_structure["foreign_keys"] = foreign_keys
                
                structure[table] = table_structure
            
            conn.close()
            logging.info(f"Estructura extraída exitosamente: {len(tables)} tablas")
            return structure
            
        except sqlite3.Error as e:
            logging.error(f"Error de SQLite: {e}")
            raise
    
    def _enrich_with_llm_descriptions(self, db_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enriquece la estructura de la base de datos con descripciones generadas por un LLM, incluyendo sinónimos, casos de uso y ejemplos.
        """
        enriched = {
            "tables": {},
            "columns": {}
        }
        
        # Mostrar una barra de progreso para el proceso de enriquecimiento
        total_tables = len(db_structure)
        logging.info(f"Iniciando generación de descripciones para {total_tables} tablas...")
        
        # Para cada tabla, generar descripción con LLM, añadiendo barra de progreso
        for table_name, table_info in tqdm(db_structure.items(), desc="Generando descripciones", 
                                           total=total_tables, unit="tabla"):
            logging.info(f"Generando descripción enriquecida para tabla: {table_name}")
            
            # Generar descripción enriquecida para la tabla
            table_data = self._generate_table_description(table_name, table_info)
            
            enriched["tables"][table_name] = {
                "name": table_name,
                "description": table_data.get("description", ""),
                "synonyms": table_data.get("synonyms", []),
                "use_case": table_data.get("use_case", "")
            }
            
            # Para cada columna, generar descripción enriquecida con LLM
            column_descriptions = self._generate_column_descriptions(table_name, table_info)
            enriched["columns"][table_name] = column_descriptions
        
        logging.info("Proceso de generación de descripciones enriquecidas completado")
        return enriched

    def _extract_json_from_text(self, text: str) -> dict:
        """
        Extrae el primer bloque JSON válido de un texto, aunque esté embebido o delimitado por ```json ... ```.
        Devuelve un dict vacío si no encuentra JSON válido.
        """
        import json as _json
        # Buscar bloque entre ```json ... ```
        match = re.search(r'```json\s*(\{[\s\S]+?\})\s*```', text)
        if not match:
            # Buscar primer bloque {...} aunque no esté delimitado
            match = re.search(r'(\{[\s\S]+?\})', text)
        if match:
            try:
                return _json.loads(match.group(1))
            except Exception:
                return {}
        return {}

    def _generate_table_description(self, table_name: str, table_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera una descripción enriquecida para una tabla utilizando un LLM, incluyendo sinónimos y casos de uso.
        Devuelve un dict con descripción, sinónimos y casos de uso.
        """
        columns_text = "\n".join([
            f"- {col.get('name', 'unknown')} ({col.get('type', 'unknown')})" + \
            (", PRIMARY KEY" if col.get('primary_key', False) else "") + \
            (", NOT NULL" if not col.get('nullable', True) else "")
            for col in table_info.get("columns", [])
        ])
        foreign_keys_text = ""
        if "foreign_keys" in table_info:
            foreign_keys_text = "\n".join([
                f"- {fk.get('column', '')} → {fk.get('referenced_table', '')}.{fk.get('referenced_column', '')}"
                for fk in table_info["foreign_keys"]
            ])
            if foreign_keys_text:
                foreign_keys_text = "Claves foráneas:\n" + foreign_keys_text
        # Prompt LLM enriquecido
        system_message = (
            "Eres un experto en bases de datos y RAG. Para la tabla dada, genera:\n"
            "1. Una descripción clara y rica (máx 3 frases, en español) sobre el propósito y los datos que almacena.\n"
            "2. Una lista de 3-6 sinónimos o términos comunes (en español) que un usuario podría usar para referirse a esta tabla.\n"
            "3. Un caso de uso típico (en español) de la tabla en una consulta médica.\n"
        )
        user_message = (
            f"Tabla: {table_name}\nColumnas:\n{columns_text}\n{foreign_keys_text}\n"
            "Devuelve el resultado en formato JSON con las claves: descripcion, sinonimos, caso_uso."
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        config = {}
        try:
            response = call_llm_with_fallbacks(messages, config, step_name=f"Tabla {table_name}")
            # Intentar extraer JSON robustamente
            data = self._extract_json_from_text(response)
            if data:
                return {
                    "description": data.get("descripcion", ""),
                    "synonyms": data.get("sinonimos", []),
                    "use_case": data.get("caso_uso", "")
                }
            else:
                # Fallback: parsear manualmente
                return {
                    "description": response.strip(),
                    "synonyms": [],
                    "use_case": ""
                }
        except Exception as e:
            logging.error(f"Error al generar descripción enriquecida para tabla {table_name}: {e}")
            return {"description": f"Tabla que almacena datos relacionados con {table_name.lower().replace('_', ' ')}.", "synonyms": [], "use_case": ""}

    def _generate_column_descriptions(self, table_name: str, table_info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Genera descripciones enriquecidas para las columnas de una tabla utilizando un LLM.
        Devuelve un dict con descripción, sinónimos, ejemplo de valor y flags.
        """
        column_descriptions = {}
        if not table_info.get("columns"):
            logging.warning(f"No hay columnas definidas para la tabla {table_name}")
            return column_descriptions
        config = {}
        # Para obtener ejemplos de valor, intentamos un SELECT si es posible
        example_row = None
        try:
            conn = sqlite3.connect(self.schema_path.replace('schema_enhanced.json','database.sqlite3.db'))
            cur = conn.cursor()
            cur.execute(f"SELECT * FROM {table_name} LIMIT 1")
            example_row = cur.fetchone()
            conn.close()
        except Exception:
            pass
        for idx, column in enumerate(table_info.get("columns", [])):
            col_name = column.get("name", "")
            if not col_name:
                continue
            col_type = column.get("type", "unknown")
            is_primary = column.get("primary_key", False)
            is_nullable = column.get("nullable", True)
            # Prompt LLM enriquecido
            system_message = (
                "Eres un experto en bases de datos y RAG. Para la columna dada, genera:\n"
                "1. Una descripción breve (máx 2 frases, en español) sobre qué almacena y su uso.\n"
                "2. Una lista de 3-6 sinónimos o términos comunes (en español) que un usuario podría usar para referirse a esta columna.\n"
                "3. El tipo semántico (por ejemplo: fecha, identificador, texto, cantidad, código, etc).\n"
                "4. ¿Es sensible? (sí/no, por ejemplo si almacena datos personales o de salud).\n"
                "Devuelve el resultado en formato JSON con las claves: descripcion, sinonimos, tipo_semantico, sensible."
            )
            user_message = (
                f"Columna: {col_name}\nTipo: {col_type}\nTabla: {table_name}\n"
                f"{'Es clave primaria' if is_primary else 'No es clave primaria'}, {'Permite NULL' if is_nullable else 'No permite NULL'}\n"
            )
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            try:
                response = call_llm_with_fallbacks(messages, config, step_name=f"Columna {table_name}.{col_name}")
                import json as _json
                try:
                    data = _json.loads(response)
                    desc = data.get("descripcion", "")
                    synonyms = data.get("sinonimos", [])
                    tipo_sem = data.get("tipo_semantico", "")
                    sensible = data.get("sensible", "no")
                except Exception:
                    desc = response.strip()
                    synonyms = []
                    tipo_sem = ""
                    sensible = "no"
                # Ejemplo de valor
                example_value = None
                if example_row and col_name in example_row.keys():
                    example_value = example_row[col_name]
                column_descriptions[col_name] = {
                    "name": col_name,
                    "description": desc,
                    "synonyms": synonyms,
                    "semantic_type": tipo_sem,
                    "sensitive": sensible,
                    "is_primary_key": is_primary,
                    "is_nullable": is_nullable,
                    "sql_type": col_type,
                    "example_value": example_value
                }
            except Exception as e:
                logging.error(f"Error al generar descripción enriquecida para columna {col_name}: {e}")
                column_descriptions[col_name] = {
                    "name": col_name,
                    "description": f"Campo que almacena {col_name.lower().replace('_', ' ')}.",
                    "synonyms": [],
                    "semantic_type": "",
                    "sensitive": "no",
                    "is_primary_key": is_primary,
                    "is_nullable": is_nullable,
                    "sql_type": col_type,
                    "example_value": None
                }
        return column_descriptions

    def _extract_common_terms(self, name: str, description: str) -> List[str]:
        """
        Extrae términos comunes de un nombre y una descripción.
        
        Args:
            name: Nombre de la tabla o columna
            description: Descripción generada
            
        Returns:
            Lista de términos comunes
        """
        # Convertir nombre a formato más legible
        readable_name = name.lower().replace('_', ' ')
        terms = [readable_name]
        
        # Extraer palabras significativas de la descripción (más de 4 letras, no stopwords)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', description.lower())
        stopwords = {'para', 'como', 'donde', 'datos', 'información', 'tabla', 'columna', 
                    'campo', 'almacena', 'contiene', 'registra', 'guarda', 'cada', 'este', 'esta'}
        
        # Filtrar palabras significativas
        significant_words = [word for word in words if word not in stopwords][:3]
        
        # Añadir palabras significativas a los términos
        terms.extend(significant_words)
        
        return terms
    
    def _extract_relationships(self, db_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extrae relaciones entre tablas a partir de las claves foráneas.
        
        Args:
            db_structure: Estructura de la base de datos
            
        Returns:
            Diccionario de relaciones
        """
        relationships = {}
        
        for table_name, table_info in db_structure.items():
            if "foreign_keys" in table_info:
                table_relations = {}
                
                for fk in table_info["foreign_keys"]:
                    ref_table = fk["referenced_table"]
                    suffix = ref_table.split('_', 1)[1] if '_' in ref_table else ref_table
                    key = f"to_{suffix}"
                    
                    table_relations[key] = {
                        "left": f"{table_name}.{fk['column']}",
                        "right": f"{ref_table}.{fk['referenced_column']}"
                    }
                
                if table_relations:
                    relationships[table_name] = table_relations
        
        return relationships
    
    def _generate_table_corpus(self, enriched_structure: Dict[str, Any]) -> List[str]:
        """
        Genera un corpus de texto para las tablas con formato para RAG.
        
        Args:
            enriched_structure: Estructura enriquecida con descripciones
            
        Returns:
            Lista de strings en formato <tabla> <nombre_comun> <descripción>
        """
        corpus = []
        tables = enriched_structure.get("tables", {})
        
        for table_name, table_info in tables.items():
            # Convertir nombre a formato más legible para nombre común
            common_name = table_name.lower().replace('_', ' ')
            if len(table_info.get("common_terms", [])) > 1:
                common_name = " ".join(table_info["common_terms"][:2])
            
            description = table_info.get("description", f"Tabla que almacena datos de {common_name}.")
            
            # Crear entrada de corpus
            corpus_entry = f"{table_name} {common_name} {description}"
            corpus.append(corpus_entry)
        
        return corpus
    
    def _generate_column_corpus(self, enriched_structure: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Genera un corpus de texto para las columnas con formato para RAG.
        
        Args:
            enriched_structure: Estructura enriquecida con descripciones
            
        Returns:
            Diccionario con tablas como claves y listas de strings como valores
        """
        corpus = {}
        columns = enriched_structure.get("columns", {})
        
        for table_name, table_columns in columns.items():
            table_corpus = []
            
            for col_name, col_info in table_columns.items():
                # Convertir nombre a formato más legible
                common_name = col_name.lower().replace('_', ' ')
                description = col_info.get("description", f"Campo que almacena {common_name}.")
                
                # Crear entrada de corpus
                corpus_entry = f"{col_name} {common_name} {description}"
                table_corpus.append(corpus_entry)
            
            if table_corpus:
                corpus[table_name] = table_corpus
        
        return corpus
    
    def _validate_schema_for_rag(self, schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Verifica que el esquema generado contiene todos los campos necesarios
        para ser utilizado por el sistema RAG.
        
        Args:
            schema: Esquema a validar
            
        Returns:
            Tupla (es_válido, lista_errores)
        """
        errors = []
        
        # 1. Verificar estructura principal
        required_keys = ["schema_knowledge", "table_corpus", "table_names", "column_corpus", "column_names", "schema"]
        for key in required_keys:
            if key not in schema:
                errors.append(f"Falta campo principal '{key}' en el esquema")
        
        # 2. Verificar estructura de schema_knowledge
        if "schema_knowledge" in schema:
            schema_knowledge = schema["schema_knowledge"]
            required_subkeys = ["tables", "columns", "relationships", "synonyms", "examples"]
            for subkey in required_subkeys:
                if subkey not in schema_knowledge:
                    errors.append(f"Falta subcampo '{subkey}' en schema_knowledge")
        
        # 3. Verificar que el corpus de tablas tiene entradas para cada tabla
        if "table_corpus" in schema and "table_names" in schema:
            # Verificar que cada tabla tiene al menos una entrada en el corpus
            tables_without_corpus = [
                table for table in schema["table_names"] 
                if table not in " ".join(schema["table_corpus"])
            ]
            if tables_without_corpus:
                errors.append(f"Tablas sin entradas en el corpus: {', '.join(tables_without_corpus)}")
        
        # 4. Verificar que el corpus de columnas tiene entradas para cada tabla
        if "column_corpus" in schema and "column_names" in schema:
            for table, columns in schema["column_names"].items():
                if table not in schema["column_corpus"]:
                    errors.append(f"No hay corpus para las columnas de la tabla '{table}'")
        
        # 5. Nueva validación: Detectar tablas sin relaciones
        if "schema_knowledge" in schema and "relationships" in schema["schema_knowledge"]:
            relationships = schema["schema_knowledge"]["relationships"]
            table_names = schema.get("table_names", [])
            tables_without_relations = [
                table for table in table_names 
                if table not in relationships and not any(table in rel for rel in relationships.values())
            ]
            if tables_without_relations:
                errors.append(f"Tablas sin relaciones definidas: {', '.join(tables_without_relations[:10])}...")
                logging.warning(f"Se encontraron {len(tables_without_relations)} tablas sin relaciones, "
                              f"lo que puede afectar negativamente al rendimiento del RAG")
        
        # Determinar si el esquema es válido (no hay errores)
        is_valid = len(errors) == 0
        
        if is_valid:
            logging.info("Validación del esquema para RAG: OK")
        else:
            logging.warning(f"Validación del esquema para RAG: {len(errors)} problemas encontrados")
            for error in errors:
                logging.warning(f"  - {error}")
        
        return is_valid, errors
    
    # Método principal para generar el esquema desde una BD SQLite
    def analyze_database_with_llm(self, db_path: str, output_path: Optional[str] = None, force_regenerate: bool = False) -> str:
        """
        Analiza una base de datos SQLite y genera un esquema JSON enriquecido con un LLM.
        
        Args:
            db_path: Ruta a la base de datos SQLite
            output_path: Ruta donde guardar el archivo JSON (opcional)
            force_regenerate: Si es True, regenera el esquema incluso si ya existe
            
        Returns:
            Ruta al archivo JSON generado
        """
        logging.info(f"Iniciando análisis de base de datos con LLM: {db_path}")
        return self.generate_db_schema_json(db_path, output_path, force_regenerate)

    def load_schema(self):
        """Carga el esquema desde el archivo si existe"""
        if self.schema_path and os.path.exists(self.schema_path):
            try:
                with open(self.schema_path, 'r', encoding='utf-8') as f:
                    self.schema_data = json.load(f)
                    self.loaded = True
                    logging.info(f"Esquema cargado correctamente desde: {self.schema_path}")
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logging.error(f"Error al decodificar el esquema: {str(e)}")
                self.schema_data = {}
        else:
            logging.warning(f"El archivo de esquema no existe: {self.schema_path}")
            self.schema_data = {}

    def _is_schema_valid(self, schema: Dict[str, Any]) -> bool:
        """
        Verifica si el esquema contiene la estructura básica necesaria.
        
        Args:
            schema: Esquema a verificar
            
        Returns:
            True si el esquema es válido, False en caso contrario
        """
        # Verificar que el esquema contiene las claves necesarias
        required_keys = ["schema_knowledge", "table_corpus", "table_names", "column_corpus", "column_names"]
        
        if not all(key in schema for key in required_keys):
            return False
        
        # Verificar que schema_knowledge contiene las subclaves necesarias
        schema_knowledge = schema.get("schema_knowledge", {})
        required_subkeys = ["tables", "columns", "relationships"]
        
        if not all(subkey in schema_knowledge for subkey in required_subkeys):
            return False
        
        # El esquema parece ser válido
        return True

# Código de ejecución para la utilidad de línea de comandos
if __name__ == "__main__":
    import argparse
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description='Generador de esquema JSON para bases de datos SQLite con LLM')
    parser.add_argument('db_path', help='Ruta a la base de datos SQLite')
    parser.add_argument('--output', '-o', help='Ruta donde guardar el archivo JSON')
    parser.add_argument('--force', '-f', action='store_true', help='Forzar la regeneración del esquema aunque ya exista') # Añadido argumento --force
    
    args = parser.parse_args()
    
    # Crear instancia del enhancer
    enhancer = SchemaEnhancer(schema_path=args.output) # Usar args.output si se proporciona
    
    # Generar el esquema
    try:
        # Pasar el argumento force_regenerate
        generated_file = enhancer.generate_db_schema_json(
            args.db_path, 
            output_path=args.output, 
            force_regenerate=args.force # Usar el valor de args.force
        )
        print(f"Esquema generado exitosamente en: {generated_file}")
    except Exception as e:
        logging.error(f"Error al generar el esquema: {e}", exc_info=True)
        print(f"Error al generar el esquema. Revisa el log para más detalles.")
