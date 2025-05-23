"""
Módulo para identificar tablas relevantes en la base de datos médica a partir de una consulta en lenguaje natural.
Utiliza el esquema enriquecido (schema_enhanced.json) y devuelve sugerencias ordenadas y trazabilidad.
"""
import os
import json
import re
from typing import List, Dict

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'schema_enhanced.json')

# Cargar stopwords de NLTK en español
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('spanish'))
except Exception:
    print("[ERROR] No se pudo cargar las stopwords de NLTK. Ejecuta 'pip install nltk' y descarga las stopwords con:\nimport nltk; nltk.download('stopwords')")
    STOPWORDS = set()

def load_enhanced_schema(path=SCHEMA_PATH):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def clean_text(text):
    # Quita puntuación y pasa a minúsculas
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text

def extract_keywords(text):
    # Devuelve solo palabras significativas (sin stopwords)
    words = clean_text(text).split()
    return [w for w in words if w not in STOPWORDS and len(w) > 2]

def extract_synonyms_and_text(table_info: dict) -> List[str]:
    """
    Extrae todos los sinónimos, descripciones y casos de uso relevantes de la info de la tabla.
    """
    text_blobs = []
    # Extraer del campo description (que es un string con JSON embebido)
    desc = table_info.get('description', '')
    if desc:
        # Intenta extraer el JSON embebido
        match = re.search(r'\{.*\}', desc, re.DOTALL)
        if match:
            try:
                desc_json = json.loads(match.group(0))
                # Agrega descripción, sinónimos y caso de uso
                if 'descripcion' in desc_json:
                    text_blobs.append(desc_json['descripcion'])
                if 'sinonimos' in desc_json:
                    text_blobs.extend(desc_json['sinonimos'])
                if 'caso_uso' in desc_json:
                    text_blobs.append(desc_json['caso_uso'])
            except Exception:
                text_blobs.append(desc)
        else:
            text_blobs.append(desc)
    # También incluye los campos directos si existen
    if 'synonyms' in table_info and table_info['synonyms']:
        text_blobs.extend(table_info['synonyms'])
    if 'use_case' in table_info and table_info['use_case']:
        text_blobs.append(table_info['use_case'])
    return [t.lower() for t in text_blobs if isinstance(t, str)]

def identify_relevant_tables(user_query: str, top_k: int = 3, explain: bool = False) -> List[Dict]:
    """
    Dada una consulta de usuario, devuelve una lista de tablas candidatas ordenadas por score de relevancia.
    El score es el número de coincidencias únicas de palabras/frases clave en descripciones, sinónimos y casos de uso.
    Coincidencias exactas de frases valen más que palabras sueltas.
    """
    schema = load_enhanced_schema()
    tables = schema.get('schema_knowledge', {}).get('tables', {})
    query = user_query.lower()
    query_keywords = set(extract_keywords(query))
    results = []
    for table, info in tables.items():
        text_blobs = extract_synonyms_and_text(info)
        matched_set = set()
        # Coincidencia exacta de frase
        for blob in text_blobs:
            blob_clean = clean_text(blob)
            if blob_clean in query:
                matched_set.add(blob.strip())
            else:
                blob_keywords = set(extract_keywords(blob))
                common = query_keywords & blob_keywords
                matched_set.update(common)
        score = len(matched_set)
        if score > 0:
            explanation = f"Coincidencias: {list(matched_set)}" if explain else None
            results.append({
                'table': table,
                'score': score,
                'matched_terms': list(matched_set),
                'explanation': explanation
            })
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]

def identify_relevant_tables_with_llm(user_query: str, llm_api_key: str, llm_api_url: str, top_k: int = 3, explain: bool = False) -> List[Dict]:
    """
    Usa RAG: construye un contexto con nombres, descripciones y sinónimos de las tablas y consulta al LLM para que elija las tablas más relevantes.
    Devuelve una lista de tablas candidatas con explicación.
    """
    import requests
    schema = load_enhanced_schema()
    tables = schema.get('schema_knowledge', {}).get('tables', {})
    # Construir contexto para el LLM
    context_lines = []
    for table, info in tables.items():
        desc = ''
        # Extraer descripción enriquecida
        match = re.search(r'\{.*\}', info.get('description', ''), re.DOTALL)
        if match:
            try:
                desc_json = json.loads(match.group(0))
                desc = desc_json.get('descripcion', '')
                synonyms = ', '.join(desc_json.get('sinonimos', []))
                use_case = desc_json.get('caso_uso', '')
            except Exception:
                desc = info.get('description', '')
                synonyms = ''
                use_case = ''
        else:
            desc = info.get('description', '')
            synonyms = ''
            use_case = ''
        context_lines.append(f"- {table}: {desc} Sinónimos: {synonyms}. Caso de uso: {use_case}")
    context = '\n'.join(context_lines)
    prompt = f"""
Tienes la siguiente estructura de base de datos médica (extracto):
{context}

Pregunta del usuario: \"{user_query}\"

Reglas importantes:
- Si la pregunta menciona 'paciente', 'del paciente', 'de un paciente', 'de pacientes', incluye siempre la tabla PATI_PATIENTS (o la tabla de pacientes que corresponda) en la lista de tablas relevantes.
- Si la pregunta menciona especialidad, tipo de cita, diagnóstico, servicio o área médica, incluye también la tabla correspondiente de tipos, especialidades o servicios si existe.

Indica una lista de las tablas más relevantes para responder la pregunta y explica brevemente por qué cada una es útil. Responde en formato JSON así:
[
  {{"table": "NOMBRE_TABLA", "explicacion": "..."}},
  ...
]
Incluye todas las tablas necesarias para responder correctamente, no solo una.
"""
    headers = {
        "Authorization": f"Bearer {llm_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "Eres un experto en bases de datos médicas."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 600
    }
    try:
        response = requests.post(llm_api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            # Intentar extraer JSON (lista de tablas)
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                # data debe ser una lista de dicts
                return [
                    {
                        'table': d.get('table', ''),
                        'score': 1,
                        'matched_terms': [],
                        'explanation': d.get('explicacion', '')
                    }
                    for d in data if isinstance(d, dict)
                ][:top_k]
    except Exception as e:
        print(f"[ERROR] LLM RAG: {e}")
    return []

def identify_relevant_tables_with_llm_dictionary(user_query: str, llm_api_key: str, llm_api_url: str, top_k: int = 3, explain: bool = False) -> List[Dict]:
    """
    Variante RAG usando dictionary.json (más pequeño) como contexto para el LLM.
    Devuelve una lista de tablas candidatas con explicación.
    """
    import requests
    DICT_PATH = os.path.join(os.path.dirname(__file__), 'data', 'dictionary.json')
    with open(DICT_PATH, 'r', encoding='utf-8') as f:
        dictionary = json.load(f)
    tables = dictionary.get('tables', {})
    # Construir contexto para el LLM
    context_lines = []
    for table, info in tables.items():
        desc = info.get('description', '')
        terms = info.get('common_terms', [])
        # Limpiar términos
        clean_terms = [t for t in terms if not t.startswith('**')]
        context_lines.append(f"- {table}: {desc} Términos: {', '.join(clean_terms)}")
    context = '\n'.join(context_lines)
    prompt = f"""
Tienes la siguiente estructura de base de datos médica (extracto):
{context}

Pregunta del usuario: \"{user_query}\"

Indica una lista de las tablas más relevantes para responder la pregunta y explica brevemente por qué cada una es útil. Responde en formato JSON así:
[
  {{"table": "NOMBRE_TABLA", "explicacion": "..."}},
  ...
]
Incluye todas las tablas necesarias para responder correctamente, no solo una.
"""
    headers = {
        "Authorization": f"Bearer {llm_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "Eres un experto en bases de datos médicas."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 600
    }
    try:
        response = requests.post(llm_api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            # Intentar extraer JSON (lista de tablas)
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return [
                    {
                        'table': d.get('table', ''),
                        'score': 1,
                        'matched_terms': [],
                        'explanation': d.get('explicacion', '')
                    }
                    for d in data if isinstance(d, dict)
                ][:top_k]
    except Exception as e:
        print(f"[ERROR] LLM RAG (dictionary): {e}")
    return []

# Ejemplo de uso interactivo
if __name__ == "__main__":
    import os
    q = input("Consulta médica: ")
    modo = input("¿Modo? (1=matching clásico, 2=LLM schema_enhanced, 3=LLM dictionary): ").strip()
    if modo == '2':
        llm_api_key = os.environ.get("DEEPSEEK_API_KEY", "sk-aedf531ee17447aa95c9102e595f29ae")
        llm_api_url = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
        candidates = identify_relevant_tables_with_llm(q, llm_api_key, llm_api_url)
        if not candidates:
            print("No se encontraron tablas relevantes (LLM schema_enhanced).")
        else:
            for c in candidates:
                print(f"Tabla: {c['table']} | Score: {c['score']} | Explicación: {c['explanation']}")
    elif modo == '3':
        llm_api_key = os.environ.get("DEEPSEEK_API_KEY", "sk-aedf531ee17447aa95c9102e595f29ae")
        llm_api_url = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
        candidates = identify_relevant_tables_with_llm_dictionary(q, llm_api_key, llm_api_url)
        if not candidates:
            print("No se encontraron tablas relevantes (LLM dictionary).")
        else:
            for c in candidates:
                print(f"Tabla: {c['table']} | Score: {c['score']} | Explicación: {c['explanation']}")
    else:
        candidates = identify_relevant_tables(q, explain=True)
        if not candidates:
            print("No se encontraron tablas relevantes.")
        else:
            for c in candidates:
                print(f"Tabla: {c['table']} | Score: {c['score']} | Términos: {c['matched_terms']} | Explicación: {c['explanation']}")
