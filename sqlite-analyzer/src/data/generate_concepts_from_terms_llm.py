"""
Script para generar automáticamente conceptos de consulta complejos utilizando un LLM.
Lee el esquema de la base de datos (schema_rag_enriched.json) y los términos comunes
del diccionario (dictionary.json), y luego pide al LLM que sugiera conceptos
para cada tabla.
"""
import json
import os
import sys

# Ajustar el path para importar desde el directorio src principal
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
if src_dir not in sys.path:
    sys.path.append(src_dir)

try:
    from llm_utils import call_llm_with_fallbacks, extract_json_from_llm_response # Importar la función necesaria
    from data.update_dictionary import load_dictionary, save_dictionary
except ImportError as e:
    print(f"Error importando módulos: {e}. Asegúrate de que el script está en src/data/ y que llm_utils.py y update_dictionary.py son accesibles.")
    sys.exit(1)

# Configuración del LLM para OpenAI
llm_config = {
    "llm_api_key": os.getenv("OPENAI_API_KEY"),
    "llm_api_url": "https://api.openai.com/v1",
    "llm_model": "gpt-3.5-turbo"
}

SCHEMA_PATH = os.path.join(current_dir, "schema_enhanced.json") # Corregido para apuntar al archivo y ubicación correctos
DICT_PATH = os.path.join(current_dir, "dictionary.json") # Ya definido en update_dictionary, pero bueno tenerlo aquí

def load_schema(path):
    """Carga el archivo de esquema JSON."""
    if not os.path.exists(path):
        print(f"Error: El archivo de esquema {path} no fue encontrado.")
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            schema_data = json.load(f)
        print(f"Esquema cargado desde {path}")
        return schema_data
    except json.JSONDecodeError as e:
        print(f"Error: El archivo de esquema {path} está corrupto o no es un JSON válido: {e}")
        return None
    except IOError as e:
        print(f"Error al leer el archivo de esquema {path}: {e}")
        return None

def build_llm_prompt_for_table_concepts(table_info, common_terms):
    """Construye el prompt para el LLM para generar conceptos para una tabla dada."""
    table_name = table_info.get("table_name", "N/A")
    table_description = table_info.get("description", "No disponible")
    columns_info = []
    for col in table_info.get("columns", []):
        col_str = f"  - {col.get('name')} ({col.get('type')}): {col.get('description', 'N/D')}"
        if col.get('is_primary_key'):
            col_str += " (PK)"
        if col.get('is_foreign_key'):
            col_str += f" (FK references {col.get('references_table')}.{col.get('references_column')})"
        columns_info.append(col_str)
    
    columns_details = "\n".join(columns_info)
    terms_str = ", ".join(common_terms) if common_terms else "ninguno en particular"

    prompt = f"""
Eres un experto en análisis de datos y diseño de bases de datos médicas.
Tu tarea es generar conceptos de consulta complejos y útiles para la tabla '{table_name}' de una base de datos médica.

Información de la tabla '{table_name}':
Descripción: {table_description}
Columnas:
{columns_details}

Términos comunes que los usuarios asocian con esta tabla: {terms_str}.

Un "concepto de consulta" representa un tipo de pregunta común, un patrón de análisis o una necesidad de información que un usuario podría tener y que involucra a esta tabla (y posiblemente otras relacionadas).

Por favor, genera entre 1 y 3 conceptos de consulta para la tabla '{table_name}'.
Para cada concepto, proporciona la siguiente información en formato JSON. La respuesta DEBE ser una lista de objetos JSON, donde cada objeto representa un concepto.
Cada objeto concepto DEBE tener los siguientes campos:
- "name": Un nombre corto, descriptivo y único para el concepto (en snake_case, ej: 'pacientes_con_diagnostico_principal').
- "description": Una descripción clara de lo que el concepto representa o la pregunta que ayuda a responder.
- "tables_involved": Una lista de strings con los nombres de las tablas involucradas (incluyendo '{table_name}' y otras que consideres necesarias).
- "key_columns": Una lista de strings con las columnas clave para los JOINs o filtros importantes (ej: 'TABLA.COLUMNA').
- "join_details": Una descripción textual de cómo se unen las tablas si hay más de una (ej: 'EPISODES.ID_EPISODE = DIAGNOSES.ID_EPISODE'). Si solo hay una tabla, puede ser una cadena vacía.
- "keywords": Una lista de strings con palabras clave relevantes para este concepto.
- "example_nl_questions": Una lista de 2-3 preguntas de ejemplo en lenguaje natural que este concepto ayudaría a resolver.
- "notes": Cualquier aclaración o nota adicional sobre el concepto (puede ser una cadena vacía).

Ejemplo de formato de respuesta JSON (debe ser una lista de estos objetos):
[
  {{
    "name": "ejemplo_concepto_tabla_actual",
    "description": "Descripción del primer concepto para {table_name}.",
    "tables_involved": ["{table_name}", "OTRA_TABLA_SI_APLICA"],
    "key_columns": ["{table_name}.ID_COLUMNA", "OTRA_TABLA_SI_APLICA.FK_COLUMNA"],
    "join_details": "{table_name}.ID_COLUMNA = OTRA_TABLA_SI_APLICA.FK_COLUMNA",
    "keywords": ["palabra_clave1", "palabra_clave2"],
    "example_nl_questions": ["Pregunta de ejemplo 1 relacionada con {table_name}?", "Pregunta de ejemplo 2?"],
    "notes": "Nota adicional para el primer concepto."
  }}
  // ... más objetos de concepto si generas más de uno ...
]

Asegúrate de que la respuesta sea únicamente el JSON que representa la lista de conceptos, sin ningún texto introductorio o explicaciones adicionales fuera del JSON.
Considera las relaciones comunes en bases de datos médicas (pacientes, episodios, diagnósticos, tratamientos, etc.) al proponer tablas relacionadas y joins.
Si la tabla actual parece ser una tabla de códigos o maestra, los conceptos podrían girar en torno a cómo buscar o utilizar esos códigos.
Si la tabla es transaccional (ej: episodios), los conceptos podrían ser sobre agregaciones, conteos o búsquedas filtradas.
"""
    return prompt

def generate_concepts():
    """Función principal para generar conceptos y actualizar dictionary.json."""
    schema_data = load_schema(SCHEMA_PATH)
    if not schema_data:
        return

    # Cargar (o inicializar) dictionary.json usando DICT_PATH explícitamente
    current_dictionary_data = load_dictionary(file_path=DICT_PATH) 

    if not isinstance(current_dictionary_data.get('concepts'), dict):
        current_dictionary_data['concepts'] = {} # Asegurar que 'concepts' es un diccionario

    # Identificar tablas que ya tienen conceptos generados para evitar reprocesarlas
    processed_tables_with_concepts = set()
    if current_dictionary_data.get('concepts'): # Asegurarse de que 'concepts' existe y no es None
        for concept_data in current_dictionary_data.get('concepts', {}).values():
            if isinstance(concept_data, dict) and 'tables_involved' in concept_data:
                # Si una tabla está en 'tables_involved' de cualquier concepto existente,
                # la consideramos como que ya tiene algún procesamiento.
                for involved_table in concept_data.get('tables_involved', []):
                    processed_tables_with_concepts.add(involved_table)
    
    if processed_tables_with_concepts:
        print(f"Información: Tablas que ya están involucradas en conceptos existentes en '{DICT_PATH}' (se omitirá la llamada al LLM para estas tablas si son la tabla principal del prompt): {', '.join(sorted(list(processed_tables_with_concepts)))}")

    # Iterar sobre las tablas del esquema enriquecido
    for table_info in schema_data:
        table_name = table_info.get("table_name")
        if not table_name:
            print(f"Advertencia: Se encontró una entrada de tabla sin nombre en el esquema. Saltando.")
            continue

        print(f"\nProcesando tabla: {table_name} para generación de conceptos...")

        # Si la tabla actual ya está involucrada en algún concepto existente, saltar la llamada al LLM.
        if table_name in processed_tables_with_concepts:
            print(f"  La tabla '{table_name}' ya está involucrada en conceptos existentes. Saltando la llamada al LLM.")
            continue

        common_terms = []
        # Los common_terms se leen del current_dictionary_data (que es dictionary.json)
        if table_name in current_dictionary_data.get("tables", {}):
            raw_terms = current_dictionary_data["tables"][table_name].get("common_terms", [])
            common_terms = [term.replace("- ", "", 1).strip() for term in raw_terms if isinstance(term, str)]
        
        prompt = build_llm_prompt_for_table_concepts(table_info, common_terms)
        
        messages = [
            {"role": "system", "content": "Eres un asistente IA experto en análisis de datos y bases de datos médicas."},
            {"role": "user", "content": prompt}
        ]

        print(f"  Llamando al LLM para generar conceptos para la tabla '{table_name}'...")
        llm_response_str = None
        try:
            llm_response_str = call_llm_with_fallbacks(llm_config, messages)
        except Exception as e_llm_call:
            print(f"  Error durante la llamada al LLM para la tabla '{table_name}': {e_llm_call}. Saltando esta tabla.")
            continue # Pasar a la siguiente tabla

        if not llm_response_str or llm_response_str.startswith("ERROR:"):
            print(f"  Error del LLM o respuesta vacía para {table_name}: {llm_response_str}. Saltando.")
            continue

        generated_concepts_list = extract_json_from_llm_response(llm_response_str)

        if generated_concepts_list is None:
            print(f"  Error al decodificar la respuesta JSON del LLM para {table_name} o no se encontró JSON. Respuesta: {llm_response_str}")
            continue
            
        try: # Este try es para procesar la lista de conceptos
            if not isinstance(generated_concepts_list, list):
                print(f"  Advertencia: La respuesta del LLM para {table_name} no fue una lista JSON válida después de la extracción. Respuesta procesada: {generated_concepts_list}")
                continue 
            
            if not generated_concepts_list:
                print(f"  El LLM no generó conceptos para la tabla {table_name}.")
                continue

            print(f"  LLM generó {len(generated_concepts_list)} concepto(s) para {table_name}.")
            
            concepts_added_for_this_table = False
            for concept_idx, concept_data in enumerate(generated_concepts_list):
                if not isinstance(concept_data, dict):
                    print(f"    Advertencia: Elemento {concept_idx} en la lista de conceptos no es un diccionario. Saltando.")
                    continue

                concept_name = concept_data.get("name")
                if not concept_name:
                    print(f"    Advertencia: Concepto generado sin nombre para {table_name}. Saltando.")
                    continue
                
                original_concept_name = concept_name
                suffix = 1
                while concept_name in current_dictionary_data["concepts"]:
                    suffix += 1
                    concept_name = f"{original_concept_name}_{suffix}"
                
                if suffix > 1:
                    print(f"    Nombre de concepto original '{original_concept_name}' ya existía. Renombrado a '{concept_name}'.")
                    concept_data["name"] = concept_name 

                if not all(k in concept_data for k in ["description", "tables_involved", "key_columns", "join_details", "keywords", "example_nl_questions"]):
                    print(f"    Advertencia: Concepto '{concept_name}' no tiene todos los campos esperados. Saltando.")
                    continue
                
                current_dictionary_data["concepts"][concept_name] = concept_data
                print(f"    Concepto '{concept_name}' añadido/actualizado en el diccionario.")
                concepts_added_for_this_table = True

            if concepts_added_for_this_table:
                # Guardar el diccionario DESPUÉS de procesar exitosamente una tabla y añadir sus conceptos.
                save_dictionary(current_dictionary_data, file_path=DICT_PATH)
                print(f"  Progreso guardado en '{DICT_PATH}' después de añadir conceptos para la tabla '{table_name}'.")
                # Actualizar el conjunto de tablas procesadas para esta sesión.
                processed_tables_with_concepts.add(table_name)

        except Exception as e_concept_processing:
            print(f"  Error inesperado procesando la lista de conceptos para {table_name}: {e_concept_processing}")
            # Continuar con la siguiente tabla; los conceptos para esta tabla (si los hubo) no se guardaron si el error fue aquí.
            
    # El guardado final se elimina, confiamos en el guardado incremental.
    print(f"\nProceso de generación de conceptos completado. El archivo '{DICT_PATH}' ha sido actualizado incrementalmente si se generaron nuevos conceptos.")

if __name__ == "__main__":
    generate_concepts()
