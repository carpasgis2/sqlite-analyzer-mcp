"""
Script para actualizar el archivo dictionary.json con nuevos mapeos semánticos (términos) o conceptos complejos.

Uso para añadir términos:
    python update_dictionary.py add-term --term "alergias" --table "ALLE_INTOLERANCES" --synonyms "reacciones adversas,hipersensibilidades"
    (El argumento --column es opcional y actualmente no se almacena de forma específica con el término, pero se mantiene por compatibilidad o uso futuro)

Uso para añadir conceptos:
    python update_dictionary.py add-concept --name "concepto_ejemplo" \\
                                            --description "Describe un patrón de consulta complejo." \\
                                            --tables "TABLA1,TABLA2" \\
                                            --key-columns "TABLA1.ID,TABLA2.FK_ID" \\
                                            --join-details "TABLA1.ID = TABLA2.FK_ID" \\
                                            --keywords "palabra1,palabra2" \\
                                            --example-questions "Pregunta ejemplo 1?;Pregunta ejemplo 2?" \\
                                            --notes "Notas adicionales sobre el concepto."

Funcionalidades:
- 'add-term': Añade un término y sus sinónimos a la lista 'common_terms' de la tabla indicada.
  - Si la tabla no existe en el diccionario, se crea.
  - Si el término (o sinónimo) ya existe para esa tabla (ignorando mayúsculas/minúsculas y espacios), no se duplica.
- 'add-concept': Añade o actualiza una entrada en la sección 'concepts' del diccionario.
  - Los conceptos ayudan a definir patrones de consulta complejos que involucran múltiples tablas.
  - Si el concepto con el mismo nombre ya existe, se sobrescribe.
"""
import argparse
import json
import os
import shutil # Añadido para el backup

DEFAULT_DICT_PATH = os.path.join(os.path.dirname(__file__), 'dictionary.json')

def load_dictionary(file_path=None):
    current_path = file_path if file_path else DEFAULT_DICT_PATH
    # Asegurarse de que el directorio exista si es el default path y se va a crear el archivo
    if not os.path.exists(current_path) and file_path is None:
        dir_name = os.path.dirname(current_path)
        if not os.path.exists(dir_name):
            try:
                os.makedirs(dir_name)
                print(f"Directorio {dir_name} creado.")
            except OSError as e:
                print(f"Error al crear el directorio {dir_name}: {e}")
                # No se puede continuar si no se puede crear el directorio para el archivo default
                return {"tables": {}, "concepts": {}}
    
    if not os.path.exists(current_path):
        default_data = {"tables": {}, "concepts": {}}
        try:
            save_dictionary(default_data, current_path, backup=False) # No hacer backup al crear por primera vez
            print(f"Archivo {current_path} no encontrado. Se ha creado uno nuevo con estructura base.")
        except IOError as e:
            print(f"Error al crear el archivo {current_path}: {e}")
            return default_data 
        return default_data
    
    try:
        with open(current_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content:
                print(f"Advertencia: El archivo {current_path} está vacío. Inicializando con estructura base.")
                data = {"tables": {}, "concepts": {}}
            else:
                data = json.loads(content)

            if not isinstance(data, dict):
                raise json.JSONDecodeError("El contenido no es un diccionario JSON válido.", content, 0)

            if 'tables' not in data or not isinstance(data['tables'], dict):
                data['tables'] = {}
            if 'concepts' not in data or not isinstance(data['concepts'], dict):
                data['concepts'] = {}
            return data
    except json.JSONDecodeError as e:
        print(f"Error: El archivo {current_path} está corrupto o no es un JSON válido: {e}. Se creará uno nuevo.")
        default_data = {"tables": {}, "concepts": {}}
        try:
            save_dictionary(default_data, current_path, backup=True) # Hacer backup del corrupto antes de sobrescribir
        except IOError as e_save:
            print(f"Error al guardar el nuevo archivo {current_path} después de corrupción: {e_save}")
        return default_data
    except IOError as e:
        print(f"Error al leer el archivo {current_path}: {e}")
        return {"tables": {}, "concepts": {}}

def save_dictionary(data, file_path=None, backup=True):
    current_path = file_path if file_path else DEFAULT_DICT_PATH
    
    if backup and os.path.exists(current_path):
        backup_path = current_path + '.bak'
        try:
            shutil.copy2(current_path, backup_path)
            print(f"Backup creado en {backup_path}")
        except IOError as e:
            print(f"Error al crear el backup {backup_path}: {e}")
            # Continuar igualmente, pero advertir

    try:
        with open(current_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"Error al guardar el diccionario en {current_path}: {e}")

def update_table_terms(data, table_name, term, synonyms=None, column_name=None): # Añadido column_name
    if table_name not in data['tables']:
        data['tables'][table_name] = {}
        print(f"Nota: La tabla '{table_name}' no existía y ha sido creada en el diccionario.")
    
    table_entry = data['tables'][table_name]
    added_any_new_term = False
    all_terms_to_process = [term] + (synonyms or [])

    if column_name:
        # Nueva lógica para términos específicos de columna
        if 'column_specific_common_terms' not in table_entry or not isinstance(table_entry['column_specific_common_terms'], list):
            table_entry['column_specific_common_terms'] = []
        
        specific_terms_list = table_entry['column_specific_common_terms']
        
        # Usar un set de tuplas (término_normalizado, columna_normalizada) para evitar duplicados
        clean_existing_specific_terms = set()
        for t_entry in specific_terms_list:
            if isinstance(t_entry, dict) and 'term' in t_entry and 'column' in t_entry:
                normalized_t = t_entry['term'].strip().lower()
                normalized_c = t_entry['column'].strip().lower()
                clean_existing_specific_terms.add((normalized_t, normalized_c))

        processed_terms_for_column = False
        for new_term_candidate in all_terms_to_process:
            original_new_term = new_term_candidate.strip()
            normalized_new_term = original_new_term.lower()
            
            # Comprobar si el término ya existe para esta columna específica
            if original_new_term and (normalized_new_term, column_name.strip().lower()) not in clean_existing_specific_terms:
                term_object = {
                    "term": original_new_term,
                    "column": column_name.strip()
                }
                specific_terms_list.append(term_object)
                clean_existing_specific_terms.add((normalized_new_term, column_name.strip().lower()))
                added_any_new_term = True
                processed_terms_for_column = True
        
        if processed_terms_for_column:
            print(f"Términos específicos añadidos/actualizados para la tabla '{table_name}' y columna '{column_name}'.")

    else:
        # Lógica original para common_terms generales (sin especificar columna)
        if 'common_terms' not in table_entry or not isinstance(table_entry['common_terms'], list):
            table_entry['common_terms'] = []
        
        terms_list = table_entry['common_terms']
        
        clean_existing_terms = set()
        for t_entry in terms_list:
            if isinstance(t_entry, str):
                normalized_term = t_entry.replace("- ", "", 1).strip().lower()
                clean_existing_terms.add(normalized_term)

        processed_general_terms = False
        for new_term_candidate in all_terms_to_process:
            original_new_term = new_term_candidate.strip()
            normalized_new_term = original_new_term.lower()

            if original_new_term and normalized_new_term not in clean_existing_terms:
                terms_list.append(f"- {original_new_term}")
                clean_existing_terms.add(normalized_new_term)
                added_any_new_term = True
                processed_general_terms = True
        
        if processed_general_terms:
            print(f"Términos comunes generales añadidos/actualizados para la tabla '{table_name}'.")
            
    return added_any_new_term

def update_concept_entry(data, name, description, tables_str, key_columns_str=None, join_details=None, keywords_str=None, example_questions_str=None, notes=None):
    concept_entry = {
        "description": description,
        "tables_involved": [t.strip() for t in tables_str.split(',') if t.strip()],
        "key_columns": [kc.strip() for kc in key_columns_str.split(',')] if key_columns_str else [],
        "join_details": join_details if join_details else "",
        "keywords": [kw.strip() for kw in keywords_str.split(',')] if keywords_str else [],
        "example_nl_questions": [q.strip() for q in example_questions_str.split(';')] if example_questions_str else [],
        "notes": notes if notes else ""
    }
    
    if 'concepts' not in data: # Asegurado por load_dictionary, pero doble chequeo.
        data['concepts'] = {}
        
    data['concepts'][name] = concept_entry
    print(f"Concepto '{name}' añadido/actualizado en el diccionario.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Actualiza dictionary.json con nuevos mapeos semánticos o conceptos.")
    parser.add_argument('--dict-path', default=DEFAULT_DICT_PATH, help=f'Ruta al archivo dictionary.json (default: {DEFAULT_DICT_PATH})')
    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar', required=True)

    # Subparser para 'add-term'
    parser_add_term = subparsers.add_parser('add-term', help='Añade un término y sus sinónimos a una tabla.')
    parser_add_term.add_argument('--term', required=True, help='Término principal (ej: alergias)')
    parser_add_term.add_argument('--table', required=True, help='Nombre de la tabla (ej: ALLE_INTOLERANCES)')
    parser_add_term.add_argument('--column', required=False, help='Columna relevante (opcional, actualmente no almacenado explícitamente con el término)')
    parser_add_term.add_argument('--synonyms', required=False, help='Sinónimos separados por coma')

    # Subparser para 'add-concept'
    parser_add_concept = subparsers.add_parser('add-concept', help='Añade o actualiza un concepto complejo.')
    parser_add_concept.add_argument('--name', required=True, help='Nombre único del concepto (ej: hospitalizacion_prolongada)')
    parser_add_concept.add_argument('--description', required=True, help='Descripción del concepto.')
    parser_add_concept.add_argument('--tables', required=True, help='Nombres de las tablas involucradas, separados por coma.')
    parser_add_concept.add_argument('--key-columns', required=False, help='Columnas clave (ej: TABLE.COLUMN), separadas por coma.')
    parser_add_concept.add_argument('--join-details', required=False, help='Descripción textual de cómo se unen las tablas.')
    parser_add_concept.add_argument('--keywords', required=False, help='Palabras clave asociadas al concepto, separadas por coma.')
    parser_add_concept.add_argument('--example-questions', required=False, help='Preguntas de ejemplo en lenguaje natural, separadas por ";".')
    parser_add_concept.add_argument('--notes', required=False, help='Notas adicionales.')

    args = parser.parse_args()
    data = load_dictionary(args.dict_path)

    if args.command == 'add-term':
        synonyms_list = [s.strip() for s in args.synonyms.split(',')] if args.synonyms else []
        if not args.term.strip():
            print("Error: El término principal no puede estar vacío.")
            return

        updated = update_table_terms(data, args.table, args.term, synonyms_list, args.column) # Pasar args.column
        if updated:
            save_dictionary(data, args.dict_path)
            # Los mensajes de éxito más específicos se imprimen dentro de update_table_terms
            print(f"Procesamiento de términos para la tabla '{args.table}' completado.")
        else:
            print(f"No se realizaron nuevos añadidos de términos para la tabla '{args.table}' (podrían ya existir o el término estar vacío).")

    elif args.command == 'add-concept':
        if not args.name.strip() or not args.description.strip() or not args.tables.strip():
            print("Error: Nombre, descripción y tablas son campos obligatorios para un concepto y no pueden estar vacíos.")
            return

        updated = update_concept_entry(data, args.name, args.description, args.tables,
                                       args.key_columns, args.join_details, args.keywords,
                                       args.example_questions, args.notes)
        if updated:
            save_dictionary(data, args.dict_path)
            # El mensaje de éxito ya está en update_concept_entry
    
    # Verificar si el diccionario se cargó correctamente al principio
    if data is None : # data podría ser None si load_dictionary falla catastróficamente
        print("Error crítico: No se pudo cargar o inicializar el diccionario. No se guardarán cambios.")


if __name__ == '__main__':
    main()
