"""
Script para actualizar el archivo dictionary.json con nuevos mapeos semánticos de términos a tablas y columnas.
Uso:
    python update_dictionary.py --term "alergias" --table "ALLE_INTOLERANCES" --column "ALIN_DESCRIPTION_ES" --synonyms "alergias,reacciones adversas,hipersensibilidades"

- El script añade el término y sinónimos a la lista de common_terms de la tabla indicada.
- Si la tabla no existe, muestra un error.
- Si el término ya existe, no lo duplica.
"""
import argparse
import json
import os

DICT_PATH = os.path.join(os.path.dirname(__file__), 'dictionary.json')

def load_dictionary():
    with open(DICT_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_dictionary(data):
    with open(DICT_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def update_table_terms(data, table, term, synonyms=None):
    if table not in data['tables']:
        print(f"Error: La tabla '{table}' no existe en el diccionario.")
        return False
    terms = data['tables'][table].get('common_terms', [])
    # Limpiar términos de formato y duplicados
    clean_terms = set(t.strip('- ').lower() for t in terms if isinstance(t, str))
    added = False
    for t in [term] + (synonyms or []):
        t_clean = t.strip().lower()
        if t_clean not in clean_terms:
            terms.append(f"- {t.strip()}")
            clean_terms.add(t_clean)
            added = True
    data['tables'][table]['common_terms'] = terms
    return added

def main():
    parser = argparse.ArgumentParser(description="Actualiza dictionary.json con nuevos mapeos semánticos.")
    parser.add_argument('--term', required=True, help='Término principal (ej: alergias)')
    parser.add_argument('--table', required=True, help='Nombre de la tabla (ej: ALLE_INTOLERANCES)')
    parser.add_argument('--column', required=False, help='Columna relevante (opcional)')
    parser.add_argument('--synonyms', required=False, help='Sinónimos separados por coma')
    args = parser.parse_args()

    synonyms = [s.strip() for s in args.synonyms.split(',')] if args.synonyms else []
    data = load_dictionary()
    updated = update_table_terms(data, args.table, args.term, synonyms)
    if updated:
        save_dictionary(data)
        print(f"Término(s) añadido(s) a la tabla '{args.table}'.")
    else:
        print("No se realizaron cambios (término(s) ya presente(s) o tabla inexistente).")

if __name__ == '__main__':
    main()
