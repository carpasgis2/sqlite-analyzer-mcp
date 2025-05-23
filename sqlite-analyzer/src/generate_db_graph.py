import sqlite3
import os
import argparse
from typing import Dict, Any, List

def extract_db_structure_for_graph(db_path: str) -> Dict[str, Any]:
    """
    Extrae la estructura de la base de datos SQLite, enfocándose en tablas y claves foráneas.

    Args:
        db_path: Ruta al archivo de la base de datos SQLite.    
    
    Returns:
        Un diccionario con la estructura de la base de datos.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"No se encontró la base de datos: {db_path}")

    # print(f"Conectando a la base de datos: {db_path}")
    structure = {}
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Obtener todas las tablas
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall() if not row[0].startswith('sqlite_')] # Excluir tablas internas de SQLite
        
        for table_name in tables:
            table_info_cursor = conn.cursor()
            # Obtener claves foráneas
            # Corregido el escape de comillas para PRAGMA
            table_info_cursor.execute(f'PRAGMA foreign_key_list("{table_name}");') 
            fkeys_raw = table_info_cursor.fetchall()
            
            foreign_keys = []
            if fkeys_raw:
                for fk_row in fkeys_raw:
                    # fk_row: (id, seq, table, from, to, on_update, on_delete, match)
                    foreign_keys.append({
                        "column": fk_row[3],
                        "referenced_table": fk_row[2],
                        "referenced_column": fk_row[4]
                    })
            
            structure[table_name] = {"foreign_keys": foreign_keys}
            
        conn.close()
        # print(f"Estructura extraída exitosamente: {len(tables)} tablas procesadas para relaciones.")
        return structure
            
    except sqlite3.Error as e:
        print(f"Error de SQLite al extraer estructura para el grafo: {e}")
        raise

def generate_dot_representation(db_structure: Dict[str, Any]) -> str:
    """
    Genera una representación en formato DOT del grafo de relaciones de la base de datos.
    """
    dot_lines = [
        "digraph DatabaseRelations {",
        "    rankdir=LR; // Dirección del grafo (Left to Right), puedes probar TB (Top to Bottom)",
        "    overlap=false; // Intenta evitar solapamiento de nodos",
        "    splines=true; // Usa splines para las aristas, puedes probar 'ortho' o 'polyline'",
        # Corregido el escape de comillas en las definiciones de estilo de nodo y arista
        "    node [shape=record, style=\"rounded,filled\", fillcolor=\"lightblue\", fontname=\"Arial\", fontsize=10];",
        "    edge [fontname=\"Arial\", fontsize=8, color=\"#555555\", arrowsize=0.8];",
        "    graph [fontname=\"Arial\", fontsize=12, label=\"Diagrama de Relaciones de la Base de Datos\", labelloc=t];"
    ]
    
    all_tables = set(db_structure.keys())
    # Asegurarse de que todas las tablas referenciadas también se listen, incluso si no tienen FKs salientes
    for table_info in db_structure.values():
        if "foreign_keys" in table_info:
            for fk in table_info["foreign_keys"]:
                all_tables.add(fk["referenced_table"])

    for table_name in sorted(list(all_tables)): # Ordenar para una salida consistente
        # Corregido el escape de comillas para la etiqueta del nodo
        dot_lines.append(f'    "{table_name}" [label="{{ {table_name} }}"];') 

    processed_relations = set()

    for table_name, table_info in db_structure.items():
        if "foreign_keys" in table_info:
            for fk in table_info["foreign_keys"]:
                from_table = table_name
                to_table = fk["referenced_table"]
                
                relation_tuple = tuple(sorted((from_table, to_table))) + (fk['column'], fk['referenced_column'])

                if relation_tuple not in processed_relations:
                    label = f"{fk['column']} → {fk['referenced_column']}"
                    # Corregido el escape de comillas para la etiqueta de la arista
                    dot_lines.append(f'    "{from_table}" -> "{to_table}" [label=" {label} "];')
                    processed_relations.add(relation_tuple)
    
    dot_lines.append("}")
    return "\n".join(dot_lines)

def main():
    parser = argparse.ArgumentParser(description="Genera un archivo .dot para visualizar relaciones de una base de datos SQLite.")
    parser.add_argument("db_path", help="Ruta al archivo de la base de datos SQLite.")
    parser.add_argument("output_dot_path", help="Ruta donde guardar el archivo .dot generado.")
    args = parser.parse_args()

    try:
        output_dir = os.path.dirname(args.output_dot_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        structure = extract_db_structure_for_graph(args.db_path)
        dot_string = generate_dot_representation(structure)

        with open(args.output_dot_path, 'w', encoding='utf-8') as f:
            f.write(dot_string)
        
        print(f"Archivo .dot generado exitosamente en: {args.output_dot_path}")
        print("Para visualizar este archivo, puedes usar Graphviz. Por ejemplo, para generar una imagen PNG:")
        # Corregido el escape de comillas para los comandos de ejemplo de Graphviz
        print(f'  dot -Tpng "{args.output_dot_path}" -o "{os.path.splitext(args.output_dot_path)[0]}.png"')
        print("O para SVG (recomendado para calidad vectorial):")
        print(f'  dot -Tsvg "{args.output_dot_path}" -o "{os.path.splitext(args.output_dot_path)[0]}.svg"')

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except sqlite3.Error as e:
        print(f"Error de base de datos: {e}")
    except IOError as e:
        print(f"Error de E/S al escribir el archivo .dot: {e}")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    main()
