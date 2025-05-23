def print_results(title, results=None):
    """
    Imprime los resultados del análisis de forma organizada.
    
    Args:
        title (str): Título o descripción de los resultados
        results (dict, optional): Resultados a imprimir. Si es None, solo se imprime el título.
    """
    print("\n" + "="*50)
    print(f"{title}")
    print("="*50)
    
    if results is not None:
        if isinstance(results, dict):
            for key, value in results.items():
                if isinstance(value, dict):
                    print(f"\n{key}:")
                    for sub_key, sub_value in value.items():
                        print(f"  {sub_key}: {sub_value}")
                else:
                    print(f"{key}: {value}")
        else:
            print(results)
    
    print("\n")

def format_output(data):
    return "\n".join(str(item) for item in data)