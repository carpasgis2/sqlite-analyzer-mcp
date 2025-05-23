import os
import sys

# Añadir el directorio de sqlite-analyzer/src al path para poder importar el módulo
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "sqlite-analyzer", "src")
sys.path.append(src_dir)

# Importar la función de prueba desde el pipeline
try:
    from pipeline import test_llm_connection, setup_logging
    
    # Ejecutar la prueba
    print(f"Importando desde: {src_dir}")
    setup_logging(level="DEBUG")
    test_llm_connection()
except ImportError as e:
    print(f"Error al importar el módulo pipeline: {e}")
    print(f"Ruta de búsqueda: {sys.path}")
    print(f"Archivos en {src_dir}:", os.listdir(src_dir) if os.path.exists(src_dir) else "Directorio no encontrado")
