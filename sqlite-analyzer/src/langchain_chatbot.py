"""
LangChain Tool para usar el pipeline de chatbot existente.
Permite usar el pipeline como herramienta en un agente conversacional LangChain.
"""
import logging # Asegurar que logging se importa al principio
import os
import sys
import re
import json # Añadir importación de json
import difflib # Añadir importación de difflib
import concurrent.futures # Añadir importación de concurrent.futures

from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_deepseek import ChatDeepSeek
from langchain_core.exceptions import OutputParserException

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importa tu pipeline (ajusta el import si es necesario)
# from pipeline import handle_query_with_langchain, handle_condition_medication_doctors_subpipeline
from pipeline import chatbot_pipeline
# Modificación: Importar get_db_connector en lugar de SQLiteConnector
from db_config import get_db_connector, DBConnector # Añadido DBConnector por si se usa para type hints en otro lado

# --- INICIO CONFIGURACIÓN DE LOGGING ---
# Configurar el logger raíz para capturar logs de este script y otros módulos (ej. pipeline)
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Nivel base, puedes cambiar a logging.DEBUG para más detalle

# Definir log_formatter y log_file_path aquí para que estén disponibles globalmente en este módulo
log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] %(message)s"
)
log_file_path = os.path.join(os.path.dirname(__file__), "chatbot_agent.log")

# Evitar añadir múltiples handlers si el script se recarga (común en algunos entornos)
if not logger.handlers:
    # Handler para guardar en archivo
    # Cambiado a mode='w' para que los logs se creen en un archivo nuevo (sobrescribiendo si existe)
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8') 
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    # Handler para mostrar en consola
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
    
    logger.info(f"Logging configurado. Los logs se guardarán en: {log_file_path} (modo overwrite)")
else:
    logger.info(f"El logger ya tiene handlers. Los logs continuarán en: {log_file_path} (modo overwrite)")
# --- FIN CONFIGURACIÓN DE LOGGING ---

# Configuración de la API de Deepseek (preferiblemente desde variables de entorno)
LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-aedf531ee17447aa95c9102e595f29ae")
LLM_API_URL = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
LLM_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
LLM_PROVIDER = "deepseek"

# Instancia el conector de base de datos
# Modificación: Usar get_db_connector()
# La ruta de la base de datos ya está configurada en db_config.py (DEFAULT_DB_CONNECTION_STRING)
db_connector: DBConnector = get_db_connector()

# Herramienta para interactuar con la base de datos médica
def safe_process_results(query: str) -> str:
    """Procesa la consulta de manera segura con mejor manejo de errores y conversión de resultados. Nunca bloquea."""
    import concurrent.futures
    import re # Importar re para expresiones regulares

    try:
        original_query_for_log = query # Save original for logging comparison
        
        # 1. Strip inicial de la consulta de entrada
        stripped_input = query.strip()
        
        # Por defecto, la consulta limpiada es la entrada sin espacios al inicio/final.
        # Esto se usará si no se detecta un bloque de Markdown.
        cleaned_query = stripped_input 
        
        # 2. Patrón para buscar y extraer el contenido DENTRO de un bloque de código Markdown.
        #   ^```             Inicio del bloque de código
        #   (?:sql)?         Etiqueta opcional 'sql' (no capturada)
        #   \\s*              Cualquier espacio en blanco (incluyendo saltos de línea) después de ``` o etiqueta sql
        #   (.*?)            El contenido SQL real (grupo de captura 1, no codicioso)
        #   \\s*              Cualquier espacio en blanco antes del ``` final
        #   ```$             Fin del bloque de código
        # re.DOTALL hace que '.' coincida también con saltos de línea.
        # re.IGNORECASE para la etiqueta opcional 'sql'.
        markdown_pattern = r"^```(?:sql)?\s*(.*?)\s*```$" # Corregido: \\s* a \s*
        
        match = re.search(markdown_pattern, stripped_input, re.DOTALL | re.IGNORECASE)
        
        if match:
            # Se encontró el patrón de Markdown. El grupo 1 es el contenido SQL.
            # Aplicar strip() para eliminar espacios/saltos de línea al inicio/final del SQL extraído.
            extracted_sql = match.group(1).strip()
            if original_query_for_log != extracted_sql : # Solo loguear si hubo un cambio real
                logger.info(f"Markdown detectado y extraído. Query original: '{original_query_for_log}'")
                logger.info(f"Query después de extraer de Markdown y strip(): '{extracted_sql}'")
            else: # La query original era solo el SQL ya limpio pero envuelto en Markdown idéntico
                logger.info(f"Markdown detectado. Query extraída (sin cambios internos): '{extracted_sql}'")
            cleaned_query = extracted_sql
        else:
            # No se encontró el patrón de Markdown.
            # La consulta podría estar ya limpia, o tener un formato inesperado.
            # Si stripped_input es diferente de la query original, significa que strip() hizo algo.
            if original_query_for_log != stripped_input:
                logger.info(f"No se detectó patrón de Markdown. Query original: '{original_query_for_log}'")
                logger.info(f"Query después de strip() inicial (usada como está): '{cleaned_query}'")
            else:
                # La query original ya estaba "limpia" (sin espacios al inicio/final y sin Markdown).
                logger.info(f"No se detectó patrón de Markdown. Query procesada como está (sin cambios por strip inicial): '{cleaned_query}'")
        
        # --- ADVERTENCIA SOBRE FUNCIONES SQL INCOMPATIBLES ---
        # Detectar funciones incompatibles y advertir
        sql_incompat_warnings = []
        if re.search(r"DATEDIFF\s*\(", cleaned_query, re.IGNORECASE):
            sql_incompat_warnings.append("ADVERTENCIA: 'DATEDIFF' no es compatible con SQLite. Usa (julianday(fecha1) - julianday(fecha2)) para diferencias de fechas en días.")
        if re.search(r"CURRENT_DATE", cleaned_query, re.IGNORECASE):
            sql_incompat_warnings.append("ADVERTENCIA: 'CURRENT_DATE' no es compatible con SQLite. Usa date('now') o datetime('now') según el caso.")
        if re.search(r"GETDATE\s*\(", cleaned_query, re.IGNORECASE):
            sql_incompat_warnings.append("ADVERTENCIA: 'GETDATE()' no es compatible con SQLite. Usa date('now') o datetime('now').")
        
        logger.info(f"Procesando consulta: '{cleaned_query}' (con timeout de 90s)")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(chatbot_pipeline, cleaned_query, db_connector, logger) # Usar cleaned_query
            try:
                pipeline_result = future.result(timeout=90)
            except concurrent.futures.TimeoutError:
                logger.error("chatbot_pipeline excedió el tiempo máximo de 90s y fue cancelado")
                return "Error: El procesamiento de la consulta tardó demasiado y fue cancelado. Por favor, intenta con una consulta más simple."
            except Exception as e:
                logger.error(f"Error inesperado ejecutando chatbot_pipeline: {e}", exc_info=True)
                return f"Error inesperado al procesar la consulta: {str(e)}"

        logger.info(f"Pipeline ejecutado. Tipo de resultado: {type(pipeline_result)}")
        if pipeline_result is None:
            logger.error("Pipeline devolvió None")
            return "No pude obtener resultados para esta consulta."
        if not isinstance(pipeline_result, dict):
            logger.error(f"Pipeline devolvió un tipo inesperado: {type(pipeline_result)}")
            return f"Error: Resultado con formato inválido. Por favor reporta este error."
        
        response_message = pipeline_result.get("response", "La consulta se ejecutó pero no pude formatear la respuesta correctamente.")
        data_results = pipeline_result.get("data")

        # --- MANEJO ROBUSTO DE ERRORES DE VALIDACIÓN DE ENTIDADES ---
        if response_message and "La consulta contiene columnas o tablas que no existen" in response_message:
            # Extraer tablas mencionadas en la consulta para sugerir exploración
            tablas_mencionadas = re.findall(r"FROM\s+([A-Z0-9_]+)|JOIN\s+([A-Z0-9_]+)", cleaned_query, re.IGNORECASE)
            tablas_planas = [t for pair in tablas_mencionadas for t in pair if t]
            sugerencia = ("\nSugerencia: Antes de volver a intentar, usa la herramienta 'Listar columnas de una tabla' "
                         "para cada tabla implicada. Ejemplo: 'Listar columnas de una tabla: NOMBRE_TABLA'.\n"
                         "Tablas detectadas en tu consulta: " + ", ".join(set(tablas_planas))) if tablas_planas else (
                         "\nSugerencia: Usa la herramienta de exploración de esquema para ver las columnas reales.")
            response_message += sugerencia

            # --- ADVERTENCIA GENÉRICA PARA CONSULTAS CON SUBCONSULTAS Y FECHAS ---
            if (
                re.search(r"NOT IN|NOT EXISTS|LEFT JOIN", cleaned_query, re.IGNORECASE)
                and re.search(r"\b(VACC_VACCINES|PATI_PATIENTS|_DATE|_ID)\b", cleaned_query, re.IGNORECASE)
                and (re.search(r"'now'", cleaned_query) or re.search(r"\d{4}", cleaned_query))
            ):
                response_message += (
                    "\nADVERTENCIA: Cuando utilices subconsultas (NOT IN, NOT EXISTS) o LEFT JOIN para filtrar entidades por ausencia de registros (por ejemplo, pacientes no vacunados, usuarios sin actividad, etc.):\n"
                    "- Verifica que el año usado en la condición de fecha sea consistente con los datos reales (puede que 'now' no coincida con los datos de prueba).\n"
                    "- Si la subconsulta NOT IN falla, intenta reescribir usando LEFT JOIN ... IS NULL.\n"
                    "- Asegúrate de que los patrones de texto en condiciones LIKE sean suficientemente amplios y correctos.\n"
                    "- Consulta siempre las columnas reales de las tablas implicadas antes de generar la consulta.\n"
                    "- Si el error persiste, simplifica la consulta y prueba los componentes por separado."
                )
                logger.warning("Advertencia genérica para subconsultas/fechas activada.")

            # Nueva verificación de complejidad y advertencia específica para CTEs
            # CORREGIDO: Expresión regular para is_complex_cte
            is_complex_cte = False
            if "WITH " in cleaned_query.upper(): # Comprobación rápida antes de regex más costoso
                is_complex_cte = re.search(r"\bWITH\s+\w+\s+AS\s*\(", cleaned_query, re.IGNORECASE) is not None

            is_other_complex = False
            if any(kwd in cleaned_query.upper() for kwd in ["OVER", "PARTITION", "CASE"]): # Comprobación rápida
                is_other_complex = re.search(r"\bOVER\s*\(|PARTITION\s+BY|CASE\s+WHEN.*THEN.*END", cleaned_query, re.IGNORECASE | re.DOTALL) is not None
            
            if is_complex_cte:
                cte_warning = (
                    "\\n\\nADVERTENCIA CRÍTICA SOBRE CTEs (CLÁUSULA WITH):\\n"
                    "Su consulta utiliza una cláusula WITH (Common Table Expression).\\n"
                    "El validador de SQL actual del sistema tiene serias dificultades con las CTEs y frecuentemente reporta errores incorrectos de 'tabla o columna no existe' incluso cuando la CTE es válida.\\n"
                    "**ACCIÓN REQUERIDA:** Por favor, reescriba la consulta para EVITAR EL USO DE CTEs (WITH).\\n"
                    "Intente usar subconsultas en línea (en el FROM o WHERE), múltiples JOINs, o descomponga la lógica en pasos más simples si es posible, aunque la consulta resultante sea más larga o menos elegante.\\n"
                    "Este es un problema conocido del validador interno."
                )
                response_message += cte_warning
                logger.warning(f"Consulta compleja con WITH detectada que falló la validación. Query: {cleaned_query[:300]}... Se añadió advertencia específica para CTEs.")
            elif is_other_complex: # Solo añadir la advertencia general de complejidad si no es una CTE (ya cubierta)
                complex_query_warning = (
                    "\\n\\nADVERTENCIA ADICIONAL DE COMPLEJIDAD:\\n"
                    "Su consulta utiliza construcciones SQL complejas (como OVER(PARTITION BY), CASE WHEN anidados extensamente, etc.).\\n"
                    "Aunque los nombres de tablas y columnas parezcan correctos, el validador actual podría tener problemas con esta complejidad.\\n"
                    "Intente simplificar la consulta o reestructurarla si el problema persiste a pesar de verificar el esquema."
                )
                response_message += complex_query_warning
                logger.warning(f"Consulta compleja (no CTE) detectada que falló la validación de entidades. Query: {cleaned_query[:300]}... Se añadió advertencia general de complejidad.")

            # Mejora: sugerencias de coincidencias de esquema basadas en palabras clave
            sql_keywords = {'select','from','where','join','on','and','or','distinct','inner','left','right','case','when','then','else','end','as','group','by','order','like','count'}
            palabras = [w for w in re.findall(r"[A-Za-z]{4,}", cleaned_query) if w.lower() not in sql_keywords]
            sugerencias_esquema = []
            for kw in list(set(palabras))[:3]:
                result = search_schema(SCHEMA_PATH, kw)
                if result.get('tables') or result.get('columns'):
                    msg = f"\nPara la palabra '{kw}', encontré en el esquema:"
                    if result.get('tables'):
                        msg += "\n  Tablas: " + ", ".join(result['tables'])
                    if result.get('columns'):
                        cols = ", ".join(f"{c['table']}.{c['column']}" for c in result['columns'])
                        msg += "\n  Columnas: " + cols
                    sugerencias_esquema.append(msg)
            if sugerencias_esquema:
                response_message += "\n" + "\n".join(sugerencias_esquema)

        logger.info(f"Resultado del pipeline procesado: Mensaje='{response_message}', Datos presentes: {data_results is not None}")

        # Formatear los datos si existen
        formatted_data = ""
        MAX_ROWS_FOR_AGENT_OBSERVATION = 10 # Limitar la cantidad de datos crudos en la observación
        if data_results:
            if isinstance(data_results, list):
                num_rows = len(data_results)
                if num_rows > 0:
                    try:
                        # Convertir lista de dicts a una cadena JSON bonita para la respuesta
                        import json
                        # Truncar datos para la observación del agente si son demasiados
                        data_to_format = data_results
                        if num_rows > MAX_ROWS_FOR_AGENT_OBSERVATION:
                            logger.info(f"Truncando datos para la observación del agente de {num_rows} a {MAX_ROWS_FOR_AGENT_OBSERVATION} filas.")
                            data_to_format = data_results[:MAX_ROWS_FOR_AGENT_OBSERVATION]
                        
                        formatted_data = json.dumps(data_to_format, indent=2, ensure_ascii=False)
                        
                    except Exception as e:
                        logger.error(f"Error al formatear los datos a JSON: {e}")
                        formatted_data = str(data_results[:MAX_ROWS_FOR_AGENT_OBSERVATION]) # Fallback a string simple del subconjunto
            elif not isinstance(data_results, list): # Dato único, no lista
                 formatted_data = str(data_results)
            # Si data_results es una lista vacía, formatted_data permanecerá vacío.

        # Combinar el mensaje de respuesta con los datos formateados
        final_response_str = response_message
        if sql_incompat_warnings:
            # Anteponer advertencias de SQL si no están ya en el mensaje principal
            if not all(warn in final_response_str for warn in sql_incompat_warnings):
                 final_response_str = "\\n".join(sql_incompat_warnings) + "\\n" + final_response_str

        if formatted_data:
            # Añadir nota si los datos fueron truncados en la observación
            if isinstance(data_results, list) and len(data_results) > MAX_ROWS_FOR_AGENT_OBSERVATION:
                final_response_str += f"\\\\n(Mostrando primeros {MAX_ROWS_FOR_AGENT_OBSERVATION} registros en esta observación detallada)"
            final_response_str += f"\\nDatos:\\n{formatted_data}"
        else: # No hay formatted_data (data_results es None, lista vacía, o no es una lista y es falsy)
            # Determinar si el mensaje principal ya es un error de validación de entidades
            is_entity_validation_error = "La consulta contiene columnas o tablas que no existen" in response_message

            if not is_entity_validation_error:
                # Si no es un error de validación de entidades, entonces procesar casos de "no resultados"
                if "no devolvió filas" in response_message or "no se encontraron resultados" in response_message:
                    # El mensaje del pipeline ya indica que no hay resultados.
                    # Añadir la sugerencia de términos más generales a final_response_str.
                    final_response_str += "\\nSugerencia: Si no obtuviste resultados, intenta usar términos más generales o utiliza la herramienta 'Buscar en el esquema por palabra clave' con un término amplio, por ejemplo: 'Buscar en el esquema por palabra clave: diabetes'."
                elif response_message: # Hay un mensaje, pero no indica explícitamente "no resultados"
                    final_response_str += "\\nNo se encontraron datos para esta consulta."
                # Si response_message está vacío y no hay datos, no se añade nada extra aquí.
            # else:
                # Si es un error de validación de entidades, final_response_str ya contiene el mensaje de error
                # y las sugerencias relevantes añadidas anteriormente. No se añade "No se encontraron datos".
                pass
        
        # Asegurar que la respuesta final sea un string simple si es necesario
        if not isinstance(final_response_str, (str, int, float)):
            logger.warning(f"La respuesta combinada no es un tipo básico serializable: {type(final_response_str)}")
            try:
                final_response_str = str(final_response_str)
            except Exception as e:
                logger.error(f"Error al convertir final_response_str a string: {e}")
                return "Error al procesar la respuesta final."
                
        # Limitar la longitud total de la observación para evitar errores de contexto del LLM
        MAX_OBSERVATION_LENGTH = 10000  # Ajustar según sea necesario (en caracteres)
        if len(final_response_str) > MAX_OBSERVATION_LENGTH:
            logger.warning(f"La observación final excede los {MAX_OBSERVATION_LENGTH} caracteres ({len(final_response_str)}). Truncando...")
            final_response_str = final_response_str[:MAX_OBSERVATION_LENGTH] + "... (Observación truncada)"

        return final_response_str
    except Exception as e:
        logger.error(f"Error en safe_process_results: {e}", exc_info=True)
        return f"Error al procesar la consulta: {str(e)}"

chatbot_tool = Tool(
    name="SQLMedicalChatbot",
    func=safe_process_results,  # Usar nuestra función robusta
    description="""Responde preguntas específicas sobre la base de datos médica.

**SUGERENCIAS GENERALES PARA CONSULTAS SQL COMPLEJAS:**
- Antes de generar cualquier consulta SQL, utiliza SIEMPRE la herramienta "Listar columnas de una tabla" para cada tabla implicada. No asumas nombres de columnas ni de tablas.
- **Manejo de JOINs:**
  - Cuando necesites unir tablas y no estés seguro de las columnas a usar para el JOIN:
    1. Utiliza la herramienta "Listar columnas de una tabla" para AMBAS tablas.
    2. Busca pares de columnas que puedan servir como clave primaria y clave foránea (por ejemplo, `TABLA_A.XYZ_ID` y `TABLA_B.XYZ_ID`, o `TABLA_A.ID` y `TABLA_B.TABLA_A_ID`).
    3. Si después de listar columnas sigues sin estar seguro, puedes usar "Buscar en el esquema por palabra clave" con los nombres de ambas tablas para ver si hay descripciones o relaciones documentadas que te ayuden a inferir la unión.
  - No inventes condiciones de JOIN; si no estás seguro, investiga el esquema.
- Si tu consulta incluye subconsultas (NOT IN, NOT EXISTS) o negaciones, y falla la validación:
  - Prueba a reescribir la consulta usando LEFT JOIN ... IS NULL como alternativa.
  - Asegúrate de que los patrones de texto en condiciones LIKE sean suficientemente amplios y correctos (considera mayúsculas/minúsculas y variantes).
- Cuando filtres por fechas (años, meses, etc.):
  - Verifica que el año usado en la condición de fecha sea consistente con los datos reales de la base (por ejemplo, si los datos de prueba están en 2025, usa ese año en vez de 'now').
  - Si usas funciones de fecha, asegúrate de que sean compatibles con SQLite (por ejemplo, usa strftime o julianday).
- Si recibes un error de validación de entidades (columnas o tablas no existen) y ya has verificado los nombres:
  - **Alias en WHERE/HAVING**: No uses un alias de columna definido en la cláusula `SELECT` directamente en las cláusulas `WHERE` o `HAVING` de la *misma* consulta. En su lugar, repite la expresión completa o usa una Expresión Común de Tabla (CTE) para definir el alias y luego filtrarlo en una consulta externa.
  - Simplifica la consulta y prueba primero los componentes por separado (por ejemplo, primero filtra por edad, luego por vacunación, etc.).
  - Consulta el esquema y las relaciones entre tablas si tienes dudas.
- Evita bucles de reintentos: si tras dos intentos la consulta sigue fallando, revisa la lógica y simplifica.

**Guía para Consultas sobre Fármacos y Condiciones Médicas:**
1.  **Fármacos**: La tabla principal es `MEDI_MEDICATIONS`. Se puede unir con `MEDI_PHARMA_THERAPEUTIC_GROUPS` mediante `PHTH_ID`.
2.  **Grupos Terapéuticos**: `MEDI_PHARMA_THERAPEUTIC_GROUPS.PHTH_DESCRIPTION_ES` describe la **clasificación farmacológica** del grupo (ej: 'Inhibidores de la enzima convertidora de angiotensina'). **No uses esta columna para buscar directamente nombres de enfermedades o condiciones médicas.**
3.  **Condiciones Médicas**: Para buscar información relacionada con una condición médica específica (ej: 'hipertensión', 'diabetes'):
    a.  Identifica la condición en tablas de diagnóstico (como `EPIS_DIAGNOSTICS` - puede requerir unir con tablas de códigos como `CODR_DIAGNOSTIC_GROUPS` para obtener descripciones) o en tablas específicas de tipos de condición (ej: `HAHT_HYPERTENSION_TYPES.HAHT_DESCRIPTION_ES` para hipertensión, `HADT_DIABETES_TYPES.HADT_DESCRIPTION_ES` para diabetes).
    b.  Una vez identificada la condición o su tipo, si la pregunta es sobre fármacos para esa condición, necesitarás inferir o conocer los grupos terapéuticos (`PHTH_ID`)
 relevantes para tratarla. (El LLM debe usar su conocimiento general aquí para identificar qué tipo de grupos farmacológicos tratan una condición dada).
    c.  Luego, usa esos `PHTH_ID` (o sus descripciones de clasificación farmacológica en `MEDI_PHARMA_THERAPEUTIC_GROUPS.PHTH_DESCRIPTION_ES`) para buscar en `MEDI_MEDICATIONS`.

**Ejemplo de cómo NO hacer para 'fármacos para hipertensión':**
   NO HAGAS: `SELECT ... FROM MEDI_MEDICATIONS m JOIN MEDI_PHARMA_THERAPEUTIC_GROUPS g ON m.PHTH_ID = g.PHTH_ID WHERE g.PHTH_DESCRIPTION_ES LIKE '%hipertensión%'`
   Esto es incorrecto porque `PHTH_DESCRIPTION_ES` es una clasificación farmacológica, no una lista de enfermedades que trata.

**Sintaxis SQL Importante (Especialmente para Cláusulas `LIKE` y funciones de fecha):**
-   **Literales de Cadena**: Siempre encierra los literales de cadena entre comillas simples (ej: `'valor_texto'`).
-   **Cláusula `LIKE`**: Asegúrate de que los patrones para `LIKE` estén correctamente formados. Por ejemplo: `nombre_columna LIKE '%texto_busqueda%'`.
-   **Fechas y diferencias de fechas**: NO uses `DATEDIFF`, `CURRENT_DATE` ni `GETDATE()`. En SQLite, usa `julianday()` y `date('now')` o `datetime('now')`.

**Ejemplo de cómo PENSAR y CONSTRUIR la consulta para 'fármacos para hipertensión':**
   1. Pensamiento Inicial: 'Hipertensión' es una condición médica. Necesito fármacos para ella.
   2. Identificar Grupos Farmacológicos Relevantes: ¿Qué *tipos* de fármacos tratan la hipertensión? (ej: 'antihipertensivos', 'betabloqueantes', 'IECA', 'ARA II', 'diuréticos', etc.). Estos son los términos que SÍ puedo buscar en `MEDI_PHARMA_THERAPEUTIC_GROUPS.PHTH_DESCRIPTION_ES`.
   3. Construir la Consulta SQL:
      `SELECT T1.MEDI_DESCRIPTION_ES FROM MEDI_MEDICATIONS AS T1 INNER JOIN MEDI_PHARMA_THERAPEUTIC_GROUPS AS T2 ON T1.PHTH_ID = T2.PHTH_ID WHERE T2.PHTH_DESCRIPTION_ES LIKE '%antihipertensivo%' OR T2.PHTH_DESCRIPTION_ES LIKE '%IECA%' OR T2.PHTH_DESCRIPTION_ES LIKE '%betabloqueante%'` (añadir otros grupos relevantes según sea necesario).

Usa esta herramienta para cualquier pregunta que implique obtener datos directamente de la base de datos.
Si necesitas traducir un ID a una descripción, formula una pregunta clara para obtener esa descripción de su tabla respectiva."""
)

# --- Herramienta para información general y SinaSuite ---
def fetch_sinasuite_info(question: str) -> str:
    """
    Busca información sobre SinaSuite, la función del chatbot o responde a saludos y preguntas generales.
    Utiliza esta herramienta para preguntas que no parezcan estar relacionadas con la extracción de datos
    específicos de la base de datos médica.
    """
    question_lower = question.lower()
    
    saludos_keywords = ["hola", "buenos días", "buenas tardes", "buenas noches", "qué tal", "hey"]
    sinasuite_keywords = [
        "sinasuite", "qué es sinasuite", "que es sinasuite",
        "cuál es tu función", "cual es tu funcion", "qué haces", "que haces",
        "quién eres", "quien eres", "para qué sirves", "para que sirves",
        "ayuda", "info", "informacion"
    ]

    if any(saludo in question_lower for saludo in saludos_keywords):
        return "¡Hola! Soy un asistente virtual. Puedo ayudarte con consultas sobre la base de datos médica o proporcionarte información general sobre SinaSuite. ¿En qué puedo ayudarte hoy?"

    if any(keyword in question_lower for keyword in sinasuite_keywords):
        return "Soy un asistente virtual con dos funciones principales: 1) Ayudarte a consultar información específica de la base de datos médica. 2) Proporcionarte información general sobre SinaSuite, que es una plataforma integral para la gestión de datos médicos. Para más detalles sobre SinaSuite, puedes visitar https://www.sinasuite.com/."

    # Si el agente eligió esta herramienta pero no es un saludo ni sobre SinaSuite,
    # podría ser un error del agente o una pregunta muy general.
    return "Puedo ayudarte a consultar la base de datos médica o darte información sobre SinaSuite. ¿Tienes alguna pregunta específica sobre estos temas?"

sinasuite_tool = Tool(
    name="SinaSuiteAndGeneralInformation",  # Nombre sin espacios
    func=fetch_sinasuite_info,
    description="""Útil para responder a saludos, preguntas generales sobre la función del chatbot, o consultas sobre SinaSuite.
No uses esta herramienta para consultas que requieran acceder o buscar datos en la base de datos médica.
Ejemplos de cuándo usarla: 'Hola', '¿Qué es SinaSuite?', '¿Quién eres?', 'Ayuda'."""
)

# --- TOOLS DE ESQUEMA ---
from src.sql_utils import list_tables, list_columns, search_schema
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "data", "dictionary.json")

def tool_list_tables(_:str=None) -> str:
    """Devuelve la lista de tablas reales del esquema."""
    tablas = list_tables(SCHEMA_PATH)
    return "Tablas disponibles:\n" + "\n".join(tablas)

def tool_list_columns(table:str) -> str:
    """Devuelve las columnas reales de una tabla. Si no se encuentran, da un mensaje informativo y sugiere nombres similares."""
    cols = list_columns(SCHEMA_PATH, table)
    # Log temporal para depuración
    logger.info(f"[DEBUG] tool_list_columns: tabla solicitada='{table}', columnas encontradas={cols}")
    if cols:
        return f"Columnas de {table}:\n" + "\n".join(cols)
    # Si no hay columnas, comprobar si la tabla existe realmente
    tablas = list_tables(SCHEMA_PATH)
    tablas_lower = [t.lower() for t in tablas]
    if table.lower() in tablas_lower:
        return (f"La tabla '{table}' existe pero no se encontraron columnas en el esquema. "
                f"Puede ser un problema de formato o de nombres. Por favor, revisa el esquema o pide ayuda.\n"
                f"ADVERTENCIA: Si tu consulta SQL falla, revisa que los nombres de columnas y tablas sean exactos y que las funciones SQL sean compatibles con SQLite. "
                f"Por ejemplo, usa 'julianday' en vez de 'DATEDIFF', y 'date('now')' en vez de 'CURRENT_DATE'.")
    # Sugerir nombres similares si la tabla no existe
    import difflib
    sugerencias = difflib.get_close_matches(table, tablas, n=3, cutoff=0.6)
    if sugerencias:
        return f"No se encontraron columnas para la tabla '{table}'. ¿Quizás quisiste decir: {', '.join(sugerencias)}?"
    return (f"No se encontraron columnas para la tabla '{table}'. "
            f"ADVERTENCIA: Si tu consulta SQL falla, revisa que los nombres de columnas y tablas sean exactos y que las funciones SQL sean compatibles con SQLite. "
            f"Por ejemplo, usa 'julianday' en vez de 'DATEDIFF', y 'date('now')' en vez de 'CURRENT_DATE'.")

def tool_search_schema(keyword:str) -> str:
    """Busca tablas y columnas por palabra clave en el esquema."""
    result = search_schema(SCHEMA_PATH, keyword)
    out = []
    if result.get("tables"):
        out.append("Tablas que coinciden con la palabra clave:" + ", ".join(result["tables"]))
    if result.get("columns"):
        column_details = [f"{col['table']}.{col['column']}" for col in result["columns"]]
        out.append("Columnas que coinciden con la palabra clave (tabla.columna):" + ", ".join(column_details))
    
    if not out:
        return f"No se encontraron tablas ni columnas que coincidan con la palabra clave '{keyword}' en el esquema."
    return "\\n".join(out)

def tool_list_all_columns(_:str=None) -> str:
    """Devuelve todas las columnas de todas las tablas del esquema."""
    tablas = list_tables(SCHEMA_PATH)
    out = []
    if not tablas:
        return "No se encontraron tablas en el esquema."
    for t in tablas:
        cols = list_columns(SCHEMA_PATH, t)
        if cols:
            out.append(f"Tabla: {t}\\n  Columnas: {', '.join(cols)}")
        else:
            out.append(f"Tabla: {t}\\n  Columnas: (No se encontraron columnas o la tabla está vacía en el esquema)")
    return "\\n".join(out)

# Herramienta sub-pipeline: Médicos por condición y medicación
# def tool_condition_medication_doctors(input: str) -> str:
#     """
#     Devuelve los IDs de los médicos que trataron pacientes con una condición y medicación dadas.
#     Formato de entrada: 'condición, medicación'. Ejemplo: 'diabetes tipo 2, metformina'.
#     """
#     try:
#         condition, medication = [s.strip() for s in input.split(',', 1)]
#     except ValueError:
#         return "Error: Debe proporcionar 'condición, medicación'. Ejemplo: 'diabetes tipo 2, metformina'."
#     # Construir estructura de la base de datos a partir del esquema JSON
#     tablas = list_tables(SCHEMA_PATH)
#     db_structure = {t: list_columns(SCHEMA_PATH, t) for t in tablas}
#     # Llamar al sub-pipeline
#     result = (condition, medication, db_structure, db_connector, logger)
#     return result.get('response', 'No se obtuvo respuesta del sub-pipeline.')

# Registrar tools en LangChain si se usa agente
TOOLS_SCHEMA = [
    Tool(
        name="Listar tablas del esquema",
        func=tool_list_tables,
        description="Devuelve la lista de tablas reales del esquema de la base de datos médica. Útil para evitar inventar tablas.\n**INSTRUCCIÓN:** Antes de generar SQL, consulta aquí las tablas reales."
    ),
    Tool(
        name="Listar columnas de una tabla",
        func=tool_list_columns,
        description="Devuelve la lista de columnas reales de una tabla. Argumento: nombre de la tabla.\n**INSTRUCCIÓN:** SIEMPRE usa esta herramienta antes de generar SQL para evitar inventar columnas."
    ),
    Tool(
        name="Buscar en el esquema por palabra clave",
        func=tool_search_schema,
        description="Busca tablas y columnas que contengan una palabra clave en su nombre o descripción. Argumento: palabra clave.\n**INSTRUCCIÓN:** Úsala para explorar el esquema antes de escribir SQL."
    ),
    Tool(
        name="Listar todas las columnas del esquema",
        func=tool_list_all_columns,
        description="Devuelve todas las columnas de todas las tablas del esquema. Útil para depuración y para explorar el esquema completo.\n**INSTRUCCIÓN:** Úsala si tienes dudas sobre el esquema."
    
    ),
    # Nueva herramienta para sub-pipeline médicos
    # Tool(
    #     name="Médicos por condición y medicación",
    #     func=tool_condition_medication_doctors,
    #     description="""Devuelve los IDs de los médicos que trataron pacientes con una condición y medicación dadas.\nFormatea el argumento como 'condición, medicación'. Ejemplo: 'diabetes tipo 2, metformina'.\nÚsala para preguntas como '¿Qué médicos han tratado a pacientes con 'diabetes tipo 2' que también están tomando 'metformina'?'."""
    # )
]

def custom_handle_parsing_errors(error: OutputParserException) -> str:
    """
    Genera un mensaje de error personalizado y prescriptivo cuando el LLM no sigue el formato ReAct.
    Intenta extraer la salida problemática del LLM para incluirla en el mensaje de corrección.
    """
    response_str = str(error) # Mensaje completo de la excepción

    # Intentar extraer la salida real del LLM que causó el problema
    problematic_output = getattr(error, 'llm_output', None)
    if problematic_output is None:
        # Intentar parsear desde el string de la excepción
        prefixes_to_check = [
            "Parsing LLM output produced both a final answer and a parse-able action:: ",
            "Could not parse LLM output: ",
            "Invalid Format: ",
            "Invalid tool `",
        ]
        parsed_from_str = False
        for prefix in prefixes_to_check:
            if response_str.startswith(prefix):
                if prefix == "Invalid tool `":
                    end_of_tool_name = response_str.find("`", len(prefix))
                    if end_of_tool_name != -1:
                        problematic_output = response_str[len(prefix):end_of_tool_name]
                        parsed_from_str = True
                        break
                else:
                    problematic_output = response_str[len(prefix):]
                    parsed_from_str = True
                    break
        if not parsed_from_str:
            problematic_output = response_str
    else:
        problematic_output = str(problematic_output)

    # Loguear la salida problemática específica que se enviará al LLM para corrección
    logging.error(f"Salida problemática del LLM (para corrección):\\n---\\n{problematic_output}\\n---")

    # Mensaje prescriptivo para el LLM
    # (El resto de la función que construye el mensaje de error para el LLM permanece igual)
    # ...
    # Asegurarse de que el logger de root también capture la salida problemática real que se envía al LLM
    # (el logger actual en el manejador de errores del agente ya lo hace si error.llm_output está poblado)

    # El mensaje que se devuelve al LLM para que lo corrija:
    # (Este es el formato que ya tenías y es bueno, solo nos aseguramos que `problematic_output` sea más preciso)
    # (El código original para construir el mensaje prescriptivo sigue aquí)
    # ... (resto del código de la función) ...
    # Por ejemplo:
    error_message_template = (
        "CRITICAL ERROR: Your response was not in the correct ReAct format. "
        "You MUST respond with either a valid 'Action:' line followed by an 'Action Input:' line, "
        "OR a 'Final Answer:' line. "
        "DO NOT provide explanations or conversational text outside of the 'Thought:' field. "
        "The available tools are: {tool_names}. "
        "Ensure your Action is one of these tools if you are using an action. "
        "Your problematic output was:\\n'''{problematic_llm_output}'''\\n"
        "Correct your response to strictly follow the ReAct format (Thought, Action, Action Input, or Final Answer)."
    )
    # Obtener nombres de herramientas (asumiendo que están disponibles en algún contexto o globalmente)
    # Esto es solo un ejemplo, necesitarías acceso a `self.tools` o similar si esto está en una clase.
    # Si es una función global, los nombres de las herramientas tendrían que pasarse o ser accesibles.
    # Por ahora, lo omito para mantener el cambio enfocado en la extracción de `problematic_output`.
    # tool_names_str = ", ".join([tool.name for tool in self.tools]) if hasattr(self, 'tools') else "Not available here"
    
    # Para este ejemplo, usaré un placeholder para tool_names
    tool_names_str = "[SQLMedicalChatbot, SinaSuiteAndGeneralInformation]" # Placeholder

    # Re-loguear la salida problemática que se usará en el prompt de corrección
    # logging.error(f"Salida problemática del LLM (para corrección del LLM):\\n---\\n{problematic_output}\\n---")
    # Este logging ya se hizo arriba.

    return error_message_template.format(
        tool_names=tool_names_str,
        problematic_llm_output=problematic_output
    )

def get_langchain_agent():
    """Inicializa y devuelve el agente LangChain configurado."""
    # Memoria conversacional
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # LLM de Deepseek
    llm = ChatDeepSeek(
        api_key=LLM_API_KEY,
        base_url=LLM_API_URL,
        model=LLM_MODEL
    )

    # Inicializa el agente con todas las herramientas, incluyendo las de esquema
    agent = initialize_agent(
        tools=[chatbot_tool, sinasuite_tool] + TOOLS_SCHEMA,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        memory=memory,
        verbose=False,
        handle_parsing_errors=custom_handle_parsing_errors,
        max_iterations=50,
        max_execution_time=300          # <-- Aumentado de 120 a 300 segundos
    )
    return agent

def main():
    # Definir códigos de color ANSI
    GREEN = "\033[92m"
    RESET = "\033[0m"

    logger.info("Iniciando Chatbot médico SQL (LangChain) - Modo Agente")
    logger.info("Escribe 'salir' para terminar.")
    # Se mantienen los prints para la consola con colores, pero la info principal va al log.
    print(f"{GREEN}Chatbot médico SQL (LangChain) - Modo Agente{RESET}")
    print(f"{GREEN}Escribe 'salir' para terminar.{RESET}")

    cli_agent = get_langchain_agent()  # Inicializar el agente para la CLI

    while True:
        question = ""  # Inicializar question
        try:
            question = input(f"{GREEN}Usuario: {RESET}").strip()
            if question.lower() in ("salir", "exit", "quit"):
                logger.info("Bot: ¡Hasta luego!")
                print(f"{GREEN}Bot: {RESET}¡Hasta luego!")
                break
            if not question:
                continue
            
            logger.info(f"Usuario pregunta: {question}")

            # El agente decidirá qué herramienta usar.
            agent_response = cli_agent.invoke({"input": question})
            
            # La estructura de agent_response puede variar, pero comúnmente es un diccionario
            # con una clave 'output' para la respuesta final.
            bot_response = agent_response.get("output", "No pude obtener una respuesta clara del agente.")
            logger.info(f"Respuesta del agente: {bot_response}")

        except OutputParserException as ope:
            logger.error(f"Error de parseo NO RECUPERADO por el agente tras el intento de corrección: {ope}", exc_info=True)
            bot_response = "Lo siento, tuve problemas para entender la estructura de la respuesta interna después de intentar corregirla. Por favor, intenta reformular tu pregunta."
        except Exception as e:
            logger.error(f"Error al invocar el agente para la pregunta '{question}': {e}", exc_info=True)
            bot_response = "Lo siento, ocurrió un error inesperado al procesar tu pregunta. Por favor, intenta reformularla."
            
        print(f"{GREEN}Bot: {RESET}{bot_response}")

if __name__ == "__main__":
    main()
