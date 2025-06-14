import os
import sys
import json
import time
import re
import logging
import requests # Necesario para BiomedicalAssistant y otras funciones
from typing import List, Dict, Any
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from langchain.schema.messages import BaseMessage

# Clase para colores ANSI en la consola
class Colors:
    RESET = '\033[0m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'

# Langchain imports
from langchain_openai import ChatOpenAI # MODIFICADO: Usar ChatOpenAI para modelos de chat
from langchain.chains import LLMChain, SequentialChain, TransformChain # Asegúrate que TransformChain esté importado si lo usas
from langchain.prompts import PromptTemplate
from langchain.tools import Tool

# Imports desde biochat_2_0.py (adaptados)
from metapub import PubMedFetcher
try:
    from deep_translator import GoogleTranslator
except ImportError:
    raise ImportError("Debes instalar deep-translator: pip install deep-translator")

# --- Configuración Inicial (similar a biochat_2_0.py) ---
os.environ['TORCH_DONT_CHECK_CUSTOM_CLASSES'] = '1'
if "NCBI_API_KEY" not in os.environ:
    os.environ["NCBI_API_KEY"] = "c404a87e96b8328d6bb5d34da565c1c7a308" # Reemplaza con tu API key

# Configuración de API Key para OpenAI (usada por Langchain)
# Tomada de la configuración de BiomedicalAssistant en el archivo original

fetch = PubMedFetcher()
translator = GoogleTranslator(source='es', target='en')

# --- ClinicalLogger (de biochat_2_0.py) ---
class ClinicalLogger:
    """Logger centralizado para el sistema biomédico"""
    def __init__(self, log_file="biomedical_chatbot_agent.log"):
        self.logger = logging.getLogger("BiomedicalChatbotAgent")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        
        # Crear directorio de logs si no existe
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir)
            except OSError as e:
                print(f"Error creando directorio de logs {log_dir}: {e}")

        # Eliminar handlers previos para evitar duplicados
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Añadir solo un handler de archivo y uno de consola
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, msg: str):
        self.logger.info(msg)

    def error(self, msg: str):
        self.logger.error(msg)

logger = ClinicalLogger(log_file="logs/biochat_agent.log")


# --- BiomedicalAssistant (simplificado, para funciones específicas si es necesario) ---
# Langchain's OpenAI LLM será el principal, pero BiomedicalAssistant puede ser útil
# para la lógica de extracción de keywords o si se necesita su send_chat específico.
class BiomedicalAssistant:
    def __init__(self, log_file="biomedical_chatbot_assistant.log"):
        self.logger = ClinicalLogger(log_file) # Logger separado para el assistant
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.model_name = "gpt-3.5-turbo" # Modelo para tareas auxiliares

    def send_chat(self, prompt: str, temperature=0.0, max_tokens=150) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "Eres un asistente biomédico experto."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(self.base_url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error en API OpenAI (BiomedicalAssistant): {e}")
            return f"Error en API: {e}"
        except (KeyError, IndexError) as e:
            self.logger.error(f"Error procesando respuesta de API (BiomedicalAssistant): {e}")
            return "Error procesando respuesta."

assistant_for_keywords = BiomedicalAssistant(log_file="logs/keywords_assistant.log")

# --- Funciones de biochat_2_0.py necesarias para PubMed Tool ---
def extract_keywords_with_llm(query: str, biomedical_assistant: BiomedicalAssistant) -> List[str]:
    """Extrae palabras clave usando el LLM (adaptado de biochat_2_0.py)."""
    if not query: return ["medicine"]
    clean_query = query.replace("¿", "").replace("?", "").strip()
    prompt = f"""
TAREA: Extrae términos médicos clave de la siguiente consulta en español.
INSTRUCCIONES:
1. Identifica conceptos médicos: enfermedades, medicamentos, síntomas, órganos, procedimientos.
2. Mantén términos compuestos (ej. "hipertensión arterial").
3. Para medicamentos, usa denominación USAN/FDA (ej. "acetaminophen", no "paracetamol").
4. Devuelve ÚNICAMENTE los términos separados por comas. Sin explicaciones.
EJEMPLO: hypertension, acetaminophen, diabetes type 2
Consulta: {clean_query}
Términos médicos clave:"""
    try:
        response = biomedical_assistant.send_chat(prompt, temperature=0.0, max_tokens=100)
        if response and "Error en API" not in response and "Error procesando respuesta" not in response:
            # Limpieza básica de la respuesta del LLM
            response = response.split('\n')[0] # Tomar solo la primera línea
            raw_keywords = [kw.strip() for kw in response.split(',') if kw.strip()]
            logger.info(f"Keywords extraídas por LLM: {raw_keywords}")
            return raw_keywords if raw_keywords else ["medicine", "health"]
    except Exception as e:
        logger.error(f"Error en extract_keywords_with_llm: {e}")
    return ["medicine", "health"] # Fallback

def translate_keywords(keywords: List[str]) -> List[str]:
    """Traduce keywords al inglés (adaptado de biochat_2_0.py)."""
    if not keywords: return ["disease"]
    translated_keywords = []
    for kw in keywords:
        try:
            # Limitar longitud de keyword para evitar errores de traducción
            if len(kw) > 100: kw = kw[:100]
            translated = translator.translate(kw)
            if translated:
                translated_keywords.append(translated.strip())
        except Exception as e:
            logger.error(f"Error traduciendo keyword '{kw}': {e}")
            translated_keywords.append(kw) # Mantener original si falla
    logger.info(f"Keywords traducidas: {translated_keywords}")
    return translated_keywords if translated_keywords else ["disease"]

def find_pubmed_tiab(keywords: List[str], num_of_articles=10) -> List[Dict[str, str]]:
    """Busca en PubMed usando [tiab] (adaptado de biochat_2_0.py)."""
    if not keywords: return []
    
    # Construir query avanzada
    # Prioridad a términos compuestos si existen, uniéndolos con AND
    # Luego OR con términos individuales
    query_parts = []
    for kw in keywords:
        if " " in kw: # Término compuesto
            query_parts.append(f"(\"{kw}\")[tiab]") # Comillas para frase exacta
        else:
            query_parts.append(f"({kw})[tiab]")
    
    if not query_parts: return []

    # Estrategia de consulta:
    # 1. Todos los términos con AND (si son pocos)
    # 2. Los primeros N con AND, el resto con OR
    # 3. Todos con OR como fallback
    query_str = ""
    if len(query_parts) <= 3:
        query_str = " AND ".join(query_parts)
    else:
        query_str = f"({' AND '.join(query_parts[:2])}) AND ({' OR '.join(query_parts[2:])})"

    logger.info(f"PubMed Query (Estrategia 1): {query_str}")
    pmids = fetch.pmids_for_query(query_str, retmax=num_of_articles)

    if not pmids and len(query_parts) > 1: # Fallback a OR si la primera falla
        query_str_or = " OR ".join(query_parts)
        logger.info(f"PubMed Query (Estrategia 2 - OR): {query_str_or}")
        pmids = fetch.pmids_for_query(query_str_or, retmax=num_of_articles)
        
    if not pmids:
        logger.warning("No se encontraron artículos en PubMed con las estrategias aplicadas.")
        return []

    logger.info(f"PMIDs encontrados: {pmids}")
    articles_data = []
    for pmid in pmids:
        try:
            article = fetch.article_by_pmid(pmid)
            # Verificar que el artículo tiene título y abstract
            if article and article.title and article.abstract:
                articles_data.append({
                    "id": pmid,
                    "title": article.title,
                    "abstract": article.abstract,
                    "year": article.year or "N/A",
                    "journal": article.journal or "N/A",
                    "authors": article.authors_str or "N/A"
                })
        except Exception as e:
            logger.error(f"Error procesando PMID {pmid}: {e}")
    
    if articles_data:
        logger.info(f"Datos de artículos recuperados de PubMed ({len(articles_data)} artículos):")
        for i, art_data in enumerate(articles_data):
            logger.info(f"  Artículo {i+1} (PMID: {art_data['id']}): Título: {art_data['title'][:100]}... Abstract: {art_data['abstract'][:150]}...") # Loguear título y abstract truncados
    elif pmids:
        logger.warning(f"Se encontraron PMIDs pero no se pudieron recuperar datos de artículos (título/abstract). PMIDs: {pmids}")

    return articles_data[:num_of_articles]

# NUEVA CLASE: Callback Handler para registrar llamadas al LLM
class LLMCallLogger(BaseCallbackHandler):
    """Callback Handler para registrar las interacciones con el LLM."""

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any
    ) -> Any:
        """Se ejecuta cuando la cadena que llama al modelo de chat comienza."""
        logger.info(f"{Colors.YELLOW}{Colors.BOLD}--- LLM Call Start ---{Colors.RESET}")
        # La serialización completa puede ser muy verbosa, la comentamos por defecto
        # logger.info(f"{Colors.YELLOW}Serialized LLM: {json.dumps(serialized, indent=2)}{Colors.RESET}")
        
        log_msg_list = []
        for i, message_group in enumerate(messages):
            group_log = []
            for message in message_group:
                group_log.append({"role": message.type, "content": message.content[:500] + ("..." if len(message.content) > 500 else "")})
            log_msg_list.append(group_log)
        logger.info(f"{Colors.YELLOW}Messages Sent to LLM: {json.dumps(log_msg_list, ensure_ascii=False, indent=2)}{Colors.RESET}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Se ejecuta cuando el LLM termina de ejecutarse."""
        logger.info(f"{Colors.GREEN}{Colors.BOLD}--- LLM Call End ---{Colors.RESET}")
        
        log_generations_list = []
        for generation_list in response.generations:
            generation_log = []
            for generation in generation_list:
                content = "N/A"
                if hasattr(generation, 'text') and generation.text:
                    content = generation.text[:500] + ("..." if len(generation.text) > 500 else "")
                elif hasattr(generation, 'message') and hasattr(generation.message, 'content'):
                     content = generation.message.content[:500] + ("..." if len(generation.message.content) > 500 else "")
                generation_log.append(content)
            log_generations_list.append(generation_log)
        logger.info(f"{Colors.GREEN}Response from LLM: {json.dumps(log_generations_list, ensure_ascii=False, indent=2)}{Colors.RESET}")
        
        # El llm_output puede ser muy verboso y específico de la plataforma, lo comentamos por defecto
        # if response.llm_output:
        #    logger.info(f"{Colors.GREEN}LLM Output details: {json.dumps(response.llm_output, ensure_ascii=False, indent=2)}{Colors.RESET}")

# 2. LLM base compartido para todos los agentes
# Usamos gpt-4 como en el esqueleto, que es más potente para tareas de razonamiento complejo.
llm_call_logger = LLMCallLogger() # Instanciar el logger
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.0,
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    callbacks=[llm_call_logger],
    streaming=True  # ACTIVAR STREAMING DE RESPUESTA
)

# 3. Manager Agent: planifica y orquesta el pipeline
manager_prompt_template = PromptTemplate(
    input_variables=["objective"],
    template=(
        "Eres un agente orquestador para investigación biomédica.\n"
        "Objetivo: {objective}\n"
        "Tu tarea es utilizar la herramienta 'pubmed_search' para encontrar artículos relevantes.\n"
        "La herramienta 'pubmed_search' tomará tu objetivo, extraerá keywords, las traducirá y buscará en PubMed.\n"
        "Devuelve directamente el resultado de la herramienta 'pubmed_search'."
        "Instrucción para la herramienta pubmed_search: usa el siguiente objetivo para tu búsqueda: {objective}" # Esto es más una guía para el tool.
    )
)
# Nota: LLMChain no usa 'tools' directamente. La herramienta se llama en la func de un Tool o por un AgentExecutor.
# Aquí, el manager_chain preparará la llamada o la descripción para la herramienta.
# Para simplificar, vamos a asumir que el manager_chain formula la query para la herramienta,
# y la SequentialChain se encargará de pasar la salida correcta.
# O, más directamente, el manager_chain podría simplemente pasar el objetivo a la herramienta.

manager_prompt = PromptTemplate(
    input_variables=["objective"],
    template=(
        "Objetivo de investigación: {objective}\n\n"
        "Prepara este objetivo para la herramienta de búsqueda de PubMed. La herramienta se encargará de procesarlo.\n"
        "Simplemente reformula el objetivo si es necesario para que sea claro para una búsqueda, o devuélvelo tal cual.\n"
        "Objetivo para PubMed:"
    )
)
# Este manager_chain es un poco redundante si la herramienta ya toma el objetivo.
# Lo simplificaremos: la primera etapa de SequentialChain será la herramienta directamente si es posible,
# o una LLMChain que prepare la entrada para la herramienta.
# El esqueleto original tiene manager_chain que produce "raw_results".
# Esto implica que manager_chain *usa* la herramienta.

# Para que manager_chain use la herramienta, necesitaríamos un AgentExecutor o una CustomChain.
# Pero el esqueleto usa LLMChain.

# Solución de compromiso: El manager_chain generará una "descripción de búsqueda"
# y la SequentialChain se configurará para que la herramienta use esta descripción.
# La SequentialChain puede manejar esto si la variable de salida de manager_chain
# es la variable de entrada de la herramienta.

manager_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["objective"],
        template="Dado el objetivo de investigación: \"{objective}\", formula una consulta concisa o descripción del tema para buscar en PubMed. Esta descripción será usada por una herramienta de búsqueda."
    ),
    output_key="pubmed_query_description" # Esta salida será la entrada para pubmed_tool
)

# 4. Extractor Agent: extrae datos clave de cada artículo
# El input será 'raw_pubmed_results' (la salida de pubmed_tool)
extractor_prompt = PromptTemplate(
    input_variables=["raw_pubmed_results"], # Esta será la cadena con los artículos de PubMed
    template=(
        "Eres un agente extractor de datos clínicos altamente especializado.\n"
        "Has recibido la siguiente lista de artículos de PubMed:\n"
        "---------------------\n"
        "{raw_pubmed_results}\n"
        "---------------------\n"
        "Para CADA artículo en la lista, extrae la siguiente información en formato JSON. Es CRUCIAL que intentes completar todos los campos basándote en el contenido del abstract, infiriendo cuando sea necesario pero sin inventar información no soportada por el texto.\n"
        "  - \"pubmed_id\": \"(ID del artículo, ej. 34567890)\",\n"
        "  - \"title\": \"(Título del artículo)\",\n"
        "  - \"drug_name\": \"(Nombre del fármaco, vacuna o intervención principal estudiada. Si es un estudio sobre una técnica, metodología o encuesta, indica el tema principal. Si no aplica directamente un 'drug_name', usa N/A como último recurso.)\",\n"
        "  - \"mechanism_of_action\": \"(Mecanismo de acción del fármaco/intervención, si se describe o se puede inferir claramente de su propósito. Si no se menciona o no aplica (ej. para encuestas), usa N/A.)\",\n"
        "  - \"trial_phase\": \"(Fase del ensayo clínico, ej. Phase I, Phase II. Si no es un ensayo clínico (ej. revisión, estudio observacional, encuesta, reporte de caso), indica 'N/A' o el tipo de estudio si es relevante y claro, ej. 'Estudio observacional').\",\n"
        "  - \"main_results_summary\": \"(Resumen específico y conciso de los principales resultados, hallazgos o conclusiones del estudio según el abstract. Evita generalidades. INFIERE si no está explícito. Si el abstract no ofrece conclusiones claras o resultados concretos, usa N/A como último recurso.)\",\n"
        "  - \"adverse_effects_summary\": \"(Resumen breve de los efectos adversos reportados en el abstract. Si no se mencionan efectos adversos, indica 'No se mencionan efectos adversos'. Si el estudio no es de una intervención que pueda causar efectos adversos (ej. una encuesta), también puedes usar 'No aplica' o 'No se mencionan efectos adversos'. Usa N/A solo si es totalmente imposible determinarlo.)\"\n"
        "Asegúrate de que la salida sea una lista de objetos JSON, donde cada objeto representa un artículo.\n"
        "Ejemplo de salida (lista de JSON objects):\n"
        "[ \n"
        "  {{ \"pubmed_id\": \"123\", \"title\": \"Title A\", \"drug_name\": \"DrugX\", \"mechanism_of_action\": \"Inhibidor de XYZ\", \"trial_phase\": \"Phase II\", \"main_results_summary\": \"DrugX mostró una reducción del 50% en el endpoint primario (p<0.05).\", \"adverse_effects_summary\": \"Efectos leves incluyeron náuseas (10%).\" }},\n"
        "  {{ \"pubmed_id\": \"456\", \"title\": \"Title B\", \"drug_name\": \"Encuesta sobre vacunación\", \"mechanism_of_action\": \"N/A\", \"trial_phase\": \"Encuesta\", \"main_results_summary\": \"El 70% de los encuestados apoya la vacunación.\", \"adverse_effects_summary\": \"No aplica.\" }}\n"
        "]\n"
        "Procesa todos los artículos proporcionados. Si no hay artículos o la entrada está vacía, devuelve una lista JSON vacía []."
    )
)
extractor_chain = LLMChain(llm=llm, prompt=extractor_prompt, output_key="extracted_data_json_list")

# 5. Validator Agent: filtra según eficacia y toxicidad
validator_prompt = PromptTemplate(
    input_variables=["extracted_data_json_list"], # Entrada es la lista JSON de la cadena anterior
    template=(
        "Eres un agente validador de estudios clínicos con criterios estrictos. Tu ÚNICA salida debe ser una lista JSON.\n"
        "Has recibido la siguiente lista de datos extraídos de artículos (en formato JSON string):\n"
        "---------------------\n"
        "{extracted_data_json_list}\n"
        "---------------------\n"
        "Primero, parsea la entrada '{extracted_data_json_list}' como una lista de objetos JSON. Si la entrada es una lista JSON vacía '[]' o no es un JSON válido, devuelve una lista JSON vacía '[]' INMEDIATAMENTE.\n"
        "Para CADA objeto JSON en la lista parseada, verifica si cumple TODOS los siguientes criterios:\n"
        "  1. Eficacia: El campo 'main_results_summary' debe sugerir resultados positivos, conclusiones de eficacia, o hallazgos útiles y concretos. No debe ser 'N/A' o excesivamente vago.\n"
        "  2. Toxicidad: El campo 'adverse_effects_summary' debe indicar una toxicidad aceptable. Se considera aceptable si dice 'No se mencionan efectos adversos', 'No aplica', 'N/A' (interpretado como no relevante o no mencionado para estudios no intervencionistas), o si describe efectos leves a moderados o manejables. No cumple si indica efectos graves no mitigados que claramente superen el beneficio potencial descrito en 'main_results_summary'.\n"
        "Devuelve una NUEVA lista JSON que contenga ÚNICAMENTE los objetos JSON de los artículos que CUMPLEN AMBOS criterios. Mantén el mismo formato JSON para cada artículo.\n"
        "Si ningún artículo cumple los criterios, o si la lista de entrada original estaba vacía o era inválida, devuelve una lista JSON vacía '[]'.\n"
        "NO incluyas ninguna explicación, solo la lista JSON resultante."
    )
)
validator_chain = LLMChain(llm=llm, prompt=validator_prompt, output_key="validated_data_json_list")

# 6. Expert Agent: refina con conocimiento médico especializado
expert_prompt = PromptTemplate(
    input_variables=["validated_data_json_list", "objective"], # Incluimos el objetivo original para contexto
    template=(
        "Eres un médico experto con amplio conocimiento en el área de la siguiente consulta: \"{objective}\". Tu ÚNICA salida debe ser una lista JSON.\n" # MODIFICADO: Énfasis en salida JSON
        "Has recibido la siguiente lista de datos validados de estudios (en formato JSON string):\n"
        "---------------------\n"
        "{validated_data_json_list}\n"
        "---------------------\n"
        "Primero, parsea la entrada '{validated_data_json_list}' como una lista de objetos JSON. Si la entrada es una lista JSON vacía '[]' o no es un JSON válido, devuelve una lista JSON vacía '[]' INMEDIATAMENTE.\n"
        "Para CADA objeto JSON en la lista parseada, añade las siguientes anotaciones expertas al objeto JSON existente, enfocándote en la relevancia para el objetivo \"{objective}\". No elimines campos previos:\n"
        "  - \"clinical_relevance_comment\": \"(Comentario conciso sobre la relevancia clínica del estudio/fármaco en el contexto del objetivo: '{objective}' y su perfil riesgo/beneficio actual.)\"\n"
        "  - \"potential_combinations\": \"(Sugerencia breve de posibles combinaciones terapéuticas prometedoras o áreas de investigación futura para este fármaco/hallazgo, relevantes para '{objective}'.)\"\n"
        "  - \"clinical_phase_viability\": \"(Observaciones sobre la viabilidad del fármaco/tratamiento en su fase clínica actual y próximos pasos esperados si es prometedor, en relación con '{objective}'.)\"\n"
        "Devuelve la lista JSON COMPLETA con los objetos JSON originales ACTUALIZADOS con estos nuevos campos.\n"
        "Si la lista de entrada original estaba vacía o era inválida, devuelve una lista JSON vacía '[]'.\n"
        "NO incluyas ninguna explicación ni ejemplos, solo la lista JSON resultante." # MODIFICADO: Énfasis en solo JSON
    )
)
expert_chain = LLMChain(llm=llm, prompt=expert_prompt, output_key="expert_annotations_json_list")

# 7. Summarizer Agent: genera el informe final
summarizer_prompt = PromptTemplate(
    input_variables=["expert_annotations_json_list", "objective"], # Incluimos el objetivo original
    template=(
        "Eres un agente especializado en la redacción de informes científicos concisos y claros para profesionales médicos.\n" # MODIFICADO: Generalizado
        "El objetivo de la investigación fue: \"{objective}\".\n"
        "Has recibido la siguiente lista de datos de estudios, validados y anotados por un experto (en formato JSON):\n"
        "---------------------\n"
        "{expert_annotations_json_list}\n"
        "---------------------\n"
        "Basándote EXCLUSIVAMENTE en la información proporcionada y el objetivo original, genera un informe estructurado con las siguientes secciones:\n"
        "  1. Introducción: Breve resumen del objetivo de la investigación (\"{objective}\") y el panorama terapéutico actual para la condición o tema principal de dicho objetivo, según los hallazgos (si los datos lo permiten).\n" # MODIFICADO
        "  2. Candidatos Terapéuticos Prometedores o Hallazgos Relevantes: Si se identificaron fármacos, tratamientos o hallazgos relevantes para \"{objective}\", presenta una tabla comparativa (en formato Markdown) o un resumen detallado para cada uno. Incluye:\n" # MODIFICADO
        "     - Nombre del Fármaco/Tratamiento/Hallazgo\n"
        "     - Mecanismo de Acción (resumido, si aplica)\n"
        "     - Resultados Principales (eficacia, p-value, tamaño muestral si disponible, o descripción del hallazgo)\n"
        "     - Efectos Adversos (resumido, si aplica)\n"
        "     - Comentario del Experto (relevancia clínica, combinaciones, viabilidad, en relación con \"{objective}\")\n" # MODIFICADO
        "     - Referencia (PubMed ID)\n"
        "  3. Conclusiones y Recomendaciones Prácticas: Resume los hallazgos clave en relación con \"{objective}\". ¿Qué tratamientos o enfoques parecen más prometedores? ¿Cuáles son las limitaciones de los datos presentados? ¿Qué recomendaciones prácticas se pueden ofrecer basadas en esta información para el tema \"{objective}\"?\n" # MODIFICADO
        "Si no se encontraron candidatos prometedores o hallazgos relevantes, o la lista de entrada está vacía, indica claramente que la búsqueda no arrojó resultados significativos según los criterios aplicados para el objetivo \"{objective}\" y explica brevemente por qué podría ser (ej. falta de estudios que cumplan criterios, datos insuficientes en los abstracts, etc.).\n" # MODIFICADO
        "El informe debe ser profesional, objetivo y basado en la evidencia proporcionada."
    )
)
summarizer_chain = LLMChain(llm=llm, prompt=summarizer_prompt, output_key="final_report")

# 8. Orquestación secuencial
# La SequentialChain pasará todas las variables conocidas a las siguientes cadenas.
# La primera cadena es manager_chain, su input es 'objective'.
# Su output 'pubmed_query_description' será usado por pubmed_tool.
# La salida de pubmed_tool (que llamaremos 'raw_pubmed_results') será la entrada para extractor_chain.

# Para integrar la herramienta en SequentialChain, necesitamos un paso que la llame.
# Podemos usar una clase wrapper o una función con `Chain.from_callable`.

# from langchain.chains import SimpleSequentialChain, TransformChain # Comentado si TransformChain ya se importa arriba
# Asegúrate que TransformChain se importa solo una vez. Si está en la línea 9, esta es redundante.

# Paso intermedio para llamar a la herramienta pubmed_tool
def run_pubmed_tool(inputs: dict) -> dict:
    query_description = inputs["pubmed_query_description"]
    tool_result = pubmed_tool.run(query_description)
    return {"raw_pubmed_results": tool_result}

pubmed_tool_chain = TransformChain(
    input_variables=["pubmed_query_description"],
    output_variables=["raw_pubmed_results"],
    transform=run_pubmed_tool
)

def search_pubmed_tool_func(query_objective: str) -> str:
    """
    Procesa el objetivo, extrae keywords, traduce y busca en PubMed.
    Devuelve una cadena formateada con los resultados para el LLM.
    """
    logger.info(f"Tool 'search_pubmed_tool_func' llamado con query_objective: {query_objective}")
    
    # Paso 1: Extraer keywords en español del objetivo
    keywords_es = extract_keywords_with_llm(query_objective, assistant_for_keywords)
    if not keywords_es:
        logger.warning("No se pudieron extraer keywords del objetivo.")
        return "No se pudieron extraer keywords para la búsqueda en PubMed."

    # Paso 2: Traducir keywords al inglés
    keywords_en = translate_keywords(keywords_es)
    if not keywords_en:
        logger.warning("No se pudieron traducir las keywords al inglés.")
        return "No se pudieron traducir las keywords para la búsqueda en PubMed."
    
    # Paso 3: Buscar en PubMed
    articles = find_pubmed_tiab(keywords_en, num_of_articles=10)
    if not articles:
        return "No se encontraron artículos relevantes en PubMed para las keywords proporcionadas."
    
    # Formatear resultados como una cadena para el LLM
    formatted_results = "Artículos de PubMed encontrados:\n\n"
    for i, art in enumerate(articles):
        formatted_results += f"Artículo {i+1}:\n"
        formatted_results += f"  ID: {art['id']}\n"
        formatted_results += f"  Título: {art['title']}\n"
        formatted_results += f"  Abstract: {art['abstract'][:500]}...\n"
        formatted_results += f"  Año: {art['year']}\n\n"
        
    return formatted_results



pubmed_tool = Tool(
    name="pubmed_search",
    func=search_pubmed_tool_func,
    description="Busca artículos en PubMed. La entrada debe ser una descripción del objetivo de investigación o una pregunta para la cual se necesitan artículos."
)

full_pipeline = SequentialChain(
    chains=[
        manager_chain,          # Input: objective -> Output: pubmed_query_description
        pubmed_tool_chain,      # Input: pubmed_query_description -> Output: raw_pubmed_results
        extractor_chain,        # Input: raw_pubmed_results -> Output: extracted_data_json_list
        validator_chain,        # Input: extracted_data_json_list -> Output: validated_data_json_list
        expert_chain,           # Input: validated_data_json_list, objective -> Output: expert_annotations_json_list
        summarizer_chain,       # Input: expert_annotations_json_list, objective -> Output: final_report
    ],
    input_variables=["objective"],
    output_variables=["final_report", "raw_pubmed_results", "extracted_data_json_list", "validated_data_json_list", "expert_annotations_json_list"], # Especificar todas las salidas que queremos conservar
    verbose=True
)

# --- Stub mínimo para integración con ChatMed ---
def full_pipeline(inputs: dict) -> dict:
    """
    Simula la respuesta de un pipeline biomédico. Devuelve un informe de ejemplo.
    """
    question = inputs.get("objective") or inputs.get("question") or ""
    return {
        "final_report": f"[Simulado] Respuesta biomédica para: '{question}'. (Aquí iría el informe real de PubMed o guidelines si estuviera implementado)"
    }

def biochat_reduced_pipeline(inputs: dict) -> dict:
    """
    Simula la respuesta de un pipeline reducido (solo metadatos PubMed).
    """
    question = inputs.get("objective") or inputs.get("question") or ""
    return {
        "raw_pubmed_results": f"[Simulado] Resultados PubMed para: '{question}'. (Aquí irían los artículos reales si estuviera implementado)"
    }

if __name__ == "__main__":
    logger.info("Iniciando BioChat Agent Pipeline en modo interactivo...")
    print("BioChat Agent Pipeline Interactivo. Escribe 'salir' para terminar.")

    # Definir un nombre de archivo constante para el informe
    reports_dir = "informes_biochat"
    report_filename = "informe_biochat_actual.md" # NOMBRE DE ARCHIVO CONSTANTE
    
    # Crear directorio de informes si no existe la primera vez
    if not os.path.exists(reports_dir):
        try:
            os.makedirs(reports_dir)
        except OSError as e:
            logger.error(f"Error creando directorio de informes {reports_dir}: {e}")
            reports_dir = "." # Guardar en el directorio actual si falla la creación

    full_report_path = os.path.join(reports_dir, report_filename)

    # Saludos comunes que el chatbot reconocerá
    common_greetings = ["hola", "buenos días", "buenas tardes", "buenas noches", "qué tal", "hey", "saludos"]
    greeting_responses = [
        "¡Hola! ¿En qué puedo ayudarte hoy con tu investigación biomédica?",
        "¡Saludos! Estoy listo para tu consulta biomédica.",
        "¡Hola! Introduce tu pregunta o tema de investigación.",
        "¡Buenas! ¿Qué información biomédica necesitas?"
    ]
    import random

    while True:
        objective_text = input(f"\n{Colors.BOLD}Por favor, introduce tu consulta o objetivo de investigación biomédica (o 'salir' para terminar):{Colors.RESET}\n> ")
        
        # Manejo de saludos
        if objective_text.lower().strip() in common_greetings:
            print(f"{Colors.GREEN}{random.choice(greeting_responses)}{Colors.RESET}")
            logger.info(f"Usuario saludó con: '{objective_text}'. Respuesta de saludo enviada.")
            continue

        if objective_text.lower() in ["salir", "exit", "quit", "adiós"]:
            logger.info("Saliendo del BioChat Agent Pipeline.")
            print("Gracias por usar BioChat. ¡Hasta pronto!")
            break

        if not objective_text.strip():
            print("No has introducido ninguna consulta. Inténtalo de nuevo.")
            continue

        logger.info(f"Nuevo objetivo recibido: {objective_text}")
        print(f"\nProcesando tu consulta: \"{objective_text}\"")
        print("Esto puede tardar unos momentos mientras los agentes trabajan...\n")
        
        try:
            # La SequentialChain espera un diccionario como entrada
            pipeline_outputs = full_pipeline({"objective": objective_text})
            final_report_text = pipeline_outputs["final_report"]

            logger.info("Informe final generado para la consulta.")
            print("\n" + "="*30 + " RESPUESTA DEL AGENTE BIOCHAT " + "="*30 + "\n")

            # STREAMING: Si quieres mostrar la respuesta a medida que se genera, puedes hacerlo así:
            # (esto solo funcionará si el LLM y la cadena soportan streaming y si usas invoke/generate)
            # Por compatibilidad, aquí mostramos el resultado final como antes:
            print(final_report_text)
            print("\n" + "="*80 + "\n")

            # Guardar la respuesta en el archivo .md constante (sobrescribiendo)
            # El timestamp y el nombre basado en el objetivo ya no son necesarios aquí
            # para el nombre del archivo, pero el timestamp se mantiene en el contenido.
            
            with open(full_report_path, "w", encoding="utf-8") as f:
                f.write(f"# Consulta Original: {objective_text}\n\n")
                f.write(f"# Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("## Respuesta Generada por BioChat Agent Pipeline\n\n")
                f.write(final_report_text)
            logger.info(f"Respuesta guardada en {full_report_path}")
            print(f"Respuesta también guardada en: {full_report_path}")

            # Opcional: imprimir resultados intermedios para depuración
            print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*20} INICIO DEBUGGING RESULTADOS INTERMEDIOS {'='*20}{Colors.RESET}")
            
            if 'raw_pubmed_results' in pipeline_outputs:
                print(f"\n{Colors.BLUE}{Colors.BOLD}--- Salida de PubMed Tool (raw_pubmed_results) ---{Colors.RESET}\n{pipeline_outputs['raw_pubmed_results']}\n")
            else:
                print(f"\n{Colors.YELLOW}{Colors.BOLD}--- Salida de PubMed Tool (raw_pubmed_results) ---{Colors.RESET}\nNo encontrado en pipeline_outputs.\n")

            if 'extracted_data_json_list' in pipeline_outputs:
                header = f"\n{Colors.BLUE}{Colors.BOLD}--- Salida de Extractor Agent (extracted_data_json_list) ---{Colors.RESET}"
                try:
                    # Intentar parsear y luego imprimir con indentación. Si falla, imprimir como cadena.
                    parsed_json_output = pipeline_outputs['extracted_data_json_list']
                    if isinstance(parsed_json_output, str):
                        try:
                            parsed_json = json.loads(parsed_json_output)
                            print(f"{header}\n{json.dumps(parsed_json, indent=2, ensure_ascii=False)}\n")
                        except json.JSONDecodeError:
                             print(f"{header} {Colors.YELLOW}(Error al parsear JSON string){Colors.RESET}\n{parsed_json_output}\n")
                    elif isinstance(parsed_json_output, list) or isinstance(parsed_json_output, dict):
                        print(f"{header}\n{json.dumps(parsed_json_output, indent=2, ensure_ascii=False)}\n")
                    else:
                        print(f"{header}\n{parsed_json_output}\n")
                except Exception as ex_parse: # Captura más general por si acaso
                     print(f"{header} {Colors.RED}(Excepción al procesar/imprimir JSON: {ex_parse}){Colors.RESET}\n{pipeline_outputs['extracted_data_json_list']}\n")
            else:
                print(f"\n{Colors.YELLOW}{Colors.BOLD}--- Salida de Extractor Agent (extracted_data_json_list) ---{Colors.RESET}\nNo encontrado en pipeline_outputs.\n")
            
            if 'validated_data_json_list' in pipeline_outputs:
                header = f"\n{Colors.BLUE}{Colors.BOLD}--- Salida de Validator Agent (validated_data_json_list) ---{Colors.RESET}"
                try:
                    parsed_json_output = pipeline_outputs['validated_data_json_list']
                    if isinstance(parsed_json_output, str):
                        try:
                            parsed_json = json.loads(parsed_json_output)
                            print(f"{header}\n{json.dumps(parsed_json, indent=2, ensure_ascii=False)}\n")
                        except json.JSONDecodeError:
                            print(f"{header} {Colors.YELLOW}(Error al parsear JSON string){Colors.RESET}\n{parsed_json_output}\n")
                    elif isinstance(parsed_json_output, list) or isinstance(parsed_json_output, dict):
                         print(f"{header}\n{json.dumps(parsed_json_output, indent=2, ensure_ascii=False)}\n")
                    else:
                        print(f"{header}\n{parsed_jsonOutput}\n")
                except Exception as ex_parse:
                     print(f"{header} {Colors.RED}(Excepción al procesar/imprimir JSON: {ex_parse}){Colors.RESET}\n{pipeline_outputs['validated_data_json_list']}\n")
            else:
                print(f"\n{Colors.YELLOW}{Colors.BOLD}--- Salida de Validator Agent (validated_data_json_list) ---{Colors.RESET}\nNo encontrado en pipeline_outputs.\n")

            if 'expert_annotations_json_list' in pipeline_outputs:
                header = f"\n{Colors.BLUE}{Colors.BOLD}--- Salida de Expert Agent (expert_annotations_json_list) ---{Colors.RESET}"
                try:
                    parsed_json_output = pipeline_outputs['expert_annotations_json_list']
                    if isinstance(parsed_json_output, str):
                        try:
                            parsed_json = json.loads(parsed_json_output)
                            print(f"{header}\n{json.dumps(parsed_json, indent=2, ensure_ascii=False)}\n")
                        except json.JSONDecodeError:
                            print(f"{header} {Colors.YELLOW}(Error al parsear JSON string){Colors.RESET}\n{parsed_json_output}\n")
                    elif isinstance(parsed_json_output, list) or isinstance(parsed_json_output, dict):
                        print(f"{header}\n{json.dumps(parsed_json_output, indent=2, ensure_ascii=False)}\n")
                    else:
                        print(f"{header}\n{parsed_jsonOutput}\n")
                except Exception as ex_parse:
                    print(f"{header} {Colors.RED}(Excepción al procesar/imprimir JSON: {ex_parse}){Colors.RESET}\n{pipeline_outputs['expert_annotations_json_list']}\n")
            else:
                print(f"\n{Colors.YELLOW}{Colors.BOLD}--- Salida de Expert Agent (expert_annotations_json_list) ---{Colors.RESET}\nNo encontrado en pipeline_outputs.\n")
            print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*20} FIN DEBUGGING RESULTADOS INTERMEDIOS {'='*20}{Colors.RESET}\n")


        except Exception as e:
            # Considera si quieres volver a añadir la captura específica de openai.RateLimitError aquí
            # o modificar ClinicalLogger.error para aceptar exc_info y pasarlo.
            logger.error(f"Error durante la ejecución del pipeline para el objetivo '{objective_text}': {e}")
            print(f"Se produjo un error al procesar tu consulta: {e}")
            print("Por favor, intenta reformular tu pregunta o inténtalo más tarde.")
