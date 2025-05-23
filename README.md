# SinaSuite: Inteligencia de Datos Médicos Avanzada con Lenguaje Natural

Este CHATBOT incorpora un motor de análisis de bases de datos SQLite de última generación, diseñado para empoderar a los profesionales de la salud y analistas. Esta capacidad permite realizar consultas intrincadas utilizando lenguaje natural, eliminando la barrera técnica de SQL y facilitando el acceso instantáneo a información vital.

Con ChatSuite, se puede:
*   **Descubrir insights profundos** ocultos en sus datos.
*   **Obtener respuestas claras y precisas** a preguntas complejas sobre pacientes, tratamientos, diagnósticos y más.
*   **Potenciar la toma de decisiones clínicas y operativas** con información relevante y oportuna.

La tecnología de IA no solo analiza esquemas y datos, sino que comprende la semántica de sus preguntas, infiere relaciones complejas entre tablas (multi-hop joins) y entrega la información crucial, cuándo y cómo la necesita. Libere el verdadero potencial de sus datos médicos con la inteligencia analítica de SinaSuite.

## Arquitectura del Componente de Análisis de Lenguaje Natural

Este componente es una parte integral de SinaSuite, responsable de la interpretación de consultas en lenguaje natural y su traducción a SQL para interactuar con bases de datos médicas SQLite.

```
sina_mcp/
├── src/                      # Núcleo de la lógica de la aplicación
│   ├── pipeline.py           # Orquestador principal del flujo de consulta NLQ a SQL
│   ├── langchain_chatbot.py  # Integración con agente LangChain y herramientas de IA
│   ├── llm_utils.py          # Utilidades para la interacción con Modelos de Lenguaje Grandes (LLMs)
│   ├── sql_generator.py      # Generador de consultas SQL a partir de JSON estructurado
│   ├── rag_enhancements.py   # Lógica de RAG para recuperación de contexto semántico
│   ├── db_relationship_graph.py # Gestión y uso del grafo de relaciones entre tablas
│   ├── generate_enhanced_dict.py # Script para generar dictionary.json (descripciones, sinónimos, relaciones)
│   ├── schema_enhancer.py    # Script para generar schema_enhanced.json (esquema de BD enriquecido)
│   ├── ...                   # Otros módulos clave (validadores, conectores BD, preprocesadores, etc.)
├── data/                     # Archivos de datos generados y utilizados por el sistema (generalmente no versionados si son específicos de una BD)
│   ├── dictionary.json       # Diccionario enriquecido de tablas, columnas, relaciones (ejemplo o plantilla)
│   ├── schema_enhanced.json  # Esquema de la BD enriquecido con metadatos semánticos (ejemplo o plantilla)
│   └── (database.sqlite3.db) # Base de datos de ejemplo o de desarrollo (si aplica)
├── docs/                     # Documentación del proyecto
│   ├── README.md             # Este archivo (documentación principal del componente)
│   └── flujo_tecnico.html    # Descripción detallada del flujo técnico interno
├── tests/                    # Pruebas unitarias y de integración para asegurar la calidad
├── requirements.txt          # Dependencias del proyecto Python
└── .gitignore                # Archivos y carpetas a ignorar por el control de versiones Git
```

## Arquitectura MCP (Model-Controller-Pipeline)

El componente de análisis de lenguaje natural de SinaSuite se estructura siguiendo un patrón conceptual que denominamos MCP (Model-Controller-Pipeline). Este enfoque organiza la lógica del sistema en tres capas principales para facilitar la modularidad, el mantenimiento y la escalabilidad:

1.  **Model (Modelo):**
    *   **Definición:** Representa los datos, el conocimiento del dominio y la lógica de negocio para interactuar con la base de datos. Incluye el esquema de la base de datos (`schema_enhanced.json`), el diccionario de datos enriquecido (`dictionary.json` con descripciones, sinónimos y relaciones), y los módulos responsables de la generación y validación de SQL (`sql_generator.py`, `db_relationship_graph.py`).
    *   **Función:** Proporciona las herramientas y la información necesaria para comprender la estructura de los datos médicos y cómo consultarlos eficazmente. Es la base sobre la cual se construyen las consultas SQL.

2.  **Controller (Controlador):**
    *   **Definición:** Actúa como el intermediario entre el usuario (o el sistema que consume este componente) y el `Pipeline`. El script `langchain_chatbot.py` es el principal exponente de esta capa.
    *   **Función:** Gestiona la interacción con el usuario (entrada de preguntas en lenguaje natural), inicializa y coordina el agente LangChain, maneja el historial de la conversación, invoca el `Pipeline` para procesar la consulta y presenta la respuesta final al usuario. Es el punto de entrada y orquestación de alto nivel.

3.  **Pipeline (Tubería de Procesamiento):**
    *   **Definición:** Es el núcleo del procesamiento de la consulta, encapsulado principalmente en `pipeline.py`.
    *   **Función:** Orquesta la secuencia de pasos necesarios para traducir una pregunta en lenguaje natural a una consulta SQL ejecutable y luego a una respuesta comprensible. Esto incluye el preprocesamiento de la pregunta, la recuperación de información relevante mediante RAG (Retrieval Augmented Generation), la interacción con el LLM para la interpretación y estructuración, la generación de SQL (utilizando componentes del `Model`), la ejecución de la consulta y la formulación de la respuesta final. El `Pipeline` es invocado por el `Controller` (`langchain_chatbot.py`) y utiliza el `Model` para acceder a la lógica de datos y generación SQL.

**Relación con `langchain_chatbot.py`:**

`langchain_chatbot.py` funciona como la capa de `Controller`. Inicia el sistema, gestiona la interfaz con el usuario (en este caso, una CLI), y lo más importante, configura y ejecuta el agente LangChain. Este agente, a su vez, utiliza herramientas personalizadas que internamente llaman al `Pipeline` (`pipeline.py`). El `Pipeline` luego utiliza los componentes del `Model` para realizar su tarea de convertir la pregunta del usuario en una consulta SQL, ejecutarla y devolver los resultados para que el `Controller` los presente.

Esta arquitectura MCP permite una clara separación de responsabilidades, haciendo que el sistema sea más robusto y fácil de evolucionar.

## Requisitos Previos

Asegúrate de tener instaladas las siguientes dependencias:

- sqlite3
- Otras bibliotecas necesarias (especificadas en `requirements.txt`)

## Instalación y Configuración (Contexto de Desarrollo/Integración)

Estos pasos son relevantes para desarrolladores o para la integración de este componente dentro de entornos específicos de SinaSuite.

1.  **Clonación del Repositorio (si aplica):**
    ```bash
    git clone https://github.com/carpasgis2/sqlite-analyzer-mcp.git
    cd sqlite-analyzer-mcp
    ```

2.  **Gestión de Dependencias:**
    Se recomienda el uso de entornos virtuales (por ejemplo, `venv` o `conda`).
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Linux/macOS
    # venv\\Scripts\\activate   # En Windows
    pip install -r requirements.txt
    ```

3.  **Configuración de Variables de Entorno:**
    Este componente requiere la configuración de ciertas variables de entorno, como claves API para los LLMs (ej. `DEEPSEEK_API_KEY`). Consulte la documentación específica de los módulos que interactúan con servicios externos.

## Uso del Componente (Interfaz Principal)

La interacción principal con este componente se realiza a través del script `langchain_chatbot.py` o integrando el `pipeline.py` en otras aplicaciones de SinaSuite.

Para ejecutar la interfaz de línea de comandos (CLI) de ejemplo:
```bash
python src/langchain_chatbot.py
```
Asegúrese de que los archivos de configuración necesarios (como `dictionary.json`, `schema_enhanced.json`) y la base de datos SQLite estén accesibles y correctamente referenciados en la configuración del `db_connector` y el `pipeline`.

## Desarrollo y Contribuciones (Interno Laberit)

El desarrollo de este componente sigue las directrices internas de Laberit. Para contribuir:
*   Asegúrese de seguir las convenciones de código y estilo.
*   Escriba pruebas unitarias para nuevas funcionalidades o correcciones.
*   Documente los cambios significativos.
*   Utilice el flujo de trabajo de Git establecido (ramas, pull requests) para la revisión de código.

## Preguntas Objetivo a Contestar por el Sistema

Este sistema está diseñado para interpretar preguntas complejas en lenguaje natural y traducirlas a consultas SQL precisas. A continuación, se muestran ejemplos del tipo de preguntas que el sistema aspira a contestar, incluyendo aquellas que requieren múltiples joins (directos e indirectos), agregaciones, y comprensión semántica:

### Consultas con Joins Múltiples (Multi-Hop)
- "¿Qué médicos han tratado a pacientes diagnosticados con 'diabetes tipo 2' que también están tomando 'metformina'?"
- "Listar las provincias donde se han registrado más de 10 casos de 'gripe A' en pacientes mayores de 60 años durante el último invierno."
- "Mostrar los tratamientos aplicados a pacientes que previamente fueron diagnosticados con 'hipertensión' y posteriormente desarrollaron 'insuficiencia renal', y qué médico supervisó el tratamiento para la insuficiencia renal."

### Consultas con Agregaciones y Condiciones Complejas
- "¿Cuál es la edad promedio de los pacientes que fueron diagnosticados con 'cáncer de pulmón' en cada hospital durante 2023, y cuántos de ellos eran fumadores?"
- "¿Cuántos pacientes distintos recibieron al menos tres tipos diferentes de medicación el mes pasado, y cuál fue el coste total de esas medicaciones para cada paciente?"
- "Comparar el número de reingresos hospitalarios dentro de los 30 días posteriores al alta para pacientes con 'insuficiencia cardíaca' versus aquellos con 'neumonía', desglosado por grupo de edad (menores de 65 y 65 o más)."

### Consultas que Requieren Comprensión Semántica y de Sinónimos
- "Pacientes que reportaron 'fuerte dolor de cabeza' y que finalmente fueron diagnosticados con 'migraña crónica'."
- "¿Qué fármacos se utilizan comúnmente para tratar la 'hipertensión arterial sistémica' (HTA)?"

### Consultas con Negaciones o Exclusiones
- "Listar todos los pacientes que NO han recibido la vacuna contra la 'influenza' este año y tienen más de 70 años."
- "¿Qué medicamentos para la 'alergia' no contienen 'pseudoefedrina' y son aptos para niños menores de 12 años?"

Estas preguntas ilustran la capacidad del sistema para navegar por relaciones complejas en la base de datos, interpretar la intención del usuario y generar consultas SQL que reflejen con precisión la solicitud original.

# Flujo de Procesamiento: Consulta en Lenguaje Natural a SQL

## Diagrama de Flujo Simplificado

```
[USUARIO] → Pregunta en lenguaje natural
    ↓
[PREPROCESAMIENTO] → Corrige typos, normaliza texto
    ↓
[RAG] → Recupera contexto de esquema relevante
    ↓
[LLM - PASO 1] → Convierte a estructura JSON
    ↓
[RAG - PASO 2] → Enriquece con conocimiento de base de datos
    ↓
[SQL GENERATOR] → Genera consulta SQL válida
    ↓
[VALIDADOR SQL] → Verifica seguridad y estructura
    ↓
[BASE DE DATOS] → Ejecuta consulta y obtiene resultados
    ↓
[LLM - PASO 3] → Formula respuesta natural con los resultados
    ↓
[USUARIO] → Recibe respuesta en lenguaje natural
```

## Descripción Detallada del Proceso

1. **Entrada de Consulta**
   - El usuario hace una pregunta en lenguaje natural (ej: "¿Cuántos pacientes fueron diagnosticados con diabetes en 2023?")
   - La pregunta se añade al historial de conversación

2. **Preprocesamiento y Enriquecimiento**
   - Se corrigen posibles errores tipográficos usando diccionarios de términos médicos
   - Se detecta el tipo de consulta (COUNT, SELECT, AVG, etc.)
   - Se identifican términos clave y se enriquece la pregunta con metadatos

3. **Recuperación de Contexto (RAG)**
   - Se utiliza el sistema RAG para identificar las tablas más relevantes para la consulta
   - El RAG proporciona contexto del esquema de la base de datos (descripciones de tablas y columnas)
   - Este contexto ayuda al LLM a entender mejor la estructura de datos

4. **Paso 1: Generación de Estructura JSON**
   - Se envía la pregunta enriquecida y el contexto del esquema al LLM
   - El LLM extrae información estructurada en formato JSON con:
     * Tablas relevantes
     * Acciones a realizar (SELECT, COUNT, etc.)
     * Columnas necesarias
     * Condiciones para la cláusula WHERE
     * Posibles JOINs entre tablas

5. **Paso 2: Búsqueda RAG y Generación SQL**
   - El sistema RAG enriquece la información extrayendo relaciones entre tablas
   - Se mapean términos del lenguaje natural a elementos reales de la base de datos
   - El SQLGenerator crea una consulta SQL válida basada en la estructura JSON
   - El SQLValidator verifica que la consulta cumpla con las reglas de seguridad

6. **Ejecución de Consulta**
   - Se ejecuta la consulta parametrizada en la base de datos
   - Se capturan los resultados (filas y columnas)

7. **Paso 3: Generación de Respuesta**
   - Se envían al LLM:
     * La pregunta original
     * La consulta SQL ejecutada
     * Los resultados obtenidos
   - El LLM formula una respuesta en lenguaje natural que responde directamente a la pregunta original
   - La respuesta se añade al historial de conversación

8. **Salida de Respuesta**
   - El usuario recibe la respuesta en lenguaje natural
   - También se proporcionan metadatos sobre el rendimiento (tiempos de procesamiento)

## Ejemplo del Flujo Completo

**Entrada del usuario:**
> "¿Cuántos pacientes con cáncer de pulmón fueron atendidos el año pasado?"

**Preprocesamiento:**
- Tipo de consulta detectado: COUNT
- Términos clave identificados: "pacientes", "cáncer de pulmón", "año pasado"

**Contexto RAG:**
- Tablas principales detectadas: PATI_PATIENTS, ONCO_EVENT_INDICATIONS
- Columnas relevantes: PATI_ID, EVIN_DESCRIPTION_ES, EVIN_DATE

**JSON Generado (Paso 1):**
```json
{
  "tables": ["PATI_PATIENTS", "ONCO_EVENT_INDICATIONS"],
  "actions": ["COUNT"],
  "columns": ["PATI_ID"],
  "conditions": ["EVIN_DESCRIPTION_ES LIKE '%cáncer de pulmón%'", "YEAR(EVIN_DATE) = 2023"],
  "joins": ["INNER JOIN ONCO_EVENT_INDICATIONS ON PATI_PATIENTS.PATI_ID = ONCO_EVENT_INDICATIONS.PATI_ID"]
}
```

**SQL Generado (Paso 2):**
```sql
SELECT COUNT(DISTINCT PATI_PATIENTS.PATI_ID) 
FROM PATI_PATIENTS 
INNER JOIN ONCO_EVENT_INDICATIONS ON PATI_PATIENTS.PATI_ID = ONCO_EVENT_INDICATIONS.PATI_ID 
WHERE EVIN_DESCRIPTION_ES LIKE '%cáncer de pulmón%' AND YEAR(EVIN_DATE) = 2023
```

**Respuesta al Usuario (Paso 3):**
> "En el año 2023 fueron atendidos 157 pacientes con diagnóstico de cáncer de pulmón según los registros de la base de datos."

## Componentes Clave del Sistema

- **RAG (Retrieval Augmented Generation)**: Proporciona contexto sobre el esquema de la base de datos y mejora la precisión de las consultas al mapear términos en lenguaje natural a elementos de la base de datos.

- **LLM (Large Language Model)**: Se utiliza en tres pasos cruciales:
  1. Extracción de información estructurada de la pregunta
  2. Enriquecimiento con conocimiento del dominio médico
  3. Generación de respuestas naturales a partir de resultados técnicos

- **SQLGenerator y SQLValidator**: Generan consultas SQL válidas y seguras a partir de la información estructurada.

- **Pipeline de Procesamiento**: Orquesta todos los componentes y mantiene un flujo coherente desde la pregunta hasta la respuesta.



[USUARIO] → Pregunta en lenguaje natural
    ↓
[PREPROCESAMIENTO]
    ↓
    ├─ Corrección de errores tipográficos
    ├─ Normalización de texto
    ├─ Detección de patrones (IDs, entidades)
    └─ Enriquecimiento con metadatos
    ↓
[RAG - RECUPERACIÓN DE CONTEXTO]
    ↓
    ├─ Identificación de tablas relevantes
    ├─ Obtención de columnas relacionadas
    └─ Generación de contexto de esquema
    ↓
[PASO 1: LLM - EXTRACCIÓN ESTRUCTURAL]
    ↓
    ├─ Generación de estructura JSON
    ├─ Extracción de tablas, acciones, condiciones
    └─ Mapeo inicial de elementos de la BD
    ↓
[PROCESAMIENTO ESTRUCTURAL]
    ↓
    ├─ Normalización de la estructura
    ├─ Validación de relaciones entre tablas
    ├─ Inferencia de relaciones faltantes
    └─ Estrategias de fallback si no hay tablas
    ↓
[PASO 2: GENERACIÓN SQL]
    ↓
    ├─ Creación de consulta SQL válida
    ├─ Aplicación de aliases para tablas
    └─ Optimización de la consulta
    ↓
[EJECUCIÓN Y AUTOCORRECCIÓN]
    ↓
    ├─ Ejecución de la consulta SQL inicial
    ├─ Manejo de errores comunes (ambigüedad, sintaxis)
    ├─ Autocorrección automática o con LLM
    └─ Re-ejecución si es necesario
    ↓
[PASO 3: LLM - GENERACIÓN DE RESPUESTA]
    ↓
    ├─ Formulación de respuesta en lenguaje natural
    ├─ Extracción y ejecución de SQL corregido (si existe)
    └─ Regeneración de respuesta con resultados mejorados
    ↓
[USUARIO] → Recibe respuesta en lenguaje natural
