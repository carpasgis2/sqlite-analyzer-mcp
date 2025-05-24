# SQLite Analyzer

Este proyecto es una herramienta para analizar bases de datos SQLite. Permite a los usuarios conectarse a una base de datos SQLite, extraer información sobre su esquema y realizar análisis de datos.

## Estructura del Proyecto

```
sqlite-analyzer
├── src
│   ├── main.py               # Punto de entrada del script
│   ├── db
│   │   ├── connection.py      # Gestión de la conexión a la base de datos
│   │   └── queries.py         # Consultas SQL para extraer datos
│   ├── analysis
│   │   ├── schema_analyzer.py # Análisis del esquema de la base de datos
│   │   └── data_analyzer.py   # Análisis de los datos dentro de las tablas
│   └── utils
│       └── helpers.py         # Funciones auxiliares
├── data
│   └── database.sqlite3.db    # Base de datos SQLite a analizar
├── requirements.txt            # Dependencias del proyecto
└── README.md                   # Documentación del proyecto
```

## Requisitos

Asegúrate de tener instaladas las siguientes dependencias:

- sqlite3
- Otras bibliotecas necesarias (especificadas en `requirements.txt`)

## Instalación

1. Clona el repositorio:
   ```
   git clone <URL_DEL_REPOSITORIO>
   cd sqlite-analyzer
   ```

2. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```

## Uso

Para ejecutar el script, utiliza el siguiente comando:

```
python src/main.py
```

Asegúrate de que el archivo `database.sqlite3.db` esté en la carpeta `data` antes de ejecutar el script.

## Contribuciones

Las contribuciones son bienvenidas. Si deseas contribuir, por favor abre un issue o envía un pull request.

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