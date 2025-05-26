# Preguntas de 1 Salto para el Chatbot

Aquí hay una lista de preguntas que puedes usar para probar el chatbot. Estas preguntas están diseñadas para requerir aproximadamente "1 salto" en la base de datos, lo que significa que generalmente involucran la unión de dos tablas directamente relacionadas o la búsqueda de información en una tabla que está directamente vinculada a la entidad principal de la pregunta.

## Preguntas Sugeridas

1.  **Sobre Pacientes y sus Diagnósticos directos:**
    *   "¿Cuáles son las descripciones de los diagnósticos registrados para el paciente con ID 1001?"
    *   "Muéstrame los diagnósticos del paciente con ID 752."

2.  **Sobre Pacientes y sus Alergias (descripción general):**
    *   "Para el paciente con ID 1001, ¿cuáles son las descripciones de las alergias que tiene registradas?"
    *   "¿Qué alergias tiene el paciente con ID 340?"

3.  **Sobre Citas y su Ubicación:**
    *   "¿En qué unidad o departamento está programada la cita con ID 2503?"
    *   "Dime la ubicación de la cita 8810."

4.  **Sobre Citas y su Estado:**
    *   "¿Cuál es el estado actual de la cita con ID 2503?"
    *   "¿Cuál es el estado de la cita 5050?"

5.  **Sobre Episodios Médicos y su Ubicación:**
    *   "¿En qué unidad o departamento se atendió el episodio médico con ID 607?"
    *   "¿Dónde ocurrió el episodio 1123?"

6.  **Sobre Medicamentos y su Grupo Farmacoterapéutico:**
    *   "¿A qué grupo farmacoterapéutico pertenece el medicamento con ID 55?"
    *   "Clasificación farmacoterapéutica del medicamento 102."

7.  **Sobre Medicamentos y su Forma Farmacéutica:**
    *   "¿Cuál es la forma farmacéutica del medicamento con ID 55?"
    *   "Forma del medicamento 215."

8.  **Sobre Casos de Cáncer y su Localización Tumoral:**
    *   "¿Cuál es la localización del tumor para el caso de cáncer con ID 3?"
    *   "Localización del tumor en el caso oncológico 15."

9.  **Sobre Casos de Cáncer y su Tipo Histológico:**
    *   "Describa el tipo histológico del cáncer registrado con ID 3."
    *   "Tipo histológico del cáncer 28."

**Nota:** Reemplaza los IDs (`1001`, `2503`, `55`, `3`, etc.) con IDs válidos de tu base de datos para obtener resultados reales.



# Orden de Preguntas (de más fácil a más difícil según saltos y complejidad)

1.  **"Como se llama el paicente 1001"?"**
    *   Salto estimado: 0-2. Complejidad: Baja (búsqueda simple).
1.  **"¿Qué fármacos se utilizan comúnmente para tratar la 'hipertensión arterial sistémica' (HTA)?"**
    *   Salto estimado: 0-1. Complejidad: Baja (búsqueda o join simple).

## Sobre Pacientes y sus Diagnósticos directos:

> "¿Cuáles son las descripciones de los diagnósticos registrados para el paciente con ID [ID_DEL_PACIENTE]?"

## Sobre Pacientes y sus Alergias (descripción general):

> "Para el paciente con ID [ID_DEL_PACIENTE], ¿cuáles son las descripciones de las alergias que tiene registradas?"

## Sobre Citas y su Ubicación:

> "¿En qué unidad o departamento está programada la cita con ID [ID_DE_LA_CITA]?"

## Sobre Citas y su Estado:

> "¿Cuál es el estado actual de la cita con ID [ID_DE_LA_CITA]?"

## Sobre Episodios Médicos y su Ubicación:

> "¿En qué unidad o departamento se atendió el episodio médico con ID [ID_DEL_EPISODIO]?"

## Sobre Medicamentos y su Grupo Farmacoterapéutico:

> "¿A qué grupo farmacoterapéutico pertenece el medicamento con ID [ID_DEL_MEDICAMENTO]?"

## Sobre Medicamentos y su Forma Farmacéutica:

> "¿Cuál es la forma farmacéutica del medicamento con ID [ID_DEL_MEDICAMENTO]?"

## Sobre Casos de Cáncer y su Localización Tumoral:

> "¿Cuál es la localización del tumor para el caso de cáncer con ID [ID_DEL_CASO_ONCOLOGICO]?"

## Sobre Casos de Cáncer y su Tipo Histológico:

> "Describa el tipo histológico del cáncer registrado con ID [ID_DEL_CASO_ONCOLOGICO]."

---

**Cuáles son las descripciones de los diagnósticos registrados para el paciente con ID 1001?**

---

2.  **"Pacientes que reportaron 'fuerte dolor de cabeza' y que finalmente fueron diagnosticados con 'migraña crónica'."**
    *   Salto estimado: 1-2. Complejidad: Baja-Media (joins, semántica).
3.  **"Listar todos los pacientes que NO han recibido la vacuna contra la 'influenza' este año y tienen más de 70 años."**
    *   Salto estimado: 1-2. Complejidad: Media (join/subconsulta, negación, filtros).

    Excel C:\Users\cpascual\PycharmProjects\pythonProject\cursos_actividades\sina_mcp\sqlite-analyzer\src\data\Pacientes_mayores_de_70_sin_vacuna_de_influenza_en_2025.csv
4.  **"¿Qué medicamentos para la 'alergia' no contienen 'pseudoefedrina' y son aptos para niños menores de 12 años?**
    *   Salto estimado: 1-2. Complejidad: Media (joins, negación, filtros)."
5.  **"¿Qué médicos han tratado a pacientes diagnosticados con 'diabetes tipo 2' que también están tomando 'metformina'?"**
    *   Salto estimado: 2-3 (multi-hop). Complejidad: Media-Alta (múltiples entidades y condiciones).
6.  **"Listar las provincias donde se han registrado más de 10 casos de 'gripe A' en pacientes mayores de 60 años durante el último invierno."**
    *   Salto estimado: 2-3 (multi-hop). Complejidad: Alta (agregación, group by, filtros múltiples).
7.  **"¿Cuál es la edad promedio de los pacientes que fueron diagnosticados con 'cáncer de pulmón' en cada hospital durante 2023, y cuántos de ellos eran fumadores?"**
    *   Salto estimado: 2-3 (multi-hop). Complejidad: Alta (múltiples agregaciones, group by, filtros).
8.  **"¿Cuántos pacientes distintos recibieron al menos tres tipos diferentes de medicación el mes pasado, y cuál fue el coste total de esas medicaciones para cada paciente?"**
    *   Salto estimado: 1-2 (pero con lógica compleja). Complejidad: Alta (agregaciones complejas, group by, subconsultas implícitas).
9.  **"Mostrar los tratamientos aplicados a pacientes que previamente fueron diagnosticados con 'hipertensión' y posteriormente desarrollaron 'insuficiencia renal', y qué médico supervisó el tratamiento para la insuficiencia renal."**
    *   Salto estimado: 3+ (multi-hop). Complejidad: Muy Alta (secuencia temporal, múltiples entidades y relaciones).
10. **"Comparar el número de reingresos hospitalarios dentro de los 30 días posteriores al alta para pacientes con 'insuficiencia cardíaca' versus aquellos con 'neumonía', desglosado por grupo de edad (menores de 65 y 65 o más)."**
    *   Salto estimado: 2-3 (multi-hop). Complejidad: Muy Alta (comparación de cohortes, condiciones temporales complejas, agregaciones, group by).
