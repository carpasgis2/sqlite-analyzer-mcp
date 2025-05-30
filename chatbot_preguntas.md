Hola, cual es tu fucnion

que es sina

cuantos pacientes hay 

cual es el paciente con el pronostico mas grave

🔎 Nivel Básico (1 salto)

✅ Consultas directas sobre pacientes, diagnósticos o alergias

**“¿Cómo se llama el paciente con ID 1001?”**

“¿Cuáles son las descripciones de los diagnósticos registrados para el paciente con ID 1001?”

“Muéstrame los diagnósticos del paciente con ID 752.”

“¿Qué alergias tiene el paciente con ID 340?”

“¿Cuáles son las descripciones de las alergias del paciente con ID 1001?”

📅 Citas

**"dame un id de cita de algun paciente, idnicaando el nombre del pacientE"**

“¿En qué unidad o departamento está programada la cita con ID 2503?”

**“Dime la ubicación de la cita 294.”**

**“¿Cuál es el estado actual de la cita con ID 294?”**

“¿Cuál es el estado de la cita 5050?”

🏥 Episodios

“¿Dónde ocurrió el episodio 1123?”

“¿En qué unidad se llevó a cabo el episodio con ID 607?”

🍆 Medicamentos

**“¿A qué grupo farmacoterapéutico pertenece el medicamento con ID 55 y como se llama ?”**  -- El medicamento con ID 55 corresponde a “ADOLONTA RETARD comp 200 mg C/20” y, en la base de datos, no tiene asignado ningún grupo farmacoterapéutico (el campo PHTH_ID está a NULL)

“¿Cuál es la forma farmacéutica del medicamento con ID 215?”

⚖️ Casos Oncológicos

“¿Cuál es la localización del tumor para el caso de cáncer con ID 3?”

“Tipo histológico del cáncer 28.”

🔢 Consultas generales simples

“¿Qué paciente tiene más citas registradas?”

“¿Cuántos diagnósticos diferentes tiene el paciente con ID 875?”





🔍 Nivel Intermedio (1-2 saltos)

🩸 Evolución clínica o cobertura

“Pacientes que reportaron 'fuerte dolor de cabeza' y que finalmente fueron diagnosticados con 'migraña crónica'.”

**“Listar todos los pacientes que NO han recibido la vacuna contra la 'influenza' este año y tienen más de 70 años.”**


"¿Cuantos medicamentos para la 'alergia' no contienen 'amoxicilina' y son aptos para niños menores de 12 años?”


Cuáles son las formas farmacéuticas disponibles para el medicamento con ID 102?

**¿En qué unidad clínica se atendieron los episodios del paciente con ID 1001  durante los ultimos dos años?**


👨‍⚕️ Medicación y diagnóstico






🔌 Nivel Avanzado (2-3 saltos)

♿️ Cohortes y reingresos


**¿Cuál es la duración promedio de hospitalización para pacientes cuyo diagnóstico principal contenga “cardiaca” vs. “neumonía”?**


1. ¿Qué grupos de diagnóstico aparecen en episodios de pacientes mayores de 75 años atendidos en el servicio de urgencias durante 2023?



2. ¿Qué pacientes con antecedente de hipertensión fueron además diagnosticados de insuficiencia renal aguda en una estancia en la unidad de cuidados intensivos, y cuál fue la duración de su hospitalización?
   
3. ¿Qué medicamentos antihipertensivos con ingrediente activo lisinopril se prescribieron a pacientes menores de 18 años durante sus hospitalizaciones?
   
4. ¿Qué tipos de vehículo participaron en accidentes con al menos un fallecido, en los que había un trabajador conductor, y que ocurrieron en la provincia de Madrid?

5. ¿Cuáles son los códigos de diagnóstico y sus descripciones para episodios en los que se diagnosticó neumonía y se administró un antibiótico durante la misma hospitalización?



### 1. Clasificación y recuento de fármacos por grupo terapéutico

**¿Cuántos medicamentos vigentes (no eliminados) hay en cada grupo terapéutico, y cuáles son los cinco grupos con más fármacos?**

* **Tablas implicadas:**

  * `MEDI_MEDICATIONS` (filtro `MEDI_DELETED = 0`)
  * `MEDI_PHARMA_THERAPEUTIC_GROUPS` (join por `PHTH_ID`)
* **Tipo de consulta:** agregación y ordenación.

```sql
SELECT 
  g.PHTH_DESCRIPTION_ES AS grupo,
  COUNT(*) AS numero_medicamentos
FROM MEDI_MEDICATIONS m
JOIN MEDI_PHARMA_THERAPEUTIC_GROUPS g
  ON m.PHTH_ID = g.PHTH_ID
WHERE m.MEDI_DELETED = 0
GROUP BY g.PHTH_DESCRIPTION_ES
ORDER BY numero_medicamentos DESC
LIMIT 5;
```

---

### 2. Estadísticas de dosis por forma farmacéutica

**¿Cuál es la dosis media y la desviación típica de los medicamentos según su forma farmacéutica (tableta, jarabe, inyectable…)?**

* **Tablas implicadas:**

  * `MEDI_MEDICATIONS` (`MEDI_DOSE`, `MEUN_ID_DOSE`)
  * `MEDI_PHARMACEUTICAL_FORMS` (join por `PFOR_ID`)
* **Tipo de consulta:** agregaciones estadísticas.

---

### 3. Perfil de alergias de pacientes

que tipos de alergias hay cuáles son los niveles de severidad de esas alergias?

**¿Qué pacientes tienen tres o más alergias registradas, y cuáles son los tipos y niveles de severidad de esas alergias?**

* **Tablas implicadas:**

  * `PATI_PATIENTS` (datos básicos de paciente)
  * `PATI_PATIENT_ALLERGIES` (join por `PATI_ID`)
  * `ALLE_ALLERGY_TYPES`, `ALLE_ALLERGY_SEVERITY_LEVELS` (join por `ALLT_ID`, `ALSE_ID`)
* **Tipo de consulta:** conteo por paciente, filtrado por having.

```sql
SELECT 
  p.PATI_ID,
  p.PATI_FULL_NAME,
  COUNT(*) AS num_alergias
FROM PATI_PATIENTS p
JOIN PATI_PATIENT_ALLERGIES a
  ON p.PATI_ID = a.PATI_ID
GROUP BY p.PATI_ID, p.PATI_FULL_NAME
HAVING COUNT(*) >= 3;
```

---

### 4. Medicación habitual de pacientes

**¿Cuáles son los cinco principios activos más comunes entre la medicación habitual de todos los pacientes?**

* **Tablas implicadas:**

  * `PATI_USUAL_MEDICATION` (registros de uso habitual, campo `ACIN_ID`)
  * `MEDI_ACTIVE_INGREDIENTS` (join por `ACIN_ID`)
* **Tipo de consulta:** agregación, ordenación y ranking.

```sql
SELECT 
  ai.ACIN_DESCRIPTION_ES AS ingrediente,
  COUNT(*) AS veces_en_uso
FROM PATI_USUAL_MEDICATION um
JOIN MEDI_ACTIVE_INGREDIENTS ai
  ON um.ACIN_ID = ai.ACIN_ID
GROUP BY ai.ACIN_DESCRIPTION_ES
ORDER BY veces_en_uso DESC
LIMIT 5;
```

---

### 5. Evolución temporal de alta de nuevos fármacos

**¿Cuántos medicamentos se han dado de alta (creado) cada mes durante el último año?**

* **Tablas implicadas:**

  * `MEDI_MEDICATIONS` (`MEDI_CREATED_DATE`)
* **Tipo de consulta:** series temporales, extracción de año/mes.

```sql
SELECT 
  STRFTIME('%Y-%m', MEDI_CREATED_DATE) AS mes,
  COUNT(*) AS nuevos_medicamentos
FROM MEDI_MEDICATIONS
WHERE MEDI_CREATED_DATE >= DATE('now','-1 year')
GROUP BY mes
ORDER BY mes;
```

---

### 6. Pacientes con intolerancias específicas


**¿Qué pacientes tienen registrada una alergia al polen?  VS CHATGTP**

**¿Qué pacientes han registrado intolerancia a un alérgeno no medicinal concreto ?**





* **Tablas implicadas:**

  * `ALLE_NOT_MEDICINAL_ALLERGENS`
  * `ALLE_INTOLERANCES` (join por `ALLN_ID`)
  * `PATI_PATIENTS` (si se hace join adicional con paciente)

---

### 7. Análisis de ausencias en citas

**¿Qué porcentaje de citas terminan en ausencia por cada motivo , y cómo varía eso por tipo de cita ?**

* **Tablas implicadas:**

  * `APPO_APPOINTMENTS` (estado de la cita)
  * `APPO_ABSENT_REASONS`, `APPO_ADMISSION_TYPES` (joins correspondientes)
* **Tipo de consulta:** joins múltiples, cálculos de porcentaje y agrupaciones cruzadas.

---
