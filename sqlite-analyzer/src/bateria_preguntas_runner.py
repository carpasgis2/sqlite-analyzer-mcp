# Script para pasar batería de preguntas por el chatbot LangChain
# Guarda resultados en bateria_respuestas.json y los imprime en pantalla
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from src.langchain_chatbot import get_langchain_agent

# Lista de preguntas estructurada por apartados para exportar los títulos en el resultado
PREGUNTAS_POR_APARTADO = [
    ("Información básica de pacientes", [
        "¿Cuál es el número total de pacientes que hay en la base de datos?",
        "¿Cuál es el nombre completo del paciente con ID 1001?",
        "¿Cuál es la fecha de nacimiento del paciente con ID 1005?",
    ]),
    ("Vacunas y fecha", [
        "¿Qué vacunas ha recibido el paciente con ID 1445 y en qué fechas?",
    ]),
    ("Alergias", [
        "¿Qué pacientes tienen alergias registradas y cuáles son esas alergias?",
        "¿Qué pacientes tienen alergias a medicamentos anticoagulantes?",
    ]),
    ("Procedimientos y oncología", [
        "¿Qué procedimientos médicos se han realizado en pacientes con diagnóstico de cáncer de pulmón?",
        "¿Qué pacientes presentan diagnóstico relacionado con cáncer de riñón y cuál es el tipo de riñón associada al diagnóstico?",
        "¿Qué pacientes tienen un marcador tumoral positivo en sus registros y cuáles son los valores asociados?",
    ]),
    ("Intervenciones y anestesia", [
        "¿Cuál es la distribución de tipos de intervención quirúrgica según el tipo de anestesia utilizada?",
    ]),
    ("Fusiones de pacientes", [
        "¿Cuántos pacientes han sido fusionados y en qué fechas se realizaron las fusiones?",
    ]),
    ("Autorizaciones de procedimientos", [
        "¿Qué procedimientos médicos están autorizados y a qué tipo pertenecen?",
    ]),
    ("Unidades médicas y procedimientos", [
        "¿Qué unidades médicas están relacionadas con los procedimientos realizados recientemente?",
    ]),
    ("Oncología avanzada", [
        "¿Qué pacientes tienen tratamiento de hormonoterapia y qué tipo de hormonoterapia reciben?",
        "¿Qué pacientes han recibido radioterapia y qué método de irradiación se utilizó?"
    ]),
    ("Consultas clínicas complejas", [
        "¿Qué médicos han tratado a pacientes diagnosticados con diabetes tipo 2 que también están tomando metformina?",
        "¿Cuál es la prevalencia de alergias alimentarias frente a no alimentarias en hombres y mujeres, por rangos de edad?",
        "¿Qué medicamentos para la alergia no contienen pseudoefedrina y son aptos para niños menores de 12 años?",
        "¿Qué médicos han tratado a pacientes diagnosticados con diabetes tipo 2 que también están tomando metformina?",
        "Listar las provincias donde se han registrado más de 10 casos de gripe A en pacientes mayores de 60 años durante el último invierno.",
        "¿Cuál es la edad promedio de los pacientes que fueron diagnosticados con cáncer de pulmón en cada hospital durante 2023, y cuántos de ellos eran fumadores?",
        "¿Cuántos pacientes distintos recibieron al menos tres tipos diferentes de medicación el mes pasado, y cuál fue el coste total de esas medicaciones para cada paciente?",
        "Mostrar los tratamientos aplicados a pacientes que previamente fueron diagnosticados con hipertensión y posteriormente desarrollaron insuficiencia renal, y qué médico supervisó el tratamiento para la insuficiencia renal.",
        "Comparar el número de reingresos hospitalarios dentro de los 30 días posteriores al alta para pacientes con insuficiencia cardíaca versus aquellos con neumonía, desglosado por grupo de edad (menores de 65 y 65 o más)."
    ]),
    ("Diagnósticos y episodios", [
        "¿Cuántos diagnósticos principales y secundarios se registraron en cada tipo de episodio durante el año 2024?",
    ]),
    ("Alergias y prevalencia", [
        "¿Qué tipo de alergias hay? Hay pacientes con alergia al polen?",
        "¿Cuál es la prevalencia de alergias alimentarias frente a no alimentarias en hombres y mujeres, por rangos de edad?",
        "¿Cuál es la distribución de severidad de alergias por tipo de alergeno en pacientes activos?",
    ]),
    ("Indicadores y factores de riesgo", [
        "¿Qué pacientes tienen al menos un indicador de diabetes positivo y un indicador de hipertensión positivo, y qué edad tenían al detectarse estos indicadores?",
    ]),
    ("Intervenciones y tiempos de ingreso", [
        "¿Cuál es el tiempo medio de ingreso (desde la fecha de ingreso hasta la fecha de alta) para cada combinación de tipo de intervención quirúrgica y tipo de anestesia?",
    ]),
    ("Vacunación y grupos sanguíneos", [
        "¿Cuál es la tasa de pacientes vacunados contra la gripe en cada grupo sanguíneo durante los últimos dos años?",
    ]),
    ("Marcadores tumorales y resultados", [
        "¿Qué pacientes presentan un resultado positivo en un marcador tumoral específico, y cuál fue la fecha y el valor de dicho resultado?",
    ]),
    ("Procedimientos oncológicos", [
        "¿Qué procedimientos oncológicos (biopsia, radioterapia o cirugía) se han realizado en pacientes diagnosticados con cáncer de pulmón, y en qué fechas?",
    ]),
    ("Hormonoterapia y tratamientos", [
        "¿Qué pacientes reciben hormonoterapia y qué tipo de pautas de hormonoterapia están siguiendo actualmente?",
    ]),
    ("Intervenciones médicas autorizadas", [
        "¿Qué intervenciones médicas están autorizadas y cuántos pacientes han recibido cada una de ellas en el último año?",
    ]),
    ("Fusiones y duplicados", [
        "¿Qué pacientes han sido fusionados como registros duplicados, cuántas fusiones se realizaron en 2023 y en qué fechas específicas ocurrieron?",
    ]),
]

def main():
    agent = get_langchain_agent()
    resultados = []
    for apartado, preguntas in PREGUNTAS_POR_APARTADO:
        print(f"\n=== {apartado} ===")
        resultados.append({"apartado": apartado})
        for pregunta in preguntas:
            print(f"\nPregunta: {pregunta}")
            try:
                respuesta = agent.invoke({"input": pregunta})
                output = respuesta.get("output", "Sin respuesta clara del agente.")
            except Exception as e:
                output = f"Error: {str(e)}"
            print(f"Respuesta: {output}")
            resultados.append({"pregunta": pregunta, "respuesta": output})
    # Guardar en archivo TXT con apartados visibles
    with open("bateria_respuestas.txt", "w", encoding="utf-8") as f:
        for item in resultados:
            if "apartado" in item:
                f.write(f"\n=== {item['apartado']} ===\n")
            else:
                f.write(f"Pregunta: {item['pregunta']}\nRespuesta: {item['respuesta']}\n\n")
    # Guardar también en JSON
    with open("bateria_respuestas.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print("\nRespuestas guardadas en bateria_respuestas.txt y bateria_respuestas.json")

if __name__ == "__main__":
    main()
