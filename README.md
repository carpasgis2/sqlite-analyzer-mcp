# SQL Medical Chatbot LangChain Tool

Este repositorio contiene una **Herramienta LangChain** que envuelve tu **pipeline de chatbot médico** (pipeline.py) para consultarlo desde un agente conversacional LangChain.

---

## 🚀 Características

- **SQLMedicalChatbot**  
  Transforma preguntas en lenguaje natural (p. ej. “¿Qué alergias tiene el paciente 1931?”) en consultas SQL seguras y las ejecuta contra una base de datos SQLite.
- **SinaSuiteAndGeneralInformation**  
  Atiende saludos, preguntas sobre la función del asistente o información de la plataforma “SinaSuite”.
- **Reintentos y manejo de errores**  
  Detecta fallos de parseo ReAct y guía al LLM para autocorregirse.
- **Memoria de conversación**  
  Mantiene el historial durante la sesión.

---

## 📦 Estructura del proyecto

.
├── src/
│ ├── langchain_chatbot.py # CLI + configuración del agente LangChain
│ ├── pipeline.py # Lógica de NLU → SQL → ejecución → NLG
│ ├── db_config.py # Conector SQLite (DBConnector)
│ └── data/ # Diccionarios y esquemas JSON
├── db/
│ └── database_new.sqlite3.db # Ejemplo de base de datos médica
├── tests/ # Pruebas unitarias (opcional)
└── requirements.txt # Dependencias Python

yaml
Copiar
Editar

---

## ⚙️ Instalación

1. **Clonar** el repositorio  
   ```bash
   git clone https://github.com/tu-usuario/sqlite-analyzer-mcp.git
   cd sqlite-analyzer-mcp
Crear y activar un entorno virtual

bash
Copiar
Editar
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.\.venv\Scripts\activate    # Windows
Instalar dependencias

bash
Copiar
Editar
pip install -r requirements.txt
Variables de entorno
Crear un archivo .env en la raíz con tus credenciales Deepseek:

ini
Copiar
Editar
DEEPSEEK_API_KEY=sk-…
DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions
DEEPSEEK_MODEL=deepseek-chat
🎯 Uso
Modo interactivo (CLI)
bash
Copiar
Editar
python src/langchain_chatbot.py
Escribe tus preguntas en la terminal.

Para salir, ingresa salir o quit.

Como módulo
python
Copiar
Editar
from src.langchain_chatbot import get_langchain_agent

agent = get_langchain_agent()
response = agent.invoke({"input": "¿Qué diagnósticos tiene el paciente 456?"})
print(response["output"])
🛠 Configuración de la herramienta
SQLMedicalChatbot

Nombre: SQLMedicalChatbot

Función: chatbot_pipeline(q, db_connector)["response"]

Descripción: preguntas específicas de la base de datos médica.

SinaSuiteAndGeneralInformation

Nombre: SinaSuiteAndGeneralInformation

Función: fetch_sinasuite_info(question)

Descripción: saludos y consultas generales.

Manejador de errores

custom_handle_parsing_errors captura OutputParserException y emite guía ReAct.

🔍 Detalles internos
pipeline.py

Preprocesa la pregunta (typos, enriquecimiento semántico).

Construye structured_info (tablas, columnas, condiciones, joins).

Genera SQL con SQLGenerator, valida y ejecuta.

Formatea resultados y genera respuesta en lenguaje natural.

db_config.py

SQLiteConnector: obtiene esquema y ejecuta queries.

🤝 Contribuir
Abre un issue describiendo tu sugerencia o bug.

Crea un fork y una rama feature/… o bugfix/….

Envía un pull request con descripción clara de los cambios.

📄 Licencia
Este proyecto está bajo la MIT License.
