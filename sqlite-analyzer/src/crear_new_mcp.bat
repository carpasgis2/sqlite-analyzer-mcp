@echo off
REM Crear la carpeta destino
mkdir ..\..\..\new_mcp\minimal_pipeline

REM Copiar archivos Python esenciales desde el directorio actual
copy pipeline.py ..\..\..\new_mcp\minimal_pipeline\
copy db_connector.py ..\..\..\new_mcp\minimal_pipeline\
copy sql_generator.py ..\..\..\new_mcp\minimal_pipeline\
copy sql_utils.py ..\..\..\new_mcp\minimal_pipeline\
copy whitelist_validator.py ..\..\..\new_mcp\minimal_pipeline\
copy db_relationship_graph.py ..\..\..\new_mcp\minimal_pipeline\
copy flexible_search_config.py ..\..\..\new_mcp\minimal_pipeline\

REM Copiar el archivo de interfaz (ajusta el nombre si no es main.py)
if exist main.py copy main.py ..\..\..\new_mcp\minimal_pipeline\
if exist streamlit_interface.py copy streamlit_interface.py ..\..\..\new_mcp\minimal_pipeline\

REM Copiar la base de datos y el JSON de relaciones si existen
if exist db\database_new.sqlite3.db copy db\database_new.sqlite3.db ..\..\..\new_mcp\minimal_pipeline\
if exist ..\table_relationships.json copy ..\table_relationships.json ..\..\..\new_mcp\minimal_pipeline\

echo.
echo Proceso finalizado. Revisa la carpeta new_mcp\minimal_pipeline
pause