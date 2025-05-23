# Documentación: Cambio de política de aliasado SQL en el pipeline del chatbot médico

## Contexto
A partir de mayo de 2025, el pipeline de generación de SQL para el chatbot médico implementa una política de aliasado robusto y sistemático para todas las tablas y columnas en las consultas generadas. Esta decisión se tomó para garantizar la validez, seguridad y ausencia de ambigüedades en los SQL generados, especialmente en escenarios complejos como self-joins, palabras reservadas, columnas ambiguas y duplicidad de JOINs.

## ¿Qué cambia?
- **Alias únicos y sistemáticos:** Todas las tablas reciben un alias generado del tipo `t1_a`, `t2_b`, etc., independientemente de si hay ambigüedad o no.
- **Referencias totalmente calificadas:** Todas las columnas en SELECT, ON, WHERE, etc., se referencian como `alias.columna` (ejemplo: `t1_a.id`, `t2_b.y`).
- **Expansión de SELECT *:** Cuando se usa `SELECT *`, se expande a la lista completa de columnas totalmente calificadas de todas las tablas involucradas.
- **JOINs normalizados:** Se eliminan JOINs duplicados, se corrigen JOINs sin ON, y se evita el uso de palabras reservadas como alias.
- **Compatibilidad con edge cases:** El sistema maneja correctamente self-joins, alias duplicados, palabras reservadas, columnas ambiguas y otros casos complejos.

## Impacto en los tests
- Los tests unitarios y de integración deben esperar y validar la presencia de alias generados (`t1_a`, `t2_b`, etc.) y referencias totalmente calificadas, en vez de la sintaxis clásica (`A.id`, `B.y`).
- Los asserts en los tests han sido adaptados para aceptar cualquier alias generado que cumpla las reglas de unicidad y no-reservado, y para verificar que las columnas estén siempre calificadas por alias.

## Ejemplo de SQL generado (nuevo formato)
```sql
SELECT t1_a.id, t2_b.y FROM A t1_a INNER JOIN B t2_b ON t1_a.id = t2_b.a_id WHERE t1_a.id = ?
```

## Razón del cambio
- **Robustez:** Evita ambigüedades y errores en consultas complejas.
- **Seguridad:** Reduce el riesgo de colisiones de nombres y problemas con palabras reservadas.
- **Mantenimiento:** Facilita la extensión y depuración del pipeline SQL.

## Archivo relevante
- `src/pipeline.py` (función `apply_table_aliases`)
- `src/sql_generator.py`
- `tests/test_sql_generator.py`

---

**Nota:** Si encuentras un test que espera la sintaxis clásica, debes actualizarlo para aceptar el nuevo formato robusto de aliasado.
