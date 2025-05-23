import re
import pytest
from src.pipeline import apply_table_aliases, normalize_query_structure

@ pytest.fixture
# Structured info fixture for simple tables
def simple_info():
    return {"tables": ["A"], "columns": ["*"], "joins": [], "conditions": []}


def test_missing_on_join_rendering():
    sql = "SELECT * FROM A JOIN B"
    info = {"tables": ["A", "B"], "columns": ["*"], "joins": [], "conditions": []}
    result = apply_table_aliases(sql, info.get("tables", []))
    # Expect a safe ON clause inserted for missing alias
    assert re.search(r"JOIN B \w+ ON 1=1", result)
    assert "ON ON" not in result


def test_self_join_aliases_unique():
    # Self-join on same table
    sql = "SELECT * FROM A INNER JOIN A ON A.id=A.id"
    info = {"tables": ["A"], "columns": ["*"], "joins": [], "conditions": []}
    result = apply_table_aliases(sql, info.get("tables", []))
    # Should have two unique aliases for A
    aliases = re.findall(r"(t\d+_a)", result)
    assert len(set(aliases)) == 2


def test_reserved_word_alias_not_used():
    # Table named 'JOIN' should get a safe alias, not literal "JOIN"
    sql = "SELECT * FROM JOIN"
    info = {"tables": ["JOIN"], "columns": ["*"], "joins": [], "conditions": []}
    result = apply_table_aliases(sql, info.get("tables", []))
    # Alias should not be the reserved word JOIN
    assert re.search(r"FROM JOIN (?!JOIN)\w+", result)


def test_no_duplicate_aliases_on_from_join_cleanup():
    # Using alias twice should collapse duplicates
    sql = "SELECT * FROM T T T"
    info = {"tables": ["T"], "columns": ["*"], "joins": [], "conditions": []}
    result = apply_table_aliases(sql, info.get("tables", []))
    # Ensure no repeated alias sequence
    assert "T T T" not in result


def test_normalize_query_structure_cols_and_joins():
    raw = {"tables": "A", "columns": "c1", "joins": None, "conditions": "col=1"}
    norm = normalize_query_structure(raw)
    assert isinstance(norm["tables"], list)
    assert isinstance(norm["columns"], list)
    assert isinstance(norm["joins"], list)
    assert isinstance(norm["conditions"], list)
