from pathlib import Path


def test_database_tools_use_parameterized_static_queries():
    project = Path(__file__).parents[1]
    source = (project / "agentic" / "tools" / "database.py").read_text(encoding="utf-8")
    db_source = (project / "agentic" / "db.py").read_text(encoding="utf-8")
    assert "cur.execute(query, params)" in db_source
    assert "options=\"-c default_transaction_read_only=on\"" in db_source
    assert "%s" in source
    assert "fetch_all(sql, (" in source
    assert "raw_sql" not in source.lower()


def test_agent_database_role_is_select_only():
    schema = (Path(__file__).parents[1] / "database" / "001_schema.sql").read_text(encoding="utf-8").lower()
    assert "create role finrag_reader" in schema
    assert "grant select on all tables" in schema
    assert "grant insert" not in schema
    assert "grant update" not in schema
    assert "grant delete" not in schema


def test_cashflow_tool_is_parameterized():
    source = (Path(__file__).parents[1] / "agentic" / "tools" / "database.py").read_text(encoding="utf-8")
    assert "def get_initiative_cashflows" in source
    assert "WHERE initiative_id = %s AND scenario = %s" in source
    registry = (Path(__file__).parents[1] / "agentic" / "tools" / "registry.py").read_text(encoding="utf-8")
    assert "calculate_npv_from_cashflows" in registry
    assert "get_initiative_cashflows" in registry
