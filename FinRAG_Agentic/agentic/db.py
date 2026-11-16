from __future__ import annotations
from contextlib import contextmanager
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row

from agentic.config import settings


@contextmanager
def get_connection() -> Iterator[psycopg.Connection[Any]]:
    conn = psycopg.connect(settings.database_url, row_factory=dict_row, connect_timeout=5, options="-c default_transaction_read_only=on")
    try:
        yield conn
    finally:
        conn.close()


def fetch_all(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    """Execute a trusted, static SELECT query with bound parameters."""
    if not query.lstrip().lower().startswith("select") and not query.lstrip().lower().startswith("with"):
        raise ValueError("Разрешены только заранее определённые SELECT/CTE запросы")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return [dict(row) for row in cur.fetchall()]
