"""
Persistence layer for the feedback/bandit loop and request logging.

Uses SQLite for zero-setup local development. In production, swap this
for PostgreSQL (matching the JD's stack) — the schema below is
database-agnostic; only the connection setup would change
(e.g. psycopg2/asyncpg instead of sqlite3).
"""
import sqlite3
import time
import uuid
from contextlib import contextmanager

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS variant_stats (
    agent TEXT NOT NULL,
    variant TEXT NOT NULL,
    uses INTEGER NOT NULL DEFAULT 0,
    ups INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (agent, variant)
);

CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    variant TEXT NOT NULL,
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    rating INTEGER,
    created_at REAL NOT NULL
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(config.DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def record_use(agent: str, variant: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO variant_stats (agent, variant, uses, ups) VALUES (?, ?, 1, 0) "
            "ON CONFLICT(agent, variant) DO UPDATE SET uses = uses + 1",
            (agent, variant),
        )


def record_request(request_id: str, agent: str, variant: str, query: str, answer: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO requests (request_id, agent, variant, query, answer, rating, created_at) "
            "VALUES (?, ?, ?, ?, ?, NULL, ?)",
            (request_id, agent, variant, query, answer, time.time()),
        )


def record_feedback(request_id: str, rating: int) -> bool:
    """rating: 1 for thumbs-up, 0 for thumbs-down. Returns False if request_id unknown."""
    with get_conn() as conn:
        cur = conn.execute("SELECT agent, variant FROM requests WHERE request_id = ?", (request_id,))
        row = cur.fetchone()
        if not row:
            return False
        agent, variant = row
        conn.execute("UPDATE requests SET rating = ? WHERE request_id = ?", (rating, request_id))
        if rating == 1:
            conn.execute(
                "UPDATE variant_stats SET ups = ups + 1 WHERE agent = ? AND variant = ?",
                (agent, variant),
            )
    return True


def get_variant_stats(agent: str) -> dict:
    """Returns {variant: (uses, ups)} for a given agent."""
    with get_conn() as conn:
        cur = conn.execute("SELECT variant, uses, ups FROM variant_stats WHERE agent = ?", (agent,))
        return {row[0]: (row[1], row[2]) for row in cur.fetchall()}


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]
