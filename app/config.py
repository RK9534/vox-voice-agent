import os

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL_NAME = os.environ.get("AGENT_MODEL", "claude-sonnet-4-6")
MAX_TOOL_ITERATIONS = int(os.environ.get("MAX_TOOL_ITERATIONS", "5"))
DOCS_DIR = os.environ.get("DOCS_DIR", os.path.join(
    os.path.dirname(__file__), "..", "data", "docs"
))
TOP_K_CHUNKS = int(os.environ.get("TOP_K_CHUNKS", "3"))
DB_PATH = os.environ.get("DB_PATH", os.path.join(
    os.path.dirname(__file__), "..", "vox.db"
))
# In production: swap DB_PATH/sqlite3 usage in app/db.py for a PostgreSQL
# connection (e.g. via asyncpg or psycopg2) — schema is identical.
EPSILON = float(os.environ.get("BANDIT_EPSILON", "0.2"))  # exploration rate
