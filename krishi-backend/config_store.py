import os
import sqlite3
from pathlib import Path

CONFIG_DB_PATH = Path(os.getenv("CONFIG_DB_PATH", Path(__file__).resolve().parent / "config.db"))
CONFIG_KEYS = (
    "GEMINI_API_KEY",
    "WEATHER_API_KEY",
    "AGMARKNET_KEY",
    "ELEVENLABS_API_KEY",
    "ELEVENLABS_STT_URL",
    "ELEVENLABS_STT_MODEL",
    "GOV_SCHEMES_URL",
    "MODEL_PATH",
    "USE_MOCK_MODEL",
)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(CONFIG_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS api_config (
                key_name TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def sync_env_to_db(env_values: dict[str, str] | None = None) -> dict[str, str]:
    init_db()
    env_values = env_values or dict(os.environ)

    conn = _connect()
    try:
        for key_name in CONFIG_KEYS:
            value = env_values.get(key_name, "")
            if value is None:
                value = ""
            conn.execute(
                """
                INSERT INTO api_config (key_name, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key_name) DO UPDATE SET
                    value = excluded.value,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (key_name, value),
            )
        conn.commit()
    finally:
        conn.close()

    return get_all_config()


def get_api_key(key_name: str) -> str:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT value FROM api_config WHERE key_name = ?",
            (key_name,),
        ).fetchone()
        return row["value"] if row is not None else ""
    finally:
        conn.close()


def get_all_config() -> dict[str, str]:
    conn = _connect()
    try:
        rows = conn.execute("SELECT key_name, value FROM api_config").fetchall()
        return {row["key_name"]: row["value"] for row in rows}
    finally:
        conn.close()


def set_api_key(key_name: str, value: str) -> str:
    if key_name not in CONFIG_KEYS:
        raise ValueError(f"Unsupported config key: {key_name}")

    init_db()
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO api_config (key_name, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key_name) DO UPDATE SET
                value = excluded.value,
                updated_at = CURRENT_TIMESTAMP
            """,
            (key_name, value),
        )
        conn.commit()
    finally:
        conn.close()

    return get_api_key(key_name)


def get_config_status() -> dict[str, bool]:
    settings = get_all_config()
    return {
        "GEMINI_API_KEY":      bool(settings.get("GEMINI_API_KEY", "")),
        "WEATHER_API_KEY":     bool(settings.get("WEATHER_API_KEY", "")),
        "AGMARKNET_KEY":       bool(settings.get("AGMARKNET_KEY", "")),
        "ELEVENLABS_API_KEY":  bool(settings.get("ELEVENLABS_API_KEY", "")),
    }


def purge_stale_keys() -> list[str]:
    """Remove rows whose key_name is no longer in CONFIG_KEYS."""
    conn = _connect()
    removed: list[str] = []
    try:
        rows = conn.execute("SELECT key_name FROM api_config").fetchall()
        for row in rows:
            if row["key_name"] not in CONFIG_KEYS:
                conn.execute("DELETE FROM api_config WHERE key_name = ?", (row["key_name"],))
                removed.append(row["key_name"])
        conn.commit()
    finally:
        conn.close()
    return removed
