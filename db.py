"""SQLite persistence for agent sessions."""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

DB_PATH = Path.home() / ".ai_cli" / "sessions.db"
DB_PATH.parent.mkdir(exist_ok=True)


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id        TEXT PRIMARY KEY,
            title     TEXT NOT NULL,
            history   TEXT NOT NULL,
            work_dir  TEXT NOT NULL DEFAULT '',
            created   TEXT NOT NULL,
            updated   TEXT NOT NULL
        )
    """)
    # migration: add work_dir column if missing
    cols = [r[1] for r in c.execute("PRAGMA table_info(sessions)").fetchall()]
    if "work_dir" not in cols:
        c.execute("ALTER TABLE sessions ADD COLUMN work_dir TEXT NOT NULL DEFAULT ''")
    c.commit()
    return c


def new_session_id() -> str:
    return uuid.uuid4().hex[:8]


def save_session(session_id: str, title: str, history: list, work_dir: str = ""):
    now = datetime.now().isoformat(timespec="seconds")
    with _conn() as c:
        c.execute("""
            INSERT INTO sessions (id, title, history, work_dir, created, updated)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title,
                history=excluded.history,
                work_dir=excluded.work_dir,
                updated=excluded.updated
        """, (session_id, title, json.dumps(history), work_dir, now, now))


def load_session(session_id: str) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    if not row:
        return None
    return {
        "id":      row["id"],
        "title":   row["title"],
        "history": json.loads(row["history"]),
        "work_dir": row["work_dir"],
        "created": row["created"],
        "updated": row["updated"],
    }


def list_sessions() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, title, work_dir, updated FROM sessions ORDER BY updated DESC"
        ).fetchall()
    return [{"id": r["id"], "title": r["title"], "work_dir": r["work_dir"], "updated": r["updated"]} for r in rows]


def latest_session_for_dir(work_dir: str) -> dict | None:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM sessions WHERE work_dir=? ORDER BY updated DESC LIMIT 1",
            (work_dir,)
        ).fetchone()
    if not row:
        return None
    return {
        "id":      row["id"],
        "title":   row["title"],
        "history": json.loads(row["history"]),
        "work_dir": row["work_dir"],
        "created": row["created"],
        "updated": row["updated"],
    }


def title_from_history(history: list) -> str:
    for msg in history:
        if msg.get("role") == "user":
            text = msg.get("content", "")
            return text[:60] + ("…" if len(text) > 60 else "")
    return "untitled"


# ── prompt_toolkit input history (↑/↓ recall) ─────────────────────────────────

def _ensure_prompt_history_table(c: sqlite3.Connection):
    c.execute("""
        CREATE TABLE IF NOT EXISTS prompt_history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            cwd        TEXT NOT NULL,
            role       TEXT NOT NULL DEFAULT 'user',
            content    TEXT NOT NULL,
            timestamp  TEXT NOT NULL
        )
    """)
    c.commit()


def save_prompt_input(session_id: str, cwd: str, text: str):
    """Append a user-typed input string to the prompt history table."""
    now = datetime.now().isoformat(timespec="seconds")
    with _conn() as c:
        _ensure_prompt_history_table(c)
        c.execute(
            "INSERT INTO prompt_history (session_id, cwd, role, content, timestamp) VALUES (?,?,?,?,?)",
            (session_id, cwd, "user", text, now),
        )


def load_prompt_inputs(session_id: str = None, cwd: str = None) -> list[str]:
    """Return all recorded user inputs for the given session/cwd, oldest first."""
    with _conn() as c:
        _ensure_prompt_history_table(c)
        if session_id and cwd:
            rows = c.execute(
                "SELECT content FROM prompt_history WHERE session_id=? AND cwd=? ORDER BY id ASC",
                (session_id, cwd),
            ).fetchall()
        elif cwd:
            rows = c.execute(
                "SELECT content FROM prompt_history WHERE cwd=? ORDER BY id ASC",
                (cwd,),
            ).fetchall()
        elif session_id:
            rows = c.execute(
                "SELECT content FROM prompt_history WHERE session_id=? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT content FROM prompt_history ORDER BY id ASC"
            ).fetchall()
    return [r["content"] for r in rows]
