#!/usr/bin/env python3
"""ai_cli — Interactive streaming AI REPL for coding tasks.
Start interactive REPL or run single prompts.
Commands: /help /modes /clear /history /quit /claude /kivi
Use bash, read, edit, glob, grep tools.

"""

import os, sys, json, subprocess, threading, time, itertools
from pathlib import Path
from typing import Literal, Union

import readline, tty, termios, fcntl
import re as _re

# ── Pure-ANSI markdown renderer ───────────────────────────────────────────────
_RST  = "\033[0m"
_BOLD = "\033[1m"
_DIM  = "\033[2m"
_ITAL = "\033[3m"
_UL   = "\033[4m"
_STRIKE = "\033[9m"
_H1   = "\033[38;2;130;190;255m"
_H2   = "\033[38;2;100;220;180m"
_H3   = "\033[38;2;200;200;100m"
_H4   = "\033[38;2;220;180;100m"
_CODE = "\033[38;2;180;140;255m"
_CBKG = "\033[48;2;22;22;30m"
_BULL = "\033[38;2;255;160;80m"
_NUM  = "\033[38;2;255;200;80m"
_BORD = "\033[38;2;70;70;70m"
_LINK = "\033[38;2;80;180;255m"
_LINK_URL = "\033[38;2;80;80;120m"
_QUOTE_BAR = "\033[38;2;80;120;80m"
_QUOTE_TXT = "\033[38;2;160;200;160m"
# table
_TB  = "\033[38;2;60;100;140m"        # border
_TH  = "\033[38;2;130;190;255m"       # header text
_THB = "\033[48;2;20;35;55m"          # header bg
_TA  = "\033[48;2;18;18;26m"          # alt row bg

_USD_TO_INR = 92.60
_work_dir = "."
_python_env = sys.executable



def _strip_ansi(s: str) -> str:
    return _re.sub(r'\033\[[0-9;]*m', '', s)


def _inline(text: str) -> str:
    # order matters: bold before italic, code first (no nesting inside code)
    # inline code
    text = _re.sub(r'`([^`]+)`', lambda m: f"{_CBKG}{_CODE} {m.group(1)} {_RST}", text)
    # strikethrough ~~text~~
    text = _re.sub(r'~~(.+?)~~', lambda m: f"{_STRIKE}{m.group(1)}{_RST}", text)
    # bold+italic ***
    text = _re.sub(r'\*\*\*(.+?)\*\*\*', lambda m: f"{_BOLD}{_ITAL}{m.group(1)}{_RST}", text)
    # bold **text** or __text__
    text = _re.sub(r'\*\*(.+?)\*\*|__(.+?)__', lambda m: f"{_BOLD}{m.group(1) or m.group(2)}{_RST}", text)
    # italic *text* or _text_
    text = _re.sub(r'\*(.+?)\*|(?<!\w)_(.+?)_(?!\w)', lambda m: f"{_ITAL}{m.group(1) or m.group(2)}{_RST}", text)
    # links [text](url)
    text = _re.sub(r'\[([^\]]+)\]\(([^)]+)\)', lambda m: f"{_LINK}{m.group(1)}{_RST} {_LINK_URL}({m.group(2)}){_RST}", text)
    # bare URLs
    text = _re.sub(r'(?<!\()https?://\S+', lambda m: f"{_LINK}{m.group(0)}{_RST}", text)
    return text


def _parse_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_sep(row: list[str]) -> bool:
    return bool(row) and all(_re.match(r'^:?-{1,}:?$', c) for c in row if c)


def _ratio_distribute(total: int, widths: list[int]) -> list[int]:
    """Distribute `total` extra chars equally across columns (round-robin for remainder)."""
    if not widths or total <= 0:
        return widths[:]
    n       = len(widths)
    each    = total // n
    leftover = total % n
    return [w + each + (1 if i < leftover else 0) for i, w in enumerate(widths)]


def _render_table(all_lines: list[str]) -> list[str]:
    parsed  = [_parse_row(l) for l in all_lines]
    sep_idx = next((i for i, r in enumerate(parsed) if _is_sep(r)), 1)
    headers = list(parsed[0]) if parsed else []
    rows    = [r for i, r in enumerate(parsed) if i != 0 and i != sep_idx]
    ncols   = max(len(headers), max((len(r) for r in rows), default=0))
    while len(headers) < ncols: headers.append("")
    rows = [r + [""] * (ncols - len(r)) for r in rows]

    # ── Step 1: natural column widths (content only, no padding) ────────────
    nat = [len(headers[c]) for c in range(ncols)]
    for row in rows:
        for c in range(ncols):
            nat[c] = max(nat[c], len(row[c]))

    # ── Step 2: fit to terminal ──────────────────────────────────────────────
    # total width = sum(nat) + 1 padding each side per col + (ncols+1) borders
    try:
        term_w = os.get_terminal_size().columns
    except OSError:
        term_w = 80
    border_overhead = ncols + 1          # one │ per column + leading │
    padding_overhead = ncols * 2         # 1 space each side per column
    available = term_w - border_overhead - padding_overhead

    # shrink: if natural widths exceed available, reduce widest columns first
    while sum(nat) > available and max(nat) > 1:
        mx = max(nat)
        for i in range(ncols):
            if nat[i] == mx:
                nat[i] -= 1
                break

    # ── Step 3: expand to fill full terminal width ───────────────────────────
    slack = available - sum(nat)
    if slack > 0:
        nat = _ratio_distribute(slack, nat)

    widths = nat

    def hline(l, m, r, f="─"):
        return f"{_TB}{l}{m.join(f * (w + 2) for w in widths)}{r}{_RST}"

    def fmt_row(cells, header=False, bg=""):
        parts = []
        for c, w in enumerate(widths):
            raw    = cells[c] if c < len(cells) else ""
            pad    = w - len(raw)
            styled = _inline(raw)
            if header:
                parts.append(f" {_THB}{_TH}{_BOLD}{styled}{_RST}{_THB}{' ' * pad} {_RST}")
            else:
                parts.append(f"{bg} {styled}{' ' * pad} {_RST}")
        return f"{_TB}│{_RST}" + f"{_TB}│{_RST}".join(parts) + f"{_TB}│{_RST}"

    out = [hline("╭", "┬", "╮")]
    out.append(fmt_row(headers, header=True))
    out.append(hline("├", "┼", "┤"))
    for i, row in enumerate(rows):
        out.append(fmt_row(row, bg=_TA if i % 2 == 0 else ""))
    out.append(hline("╰", "┴", "╯"))
    return out


def _print_markdown(text: str):
    lines  = text.split("\n")
    out    = []
    i      = 0
    in_code = False
    code_lang = ""

    while i < len(lines):
        line = lines[i]

        # ── fenced code block ────────────────────────────────────────────────
        if line.startswith("```"):
            if not in_code:
                in_code   = True
                code_lang = line[3:].strip()
                label     = f" {code_lang}" if code_lang else ""
                try: tw = os.get_terminal_size().columns - 4
                except: tw = 76
                out.append(f"{_CBKG}{_BORD}  ┌{'─' * tw}┐{_RST}")
                if code_lang:
                    out.append(f"{_CBKG}{_DIM}  │ {code_lang:<{tw-2}}│{_RST}")
                    out.append(f"{_CBKG}{_BORD}  ├{'─' * tw}┤{_RST}")
            else:
                try: tw = os.get_terminal_size().columns - 4
                except: tw = 76
                out.append(f"{_CBKG}{_BORD}  └{'─' * tw}┘{_RST}")
                in_code = False
            i += 1; continue

        if in_code:
            out.append(f"{_CBKG}{_CODE}  │ {line}{_RST}")
            i += 1; continue

        # ── headings ─────────────────────────────────────────────────────────
        if line.startswith("#### "):
            out.append(f"{_H4}{_BOLD}{line[5:]}{_RST}"); i += 1; continue
        if line.startswith("### "):
            out.append(f"\n{_H3}{_BOLD}{line[4:]}{_RST}"); i += 1; continue
        if line.startswith("## "):
            out.append(f"\n{_H2}{_BOLD}{line[3:]}{_RST}\n"); i += 1; continue
        if line.startswith("# "):
            try: tw = os.get_terminal_size().columns
            except: tw = 80
            title = line[2:]
            out.append(f"\n{_H1}{_BOLD}{_UL}{title}{_RST}")
            out.append(f"{_H1}{'─' * min(len(title), tw)}{_RST}\n")
            i += 1; continue

        # ── horizontal rule ──────────────────────────────────────────────────
        if _re.match(r'^\s*[-*_]{3,}\s*$', line) and not line.strip().startswith("*") or _re.match(r'^\s*---+\s*$', line):
            try: tw = os.get_terminal_size().columns
            except: tw = 80
            out.append(f"{_BORD}{'─' * tw}{_RST}"); i += 1; continue

        # ── table: collect consecutive pipe-containing lines ─────────────────
        # allow blank lines between rows (model sometimes adds them)
        if "|" in line:
            tbl = []
            while i < len(lines):
                l = lines[i]
                if "|" in l:
                    tbl.append(l); i += 1
                elif l.strip() == "" and i + 1 < len(lines) and "|" in lines[i + 1]:
                    i += 1  # skip blank line between rows
                else:
                    break
            if len(tbl) >= 2 and _is_sep(_parse_row(tbl[min(1, len(tbl)-1)])):
                out.extend(_render_table(tbl))
            else:
                for tl in tbl:
                    out.append(_inline(tl))
            continue

        # ── blockquote ───────────────────────────────────────────────────────
        if line.startswith(">"):
            inner = line[1:].lstrip()
            out.append(f"{_QUOTE_BAR}▌{_RST} {_QUOTE_TXT}{_inline(inner)}{_RST}")
            i += 1; continue

        # ── bullets ──────────────────────────────────────────────────────────
        m = _re.match(r'^(\s*)([-*+]) (.*)', line)
        if m:
            depth  = len(m.group(1)) // 2
            glyphs = ["•", "◦", "▸", "▹"]
            glyph  = glyphs[min(depth, len(glyphs)-1)]
            indent = "  " * depth
            out.append(f"{indent}{_BULL}{glyph}{_RST} {_inline(m.group(3))}")
            i += 1; continue

        # ── numbered list ────────────────────────────────────────────────────
        m = _re.match(r'^(\s*)(\d+)\. (.*)', line)
        if m:
            indent = m.group(1)
            num    = m.group(2)
            out.append(f"{indent}{_NUM}{num}.{_RST} {_inline(m.group(3))}")
            i += 1; continue

        # ── normal line ──────────────────────────────────────────────────────
        out.append(_inline(line))
        i += 1

    print("\n".join(out))



# ── Copilot models ────────────────────────────────────────────────────────────
COPILOT_MODELS = {
    "opus": {"id": "claude-opus-4.6", "name": "Claude Opus 4.6", "tier": "Premium"},
    "sonnet": {
        "id": "claude-sonnet-4.6",
        "name": "Claude Sonnet 4.6",
        "tier": "Standard",
    },
    "haiku": {
        "id": "claude-haiku-4.5",
        "name": "Claude Haiku 4.5",
        "tier": "Fast/Cheap",
    },
    "gpt5": {"id": "gpt-5.4", "name": "GPT-5.4", "tier": "Standard"},
    "gpt5m": {"id": "gpt-5-mini", "name": "GPT-5 mini", "tier": "Fast/Cheap"},
    "gpt4": {"id": "gpt-4.1", "name": "GPT-4.1", "tier": "Fast/Cheap"},
}

# ── Plan/Build mode toggle ───────────────────────────────────────────────────
PLANNER_SYSTEM_PROMPT = """
You are in PLAN MODE. Your task is to create a detailed execution plan for the user's request.
DO NOT execute any tools. Only output a structured plan describing what steps should be taken.
Your plan should include:
- What files need to be read or modified
- What commands need to be run
- What tools need to be called
- The expected outcome of each step

Once you've created the plan, ask the user if they want to proceed with execution.
"""
_copilot_model: str | None = None  # None = copilot default



def _env():
    env = os.environ.copy()
    bin_dir = str(Path(_python_env).parent)
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    env["VIRTUAL_ENV"] = str(Path(_python_env).parent.parent)
    return env

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


import os
import json
import time
import inspect
import concurrent.futures
import urllib.request
import urllib.error
import http.client


# ── Pure-Python OpenAI-compatible client ─────────────────────────────────────

class _Obj:
    """Recursively converts a dict into attribute-accessible object."""
    def __init__(self, d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, _Obj(v))
            elif isinstance(v, list):
                setattr(self, k, [_Obj(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, v)
    def __repr__(self):
        return f"_Obj({self.__dict__})"
    # safe attribute access — return None for missing keys like the SDK does
    def __getattr__(self, name):
        return None


class _SSEStream:
    """Iterates over SSE lines from an http.client.HTTPResponse, yields _Obj chunks."""

    def __init__(self, resp: http.client.HTTPResponse):
        self._resp = resp

    def __iter__(self):
        buf = b""
        try:
            while True:
                chunk = self._resp.read(1)
                if not chunk:
                    break
                buf += chunk
                # SSE events are separated by double newline
                while b"\n\n" in buf:
                    event_bytes, buf = buf.split(b"\n\n", 1)
                    data = None
                    for raw in event_bytes.splitlines():
                        line = raw.decode("utf-8", errors="replace")
                        if line.startswith("data:"):
                            data = line[5:].lstrip(" ")
                    if data is None:
                        continue
                    if data.startswith("[DONE]"):
                        return
                    try:
                        yield _Obj(json.loads(data))
                    except json.JSONDecodeError:
                        continue
        finally:
            self._resp.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._resp.close()


class _Completions:
    def __init__(self, base_url: str, api_key: str, timeout: float):
        self._base_url = base_url.rstrip("/")
        self._api_key  = api_key
        self._timeout  = timeout

    def create(self, *, model: str, messages: list, stream: bool = False,
               tools=None, tool_choice=None, temperature=None, top_p=None,
               max_tokens=None, extra_body: dict = None, **kwargs) -> "_SSEStream | _Obj":

        body: dict = {"model": model, "messages": messages, "stream": stream}
        if tools       is not None: body["tools"]       = tools
        if tool_choice is not None: body["tool_choice"] = tool_choice
        if temperature is not None: body["temperature"] = temperature
        if top_p       is not None: body["top_p"]       = top_p
        if max_tokens  is not None: body["max_tokens"]  = max_tokens
        # pass through any extra standard kwargs
        for k, v in kwargs.items():
            if v is not None:
                body[k] = v
        # extra_body fields are merged at top level (same as SDK behaviour)
        if extra_body:
            body.update(extra_body)

        payload = json.dumps(body).encode()
        url = f"{self._base_url}/chat/completions"

        # parse host/port/path from url
        if url.startswith("https://"):
            scheme, rest = "https", url[8:]
        else:
            scheme, rest = "http", url[7:]
        host_part, _, path = rest.partition("/")
        path = "/" + path
        host, _, port_s = host_part.partition(":")
        port = int(port_s) if port_s else (443 if scheme == "https" else 80)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "text/event-stream" if stream else "application/json",
        }

        if scheme == "https":
            import ssl
            conn = http.client.HTTPSConnection(host, port, timeout=self._timeout,
                                               context=ssl.create_default_context())
        else:
            conn = http.client.HTTPConnection(host, port, timeout=self._timeout)

        conn.request("POST", path, body=payload, headers=headers)
        resp = conn.getresponse()

        if resp.status not in (200, 201):
            body_err = resp.read().decode(errors="replace")
            raise RuntimeError(f"HTTP {resp.status}: {body_err}")

        if stream:
            return _SSEStream(resp)
        else:
            return _Obj(json.loads(resp.read().decode()))


class _Chat:
    def __init__(self, base_url, api_key, timeout):
        self.completions = _Completions(base_url, api_key, timeout)


class OpenAI:
    """Pure-Python OpenAI-compatible client (no httpx/openai dependency)."""

    def __init__(self, *, base_url: str = "https://api.openai.com/v1",
                 api_key: str = "EMPTY", timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.api_key  = api_key
        self.chat     = _Chat(self.base_url, api_key, timeout)

# ─────────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, Any, Type

modes = {
    "thinking_general": {
        "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
        "presence_penalty": 1.5, "repetition_penalty": 1.0, "enable_thinking": True
    },
    "thinking_coding": {
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
        "presence_penalty": 0.0, "repetition_penalty": 1.0, "enable_thinking": True,
    },
    "instruct_general": {
        "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
        "presence_penalty": 1.5, "repetition_penalty": 1.0, "enable_thinking": False,
    },
    "instruct_coding": {
        "temperature": 0.3,"top_p": 0.85,"top_k": 10,"min_p": 0.05,
        "presence_penalty": 0.0,"repetition_penalty": 1.05,"enable_thinking": False
    },
    "instruct_reasoning": {
        "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
        "presence_penalty": 1.5, "repetition_penalty": 1.0, "enable_thinking": False,
    }
}

@dataclass
class StructuredOutput:
    choice: Optional[list] = None
    regex: Optional[str] = None
    json: Optional[Any] = None
    grammar: Optional[str] = None

@dataclass
class AIConfig:
    base_url: str = None
    api_key: str = "EMPTY"

    def __post_init__(self):
        if self.base_url is None:
            self.base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")

@dataclass
class AICompletionConfig:
    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    repetition_penalty: float
    enable_thinking: bool

@dataclass(frozen=True)
class Text:
    content: str
    id: str = None

@dataclass(frozen=True)
class Assistant:
    content: str
    id: str = None

@dataclass(frozen=True)
class ToolCall:
    name: str
    id: str
    arguments: str  # always complete JSON string

@dataclass(frozen=True)
class ToolResult:
    name: str
    id: str
    arguments: str
    result: str

@dataclass(frozen=True)
class StepResult:
    """Emitted at the end of each LLM call (AIAgent) or opencode step_finish."""
    step: int                        # 1-based iteration index
    text: str                        # assistant text produced this step
    tool_calls: list                 # list[ToolCall] requested this step
    tool_results: list               # list[ToolResult] executed this step
    messages: list                   # chat snapshot (empty list for OpenCodeAgent)
    stop_reason: str                 # "tool_use" | "end_turn" | "tool-calls" | "stop"
    input_tokens: int = None         # prompt tokens for this step
    output_tokens: int = None        # completion tokens for this step
    meta: dict = None                # extra backend-specific data (cost, cache, session_id, reasoning tokens…)

@dataclass(frozen=True)
class AgentResult:
    """Emitted once when the full run finishes (AIAgent or OpenCodeAgent)."""
    steps: int                       # total iterations
    answer: str                      # last assistant text (the final response)
    tool_calls_total: int            # total tool calls across all steps
    messages: list                   # complete final chat history (empty list for OpenCodeAgent)
    elapsed_s: float                 # wall-clock seconds for the whole run
    input_tokens_total: int = None   # cumulative prompt tokens
    output_tokens_total: int = None  # cumulative completion tokens
    meta: dict = None                # extra: cost_total, cache totals, session_id, reasoning tokens…

@dataclass(frozen=True)
class DoneEvent:
    pass


class Chat:
    def __init__(self, text: str = None, *, role: str = "user",
                 images: list = None, videos: list = None):
        self._messages: list[dict] = []
        if text is not None or images or videos:
            self._append(text, role=role, images=images, videos=videos)

    @property
    def messages(self) -> list[dict]:
        return self._messages

    @property
    def answer(self) -> str:
        for msg in reversed(self._messages):
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
                return msg["content"]
        return ""

    def __repr__(self) -> str:
        return f"Chat(messages={json.dumps(self._messages, indent=2)})"

    def add(self, item, *, role: str = "user", images: list = None, videos: list = None):
        """
        Flexible add — accepts:
          - str        (role="user")      → user text message
          - str        (role="assistant") → assistant text message
          - str        (role="tool")      → tool result; auto-resolves last tool call id
          - ToolCall                      → saves assistant tool_call message
          - ToolResult                    → saves tool_call + tool result messages
        """
        if isinstance(item, str):
            if role == "tool":
                self._append_tool_result_auto(item)
            else:
                self._append(item, role=role, images=images, videos=videos)
        elif isinstance(item, ToolResult):
            self._append_tool_call(item)
            self._append_tool_result(item, item.result)
        elif isinstance(item, ToolCall):
            self._append_tool_call(item)
        elif isinstance(item, Assistant):
            self._append(item.content, role="assistant")
        else:
            raise TypeError(f"Unsupported item type: {type(item)}")
        return self

    def _append_tool_result_auto(self, result: str):
        for msg in reversed(self._messages):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tool_call_id = msg["tool_calls"][-1]["id"]
                self._messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": str(result),
                })
                return
        raise ValueError("No previous tool call found in message history")

    @staticmethod
    def _build_content(text: str, images: list = None, videos: list = None) -> list | str:
        parts = []
        for img in (images or []):
            if img.startswith("data:"):
                url = img
            elif img.startswith("http://") or img.startswith("https://"):
                url = img
            else:
                import base64, mimetypes
                mime, _ = mimetypes.guess_type(img)
                mime = mime or "image/jpeg"
                with open(img, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                url = f"data:{mime};base64,{b64}"
            parts.append({"type": "image_url", "image_url": {"url": url}})

        for vid in (videos or []):
            if vid.startswith("http://") or vid.startswith("https://"):
                url = vid
            else:
                url = f"file://{vid}"
            parts.append({"type": "video_url", "video_url": {"url": url}})

        if text:
            parts.append({"type": "text", "text": text})

        return parts if len(parts) > 1 or (parts and parts[0]["type"] != "text") else text or ""

    def _append(self, text: str, *, role: str, images: list = None, videos: list = None):
        content = self._build_content(text, images, videos)
        self._messages.append({"role": role, "content": content})

    def _append_tool_call(self, tool: "Tool"):
        try:
            args = json.loads(tool.arguments) if tool.arguments else {}
        except json.JSONDecodeError:
            args = tool.arguments
        self._messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": tool.id,
                "type": "function",
                "function": {
                    "name": tool.name,
                    "arguments": json.dumps(args) if isinstance(args, dict) else args,
                }
            }]
        })

    def _append_tool_result(self, tool: "Tool", result: str):
        self._messages.append({
            "role": "tool",
            "tool_call_id": tool.id,
            "content": str(result),
        })


class StreamManager:
    @staticmethod
    def run(stream):
        pending: dict[int, dict] = {}  # index -> {name, id, arguments}
        text_buffer = ""
        for chat_completion_chunk in stream:
            chunk = chat_completion_chunk.choices[0].delta
            if chunk.tool_calls:
                if text_buffer:
                    yield Assistant(content=text_buffer)
                    text_buffer = ""
                for tc in chunk.tool_calls:
                    idx = tc.index if hasattr(tc, 'index') else 0
                    if tc.function.name:
                        # new tool starting — flush any previously completed tool at this index
                        if idx in pending:
                            p = pending[idx]
                            yield ToolCall(name=p["name"], id=p["id"], arguments=p["arguments"])
                        pending[idx] = {"name": tc.function.name, "id": tc.id or "", "arguments": tc.function.arguments or ""}
                    elif idx in pending:
                        pending[idx]["arguments"] += tc.function.arguments or ""
            else:
                text_buffer += chunk.content or ""
                yield Text(content=chunk.content or "")

        if text_buffer:
            yield Assistant(content=text_buffer)
        for idx in sorted(pending):
            p = pending[idx]
            yield ToolCall(name=p["name"], id=p["id"], arguments=p["arguments"])


class AIAgent:
    @staticmethod
    def _ensure_chat(messages) -> Chat:
        """Coerce input to Chat: str -> Chat, list[dict] -> Chat, otherwise return as-is."""
        if isinstance(messages, str):
            return Chat(messages)
        if isinstance(messages, list):
            c = Chat.__new__(Chat)
            c._messages = messages
            return c
        return messages

    def __init__(self, config: AIConfig = None, tools=None, name = None, description = None) -> None:
        self._name = name
        self._description = description
        if config is None:
            config = AIConfig()
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self._global_tools: dict[str, dict] = {}
        self._fn_registry: dict[str, Callable] = {}
        self._resolve_global_tools(tools)

    def _resolve_global_tools(self, tools):
        if tools is None:
            return
        for t in tools:
            if isinstance(t, AIAgent):
                t = t.to_tool()
            if isinstance(t, dict):
                self._global_tools[t['function']['name']] = t
            elif isinstance(t, str):
                raise NotImplementedError("Resolving tools from string is not implemented yet")
            elif callable(t):
                name = getattr(t, '__name__', None) or type(t).__name__.lower()
                self._global_tools[name] = self.fn_to_tool(t)
                self._fn_registry[name] = t
            else:
                raise ValueError(f"Unsupported tool type: {type(t)}")

    def to_tool(self):
        """Wrap this agent as a callable tool (single `input: str` param)."""
        tool_name = self._name or type(self).__name__.lower()
        tool_desc = self._description or f"Agent: {tool_name}"
        agent = self

        import textwrap
        def _run(input: str) -> str:
            chat = Chat(input)
            parts = []
            for event in agent.forward(chat):
                if isinstance(event, Text) and event.id is None:
                    parts.append(event.content)
            return "".join(parts)

        fn_src = textwrap.dedent(f"""
            def {tool_name}(input: str) -> str:
                \"\"\"{tool_desc}\"\"\"
                return _run(input)
        """)
        ns = {"_run": _run}
        exec(fn_src, ns)
        return ns[tool_name]

    def fn_to_tool(self, fn: Callable) -> dict:
        sig = inspect.signature(fn)
        doc = inspect.getdoc(fn) or ""

        # Parse Args: section from docstring to get per-param descriptions
        param_docs: dict[str, str] = {}
        in_args = False
        current_param = None
        for line in doc.splitlines():
            stripped = line.strip()
            if stripped.lower() in ("args:", "arguments:", "parameters:"):
                in_args = True
                continue
            if in_args:
                # blank line or new section header (no leading space) ends Args block
                if not stripped:
                    in_args = False; current_param = None; continue
                if not line.startswith(" ") and not line.startswith("\t"):
                    in_args = False; current_param = None; continue
                # "    param_name: description" or "    param_name (type): description"
                if ":" in stripped and not stripped.startswith("-"):
                    pname, _, pdesc = stripped.partition(":")
                    pname = pname.split("(")[0].strip()
                    if pname in sig.parameters:
                        param_docs[pname] = pdesc.strip()
                        current_param = pname
                        continue
                # continuation line for current param
                if current_param:
                    param_docs[current_param] = param_docs[current_param] + " " + stripped

        # Strip Args: block from description so it's not duplicated
        desc_lines = []
        in_args2 = False
        for line in doc.splitlines():
            stripped = line.strip()
            if stripped.lower() in ("args:", "arguments:", "parameters:",
                                    "# capabilities", "capabilities:"):
                in_args2 = True; continue
            if in_args2:
                if not stripped and not line.startswith(" ") and not line.startswith("\t"):
                    in_args2 = False
                continue
            desc_lines.append(line)
        description = "\n".join(desc_lines).strip()

        props = {}
        required = []
        for name, param in sig.parameters.items():
            ann = param.annotation
            if ann is inspect.Parameter.empty:
                ptype = "string"
            elif ann in (int,):
                ptype = "integer"
            elif ann in (float,):
                ptype = "number"
            elif ann in (bool,):
                ptype = "boolean"
            else:
                ptype = "string"
            prop: dict = {"type": ptype}
            if name in param_docs:
                prop["description"] = param_docs[name]
            props[name] = prop
            if param.default is inspect.Parameter.empty:
                required.append(name)
        return {
            "type": "function",
            "function": {
                "name": getattr(fn, '__name__', None) or type(fn).__name__.lower(),
                "description": description,
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        }

    def _resolve_completion_args(self, mode):
        if isinstance(mode, str):
            if mode not in modes:
                raise ValueError(f"Unsupported mode: {mode}")
            mode_config = modes[mode]
        elif isinstance(mode, AICompletionConfig):
            mode_config = {
                "temperature": mode.temperature, "top_p": mode.top_p,
                "top_k": mode.top_k, "presence_penalty": mode.presence_penalty,
                "repetition_penalty": mode.repetition_penalty,
                "enable_thinking": mode.enable_thinking,
            }
        else:
            raise ValueError(f"Unsupported mode type: {type(mode)}")

        return {
            "temperature": mode_config["temperature"],
            "top_p": mode_config["top_p"],
            "presence_penalty": mode_config["presence_penalty"],
            "extra_body": {
                "top_k": mode_config["top_k"],
                "min_p": mode_config.get("min_p", 0.0),
                "repetition_penalty": mode_config["repetition_penalty"],
                "chat_template_kwargs": {"enable_thinking": mode_config["enable_thinking"]}
            }
        }

    def _resolve_tools(self, tools):
        if not tools:
            return []
        _resolved_tools = []
        for t in tools:
            if isinstance(t, AIAgent):
                t = t.to_tool()
            if isinstance(t, dict):
                _resolved_tools.append(t)
            elif isinstance(t, str):
                if t in self._global_tools:
                    _resolved_tools.append(self._global_tools[t])
                elif t:
                    matched_tools = [tool for name, tool in self._global_tools.items() if name.startswith(t)]
                    _resolved_tools.extend(matched_tools)
                else:
                    raise ValueError(f"Tool name {t} not found in global tools")
            elif callable(t):
                _resolved_tools.append(self.fn_to_tool(t))
            else:
                raise ValueError(f"Unsupported tool type: {type(t)}")
        return _resolved_tools

    def _resolve_structured_output(self, structured_output) -> dict:
        if structured_output is None:
            return {}
        try:
            from pydantic import BaseModel as PydanticBaseModel
            if isinstance(structured_output, type) and issubclass(structured_output, PydanticBaseModel):
                structured_output = StructuredOutput(json=structured_output.model_json_schema())
            elif isinstance(structured_output, PydanticBaseModel):
                structured_output = StructuredOutput(json=type(structured_output).model_json_schema())
        except ImportError:
            pass

        if not isinstance(structured_output, StructuredOutput):
            raise TypeError(f"Unsupported structured_output type: {type(structured_output)}")

        if structured_output.choice is not None:
            return {"extra_body_structured": {"choice": structured_output.choice}}
        if structured_output.regex is not None:
            return {"extra_body_structured": {"regex": structured_output.regex}}
        if structured_output.grammar is not None:
            return {"extra_body_structured": {"grammar": structured_output.grammar}}
        if structured_output.json is not None:
            schema = structured_output.json
            try:
                from pydantic import BaseModel as PydanticBaseModel
                if isinstance(schema, type) and issubclass(schema, PydanticBaseModel):
                    schema = schema.model_json_schema()
                elif isinstance(schema, PydanticBaseModel):
                    schema = type(schema).model_json_schema()
            except ImportError:
                pass
            return {"response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema.get("title", "structured-output"), "schema": schema}
            }}
        return {}

    def _merge_structured_into_kwargs(self, base_kwargs: dict, structured_kwargs: dict) -> dict:
        result = dict(base_kwargs)
        if "extra_body_structured" in structured_kwargs:
            eb = dict(result.get("extra_body", {}))
            eb["structured_outputs"] = structured_kwargs["extra_body_structured"]
            result["extra_body"] = eb
        if "response_format" in structured_kwargs:
            result["response_format"] = structured_kwargs["response_format"]
        return result

    def step(self, messages, model="", max_tokens=None, tools=None,
             mode: AICompletionConfig | str = "thinking_coding",
             structured_output: "StructuredOutput | None" = None, **kwargs):
        messages = self._ensure_chat(messages)
        _messages = messages.messages
        resolve_args = self._resolve_completion_args(mode)
        resolve_args = self._merge_structured_into_kwargs(
            resolve_args, self._resolve_structured_output(structured_output)
        )
        _tools = self._resolve_tools(tools)
        stream = self.client.chat.completions.create(
            messages=_messages, model=model, max_tokens=max_tokens,
            stream=True,
            tools=_tools or None,
            **resolve_args,
            **kwargs
        )
        yield from StreamManager.run(stream)

    def forward(self, chat: Chat, model="", max_tokens=None, tools=None,
                mode: AICompletionConfig | str = "instruct_general",
                loop_stop_condition=None, max_steps=None,
                additional_prompts: list = None,
                structured_output: "StructuredOutput | None" = None, **kwargs):
        """Agentic loop — streams, auto-mutates chat, runs tools sequentially."""
        chat = self._ensure_chat(chat)
        _effective_tools = tools if tools is not None else list(self._global_tools.values())
        _tool_schemas = self._resolve_tools(_effective_tools)
        _fn_registry: dict[str, Callable] = {**self._fn_registry}
        for t in (tools or []):
            if callable(t):
                _fn_registry[t.__name__] = t
        for schema in _tool_schemas:
            name = schema['function']['name']
            if name not in _fn_registry:
                raise ValueError(f"Tool '{name}' has no callable — pass the function, not just a schema dict")

        resolve_args = self._resolve_completion_args(mode)
        resolve_args = self._merge_structured_into_kwargs(
            resolve_args, self._resolve_structured_output(structured_output)
        )
        step = 0
        total_tool_calls = 0
        total_input_tokens = 0
        total_output_tokens = 0
        t_start = time.monotonic()

        additional_prompts = list(additional_prompts or [])

        while True:
            if loop_stop_condition is not None and loop_stop_condition(chat):
                return
            if max_steps is not None and step >= max_steps:
                if additional_prompts:
                    chat.add(additional_prompts.pop(0), role="user")
                    step = 0
                else:
                    return
            step += 1

            _call_kwargs = dict(kwargs)
            _effective_schemas = _tool_schemas or None
            if not _effective_schemas:
                _call_kwargs.pop("tool_choice", None)
            stream = self.client.chat.completions.create(
                messages=chat.messages, model=model, max_tokens=max_tokens,
                stream=True,
                tools=_effective_schemas,
                **resolve_args,
                **_call_kwargs
            )

            tool_calls: list[ToolCall] = []
            step_text = ""
            for chunk in StreamManager.run(stream):
                if isinstance(chunk, ToolCall):
                    tool_calls.append(chunk)
                    # don't add to chat yet — batch all tool calls into one assistant message below
                    yield chunk
                elif isinstance(chunk, Assistant):
                    chat.add(chunk)
                    yield chunk
                elif isinstance(chunk, Text) and chunk.id is None:
                    step_text += chunk.content
                    yield chunk
                else:
                    yield chunk

            # Emit all tool calls as a single assistant message so the history is valid
            if tool_calls:
                chat._messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                })

            def _exec_tool(tc: ToolCall):
                fn = _fn_registry[tc.name]
                args = json.loads(tc.arguments) if tc.arguments else {}
                raw = fn(**args)
                if inspect.isgenerator(raw):
                    parts = []
                    for chunk in raw:
                        if isinstance(chunk, (Text, Assistant)):
                            parts.append(chunk.content)
                    return "".join(parts)
                return str(raw)

            step_results: list[ToolResult] = []
            if len(tool_calls) == 1:
                tc = tool_calls[0]
                result = _exec_tool(tc)
                chat._append_tool_result(tc, result)
                tr = ToolResult(name=tc.name, id=tc.id, arguments=tc.arguments, result=result)
                step_results.append(tr)
                yield tr
            elif tool_calls:
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(tool_calls)) as pool:
                    futures = [pool.submit(_exec_tool, tc) for tc in tool_calls]
                    # collect results in original call order so chat history stays consistent
                    results = [f.result() for f in futures]
                for tc, result in zip(tool_calls, results):
                    chat._append_tool_result(tc, result)
                    tr = ToolResult(name=tc.name, id=tc.id, arguments=tc.arguments, result=result)
                    step_results.append(tr)
                    yield tr

            total_tool_calls += len(tool_calls)
            stop_reason = "tool_use" if tool_calls else "end_turn"
            yield StepResult(
                step=step,
                text=step_text,
                tool_calls=list(tool_calls),
                tool_results=step_results,
                messages=list(chat.messages),
                stop_reason=stop_reason,
            )

            if not tool_calls:
                yield AgentResult(
                    steps=step,
                    answer=step_text,
                    tool_calls_total=total_tool_calls,
                    messages=list(chat.messages),
                    elapsed_s=round(time.monotonic() - t_start, 3),
                )
                yield DoneEvent()
                return

    # ── high-level API ────────────────────────────────────────────────────────

    def task(self, prompt: "str | Chat", **kwargs) -> "Chat":
        """Blocking run: str or Chat in → runs forward → returns Chat (.answer ready)."""
        chat = self._ensure_chat(prompt)
        for _ in self.forward(chat, **kwargs):
            pass
        return chat

    def __call__(self, prompt: "str | Chat", **kwargs) -> "Chat":
        return self.task(prompt, **kwargs)

    def batch(self, prompts: list, **kwargs) -> "list[Chat]":
        """Run prompts in parallel. Returns list[Chat] in same order as input."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            futures = [pool.submit(self.task, p, **kwargs) for p in prompts]
            return [f.result() for f in futures]

    def compress(self, chat: "Chat", keep_last: int = 4) -> "Chat":
        """Summarize old history, keep last keep_last exchange pairs. Returns new Chat."""
        system_msgs = [m for m in chat.messages if m["role"] == "system"]
        non_system = [m for m in chat.messages if m["role"] != "system"]
        cutoff = keep_last * 2
        if len(non_system) <= cutoff:
            return chat
        old, recent = non_system[:-cutoff], non_system[-cutoff:]
        history_text = "\n".join(
            f"{m['role']}: {(m.get('content') or '')[:400]}"
            for m in old
            if isinstance(m.get("content"), str)
        )
        summary = self.task(
            f"Summarize this conversation in 2-4 sentences:\n\n{history_text}"
        ).answer
        compressed = Chat.__new__(Chat)
        compressed._messages = (
            system_msgs
            + [{"role": "user", "content": f"[Prior conversation summary]: {summary}"},
               {"role": "assistant", "content": "Got it, I have context from our earlier conversation."}]
            + recent
        )
        return compressed

    def pipe(self, *agents: "AIAgent") -> "PipelineAgent":
        """Chain this agent with more agents: each agent's answer feeds the next."""
        return PipelineAgent([self] + list(agents))

    def evaluate(self, chat: "Chat", rubric: str) -> float:
        """Judge chat.answer against rubric. Returns score 0.0–1.0."""
        import re
        raw = self.task(
            f"Rate this answer 0–10 (reply with a single number only).\n"
            f"Rubric: {rubric}\nAnswer: {chat.answer}"
        ).answer.strip()
        m = re.search(r"\d+(?:\.\d+)?", raw)
        try:
            return round(min(max(float(m.group()) / 10.0, 0.0), 1.0), 3) if m else 0.0
        except ValueError:
            return 0.0

    def structured(self, prompt: "str | Chat", schema, **kwargs):
        """Run prompt with JSON schema constraint. Returns schema instance if Pydantic, else dict."""
        chat = self._ensure_chat(prompt)
        result = self.task(chat, structured_output=StructuredOutput(json=schema), **kwargs)
        try:
            data = json.loads(result.answer)
        except (json.JSONDecodeError, ValueError):
            return {"raw": result.answer}
        try:
            from pydantic import BaseModel as _PydanticBase
            if isinstance(schema, type) and issubclass(schema, _PydanticBase):
                return schema(**data)
        except (ImportError, Exception):
            pass
        return data


class PipelineAgent:
    """Chain of AIAgents: each agent's .answer becomes the next agent's input."""

    def __init__(self, agents: list):
        self._agents = agents

    def task(self, prompt: "str | Chat", **kwargs) -> "Chat":
        chat = AIAgent._ensure_chat(prompt)
        for i, agent in enumerate(self._agents):
            chat = agent.task(chat.answer if i > 0 else chat, **kwargs)
        return chat

    def __call__(self, prompt: "str | Chat", **kwargs) -> "Chat":
        return self.task(prompt, **kwargs)

    def pipe(self, *more: "AIAgent") -> "PipelineAgent":
        return PipelineAgent(self._agents + list(more))


class OpenCodeAgent:
    """
    Wraps `opencode run --format json` as a streaming Python generator.
    Yields the same event types as AIAgent.forward() so consumers are identical:
      Text        — streaming text chunks
      ToolCall    — tool announced (pre-execution)
      ToolResult  — tool completed (or errored); result field has output or error msg
      StepResult  — per-step summary; meta dict carries {cost, cache_read, cache_write,
                    tokens_reasoning, session_id, tool, title, status, message_id}
      AgentResult — full run summary; meta carries cumulative cost/cache/reasoning + session_id
      DoneEvent   — terminal sentinel (same as AIAgent)

    Usage:
        agent = OpenCodeAgent()
        for event in agent.run("list files"):
            if isinstance(event, Text):
                print(event.content, end="", flush=True)
            elif isinstance(event, ToolCall):
                print(f"\\n[{event.name}] {event.arguments}")
            elif isinstance(event, ToolResult):
                print(f"  → {event.result[:80]}")
            elif isinstance(event, AgentResult):
                print(f"done {event.elapsed_s}s cost=${event.meta['cost_total']:.6f}")
    """

    def __init__(
        self,
        model: str = None,
        agent: str = None,
        working_dir: str = None,
        skip_permissions: bool = False,
    ):
        self.model = model
        self.agent = agent
        self.working_dir = working_dir
        self.skip_permissions = skip_permissions
        self._session_id: str = None  # set after first run, reused for continuity

    def _build_cmd(self, prompt: str, session_id: str = None, continue_last: bool = False) -> list[str]:
        cmd = ["opencode", "run", "--format", "json"]
        if self.model:
            cmd += ["--model", self.model]
        if self.agent:
            cmd += ["--agent", self.agent]
        if self.working_dir:
            cmd += ["--dir", self.working_dir]
        if self.skip_permissions:
            cmd += ["--dangerously-skip-permissions"]
        if session_id:
            cmd += ["--session", session_id]
        elif continue_last:
            cmd += ["--continue"]
        cmd.append(prompt)
        return cmd

    def run(self, prompt: str, session_id: str = None, continue_last: bool = False):
        import subprocess
        cmd = self._build_cmd(prompt, session_id=session_id, continue_last=continue_last)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

        step = 0
        tool_calls_total = 0
        t_start = time.monotonic()
        answer_parts: list[str] = []
        step_tool_calls: list[ToolCall] = []
        step_tool_results: list[ToolResult] = []
        step_text = ""
        step_message_id = ""
        current_session_id = ""

        # per-step token accumulators
        tok_in = tok_out = tok_reason = cache_w = cache_r = step_cost = 0

        # cumulative
        total_input = total_output = total_reason = 0
        total_cache_w = total_cache_r = 0
        total_cost = 0.0

        try:
            for raw_line in proc.stdout:
                raw_line = raw_line.strip()
                if not raw_line or not raw_line.startswith("{"):
                    continue
                try:
                    ev = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                etype = ev.get("type")
                part = ev.get("part", {})
                current_session_id = ev.get("sessionID", current_session_id)

                if etype == "step_start":
                    step += 1
                    step_message_id = part.get("messageID", "")
                    step_text = ""
                    step_tool_calls = []
                    step_tool_results = []
                    tok_in = tok_out = tok_reason = cache_w = cache_r = step_cost = 0

                elif etype == "text":
                    text = part.get("text", "")
                    step_text += text
                    answer_parts.append(text)
                    yield Text(content=text)

                elif etype == "tool_use":
                    state = part.get("state", {})
                    t = state.get("time", {})
                    tool_name = part.get("tool", "")
                    call_id   = part.get("callID", "")
                    inp       = json.dumps(state.get("input", {}))
                    status    = state.get("status", "")
                    output    = state.get("output", "") or state.get("error", "")
                    title     = part.get("title", tool_name)
                    tool_calls_total += 1

                    tc = ToolCall(name=tool_name, id=call_id, arguments=inp)
                    tr = ToolResult(name=tool_name, id=call_id, arguments=inp, result=output)
                    step_tool_calls.append(tc)
                    step_tool_results.append(tr)

                    yield tc
                    yield tr

                elif etype == "step_finish":
                    tokens    = part.get("tokens", {})
                    cache     = tokens.get("cache", {})
                    tok_in    = tokens.get("input", 0)
                    tok_out   = tokens.get("output", 0)
                    tok_reason = tokens.get("reasoning", 0)
                    cache_w   = cache.get("write", 0)
                    cache_r   = cache.get("read", 0)
                    step_cost = part.get("cost", 0.0)
                    reason    = part.get("reason", "")

                    total_input   += tok_in
                    total_output  += tok_out
                    total_reason  += tok_reason
                    total_cache_w += cache_w
                    total_cache_r += cache_r
                    total_cost    += step_cost

                    yield StepResult(
                        step=step,
                        text=step_text,
                        tool_calls=list(step_tool_calls),
                        tool_results=list(step_tool_results),
                        messages=[],
                        stop_reason=reason,
                        input_tokens=tok_in,
                        output_tokens=tok_out,
                        meta={
                            "session_id": current_session_id,
                            "message_id": step_message_id,
                            "tokens_reasoning": tok_reason,
                            "cache_write": cache_w,
                            "cache_read": cache_r,
                            "cost": step_cost,
                        },
                    )

        finally:
            proc.wait()

        final_answer = "".join(answer_parts).lstrip("\n")
        if current_session_id:
            self._session_id = current_session_id

        yield AgentResult(
            steps=step,
            answer=final_answer,
            tool_calls_total=tool_calls_total,
            messages=[],
            elapsed_s=round(time.monotonic() - t_start, 3),
            input_tokens_total=total_input,
            output_tokens_total=total_output,
            meta={
                "session_id": current_session_id,
                "tokens_reasoning_total": total_reason,
                "cache_write_total": total_cache_w,
                "cache_read_total": total_cache_r,
                "cost_total": total_cost,
            },
        )
        yield DoneEvent()





# ── file operation tools ──────────────────────────────────────────────────────



def _resolve(path: str) -> Path:
    return Path(path) if Path(path).is_absolute() else Path(_work_dir) / path


def read(path: str) -> str:
    """Read a file and return its contents."""
    try:
        return _resolve(path).read_text()
    except Exception as e:
        return f"[read error] {e}"


def write(path: str, content: str) -> str:
    """Write content to a file, creating or overwriting it."""
    try:
        p = _resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"[wrote {len(content)} chars to {path}]"
    except Exception as e:
        return f"[write error] {e}"


def edit(path: str, old_string: str, new_string: str) -> str:
    """Replace old_string with new_string in file (first occurrence)."""
    try:
        import difflib

        p = _resolve(path)
        text = p.read_text()
        if old_string not in text:
            return f"[edit error] old_string not found in {path}"
        new_text = text.replace(old_string, new_string, 1)
        p.write_text(new_text)
        diff = list(
            difflib.unified_diff(
                text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )
        return "".join(diff) if diff else "[no changes]"
    except Exception as e:
        return f"[edit error] {e}"


def bash(command: str) -> str:
    """Run a shell command and return stdout+stderr (max 8000 chars)."""
    try:
        r = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=_work_dir,
            env=_env(),
        )
        out = (r.stdout + r.stderr).strip()
        return out[:8000] if out else "[no output]"
    except subprocess.TimeoutExpired:
        return "[bash error] timed out (60s)"
    except Exception as e:
        return f"[bash error] {e}"


def glob(pattern: str) -> str:
    """Find files matching a glob pattern under the working directory."""
    try:
        matches = sorted(str(p) for p in Path(_work_dir).glob(pattern))
        return "\n".join(matches) if matches else "[glob] no matches"
    except Exception as e:
        return f"[glob error] {e}"


def grep(pattern: str, path: str = ".") -> str:
    """Search for regex pattern in files. path is relative to working dir."""
    try:
        search_path = path if Path(path).is_absolute() else str(Path(_work_dir) / path)
        r = subprocess.run(
            ["grep", "-rn", pattern, search_path],
            capture_output=True,
            text=True,
            timeout=30,
            env=_env(),
        )
        out = (r.stdout + r.stderr).strip()
        return out[:8000] if out else "[grep] no matches"
    except Exception as e:
        return f"[grep error] {e}"


def web_search(query: str) -> str:
    """Search the web and return top results as text."""
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=4))
        return "\n\n".join(
            f"{r['title']}\n{r.get('href', '')}\n{r.get('body', '')}" for r in raw
        )
    except Exception as e:
        return f"Search error: {e}"


def web_fetch(url: str) -> str:
    """Fetch a URL and return its markdown content."""
    try:
        from typing import cast
        from scrapling.fetchers import Fetcher
        from scrapling.core.shell import Convertor
        from scrapling.engines._browsers._types import ImpersonateType
        from scrapling.core._types import extraction_types

        page = Fetcher.get(
            url,
            timeout=30,
            retries=3,
            retry_delay=1,
            impersonate=cast(ImpersonateType, "chrome"),
        )
        content = list(
            Convertor._extract_content(
                page,
                css_selector=None,
                extraction_type=cast(extraction_types, "markdown"),
                main_content_only=True,
            )
        )
        return "\n".join(content)
    except Exception as e:
        return f"Error fetching {url}: {e}"


def kivi(prompt: str) -> str:
    """Launch a kivi sub-agent with the given prompt and return its full output.
    Use this to delegate tasks to a parallel AI agent that can use all the same tools."""
    try:
        base_url = os.environ.get("OPENAI_BASE_URL", "http://192.168.170.76:8000/v1")
        sub_chat = Chat()
        sub_tools = [read, write, edit, bash, glob, grep, web_search, web_fetch]
        sub_chat._messages.append({"role": "system", "content": _build_system_prompt(sub_tools)})
        sub_chat.add(prompt)
        agent = AIAgent(config=AIConfig(base_url=base_url), tools=sub_tools)
        parts = []
        for event in agent.forward(sub_chat, mode="instruct_coding", tool_choice="auto"):
            if isinstance(event, Text) and event.id is None:
                parts.append(event.content)
        result = "".join(parts).strip()
        return result[:16000] if result else "[no output]"
    except Exception as e:
        return f"[kivi error] {e}"


class ClaudeTool:
    """Call Claude (claude-sonnet-4-6) as a powerful sub-agent for complex tasks.
    Use for: judging, planning, advising, OOD solving, complex decomposition.
    Costly — prefer the base agent for simple tasks.

    Args:
        prompt: The task or question to send to Claude.
        use_full_chat_history: If true, prepend the full conversation as context.
        pass_parent_prompt: If true, append the last user message as context.

    Session strategy: always fresh — sub-agent calls are independent tasks,
    resuming an unrelated previous session pollutes context and wastes tokens.
    Cache benefit happens naturally when kivi calls claude multiple times within
    the same turn (< 5 min apart), without us forcing session reuse.
    """

    __name__ = "claude"
    _MODEL = "claude-sonnet-4-6"

    def __init__(self, chat: "Chat"):
        self._chat = chat

    def __call__(
        self,
        prompt: str = "",
        use_full_chat_history: bool = False,
        pass_parent_prompt: bool = False,
    ) -> str:
        import asyncio
        from claude_agent_sdk import (
            query,
            ClaudeAgentOptions,
            AssistantMessage,
            ResultMessage,
            TextBlock,
        )
        import threading

        parent = ""
        for m in reversed(self._chat.messages):
            if m["role"] == "user":
                parent = m["content"]
                break

        if not prompt and pass_parent_prompt:
            final_prompt = parent
        else:
            context_parts = []
            if use_full_chat_history:
                context_parts.append(
                    "\n".join(
                        f"{m['role']}: {m['content']}" for m in self._chat.messages
                    )
                )
            if pass_parent_prompt and parent:
                context_parts.append(f"[PARENT_PROMPT]: {parent}")
            final_prompt = (
                ("Context:\n" + "\n\n".join(context_parts) + f"\n\nTask: {prompt}")
                if context_parts
                else prompt
            )

        # Always fresh — independent sub-agent tasks must not share context
        opts = ClaudeAgentOptions(
            model=self._MODEL,
            permission_mode="bypassPermissions",
            cwd=_work_dir,
        )

        async def _run():
            text = ""
            async for msg in query(prompt=final_prompt, options=opts):
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            text += block.text
            return text

        try:
            # asyncio.run() fails if called from a thread with a running loop;
            # run in a dedicated thread to always get a fresh event loop.
            result_box: list = []
            exc_box: list = []
            def _thread_run():
                try:
                    result_box.append(asyncio.run(_run()))
                except Exception as e:
                    exc_box.append(e)
            t = threading.Thread(target=_thread_run, daemon=True)
            t.start()
            t.join()
            if exc_box:
                raise exc_box[0]
            text = result_box[0] if result_box else ""
        except Exception as e:
            return f"[claude error] {e}"

        return text or "[no response]"


# ── Claude usage tracking ─────────────────────────────────────────────────────

_USAGE_DB = Path.home() / ".ai_cli" / "claude_usage.db"


def _init_usage_db():
    import sqlite3
    _USAGE_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(_USAGE_DB) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS turns (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            ts            TEXT NOT NULL,
            cwd           TEXT,
            session_id    TEXT,
            resumed       INTEGER DEFAULT 0,
            prompt_preview TEXT,
            input_tokens  INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cache_read    INTEGER DEFAULT 0,
            cache_write   INTEGER DEFAULT 0,
            cost_usd      REAL DEFAULT 0,
            cost_inr      REAL DEFAULT 0,
            limits_before TEXT,
            limits_after  TEXT
        )""")


def _fetch_limits_json() -> str | None:
    """Fetch current usage limits from Anthropic API, return raw JSON string or None."""
    creds_path = Path.home() / ".claude" / ".credentials.json"
    try:
        token = json.loads(creds_path.read_text())["claudeAiOauth"]["accessToken"]
    except Exception:
        return None
    import urllib.request, urllib.error
    req = urllib.request.Request(
        "https://api.anthropic.com/api/oauth/usage",
        headers={
            "Authorization": f"Bearer {token}",
            "anthropic-beta": "oauth-2025-04-20",
            "User-Agent": "claude-code/2.0.32",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return resp.read().decode()
    except Exception:
        return None


def _log_turn(*, cwd: str, session_id: str, resumed: bool, prompt_preview: str,
               input_tokens: int, output_tokens: int, cache_read: int, cache_write: int,
               cost_usd: float, limits_before: str | None, limits_after: str | None):
    import sqlite3
    from datetime import datetime
    _init_usage_db()
    cost_inr = cost_usd * _USD_TO_INR
    with sqlite3.connect(_USAGE_DB) as con:
        con.execute(
            """INSERT INTO turns
               (ts, cwd, session_id, resumed, prompt_preview,
                input_tokens, output_tokens, cache_read, cache_write,
                cost_usd, cost_inr, limits_before, limits_after)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (datetime.now().isoformat(), cwd, session_id, int(resumed), prompt_preview,
             input_tokens, output_tokens, cache_read, cache_write,
             cost_usd, cost_inr, limits_before, limits_after),
        )


class ClaudeDirectAgent:
    """Direct Claude mode — 30-min time-windowed session reuse for cache warmth."""

    _MODEL = "claude-sonnet-4-6"
    _DB = Path.home() / ".ai_cli" / "claude_direct_sessions.db"
    _SESSION_TTL_MINUTES = 30

    def __init__(self):
        self._DB.parent.mkdir(parents=True, exist_ok=True)
        import sqlite3
        with sqlite3.connect(self._DB) as con:
            con.execute("""CREATE TABLE IF NOT EXISTS sessions (
                cwd TEXT PRIMARY KEY, session_id TEXT, updated TEXT
            )""")
        # Limits fetched once per session start/resume, not per call
        self._session_limits: str | None = None
        self._last_known_session_id: str | None = None

    def _get_session(self) -> tuple[str | None, bool]:
        """Return (session_id, resumed). Expires after TTL; fresh otherwise."""
        import sqlite3
        from datetime import datetime, timedelta
        with sqlite3.connect(self._DB) as con:
            row = con.execute(
                "SELECT session_id, updated FROM sessions WHERE cwd=?",
                (str(Path(_work_dir).resolve()),),
            ).fetchone()
        if not row:
            return None, False
        session_id, updated_str = row
        try:
            updated = datetime.fromisoformat(updated_str)
            if datetime.now() - updated > timedelta(minutes=self._SESSION_TTL_MINUTES):
                return None, False  # cache cold — start fresh
        except Exception:
            return None, False
        return session_id, True

    def _put_session(self, session_id: str):
        import sqlite3
        from datetime import datetime
        cwd = str(Path(_work_dir).resolve())
        with sqlite3.connect(self._DB) as con:
            con.execute(
                "INSERT OR REPLACE INTO sessions (cwd, session_id, updated) VALUES (?,?,?)",
                (cwd, session_id, datetime.now().isoformat()),
            )

    def __call__(
        self, prompt: str, chat: "Chat", plan_mode: bool = False
    ) -> tuple[str, str | None]:
        """Stream Claude response. Returns (full_text, session_id)."""
        import asyncio
        from claude_agent_sdk import (
            query,
            ClaudeAgentOptions,
            AssistantMessage,
            ResultMessage,
            SystemMessage,
            StreamEvent,
            TextBlock,
        )

        context_lines = []
        for m in chat.messages:
            if m["role"] == "system":
                continue
            content = m.get("content", "")
            if isinstance(content, str):
                context_lines.append(f"{m['role']}: {content.split(chr(10))[0][:100]}")
        context = "\n".join(context_lines[-10:]) if context_lines else ""
        final_prompt = (
            f"[Chat context]\n{context}\n\n[New message]\n{prompt}" if context else prompt
        )

        resume, resumed = self._get_session()

        # Fetch limits only when session changes (start or resume after expiry)
        if resume != self._last_known_session_id:
            self._session_limits = _fetch_limits_json()
            self._last_known_session_id = resume
        limits_before = self._session_limits
        resume_label = f"{DIM}[resuming session]{RESET}" if resumed else f"{DIM}[fresh session]{RESET}"
        print(resume_label, flush=True)

        opts = ClaudeAgentOptions(
            model=self._MODEL,
            permission_mode="bypassPermissions",
            resume=resume,
            cwd=_work_dir,
            disallowed_tools=["Edit"] if plan_mode else [],
            include_partial_messages=True,
        )

        usage_data: dict = {}

        async def _run():
            text, sid = "", None
            in_text_block = False
            cur_live: StreamText | None = None
            async for msg in query(prompt=final_prompt, options=opts):
                if isinstance(msg, StreamEvent):
                    ev = msg.event
                    ev_type = ev.get("type")
                    if ev_type == "content_block_start":
                        cb = ev.get("content_block", {})
                        if cb.get("type") == "tool_use":
                            in_text_block = False
                            print(f"\n{DIM}[claude:{cb['name']}]{RESET} ", end="", flush=True)
                        elif cb.get("type") == "text":
                            in_text_block = True
                            cur_live = StreamText()
                            cur_live.start()
                    elif ev_type == "content_block_stop":
                        if not in_text_block:
                            print(flush=True)
                        elif cur_live:
                            cur_live.stop()
                            cur_live = None
                        in_text_block = False
                    elif ev_type == "content_block_delta":
                        delta = ev.get("delta", {})
                        if delta.get("type") == "text_delta" and in_text_block:
                            chunk = delta.get("text", "")
                            text += chunk
                            if cur_live:
                                cur_live.append(chunk)
                        elif delta.get("type") == "input_json_delta":
                            print(f"{DIM}{delta.get('partial_json','')}{RESET}", end="", flush=True)
                elif isinstance(msg, SystemMessage) and msg.subtype == "init":
                    sid = msg.data.get("session_id")
                elif isinstance(msg, ResultMessage):
                    sid = msg.session_id
                    u = msg.usage or {}
                    cost = msg.total_cost_usd or 0.0
                    usage_data.update(
                        input_tokens=u.get("input_tokens", 0),
                        output_tokens=u.get("output_tokens", 0),
                        cache_read=u.get("cache_read_input_tokens", 0),
                        cache_write=u.get("cache_creation_input_tokens", 0),
                        cost_usd=cost,
                    )
                    inp = u.get("input_tokens", 0)
                    out = u.get("output_tokens", 0)
                    cr  = u.get("cache_read_input_tokens", 0)
                    cw  = u.get("cache_creation_input_tokens", 0)
                    parts = [f"in={inp}", f"out={out}"]
                    if cr: parts.append(f"cache_read={cr}")
                    if cw: parts.append(f"cache_write={cw}")
                    cost_str = f"  ${cost:.6f} (₹{cost * _USD_TO_INR:.4f})" if cost else ""
                    print(f"\n{DIM}[claude · {'  '.join(parts)}{cost_str}]{RESET}", flush=True)
            return text, sid

        try:
            text, sid = asyncio.run(_run())
        except Exception as e:
            return f"[claude error] {e}", None

        if sid:
            self._put_session(sid)

        _log_turn(
            cwd=str(Path(_work_dir).resolve()),
            session_id=sid or "",
            resumed=resumed,
            prompt_preview=prompt[:120],
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            cache_read=usage_data.get("cache_read", 0),
            cache_write=usage_data.get("cache_write", 0),
            cost_usd=usage_data.get("cost_usd", 0.0),
            limits_before=limits_before,
            limits_after=None,
        )

        return text or "[no response]", sid


class OpenCodeTool:
    """Run opencode as a sub-agent with the given prompt, streaming its output.
    Use for: delegating coding tasks to opencode (agentic, file-aware).

    Args:
        prompt: The task or question to send to opencode.
    """

    __name__ = "opencode"

    def __call__(self, prompt: str = "") -> str:
        agent = OpenCodeAgent(working_dir=_work_dir, skip_permissions=False)
        text_parts = []
        try:
            for event in agent.run(prompt):
                if isinstance(event, Text):
                    text_parts.append(event.content)
                elif isinstance(event, AgentResult):
                    pass
        except FileNotFoundError:
            return "[opencode error] opencode not found in PATH"
        result = "".join(text_parts).lstrip("\n")
        return result[:16000] if result else "[no output]"


STATIC_TOOLS = [
    read,
    write,
    edit,
    bash,
    glob,
    grep,
    web_search,
    web_fetch,
    kivi,
    OpenCodeTool(),
]


# ── Copilot Agent ─────────────────────────────────────────────────────────────


class CopilotAgent:
    """Wrapper around the GitHub Copilot CLI (`copilot`) for use as an agent."""

    def __init__(self, working_dir: str = "."):
        self.working_dir = working_dir
        self._session_id: str | None = None

    def run(self, prompt: str, *, resume: bool = False):
        """Stream copilot output line-by-line."""
        cmd = ["copilot", "-p", prompt, "--allow-all", "-s", "--stream", "on"]
        if _copilot_model:
            cmd += ["--model", _copilot_model]
        if resume and self._session_id:
            cmd += ["--resume", self._session_id]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=self.working_dir,
        )
        return proc


# ── colors ────────────────────────────────────────────────────────────────────

RESET, BOLD, CYAN, YELLOW = "\033[0m", "\033[1m", "\033[36m", "\033[33m"
GREEN, DIM, RED = "\033[32m", "\033[2m", "\033[31m"

# Anthropic brand palette
CORAL = "\033[38;2;217;119;87m"  # #D97757  kivi / primary
PURPLE = "\033[38;2;139;92;246m"  # #8B5CF6  claude
TEAL = "\033[38;2;6;182;212m"  # #06B6D4  opencode
LIME = "\033[38;2;34;197;94m"  # #22C55E  copilot
CREAM = "\033[38;2;245;230;211m"  # #F5E6D3  accent text

AGENT_COLOR = {"kivi": CORAL, "claude": PURPLE, "opencode": TEAL, "copilot": LIME}

# ── tool arg formatting ────────────────────────────────────────────────────────


def _fmt_args(name: str, arguments: str) -> str:
    """Format tool call arguments for display. Edit shows only path; others show full args."""
    try:
        args = json.loads(arguments) if arguments else {}
    except Exception:
        return f"({arguments})"
    if name == "edit":
        return f"({args.get('path', '')})"
    parts = []
    for k, v in args.items():
        sv = str(v)
        if len(sv) > 120:
            sv = sv[:117] + "…"
        parts.append(f"{k}={sv!r}")
    inner = ", ".join(parts)
    return f"({inner})"


# ── thinking spinner ──────────────────────────────────────────────────────────


class ThinkingSpinner:
    _frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self):
        self._stop = threading.Event()
        self._thread = None
        self._chars = 0

    def start(self):
        self._stop.clear()
        self._chars = 0

        def _spin():
            for f in itertools.cycle(self._frames):
                if self._stop.is_set():
                    break
                sys.stdout.write(
                    f"\r{DIM}{f} thinking ({self._chars} chars)…{RESET}   "
                )
                sys.stdout.flush()
                time.sleep(0.08)

        self._thread = threading.Thread(target=_spin, daemon=True)
        self._thread.start()

    def update(self, n: int):
        self._chars = n

    def stop(self, thinking_text: str):
        self._stop.set()
        if self._thread:
            self._thread.join()
        sys.stdout.write(
            f"\r{DIM}▶ thinking ({len(thinking_text)} chars) — /think to expand{RESET}\n"
        )
        sys.stdout.flush()


def expand_thinking(text: str):
    print(f"{DIM}┌─ thinking ──────────────────────────────{RESET}")
    for line in text.splitlines():
        print(f"{DIM}│ {line}{RESET}")
    print(f"{DIM}└─────────────────────────────────────────{RESET}")



class ParallelToolDisplay:
    _frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]

    def __init__(self, tool_calls: list):
        self._tools    = tool_calls
        self._statuses: dict[str, str] = {tc.id: "running" for tc in tool_calls}
        self._results:  dict[str, str] = {}
        self._lock     = threading.Lock()
        self._stop_ev  = threading.Event()
        self._thread   = None
        self._nlines   = 0   # lines currently drawn

    def _render(self, frame: str) -> str:
        YEL  = "\033[33m"
        GRN  = "\033[32m"
        RST  = "\033[0m"
        DIM_ = "\033[2m"
        with self._lock:
            total = len(self._tools)
            done  = sum(1 for s in self._statuses.values() if s == "done")
            if done == total:
                header = f"{GRN}⚡ {total} tool{'s' if total>1 else ''} — all done{RST}"
            elif done:
                header = f"{YEL}⚡ {total} tool{'s' if total>1 else ''} in parallel  ✓{done}  {frame}{total-done} running{RST}"
            else:
                header = f"{YEL}⚡ {total} tool{'s' if total>1 else ''} in parallel  {frame}{RST}"
            rows = [header]
            for tc in self._tools:
                args_hint = _fmt_args(tc.name, tc.arguments)
                if self._statuses[tc.id] == "done":
                    res     = self._results.get(tc.id, "")
                    preview = res[:80].replace("\n", " ")
                    ellip   = "…" if len(res) > 80 else ""
                    rows.append(f"  {GRN}✓{RST} {YEL}{tc.name}{RST}{args_hint}")
                    rows.append(f"    {DIM_}→ {preview}{ellip}{RST}")
                else:
                    rows.append(f"  {DIM_}{frame} {tc.name}{args_hint}{RST}")
        return "\n".join(rows)

    def _erase(self):
        if self._nlines:
            # move up and erase each line
            sys.stdout.write(f"\033[{self._nlines}A")
            sys.stdout.write("\033[J")
            sys.stdout.flush()
            self._nlines = 0

    def _paint(self, frame: str):
        text = self._render(frame)
        lines = text.count("\n") + 1
        sys.stdout.write(text + "\n")
        sys.stdout.flush()
        self._nlines = lines

    def start(self):
        if not sys.stdout.isatty():
            print(f"{DIM}[tools running...]{RESET}", flush=True)
            return
        self._stop_ev.clear()
        self._paint(self._frames[0])
        def _loop():
            for f in itertools.cycle(self._frames):
                if self._stop_ev.is_set():
                    break
                self._erase()
                self._paint(f)
                time.sleep(0.08)
        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def add_tool(self, tc):
        with self._lock:
            self._tools.append(tc)
            self._statuses[tc.id] = "running"

    def complete(self, tool_id: str, result: str):
        with self._lock:
            self._statuses[tool_id] = "done"
            self._results[tool_id] = result

    def stop(self):
        self._stop_ev.set()
        if self._thread:
            self._thread.join()
        if sys.stdout.isatty():
            self._erase()
            self._paint("✓")
        for tc in self._tools:
            if tc.name == "edit":
                result = self._results.get(tc.id, "")
                for line in result.splitlines():
                    if line.startswith("+") and not line.startswith("+++"):
                        print(f"{GREEN}{line}{RESET}", flush=True)
                    elif line.startswith("-") and not line.startswith("---"):
                        print(f"{RED}{line}{RESET}", flush=True)
                    elif line.startswith("@@"):
                        print(f"{CYAN}{line}{RESET}", flush=True)
                    else:
                        print(f"{DIM}{line}{RESET}", flush=True)


# ── streaming text accumulator ────────────────────────────────────────────────


class StreamText:
    def __init__(self):
        self._buf = []

    def start(self):
        self._buf = []

    def append(self, chunk: str):
        self._buf.append(chunk)
        print(chunk, end="", flush=True)

    def stop(self) -> str:
        text = "".join(self._buf)
        if text:
            # clear the streamed plain text line and re-render as markdown
            sys.stdout.write("\033[2K\r")   # erase current line, go to col 0
            sys.stdout.flush()
            # count visual rows the streamed text used, move cursor to its start
            try:
                width = os.get_terminal_size().columns
            except OSError:
                width = 80
            rows = 0
            col  = 0
            for ch in text:
                if ch == "\n":
                    rows += 1
                    col   = 0
                else:
                    col += 1
                    if col >= width:
                        rows += 1
                        col   = 0
            if rows > 0:
                sys.stdout.write(f"\033[{rows}A")
            sys.stdout.write("\033[J")   # erase from cursor to end of screen
            sys.stdout.flush()
            _print_markdown(text.rstrip("\n"))
        return text


# ── shared event processing ──────────────────────────────────────────────────


def _process_turn(agent, chat, current_mode):
    """Run one agent turn, streaming thinking + text + tools. Returns thinking text or ''."""
    is_thinking = modes.get(current_mode, {}).get("enable_thinking", False)
    thinking_buf, in_thinking = "", is_thinking
    spinner = ThinkingSpinner()
    turn_thinking = ""
    if is_thinking:
        spinner.start()

    live = StreamText()
    live_on = False

    # parallel tool state
    pending_tools: dict[str, object] = {}  # id -> ToolCall (args set, awaiting result)
    display: ParallelToolDisplay | None = None
    _DEBUG = os.environ.get("KIVI_DEBUG", "")

    try:
        for event in agent.forward(chat, mode=current_mode, tool_choice="auto"):
            # ── streaming text chunk ──────────────────────────────────────────
            if isinstance(event, Text) and event.id is None:
                chunk = event.content
                if in_thinking:
                    if "</think>" in chunk:
                        before, _, rest = chunk.partition("</think>")
                        thinking_buf += before
                        turn_thinking = thinking_buf
                        thinking_buf = ""
                        in_thinking = False
                        spinner.stop(turn_thinking)
                        if rest:
                            live.start()
                            live_on = True
                            live.append(rest)
                    else:
                        thinking_buf += chunk
                        spinner.update(len(thinking_buf))
                else:
                    if not live_on:
                        live.start()
                        live_on = True
                    live.append(chunk)

            # ── tool announced with full arguments (pre-execution) ────────────
            elif isinstance(event, ToolCall):
                if live_on:
                    live.stop()
                    live_on = False
                if _DEBUG:
                    print(
                        f"\n{DIM}[debug] tool announced: id={event.id!r} name={event.name} args={event.arguments[:80]!r}{RESET}",
                        flush=True,
                    )
                pending_tools[event.id] = event
                if display is None:
                    display = ParallelToolDisplay(list(pending_tools.values()))
                    display.start()
                else:
                    display.add_tool(event)

            # ── tool result arrived ───────────────────────────────────────────
            elif isinstance(event, ToolResult):
                if _DEBUG:
                    print(
                        f"\n{DIM}[debug] tool result: id={event.id!r} name={event.name} display={display is not None} pending_ids={list(pending_tools.keys())}{RESET}",
                        flush=True,
                    )
                if display is None:
                    print(
                        f"{YELLOW}{BOLD}[tool] {event.name}{RESET} {GREEN}✓{RESET}",
                        flush=True,
                    )
                    pending_tools.clear()
                    live = StreamText()
                    live_on = False
                    continue

                display.complete(event.id, event.result)

                if all(display._statuses[i] == "done" for i in display._statuses):
                    display.stop()
                    display = None
                    pending_tools.clear()

                live = StreamText()
                live_on = False

            # ── step/agent summary (ignore in REPL display) ──────────────────
            elif isinstance(event, (StepResult, AgentResult)):
                pass

    except KeyboardInterrupt:
        if in_thinking:
            spinner.stop(thinking_buf)
        if live_on:
            live.stop()
            live_on = False
        if display:
            display.stop()
        print(f"\n{DIM}[interrupted]{RESET}")
    except Exception as e:
        if in_thinking:
            spinner.stop(thinking_buf)
        if live_on:
            live.stop()
            live_on = False
        if display:
            display.stop()
        print(f"\n{RED}[error] {e}{RESET}")

    if live_on:
        live.stop()
    return turn_thinking


def _build_chat_context(chat: "Chat", max_messages: int = 10) -> str:
    """Build a concise chat context string from recent non-system messages."""
    lines = []
    for m in chat.messages:
        if m["role"] == "system":
            continue
        role = m["role"]
        content = m.get("content", "")
        if not isinstance(content, str):
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        if content.strip():
            lines.append(f"{role}: {content.strip()[:200]}")
    recent = lines[-max_messages:]
    return "\n".join(recent) if recent else ""


def _process_turn_opencode(oc_agent: "OpenCodeAgent", chat: "Chat", user_input: str):
    """Run one opencode turn — same visual style as _process_turn."""
    live = StreamText()
    live_on = False
    display: ParallelToolDisplay | None = None
    pending_tools: dict[str, object] = {}
    answer_parts: list[str] = []

    # Inject chat context on first opencode turn (no session yet)
    if oc_agent._session_id is None:
        ctx = _build_chat_context(chat)
        prompt = (
            f"[Prior conversation context]\n{ctx}\n\n[New message]\n{user_input}"
            if ctx
            else user_input
        )
    else:
        prompt = user_input

    try:
        for event in oc_agent.run(prompt, session_id=oc_agent._session_id):
            if isinstance(event, Text):
                # opencode prefixes responses with \n\n — strip on first chunk
                content = event.content.lstrip("\n") if not live_on else event.content
                if not content:
                    continue
                answer_parts.append(content)
                if not live_on:
                    live.start()
                    live_on = True
                live.append(content)

            elif isinstance(event, ToolCall):
                if live_on:
                    live.stop()
                    live_on = False
                pending_tools[event.id] = event
                if display is None:
                    display = ParallelToolDisplay(list(pending_tools.values()))
                    display.start()
                else:
                    display.add_tool(event)

            elif isinstance(event, ToolResult):
                if display:
                    display.complete(event.id, event.result)
                    if all(display._statuses[i] == "done" for i in display._statuses):
                        display.stop()
                        display = None
                        pending_tools.clear()
                    live = StreamText()
                    live_on = False

            elif isinstance(event, AgentResult):
                if live_on:
                    live.stop()
                    live_on = False
                meta = event.meta or {}
                sid = meta.get("session_id", "")
                if sid:
                    oc_agent._session_id = sid
                cost = meta.get("cost_total", 0.0)
                cost_str = f"  {DIM}${cost:.6f}{RESET}" if cost else ""
                print(
                    f"\n{DIM}[{event.steps} step(s) · {event.tool_calls_total} tool(s) · {event.elapsed_s}s{cost_str}]{RESET}"
                )

    except KeyboardInterrupt:
        if live_on:
            live.stop()
        if display:
            display.stop()
        print(f"\n{DIM}[interrupted]{RESET}")
    except Exception as e:
        if live_on:
            live.stop()
        if display:
            display.stop()
        print(f"\n{RED}[error] {e}{RESET}")

    if live_on:
        live.stop()

    # inject into shared chat so other agents see the exchange
    answer = "".join(answer_parts).strip()
    if answer:
        chat.add(user_input, role="user")
        chat.add(answer, role="assistant")


def _process_turn_copilot(copilot_agent: "CopilotAgent", chat: "Chat", user_input: str):
    """Run one Copilot CLI turn, streaming output and updating chat history."""

    live = StreamText()
    live_on = False
    chunks = []

    # Inject prior chat context so copilot sees cross-agent history
    ctx = _build_chat_context(chat)
    prompt = (
        f"[Prior conversation context]\n{ctx}\n\n[New message]\n{user_input}"
        if ctx
        else user_input
    )

    try:
        proc = copilot_agent.run(prompt, resume=True)
        live.start()
        live_on = True
        for line in proc.stdout:
            live.append(line)
            chunks.append(line)
        proc.wait(timeout=300)
        if live_on:
            live.stop()
            live_on = False
        result = "".join(chunks).strip() or "(no output)"
    except KeyboardInterrupt:
        proc.kill()
        if live_on:
            live.stop()
            live_on = False
        result = "".join(chunks).strip() + "\n[interrupted]"
        print(f"\n{DIM}[interrupted]{RESET}")
    except subprocess.TimeoutExpired:
        proc.kill()
        if live_on:
            live.stop()
            live_on = False
        result = "".join(chunks).strip() + "\n[copilot error] timed out (300s)"
    except Exception as e:
        if live_on:
            live.stop()
            live_on = False
        result = f"[copilot error] {e}"
        print(f"\n{RED}[error] {e}{RESET}")

    # inject into shared chat as plain user/assistant so all agents see the exchange
    if result and not result.startswith("[copilot error]"):
        chat.add(user_input, role="user")
        chat.add(result[:16000], role="assistant")


def _process_turn_claude(claude_agent, chat, user_input, plan_mode: bool = False):
    """Run one Claude turn, streaming output. Manages shared chat context."""
    try:
        chat.add(user_input)
        print()
        text, sid = claude_agent(user_input, chat, plan_mode=plan_mode)
        if text and text != "[no response]" and not text.startswith("[claude error]"):
            chat.add(text, role="assistant")
        print()
    except KeyboardInterrupt:
        print(f"\n{DIM}[interrupted]{RESET}\n")
    except Exception as e:
        print(f"\n{RED}[error] {e}{RESET}\n")


# ── mode picker ───────────────────────────────────────────────────────────────


def pick_mode(current: str) -> str:
    import curses

    mode_list = list(modes.keys())
    idx = mode_list.index(current) if current in mode_list else 0

    def _inner(stdscr):
        nonlocal idx
        curses.curs_set(0)
        while True:
            stdscr.clear()
            _, w = stdscr.getmaxyx()
            stdscr.addstr(
                0, 0, "Select mode  (↑↓ / k j, Enter confirm, q cancel)", curses.A_BOLD
            )
            for i, m in enumerate(mode_list):
                cfg = modes[m]
                think = "thinking" if cfg.get("enable_thinking") else "no-think"
                label = f"  {'>' if i == idx else ' '} {m:<28} temp={cfg['temperature']}  {think}"
                stdscr.addstr(
                    i + 2,
                    0,
                    label[: w - 1],
                    curses.A_REVERSE if i == idx else curses.A_NORMAL,
                )
            stdscr.refresh()
            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")) and idx > 0:
                idx -= 1
            elif key in (curses.KEY_DOWN, ord("j")) and idx < len(mode_list) - 1:
                idx += 1
            elif key in (curses.KEY_ENTER, 10, 13):
                return mode_list[idx]
            elif key in (ord("q"), 27):
                return current

    return curses.wrapper(_inner)


# ── /usage — read ~/.claude/stats-cache.json ─────────────────────────────────


def _show_usage():
    stats_path = Path.home() / ".claude" / "stats-cache.json"
    try:
        data = json.loads(stats_path.read_text())
    except Exception as e:
        print(f"{RED}[usage error] {e}{RESET}")
        return

    from datetime import date
    today = str(date.today())

    # ── today's tokens ──────────────────────────────────────────────────
    today_tokens: dict[str, int] = {}
    for entry in data.get("dailyModelTokens", []):
        if entry["date"] == today:
            today_tokens = entry.get("tokensByModel", {})
            break

    # ── all-time model usage ─────────────────────────────────────────────
    model_usage: dict = data.get("modelUsage", {})

    # only show claude models
    CLAUDE_MODELS = {k: v for k, v in model_usage.items() if "claude" in k.lower()}

    print(f"\n{BOLD}Claude Usage{RESET}  {DIM}(from ~/.claude/stats-cache.json){RESET}\n")

    # today summary
    if today_tokens:
        print(f"  {BOLD}Today ({today}){RESET}")
        for model, tok in sorted(today_tokens.items()):
            if "claude" in model.lower():
                short = model.split("/")[-1]
                print(f"    {CYAN}{short:<32}{RESET} {DIM}{tok:>10,} tokens{RESET}")
        print()

    # all-time per model
    if CLAUDE_MODELS:
        print(f"  {BOLD}All-time by model{RESET}")
        total_in = total_out = total_cache_r = total_cache_w = 0
        for model, u in sorted(CLAUDE_MODELS.items()):
            short = model.split("/")[-1]
            inp   = u.get("inputTokens", 0)
            out   = u.get("outputTokens", 0)
            cr    = u.get("cacheReadInputTokens", 0)
            cw    = u.get("cacheCreationInputTokens", 0)
            total_in += inp; total_out += out; total_cache_r += cr; total_cache_w += cw
            print(f"    {CYAN}{short:<32}{RESET}  in={inp:>10,}  out={out:>10,}  cache_r={cr:>12,}  cache_w={cw:>10,}")
        print(f"    {DIM}{'TOTAL':<32}  in={total_in:>10,}  out={total_out:>10,}  cache_r={total_cache_r:>12,}  cache_w={total_cache_w:>10,}{RESET}")
        print()

    # overall stats
    print(f"  {BOLD}Overall{RESET}")
    print(f"    {DIM}Total sessions : {data.get('totalSessions', 0):,}{RESET}")
    print(f"    {DIM}Total messages : {data.get('totalMessages', 0):,}{RESET}")
    print(f"    {DIM}First session  : {data.get('firstSessionDate', 'n/a')}{RESET}")
    print(f"    {DIM}Last computed  : {data.get('lastComputedDate', 'n/a')}{RESET}")
    print()


# ── /limits — live usage limits from Anthropic OAuth API ──────────────────────

def _show_limits():
    creds_path = Path.home() / ".claude" / ".credentials.json"
    try:
        creds = json.loads(creds_path.read_text())
        token = creds["claudeAiOauth"]["accessToken"]
        sub   = creds["claudeAiOauth"].get("subscriptionType", "?")
        tier  = creds["claudeAiOauth"].get("rateLimitTier", "")
    except Exception as e:
        print(f"{RED}[limits error] can't read credentials: {e}{RESET}")
        return

    import urllib.request, urllib.error
    req = urllib.request.Request(
        "https://api.anthropic.com/api/oauth/usage",
        headers={
            "Authorization": f"Bearer {token}",
            "anthropic-beta": "oauth-2025-04-20",
            "User-Agent": "claude-code/2.0.32",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"{RED}[limits error] HTTP {e.code}: {e.reason}{RESET}")
        return
    except Exception as e:
        print(f"{RED}[limits error] {e}{RESET}")
        return

    def _bar(pct: float, width: int = 28) -> str:
        filled = int(width * pct / 100)
        bar = "█" * filled + "░" * (width - filled)
        color = RED if pct >= 80 else YELLOW if pct >= 50 else GREEN
        return f"{color}{bar}{RESET}"

    def _fmt_reset(ts: str | None) -> str:
        if not ts:
            return ""
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(ts)
            now = datetime.now(timezone.utc)
            total = int((dt - now).total_seconds())
            if total <= 0:
                return "resets now"
            h, rem = divmod(total, 3600)
            m = rem // 60
            return f"resets in {h}h {m}m  ({dt.strftime('%b %d %H:%M')} UTC)"
        except Exception:
            return ts or ""

    def _label(key: str) -> str:
        return key.replace("_", " ").title()

    tier_str = f"  tier={tier}" if tier else ""
    print(f"\n{BOLD}Claude Usage Limits{RESET}  {DIM}plan={sub}{tier_str}{RESET}\n")

    # Separate window limits from extra_usage
    skip = {"extra_usage"}
    window_keys = [k for k, v in data.items() if k not in skip and isinstance(v, dict) and v.get("utilization") is not None]
    null_keys   = [k for k, v in data.items() if k not in skip and v is None]

    if window_keys:
        col = max(len(_label(k)) for k in window_keys) + 2
        for key in window_keys:
            val = data[key]
            pct = val.get("utilization", 0.0)
            reset_str = _fmt_reset(val.get("resets_at"))
            label = _label(key)
            pct_col = RED if pct >= 80 else YELLOW if pct >= 50 else GREEN
            print(f"  {CYAN}{label:<{col}}{RESET}  {_bar(pct)}  {pct_col}{pct:>5.1f}%{RESET}  {DIM}{reset_str}{RESET}")

    if null_keys:
        print(f"\n  {DIM}not active: {', '.join(_label(k) for k in null_keys)}{RESET}")

    extra = data.get("extra_usage", {})
    if extra:
        print(f"\n  {BOLD}Extra Usage{RESET}")
        enabled = extra.get("is_enabled", False)
        print(f"    enabled       : {GREEN+'yes'+RESET if enabled else DIM+'no'+RESET}")
        if enabled or extra.get("used_credits") is not None:
            used  = extra.get("used_credits") or 0
            limit = extra.get("monthly_limit") or 0
            curr  = extra.get("currency") or "USD"
            inr   = used * _USD_TO_INR
            print(f"    used          : {used} {curr}  (₹{inr:.2f})")
            print(f"    monthly limit : {limit} {curr}")
            if extra.get("utilization") is not None:
                print(f"    utilization   : {extra['utilization']}%")

    print()


# ── /insights ────────────────────────────────────────────────────────────────

def _show_insights():
    import sqlite3
    from datetime import datetime, timedelta

    _init_usage_db()
    if not _USAGE_DB.exists():
        print(f"{DIM}no usage data yet — use /claude mode to start tracking{RESET}")
        return

    with sqlite3.connect(_USAGE_DB) as con:
        rows = con.execute(
            "SELECT ts, session_id, resumed, prompt_preview, input_tokens, output_tokens, "
            "cache_read, cache_write, cost_usd, cost_inr, limits_before, limits_after "
            "FROM turns ORDER BY ts DESC LIMIT 500"
        ).fetchall()

    if not rows:
        print(f"{DIM}no usage data yet{RESET}")
        return

    cols = ["ts","session_id","resumed","prompt_preview","input_tokens","output_tokens",
            "cache_read","cache_write","cost_usd","cost_inr","limits_before","limits_after"]
    turns = [dict(zip(cols, r)) for r in rows]
    turns.reverse()  # oldest first for trend

    # ── aggregate by day ────────────────────────────────────────────────
    from collections import defaultdict
    daily: dict[str, dict] = defaultdict(lambda: dict(cost=0.0, inp=0, out=0, cr=0, cw=0, n=0))
    for t in turns:
        day = t["ts"][:10]
        daily[day]["cost"] += t["cost_usd"] or 0
        daily[day]["inp"]  += t["input_tokens"] or 0
        daily[day]["out"]  += t["output_tokens"] or 0
        daily[day]["cr"]   += t["cache_read"] or 0
        daily[day]["cw"]   += t["cache_write"] or 0
        daily[day]["n"]    += 1

    days = sorted(daily.keys())[-14:]  # last 14 days

    def _spark(values: list[float], width: int = 1) -> str:
        bars = " ▁▂▃▄▅▆▇█"
        if not values or max(values) == 0:
            return "▁" * len(values)
        mx = max(values)
        return "".join(bars[min(8, int(v / mx * 8))] for v in values)

    # ── totals ──────────────────────────────────────────────────────────
    total_cost = sum(t["cost_usd"] or 0 for t in turns)
    total_inp  = sum(t["input_tokens"] or 0 for t in turns)
    total_out  = sum(t["output_tokens"] or 0 for t in turns)
    total_cr   = sum(t["cache_read"] or 0 for t in turns)
    total_cw   = sum(t["cache_write"] or 0 for t in turns)
    total_turns = len(turns)
    sessions = len(set(t["session_id"] for t in turns if t["session_id"]))
    resumed_count = sum(1 for t in turns if t["resumed"])
    fresh_count   = total_turns - resumed_count

    print(f"\n{BOLD}Claude Usage Insights{RESET}  {DIM}({total_turns} turns tracked){RESET}\n")

    # ── summary row ─────────────────────────────────────────────────────
    print(f"  {BOLD}All-time{RESET}")
    print(f"    cost          : {GREEN}${total_cost:.6f}{RESET}  {DIM}(₹{total_cost * _USD_TO_INR:.4f}){RESET}")
    print(f"    input tokens  : {total_inp:,}")
    print(f"    output tokens : {total_out:,}")
    print(f"    cache read    : {CYAN}{total_cr:,}{RESET}  {DIM}(saves ~${total_cr * 0.000003:.4f}){RESET}")
    print(f"    cache write   : {total_cw:,}")
    print(f"    sessions      : {sessions}  {DIM}(resumed={resumed_count}  fresh={fresh_count}){RESET}")
    print()

    # ── daily cost sparkline ─────────────────────────────────────────────
    if days:
        costs = [daily[d]["cost"] for d in days]
        spark = _spark(costs)
        print(f"  {BOLD}Cost trend (last {len(days)} days){RESET}")
        print(f"    {YELLOW}{spark}{RESET}")
        print(f"    {DIM}{days[0]}{'':>10}{days[-1]}{RESET}")
        print()

        # per-day table (last 7)
        print(f"  {BOLD}Daily breakdown{RESET}")
        header = f"    {'date':<12} {'turns':>5} {'cost':>10} {'inp':>8} {'out':>8} {'cache_r':>10}"
        print(f"{DIM}{header}{RESET}")
        for d in days[-7:]:
            dd = daily[d]
            cost_col = YELLOW if dd["cost"] > 0.01 else DIM
            print(f"    {CYAN}{d}{RESET}  {dd['n']:>5}  "
                  f"{cost_col}${dd['cost']:>8.6f}{RESET}  "
                  f"{dd['inp']:>8,}  {dd['out']:>8,}  {CYAN}{dd['cr']:>10,}{RESET}")
        print()

    # ── cache efficiency ────────────────────────────────────────────────
    total_readable = total_inp + total_cr
    cache_pct = (total_cr / total_readable * 100) if total_readable else 0
    bar_w = 30
    filled = int(bar_w * cache_pct / 100)
    bar = f"{CYAN}{'█' * filled}{DIM}{'░' * (bar_w - filled)}{RESET}"
    print(f"  {BOLD}Cache efficiency{RESET}")
    print(f"    {bar}  {CYAN}{cache_pct:.1f}%{RESET} of reads served from cache")
    print()

    # ── session strategy ────────────────────────────────────────────────
    print(f"  {BOLD}Session strategy{RESET}")
    print(f"    resumed  : {resumed_count:>4}  {DIM}(warm cache, cheaper){RESET}")
    print(f"    fresh    : {fresh_count:>4}  {DIM}(cold start, full cost){RESET}")
    print()

    # ── 5h limit trend from stored snapshots ────────────────────────────
    snapshots = []
    for t in turns[-20:]:
        for key in ("limits_after", "limits_before"):
            raw = t.get(key)
            if raw:
                try:
                    d = json.loads(raw)
                    fh = d.get("five_hour")
                    if fh and fh.get("utilization") is not None:
                        snapshots.append((t["ts"][:16], fh["utilization"]))
                        break
                except Exception:
                    pass

    if snapshots:
        utils = [u for _, u in snapshots]
        spark = _spark(utils)
        print(f"  {BOLD}5-hour limit pressure (last {len(utils)} snapshots){RESET}")
        color = RED if utils[-1] >= 80 else YELLOW if utils[-1] >= 50 else GREEN
        print(f"    {color}{spark}{RESET}  current={color}{utils[-1]:.0f}%{RESET}")
        print(f"    {DIM}{snapshots[0][0]}  →  {snapshots[-1][0]}{RESET}")
        print()

    # ── top 5 costliest turns ────────────────────────────────────────────
    costly = sorted(turns, key=lambda t: t["cost_usd"] or 0, reverse=True)[:5]
    if any(t["cost_usd"] for t in costly):
        print(f"  {BOLD}Top 5 costliest turns{RESET}")
        for t in costly:
            cost = t["cost_usd"] or 0
            if not cost:
                continue
            preview = (t["prompt_preview"] or "")[:50].replace("\n", " ")
            print(f"    {DIM}{t['ts'][:16]}{RESET}  {YELLOW}${cost:.6f}{RESET}  {DIM}{preview}{RESET}")
        print()


# ── Pure-Python prompt replacement ───────────────────────────────────────────
_SLASH_COMMANDS = [
    "/help", "/modes", "/mode", "/clear", "/history", "/cwd", "/think",
    "/sessions", "/claude", "/kivi", "/opencode", "/copilot", "/model",
    "/usage", "/limits", "/insights", "/quit",
]


class _Trie:
    """Prefix trie over history strings for O(prefix_len) autosuggestion lookup."""
    def __init__(self):
        # Each node: dict of char -> child node, plus '_end' -> original string
        self._root: dict = {}

    def insert(self, s: str):
        node = self._root
        for ch in s:
            if ch not in node:
                node[ch] = {}
            node = node[ch]
        node['_end'] = s   # store the full string at the terminal node

    def suggest(self, prefix: str) -> str | None:
        """Return the most recently inserted string with this prefix, or None."""
        node = self._root
        for ch in prefix:
            if ch not in node:
                return None
            node = node[ch]
        # DFS to find '_end' — returns first found (insertion order via dict)
        return self._dfs(node)

    def _dfs(self, node: dict) -> str | None:
        if '_end' in node:
            return node['_end']
        for ch, child in node.items():
            if ch != '_end':
                result = self._dfs(child)
                if result is not None:
                    return result
        return None


class PromptToolKit:
    """Raw-terminal prompt: history (↑/↓), tab dropdown, Shift+Tab mode toggle, ANSI color, trie autosuggestion."""

    RST  = "\033[0m"
    BOLD = "\033[1m"
    GREY = "\033[90m"
    DIM2 = "\033[2m"
    # cursor / erase
    _EL  = "\033[2K"   # erase line
    _CR  = "\r"

    def __init__(self, session_id: str, work_dir: str, repl_mode_container: list):
        self._sid  = session_id
        self._cwd  = work_dir
        self._mode = repl_mode_container
        self._completions = _SLASH_COMMANDS + [f"/mode {m}" for m in modes]
        self._history: list[str] = load_prompt_inputs(cwd=work_dir)
        # build trie from history (insert oldest first so newest wins on collision)
        self._trie = _Trie()
        for s in self._history:
            self._trie.insert(s)

    # ── ANSI helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _hex_fg(h: str) -> str:
        r, g, b = int(h[1:3],16), int(h[3:5],16), int(h[5:7],16)
        return f"\033[38;2;{r};{g};{b}m"

    def _render_prompt(self, agent: str, color: str) -> str:
        if self._mode[0] == "plan":
            label = f"{agent}{self.GREY}_plan{self.RST}{color}"
        else:
            label = agent
        return f"{self.BOLD}{color}{label}> {self.RST}"

    def _get_suggestion(self, buf: str) -> str:
        """Return the trie suggestion suffix (part after buf), or ''."""
        if not buf:
            return ""
        full = self._trie.suggest(buf)
        if full and full != buf and full.startswith(buf):
            return full[len(buf):]
        return ""

    def _redraw(self, prompt_str: str, buf: str, cur: int):
        """Rewrite the current line: prompt + buffer [+ grey suggestion], cursor at cur."""
        suggestion = self._get_suggestion(buf) if cur == len(buf) and not buf.startswith("/") else ""
        sys.stdout.write(f"{self._CR}{self._EL}{prompt_str}{buf}")
        if suggestion:
            sys.stdout.write(f"{self.GREY}{suggestion}{self.RST}")
            sys.stdout.write(f"\033[{len(suggestion)}D")
        elif cur < len(buf):
            sys.stdout.write(f"\033[{len(buf)-cur}D")
        sys.stdout.flush()
        return suggestion  # caller may need it for → acceptance

    def _show_dropdown(self, prompt_str: str, buf: str, matches: list[str], active: int = 0):
        col     = self._hex_fg("#4a9eff")
        hi_bg   = "\033[48;2;40;40;60m"   # subtle highlight bg for active row
        out = ""
        for i, m in enumerate(matches[:12]):
            if i == active:
                out += f"\r\n  {hi_bg}{col}▸ {m}{self.RST}"
            else:
                out += f"\r\n  {col}  {m}{self.RST}"
        rows = min(len(matches), 12)
        out += f"\033[{rows}A"
        sys.stdout.write(out)
        sys.stdout.flush()

    def _clear_dropdown(self, rows: int):
        if rows == 0:
            return
        sys.stdout.write("\033[s")  # save cursor
        for _ in range(rows):
            sys.stdout.write(f"\r\n{self._EL}")
        sys.stdout.write("\033[u")  # restore cursor
        sys.stdout.flush()

    # ── raw key reader ────────────────────────────────────────────────────────

    @staticmethod
    def _read_key(fd: int) -> str:
        ch = os.read(fd, 1).decode("utf-8", errors="replace")
        if ch == "\x1b":
            # read up to 5 more bytes non-blocking
            old_fl = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, old_fl | os.O_NONBLOCK)
            try:
                rest = b""
                for _ in range(5):
                    try:
                        rest += os.read(fd, 1)
                    except BlockingIOError:
                        break
            finally:
                fcntl.fcntl(fd, fcntl.F_SETFL, old_fl)
            seq = rest.decode("utf-8", errors="replace")
            return "\x1b" + seq
        return ch

    # ── main prompt loop ──────────────────────────────────────────────────────

    def prompt(self, agent: str = "kivi", ac_hex: str = "#D97757") -> str:
        color      = self._hex_fg(ac_hex)
        prompt_str = self._render_prompt(agent, color)
        buf        = []   # list of chars
        cur        = 0    # cursor position in buf
        hist_pos   = len(self._history)
        saved_buf  = []   # saved draft while browsing history
        dd_rows    = 0    # dropdown rows currently shown
        tab_matches: list[str] = []
        tab_idx    = 0
        suggestion = ""   # current trie autosuggestion suffix

        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setraw(fd)
        sys.stdout.write(prompt_str)
        sys.stdout.flush()

        try:
            while True:
                key = self._read_key(fd)

                # ── Enter ────────────────────────────────────────────────────
                if key in ("\r", "\n"):
                    self._clear_dropdown(dd_rows)
                    dd_rows = 0
                    sys.stdout.write("\r\n")
                    sys.stdout.flush()
                    text = "".join(buf).strip()
                    if text:
                        self._history.append(text)
                        self._trie.insert(text)
                        save_prompt_input(self._sid, self._cwd, text)
                    return text

                # ── Ctrl-C ───────────────────────────────────────────────────
                elif key == "\x03":
                    self._clear_dropdown(dd_rows)
                    sys.stdout.write("\r\n")
                    sys.stdout.flush()
                    raise KeyboardInterrupt

                # ── Ctrl-D ───────────────────────────────────────────────────
                elif key == "\x04":
                    self._clear_dropdown(dd_rows)
                    sys.stdout.write("\r\n")
                    sys.stdout.flush()
                    raise EOFError

                # ── Backspace ────────────────────────────────────────────────
                elif key in ("\x7f", "\x08"):
                    self._clear_dropdown(dd_rows); dd_rows = 0
                    tab_matches = []
                    if cur > 0:
                        buf.pop(cur - 1)
                        cur -= 1
                    self._redraw(prompt_str, "".join(buf), cur)

                # ── Shift+Tab  ESC [ Z ───────────────────────────────────────
                elif key == "\x1b[Z":
                    self._clear_dropdown(dd_rows); dd_rows = 0
                    self._mode[0] = "plan" if self._mode[0] == "build" else "build"
                    prompt_str = self._render_prompt(agent, color)
                    self._redraw(prompt_str, "".join(buf), cur)

                # ── Tab ──────────────────────────────────────────────────────
                elif key == "\t":
                    if dd_rows and tab_matches:
                        # dropdown open — cycle down
                        tab_idx = (tab_idx + 1) % len(tab_matches)
                        buf = list(tab_matches[tab_idx]); cur = len(buf)
                        self._clear_dropdown(dd_rows)
                        dd_rows = min(len(tab_matches), 12)
                        self._redraw(prompt_str, "".join(buf), cur)
                        self._show_dropdown(prompt_str, "".join(buf), tab_matches, tab_idx)
                    else:
                        line = "".join(buf)
                        tab_matches = [c for c in self._completions if c.startswith(line)]
                        if tab_matches:
                            tab_idx = 0
                            buf = list(tab_matches[0]); cur = len(buf)
                            dd_rows = min(len(tab_matches), 12)
                            self._redraw(prompt_str, "".join(buf), cur)
                            self._show_dropdown(prompt_str, "".join(buf), tab_matches, tab_idx)

                # ── Arrow Up ─────────────────────────────────────────────────
                elif key == "\x1b[A":
                    if dd_rows and tab_matches:
                        # navigate dropdown up
                        tab_idx = (tab_idx - 1) % len(tab_matches)
                        buf = list(tab_matches[tab_idx]); cur = len(buf)
                        self._clear_dropdown(dd_rows)
                        dd_rows = min(len(tab_matches), 12)
                        self._redraw(prompt_str, "".join(buf), cur)
                        self._show_dropdown(prompt_str, "".join(buf), tab_matches, tab_idx)
                    else:
                        if hist_pos == len(self._history):
                            saved_buf = buf[:]
                        if hist_pos > 0:
                            hist_pos -= 1
                            buf = list(self._history[hist_pos]); cur = len(buf)
                        self._redraw(prompt_str, "".join(buf), cur)

                # ── Arrow Down ───────────────────────────────────────────────
                elif key == "\x1b[B":
                    if dd_rows and tab_matches:
                        # navigate dropdown down
                        tab_idx = (tab_idx + 1) % len(tab_matches)
                        buf = list(tab_matches[tab_idx]); cur = len(buf)
                        self._clear_dropdown(dd_rows)
                        dd_rows = min(len(tab_matches), 12)
                        self._redraw(prompt_str, "".join(buf), cur)
                        self._show_dropdown(prompt_str, "".join(buf), tab_matches, tab_idx)
                    else:
                        if hist_pos < len(self._history) - 1:
                            hist_pos += 1
                            buf = list(self._history[hist_pos]); cur = len(buf)
                        elif hist_pos == len(self._history) - 1:
                            hist_pos = len(self._history)
                            buf = saved_buf[:]; cur = len(buf)
                        self._redraw(prompt_str, "".join(buf), cur)

                # ── Arrow Left ───────────────────────────────────────────────
                elif key == "\x1b[D":
                    if cur > 0:
                        cur -= 1
                        sys.stdout.write("\033[D")
                        sys.stdout.flush()

                # ── Arrow Right — move cursor or accept suggestion ────────────
                elif key == "\x1b[C":
                    if cur < len(buf):
                        cur += 1
                        sys.stdout.write("\033[C")
                        sys.stdout.flush()
                    else:
                        # accept trie suggestion
                        sug = self._get_suggestion("".join(buf))
                        if sug:
                            buf = list("".join(buf) + sug)
                            cur = len(buf)
                            self._redraw(prompt_str, "".join(buf), cur)

                # ── Home / End ───────────────────────────────────────────────
                elif key in ("\x1b[H", "\x01"):   # Home or Ctrl-A
                    cur = 0
                    self._redraw(prompt_str, "".join(buf), cur)
                elif key in ("\x1b[F", "\x05"):   # End or Ctrl-E
                    cur = len(buf)
                    self._redraw(prompt_str, "".join(buf), cur)

                # ── Delete ───────────────────────────────────────────────────
                elif key == "\x1b[3~":
                    self._clear_dropdown(dd_rows); dd_rows = 0
                    if cur < len(buf):
                        buf.pop(cur)
                    self._redraw(prompt_str, "".join(buf), cur)

                # ── Printable chars ──────────────────────────────────────────
                elif len(key) == 1 and key >= " ":
                    self._clear_dropdown(dd_rows); dd_rows = 0
                    tab_matches = []
                    buf.insert(cur, key)
                    cur += 1
                    self._redraw(prompt_str, "".join(buf), cur)
                    # auto-open dropdown when line starts with /
                    if "".join(buf).startswith("/"):
                        line = "".join(buf)
                        tab_matches = [c for c in self._completions if c.startswith(line)]
                        if tab_matches:
                            tab_idx = 0
                            dd_rows = min(len(tab_matches), 12)
                            self._show_dropdown(prompt_str, "".join(buf), tab_matches, tab_idx)

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def make_session(session_id: str, work_dir: str, repl_mode_container: list = None) -> PromptToolKit:
    return PromptToolKit(session_id, work_dir, repl_mode_container or ["build"])


def _make_keybindings(repl_mode_container):
    # No-op: Shift+Tab is handled inline in the REPL loop via _handle_shift_tab
    return repl_mode_container


# ── @ directive expansion ─────────────────────────────────────────────────────

_IGNORE_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache", ".pytest_cache", "dist", "build", ".tox"}


def _generate_tree(root: str, max_depth: int = 4) -> str:
    """Return a pretty tree string for *root* up to *max_depth* levels."""
    lines = [os.path.basename(root) or root]

    def _walk(path: str, prefix: str, depth: int):
        if depth > max_depth:
            return
        try:
            entries = sorted(os.listdir(path))
        except PermissionError:
            return
        entries = [e for e in entries if not (e.startswith(".") and e not in ()) or e == ".env"]
        # Filter ignored dirs
        visible = [e for e in entries if not (os.path.isdir(os.path.join(path, e)) and e in _IGNORE_DIRS)]
        for i, name in enumerate(visible):
            connector = "└── " if i == len(visible) - 1 else "├── "
            lines.append(f"{prefix}{connector}{name}")
            full = os.path.join(path, name)
            if os.path.isdir(full):
                extension = "    " if i == len(visible) - 1 else "│   "
                _walk(full, prefix + extension, depth + 1)

    _walk(root, "", 1)
    return "\n".join(lines)


def _expand_at_directives(prompt: str, work_dir: str) -> str:
    """Replace @-directives in *prompt* with their expanded content."""
    import re

    # @tree → directory tree of work_dir
    if "@tree" in prompt:
        tree = _generate_tree(work_dir)
        block = f"<tree>\n{tree}\n</tree>"
        prompt = prompt.replace("@tree", block)

    # @file:<path> → file contents
    def _expand_file(m: re.Match) -> str:
        fpath = m.group(1)
        full = fpath if os.path.isabs(fpath) else os.path.join(work_dir, fpath)
        try:
            content = Path(full).read_text(errors="replace")
            return f"<file path=\"{fpath}\">\n{content}\n</file>"
        except FileNotFoundError:
            return f"[file not found: {fpath}]"

    prompt = re.sub(r"@file:(\S+)", _expand_file, prompt)

    # @git → recent git log + diff stat
    if "@git" in prompt:
        try:
            log = subprocess.check_output(
                ["git", "-C", work_dir, "log", "--oneline", "-10"], text=True, stderr=subprocess.DEVNULL
            )
            diff = subprocess.check_output(
                ["git", "-C", work_dir, "diff", "--stat", "HEAD"], text=True, stderr=subprocess.DEVNULL
            )
            block = f"<git>\n## Recent commits\n{log}\n## Diff stat\n{diff}\n</git>"
        except Exception:
            block = "[git info unavailable]"
        prompt = prompt.replace("@git", block)

    return prompt


# ── REPL ──────────────────────────────────────────────────────────────────────


def _build_system_prompt(tools: list) -> str:
    """Build system prompt dynamically from the actual resolved tool schemas."""
    base_url = os.environ.get("OPENAI_BASE_URL", "http://192.168.170.76:8000/v1")
    agent = AIAgent(config=AIConfig(base_url=base_url), tools=tools)
    schemas = agent._resolve_tools(tools)

    lines = [
        "You are Kivi Agent, official CLI for Kivi model.",
        "You are an interactive agent that helps users with software engineering tasks. Use the instructions below and the tools available to you to assist the user.",
        "When multiple independent tasks can be done in parallel, call multiple tools at once — they run concurrently.",
        "",
        "## Tools",
    ]
    for s in schemas:
        fn = s["function"]
        name = fn["name"]
        desc = (fn.get("description") or "").splitlines()[0]  # first line only
        params = fn.get("parameters", {}).get("properties", {})
        required = fn.get("parameters", {}).get("required", [])
        param_parts = []
        for pname, pinfo in params.items():
            ptype = pinfo.get("type", "string")
            pdesc = pinfo.get("description", "")
            opt = "" if pname in required else ", optional"
            if pdesc:
                param_parts.append(f"    {pname} ({ptype}{opt}): {pdesc}")
            else:
                param_parts.append(f"    {pname} ({ptype}{opt})")
        sig = ", ".join((pn if pn in required else f"{pn}=...") for pn in params)
        lines.append(f"- **{name}**({sig}): {desc}")
        lines.extend(param_parts)

    lines += [
        "",
        "Be concise and direct. Always use tools to act on files/shell instead of describing what to do.",
    ]
    return "\n".join(lines)


def _render_history(history: list):
    """Re-render saved chat history in the same visual style as a live session."""
    # Pair up: find user messages and the assistant reply that follows
    i = 0
    non_system = [m for m in history if m.get("role") not in ("system",)]
    while i < len(non_system):
        msg = non_system[i]
        role = msg.get("role")
        content = msg.get("content") or ""
        if not isinstance(content, str):
            # multipart (images etc) — just grab text parts
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )

        if role == "user" and content.strip():
            print(f"{DIM}[kivi]>{RESET} {content.strip()}")
            # look for the next assistant message
            j = i + 1
            while j < len(non_system) and non_system[j].get("role") != "user":
                if non_system[j].get("role") == "assistant":
                    ac = non_system[j].get("content") or ""
                    if isinstance(ac, str) and ac.strip():
                        print()
                        _print_markdown(ac.strip())
                        print()
                j += 1
            i = j
        else:
            i += 1
    print(f"{DIM}── end of history ──────────────────────────────{RESET}\n")


def run_repl(work_dir: str, session_id: str = None, initial_history: list = None):
    global _work_dir
    _work_dir = work_dir

    base_url = os.environ.get("OPENAI_BASE_URL", "http://192.168.170.76:8000/v1")

    chat = Chat()
    # Use all tools in build mode, filter in plan mode
    all_tools = STATIC_TOOLS + [ClaudeTool(chat)]
    if initial_history:
        chat._messages = list(initial_history)
    else:
        chat._messages.append(
            {"role": "system", "content": _build_system_prompt(all_tools)}
        )

    agent = AIAgent(config=AIConfig(base_url=base_url), tools=all_tools)
    claude_agent = ClaudeDirectAgent()
    oc_agent = OpenCodeAgent(working_dir=_work_dir, skip_permissions=False)
    copilot_agent = CopilotAgent(working_dir=_work_dir)
    current_mode = "instruct_coding"
    current_agent = "kivi"
    last_thinking = ""
    # repl_mode: "build" (default) or "plan" (edit tool disabled)
    repl_mode_container = ["build"]

    if session_id is None:
        session_id = new_session_id()
    resumed = initial_history is not None
    _make_keybindings(repl_mode_container)
    session = make_session(session_id, work_dir, repl_mode_container)

    ac = AGENT_COLOR.get(current_agent, CORAL)
    print(f"""           kivi v1.0 · AI Agent
 ▐▛███▜▌   {_work_dir}
▝▜█████▛▘  {DIM}endpoint: {base_url}{RESET}
  ▘▘ ▝▝    {DIM}{'resumed ' if resumed else ''}session {RESET}{CREAM}{session_id}{RESET}"{DIM}|{RESET}  mode: {CREAM}{current_mode}{RESET}"{DIM}|{RESET}  agent: {ac}{BOLD}{current_agent}{RESET}"{DIM}""" 
    )
    if resumed and initial_history:
        _render_history(initial_history)

    _AGENT_HEX = {
        "kivi": "#D97757",
        "claude": "#8B5CF6",
        "opencode": "#06B6D4",
        "copilot": "#22C55E",
    }

    while True:
        ac_hex = _AGENT_HEX.get(current_agent, "#D97757")

        try:
            user_input = session.prompt(current_agent, ac_hex).strip()
        except KeyboardInterrupt:
            # Ctrl+C clears current input, stay in loop
            print()
            continue
        except EOFError:
            # Ctrl+D exits
            print(f"\n{DIM}bye{RESET}")
            break
        if not user_input:
            continue

        # ── ! bash shortcut (injected as tool call into chat) ─────────────
        if user_input.startswith("!"):
            cmd = user_input[1:].strip()
            if cmd:
                import uuid, json

                result = bash(cmd)
                print(f"{DIM}{result}{RESET}")
                tool_id = f"call_{uuid.uuid4().hex[:24]}"
                chat._messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": tool_id,
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": json.dumps({"command": cmd}),
                                },
                            }
                        ],
                    }
                )
                chat._messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": result,
                    }
                )
            continue
        # ── slash commands ────────────────────────────────────────────────
        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()

            if cmd == "/help":
                print(f"\n{BOLD}Commands:{RESET}")
                for c, d in [
                    ("/modes", "interactive mode picker"),
                    ("/mode <name>", "switch mode"),
                    ("/clear", "clear conversation"),
                    ("/history", "message count"),
                    ("/cwd", "working directory"),
                    ("/think", "expand thinking"),
                    ("/sessions", "list sessions"),
                    ("/claude", "switch to Claude mode"),
                    ("/kivi", "switch to Kivi mode"),
                    ("/opencode", "switch to OpenCode mode"),
                    ("/copilot [prompt]", "switch to/ask GitHub Copilot CLI"),
                    ("/model [name]", "pick copilot model"),
                    ("/usage", "show Claude token usage from stats-cache"),
                    ("/limits", "show live 5h/7d usage limits from Anthropic API"),
                    ("/insights", "show cost/token trends from usage log"),
                    ("/quit", "quit"),
                ]:
                    print(f"  {CYAN}{c:<16}{RESET} {d}")
                print()
            elif cmd == "/modes":
                sel = pick_mode(current_mode)
                current_mode = sel
                print(
                    f"{GREEN if sel != current_mode else DIM}mode → {current_mode}{RESET}"
                )
            elif cmd == "/mode":
                if len(parts) >= 2 and parts[1] in modes:
                    current_mode = parts[1]
                    print(f"{GREEN}mode → {current_mode}{RESET}")
                else:
                    print(f"{RED}unknown mode{RESET} — valid: {', '.join(modes)}")
            elif cmd == "/claude":
                current_agent = "claude"
            elif cmd == "/kivi":
                current_agent = "kivi"
            elif cmd == "/opencode":
                current_agent = "opencode"
            elif cmd == "/copilot":
                prompt_text = user_input[len(cmd) :].strip()
                if not prompt_text:
                    current_agent = "copilot"
                else:
                    # one-shot: run prompt and inject into chat
                    print()
                    _process_turn_copilot(copilot_agent, chat, prompt_text)
                    print()
            elif cmd == "/model":
                global _copilot_model
                arg = user_input[len("/model") :].strip().lower()
                if arg:
                    match = next(
                        (
                            v
                            for k, v in COPILOT_MODELS.items()
                            if k == arg or v["id"] == arg
                        ),
                        None,
                    )
                    if match:
                        _copilot_model = match["id"]
                        print(
                            f"{GREEN}copilot model → {match['name']} ({match['id']}){RESET}"
                        )
                    else:
                        print(
                            f"{RED}unknown model '{arg}'{RESET} — valid: {', '.join(COPILOT_MODELS)}"
                        )
                else:
                    cur = _copilot_model or "(default)"
                    print(f"\n{BOLD}Copilot Models:{RESET}  {DIM}current: {cur}{RESET}")
                    for key, m in COPILOT_MODELS.items():
                        marker = (
                            f"{GREEN}●{RESET}" if _copilot_model == m["id"] else " "
                        )
                        print(
                            f"  {marker} {CYAN}{key:<8}{RESET} {m['name']:<20} {DIM}{m['id']:<22} {m['tier']}{RESET}"
                        )
                    print()
            elif cmd == "/clear":
                chat = Chat()
                all_tools = STATIC_TOOLS + [ClaudeTool(chat)]
                chat._messages.append(
                    {"role": "system", "content": _build_system_prompt(all_tools)}
                )
                agent = AIAgent(config=AIConfig(base_url=base_url), tools=all_tools)
                session = make_session(session_id, work_dir, repl_mode_container)
                print(f"{DIM}cleared — new session {session_id}{RESET}")
                # Reset to build mode on clear
                repl_mode_container[0] = "build"
            elif cmd == "/history":
                print(
                    f"{DIM}{sum(1 for m in chat.messages if m['role'] != 'system')} messages{RESET}"
                )
            elif cmd == "/cwd":
                print(f"{DIM}{_work_dir}{RESET}")
            elif cmd == "/think":
                if last_thinking:
                    expand_thinking(last_thinking)
                else:
                    print(f"{DIM}no thinking from last response{RESET}")
            elif cmd == "/sessions":
                rows = list_sessions()
                if not rows:
                    print(f"{DIM}no saved sessions{RESET}")
                else:
                    for r in rows[:20]:
                        marker = f"{GREEN}*{RESET} " if r["id"] == session_id else "  "
                        print(
                            f"{marker}{CYAN}{r['id']}{RESET}  {r['updated']}  {DIM}{r['title']}{RESET}"
                        )
            elif cmd == "/usage":
                _show_usage()
            elif cmd == "/limits":
                _show_limits()
            elif cmd == "/insights":
                _show_insights()
            elif cmd in ("/quit", "/exit"):
                print(f"{DIM}bye{RESET}")
                break
            else:
                print(f"{RED}unknown: {cmd}{RESET}  (try /help)")
            continue

        # ── @ directive expansion ──────────────────────────────────────────
        expanded_input = _expand_at_directives(user_input, _work_dir)

        # ── chat turn ─────────────────────────────────────────────────────
        print()
        if current_agent == "claude":
            _process_turn_claude(
                claude_agent,
                chat,
                expanded_input,
                plan_mode=repl_mode_container[0] == "plan",
            )
        elif current_agent == "opencode":
            _process_turn_opencode(oc_agent, chat, expanded_input)
        elif current_agent == "copilot":
            _process_turn_copilot(copilot_agent, chat, expanded_input)
        else:
            if repl_mode_container[0] == "plan":
                filtered_tools = [
                    t for t in all_tools if getattr(t, "__name__", None) != "edit"
                ]
                agent_filtered = AIAgent(
                    config=AIConfig(base_url=base_url), tools=filtered_tools
                )
                base_sys = _build_system_prompt(filtered_tools)
                chat._messages[0]["content"] = PLANNER_SYSTEM_PROMPT + "\n\n" + base_sys
                chat.add(expanded_input)
                turn_thinking = _process_turn(agent_filtered, chat, current_mode)
                if turn_thinking:
                    last_thinking = turn_thinking
            else:
                chat.add(expanded_input)
                turn_thinking = _process_turn(agent, chat, current_mode)
                if turn_thinking:
                    last_thinking = turn_thinking

        # auto-save after each turn
        history = [m for m in chat.messages if m["role"] != "system"]
        if history:
            save_session(
                session_id,
                title_from_history(chat.messages),
                chat.messages,
                work_dir=_work_dir,
            )
        print()


# ── entry point ───────────────────────────────────────────────────────────────


def _run_single_prompt(prompt: str, work_dir: str = "."):
    global _work_dir
    _work_dir = work_dir

    base_url = os.environ.get("OPENAI_BASE_URL", "http://192.168.170.76:8000/v1")

    chat = Chat()
    all_tools = STATIC_TOOLS + [ClaudeTool(chat)]
    chat._messages.append(
        {"role": "system", "content": _build_system_prompt(all_tools)}
    )
    chat.add(prompt)

    agent = AIAgent(config=AIConfig(base_url=base_url), tools=all_tools)

    print(f"""
  {CORAL}▐▛███▜▌{RESET}   {BOLD}ai_cli{RESET} v1.0
  {CORAL}▝▜█████▛▘{RESET}  {CREAM}{BOLD}kivi{RESET} {DIM}· AI Agent{RESET}
  {CORAL}  ▘▘ ▝▝{RESET}    {DIM}{_work_dir}{RESET}
           {DIM}endpoint: {base_url}{RESET}""")
    print(f"  {DIM}single prompt mode{RESET}\n")

    _process_turn(agent, chat, "thinking_coding")
    print()




class AICli:
    """CLI interface for AIAgent / OpenCodeAgent."""

    BUILTIN_TOOLS = {
        "web_search": web_search,
        "web_fetch": web_fetch,
        "read": read,
        "write": write,
        "edit": edit,
        "bash": bash,
        "glob": glob,
        "grep": grep,
    }

    FILE_TOOLS = ["read", "write", "edit", "bash", "glob", "grep"]

    @staticmethod
    def build_parser():
        import argparse
        import textwrap

        parser = argparse.ArgumentParser(
            prog="agent",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent("""\
                agent — AI agent CLI. Ask questions, run agentic tasks, work on files.

                PROMPT is plain text (no quotes needed for multi-word input).
                Use @filepath to read prompt from a file, or pipe via stdin.

                ── Quick start ───────────────────────────────────────────────
                  agent hi
                  agent tell me a joke
                  agent explain this code @myfile.py

                ── Agentic modes (loop + all tools auto-enabled) ─────────────
                  agent --plan "refactor auth.py to use dataclasses"
                  agent --plan "add unit tests for utils.py" --dir ./myproject
                  agent -y "fix the bug in parser.py"          # yolo: fast, no thinking

                ── Session continuity ─────────────────────────────────────────
                  agent "first question"
                  agent --continue "follow-up question"         # auto-resume latest session
                  agent --resume abc123ef "another follow-up"   # resume specific session

                ── Mode & tools ───────────────────────────────────────────────
                  agent -m thinking_coding "design a caching layer"
                  agent "search AI news" --loop --tools web_search,web_fetch --verbose
                  agent "read config.py and summarize" --tools read
                  agent "what files changed?" --tools bash --loop

                ── Structured output ──────────────────────────────────────────
                  agent "Is this positive?" --choices "yes,no,maybe"
                  agent "Extract email" --regex "\\w+@\\w+\\.com"
                  agent "Parse person" --json-schema person.json

                ── Images / PDFs / Videos ─────────────────────────────────────
                  agent "describe this image" -i photo.jpg
                  agent "classify" -i ./shots/ --choices "cat,dog"
                  agent "summarize this PDF" -i report.pdf
                  agent "what is on page 3?" -i report.pdf --pages 3
                  agent "compare these pages" -i doc.pdf --pages 1,5
                  agent "review slides 2 to 7" -i deck.pdf --pages 2-7
                  agent "describe" -i clip.mp4

                ── Other ──────────────────────────────────────────────────────
                  agent --batch prompts.txt
                  cat error.log | agent "what is this error?"
                  agent "List files" --agent opencode

                Tools available: read, write, edit, bash, glob, grep, web_search, web_fetch
                Modes: thinking_general, thinking_coding, instruct_general, instruct_coding, instruct_reasoning
            """),
        )

        # ── Input ──────────────────────────────────────────────────────────
        inp = parser.add_argument_group("Input")
        inp.add_argument(
            "prompt", nargs="?", default=None,
            help="Prompt text. If omitted, reads from stdin.",
        )
        inp.add_argument("-p", "--prompt-flag", dest="prompt_flag", metavar="PROMPT",
                         help="Explicit prompt (alternative to positional).")
        inp.add_argument("-s", "--system", metavar="TEXT|@FILE",
                         help="System prompt text or @filepath.")
        inp.add_argument(
            "-i", "--input", nargs="+", metavar="PATH", dest="inputs",
            help="Input files: images (jpg/png/…), PDFs, or videos. Auto-detected by extension.",
        )
        inp.add_argument(
            "--pages", metavar="SPEC", default=None,
            help='PDF page selection (1-indexed): "1", "1,2", "1,5", "2-7". Default: all pages.',
        )
        inp.add_argument(
            "--batch", metavar="FILE",
            help="Run prompts in parallel. FILE has one prompt per line. Outputs answers in order.",
        )

        # ── Model / Connection ─────────────────────────────────────────────
        mdl = parser.add_argument_group("Model / Connection")
        mdl.add_argument("--base-url", metavar="URL",
                         help="OpenAI-compatible base URL (default: $OPENAI_BASE_URL or http://localhost:8000/v1).")
        mdl.add_argument("--model", default="", metavar="NAME",
                         help="Model name to pass to the backend.")
        mdl.add_argument(
            "--agent", choices=["ai", "opencode"], default="ai",
            help="Agent type: 'ai' = AIAgent (default), 'opencode' = OpenCodeAgent.",
        )

        # ── Mode / Params ──────────────────────────────────────────────────
        prm = parser.add_argument_group("Mode / Sampling Params")
        prm.add_argument(
            "--mode", "-m", default="instruct_general",
            choices=list(modes.keys()) + ["custom"],
            help="Named sampling preset (default: instruct_general). Use 'custom' with --temperature etc.",
        )
        prm.add_argument(
            "--plan", action="store_true",
            help="Plan mode: agentic loop with all file/shell tools, thinking_coding mode, uses --dir.",
        )
        prm.add_argument(
            "-y", "--yolo", action="store_true",
            help="Yolo mode: same as --plan but skips confirmation prompts (instruct_coding, max speed).",
        )
        prm.add_argument("--temperature", type=float, metavar="F",
                         help="Override temperature (implies --mode custom).")
        prm.add_argument("--top-p", type=float, metavar="F",
                         help="Override top_p.")
        prm.add_argument("--top-k", type=int, metavar="N",
                         help="Override top_k.")
        prm.add_argument("--presence-penalty", type=float, metavar="F",
                         help="Override presence_penalty.")
        prm.add_argument("--repetition-penalty", type=float, metavar="F",
                         help="Override repetition_penalty.")
        prm.add_argument("--thinking", action="store_true", default=None,
                         help="Enable thinking mode (enable_thinking=True).")
        prm.add_argument("--max-tokens", type=int, metavar="N",
                         help="Maximum tokens to generate.")

        # ── Execution ──────────────────────────────────────────────────────
        exe = parser.add_argument_group("Execution")
        exe.add_argument(
            "--loop", action="store_true",
            help="Agentic forward loop (auto tool execution). Default: single step.",
        )
        exe.add_argument(
            "--tools", metavar="NAMES",
            help="Comma-separated built-in tools: web_search, web_fetch, weather, time, read, write, edit, bash, glob, grep.",
        )
        exe.add_argument("--max-steps", type=int, metavar="N",
                         help="Max agentic loop steps (only with --loop).")
        exe.add_argument(
            "--tool-choice", metavar="auto|none|TOOL",
            help="Tool choice strategy (default: auto when tools present).",
        )
        exe.add_argument(
            "--dir", metavar="PATH", default=None,
            help="Working directory for file tools (default: cwd). Used with --plan/--yolo.",
        )
        exe.add_argument(
            "--resume", metavar="SESSION_ID",
            help="Resume a previous session by ID.",
        )
        exe.add_argument(
            "-c", "--continue", dest="continue_", action="store_true",
            help="Continue the latest session for the working directory.",
        )

        # ── Structured Output ──────────────────────────────────────────────
        so = parser.add_argument_group("Structured Output (choose at most one)")
        so.add_argument(
            "--choices", metavar="A,B,C",
            help="Constrain output to one of these comma-separated options.",
        )
        so.add_argument(
            "--regex", metavar="PATTERN",
            help="Constrain output to match this regex pattern.",
        )
        so.add_argument(
            "--json-schema", metavar="FILE|JSON",
            help="JSON schema: path to .json file or inline JSON string.",
        )
        so.add_argument(
            "--grammar", metavar="FILE|EBNF",
            help="EBNF grammar: path to file or inline grammar string.",
        )

        # ── Output ─────────────────────────────────────────────────────────
        out = parser.add_argument_group("Output")
        out.add_argument(
            "--stream", action="store_true", default=True,
            help="Stream tokens as they arrive (default: on).",
        )
        out.add_argument(
            "--no-stream", dest="stream", action="store_false",
            help="Wait for full response before printing.",
        )
        out.add_argument(
            "--output", metavar="FILE",
            help="Write final answer to this file.",
        )
        out.add_argument(
            "--format", choices=["text", "json"], default="text",
            help="Output format: 'text' (default) or 'json' (full event log).",
        )
        out.add_argument(
            "--verbose", action="store_true",
            help="Show tool calls and results during execution.",
        )

        return parser

    @staticmethod
    def _resolve_prompt(args) -> Optional[str]:
        import sys
        text = args.prompt_flag or args.prompt
        if text is None and not sys.stdin.isatty():
            stdin_text = sys.stdin.read().strip()
            text = stdin_text if stdin_text else None
        if text and text.startswith("@"):
            path = text[1:]
            with open(path) as f:
                text = f.read().strip()
        return text

    @staticmethod
    def _resolve_system(system_arg: Optional[str]) -> Optional[str]:
        if system_arg is None:
            return None
        if system_arg.startswith("@"):
            with open(system_arg[1:]) as f:
                return f.read().strip()
        return system_arg

    @staticmethod
    def _parse_pages(spec: str) -> list[int]:
        """Parse page spec string into sorted list of 1-indexed page numbers."""
        pages = set()
        for part in spec.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1)
                pages.update(range(int(a), int(b) + 1))
            else:
                pages.add(int(part))
        return sorted(pages)

    @staticmethod
    def _pdf_to_images(pdf_path: str, pages_spec: str = None) -> list[str]:
        """Convert PDF pages to base64 PNG data URIs using pdf2image."""
        import base64, io, tempfile
        try:
            from pdf2image import convert_from_path
        except ImportError:
            raise ImportError("pdf2image is required for PDF support: uv pip install pdf2image")

        kwargs = {}
        if pages_spec:
            page_nums = AICli._parse_pages(pages_spec)
            kwargs["first_page"] = page_nums[0]
            kwargs["last_page"] = page_nums[-1]

        images = convert_from_path(pdf_path, **kwargs)

        if pages_spec:
            page_nums = AICli._parse_pages(pages_spec)
            # convert_from_path returns pages first_page..last_page; filter to exact selection
            first = page_nums[0]
            images = [images[p - first] for p in page_nums if (p - first) < len(images)]

        result = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            result.append(f"data:image/png;base64,{b64}")
        return result

    @staticmethod
    def _resolve_inputs(input_args, pages_spec: str = None) -> tuple[list, list]:
        """Returns (images, videos). Images may include base64 data URIs from PDFs."""
        if not input_args:
            return [], []
        import glob as _glob

        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
        VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"}
        PDF_EXTS   = {".pdf"}

        images, videos = [], []

        for p in input_args:
            if os.path.isdir(p):
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.gif", "*.webp", "*.bmp"):
                    images.extend(sorted(_glob.glob(os.path.join(p, ext))))
                continue

            ext = os.path.splitext(p)[1].lower()
            if ext in PDF_EXTS:
                images.extend(AICli._pdf_to_images(p, pages_spec))
            elif ext in VIDEO_EXTS:
                videos.append(p)
            else:
                images.append(p)

        return images, videos

    @staticmethod
    def _resolve_tools(tools_arg: Optional[str]) -> list:
        if not tools_arg:
            return []
        result = []
        for name in tools_arg.split(","):
            name = name.strip()
            if name in AICli.BUILTIN_TOOLS:
                result.append(AICli.BUILTIN_TOOLS[name])
            else:
                raise ValueError(f"Unknown tool '{name}'. Available: {', '.join(AICli.BUILTIN_TOOLS)}")
        return result

    @staticmethod
    def _resolve_structured_output(args) -> Optional[StructuredOutput]:
        count = sum([
            args.choices is not None,
            args.regex is not None,
            args.json_schema is not None,
            args.grammar is not None,
        ])
        if count == 0:
            return None
        if count > 1:
            raise ValueError("Specify at most one of: --choices, --regex, --json-schema, --grammar")

        if args.choices:
            return StructuredOutput(choice=[c.strip() for c in args.choices.split(",")])
        if args.regex:
            return StructuredOutput(regex=args.regex)
        if args.grammar:
            gram = args.grammar
            if os.path.isfile(gram):
                with open(gram) as f:
                    gram = f.read()
            return StructuredOutput(grammar=gram)
        if args.json_schema:
            raw = args.json_schema
            if os.path.isfile(raw):
                with open(raw) as f:
                    raw = f.read()
            try:
                schema = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"--json-schema is not valid JSON: {e}")
            return StructuredOutput(json=schema)
        return None

    @staticmethod
    def _resolve_mode(args):
        has_custom = any(x is not None for x in [
            args.temperature, args.top_p, args.top_k,
            args.presence_penalty, args.repetition_penalty,
        ]) or args.thinking

        if has_custom or args.mode == "custom":
            base = modes.get(args.mode if args.mode != "custom" else "instruct_general", modes["instruct_general"])
            return AICompletionConfig(
                temperature=args.temperature if args.temperature is not None else base["temperature"],
                top_p=args.top_p if args.top_p is not None else base["top_p"],
                top_k=args.top_k if args.top_k is not None else base["top_k"],
                presence_penalty=args.presence_penalty if args.presence_penalty is not None else base["presence_penalty"],
                repetition_penalty=args.repetition_penalty if args.repetition_penalty is not None else base["repetition_penalty"],
                enable_thinking=args.thinking if args.thinking else base["enable_thinking"],
            )
        return args.mode

    @classmethod
    def run(cls, argv=None):
        import sys
        from pathlib import Path
        parser = cls.build_parser()
        args, extra = parser.parse_known_args(argv)
        if extra:
            joined = " ".join(extra)
            args.prompt = ((args.prompt + " ") if args.prompt else "") + joined

        # ── resolve base-url ───────────────────────────────────────────────
        if args.base_url:
            os.environ["OPENAI_BASE_URL"] = args.base_url

        # ── resolve working dir ────────────────────────────────────────────
        global _work_dir
        work_dir = str(Path(args.dir).resolve()) if args.dir else str(Path.cwd())
        _work_dir = work_dir

        # ── plan / yolo mode overrides ─────────────────────────────────────
        if args.plan:
            args.mode = "thinking_coding"
            args.loop = True
            if not args.tools:
                args.tools = ",".join(cls.FILE_TOOLS + ["web_search", "web_fetch"])
        elif args.yolo:
            args.mode = "instruct_coding"
            args.loop = True
            if not args.tools:
                args.tools = ",".join(cls.FILE_TOOLS + ["web_search", "web_fetch"])

        session_id = None
        prior_history = None

        if args.resume:
            rec = load_session(args.resume)
            if not rec:
                print(f"\033[31msession {args.resume!r} not found\033[0m")
                sys.exit(1)
            session_id = rec["id"]
            prior_history = rec["history"]
            _work_dir = rec.get("work_dir") or work_dir
        elif args.continue_:
            rec = latest_session_for_dir(work_dir)
            if rec:
                session_id = rec["id"]
                prior_history = rec["history"]
            else:
                print(f"\033[2mno session found for {work_dir}, starting fresh\033[0m")

        if session_id is None:
            session_id = new_session_id()

        # ── batch mode ─────────────────────────────────────────────────────
        if args.batch:
            cls._run_batch(args)
            return

        # ── resolve prompt ─────────────────────────────────────────────────
        prompt_text = cls._resolve_prompt(args)
        if not prompt_text:
            parser.print_help()
            sys.exit(0)

        system_text = cls._resolve_system(args.system)
        images, videos = cls._resolve_inputs(args.inputs, args.pages)
        tools = cls._resolve_tools(args.tools)
        structured_output = cls._resolve_structured_output(args)
        mode = cls._resolve_mode(args)

        # ── build chat (resume history or fresh) ───────────────────────────
        if prior_history:
            chat = Chat.__new__(Chat)
            chat._messages = list(prior_history)
            chat.add(prompt_text, images=images or None, videos=videos or None)
        else:
            chat = Chat(prompt_text, images=images or None, videos=videos or None)
            if system_text:
                chat._messages.insert(0, {"role": "system", "content": system_text})

        # ── opencode agent ─────────────────────────────────────────────────
        if args.agent == "opencode":
            agent = OpenCodeAgent(model=args.model or None)
            cls._run_opencode(agent, prompt_text, args)
            return

        # ── ai agent ──────────────────────────────────────────────────────
        config = AIConfig(base_url=os.environ.get("OPENAI_BASE_URL"))
        agent = AIAgent(config=config, tools=tools or None)

        forward_kwargs = dict(
            model=args.model,
            max_tokens=args.max_tokens,
            mode=mode,
            structured_output=structured_output,
        )
        if tools and args.tool_choice:
            forward_kwargs["tool_choice"] = args.tool_choice
        elif tools:
            forward_kwargs["tool_choice"] = "auto"

        events_log = []

        if args.loop:
            forward_kwargs["max_steps"] = args.max_steps
            gen = agent.forward(chat, tools=tools or None, **forward_kwargs)
        else:
            gen = agent.step(chat, tools=tools or None, **forward_kwargs)

        answer_parts = []
        for event in gen:
            if args.format == "json":
                events_log.append(cls._event_to_dict(event))
                continue

            if isinstance(event, Text) and event.id is None:
                if args.stream:
                    print(event.content, end="", flush=True)
                answer_parts.append(event.content)
            elif isinstance(event, ToolCall) and args.verbose:
                print(f"\n\033[33m[tool call]\033[0m {event.name}({event.arguments})", file=sys.stderr)
            elif isinstance(event, ToolResult) and args.verbose:
                preview = event.result[:200].replace("\n", " ")
                print(f"\033[32m[tool result]\033[0m {event.name} → {preview}...", file=sys.stderr)
            elif isinstance(event, AgentResult) and args.verbose:
                print(
                    f"\n\033[90m[done] steps={event.steps} tool_calls={event.tool_calls_total} "
                    f"elapsed={event.elapsed_s}s\033[0m",
                    file=sys.stderr,
                )

        if args.format == "json":
            print(json.dumps(events_log, indent=2))
            return

        if not args.stream:
            print("".join(answer_parts))
        else:
            print()

        if args.output:
            with open(args.output, "w") as f:
                f.write("".join(answer_parts))
            print(f"\033[90m[saved → {args.output}]\033[0m", file=sys.stderr)

        # ── save session & show session footer ─────────────────────────────
        if db_fns:
            history = [m for m in chat.messages if m["role"] != "system"]
            if history:
                save_session_fn(session_id, title_fn(chat.messages), chat.messages, work_dir=work_dir)
        #print(
            #f"\033[2msession: {session_id}  |  --resume {session_id}  |  --continue\033[0m",
            #file=sys.stderr,
        #)

    @classmethod
    def _run_opencode(cls, agent: OpenCodeAgent, prompt: str, args):
        import sys
        answer_parts = []
        events_log = []
        for event in agent.run(prompt):
            if args.format == "json":
                events_log.append(cls._event_to_dict(event))
                continue
            if isinstance(event, Text):
                if args.stream:
                    print(event.content, end="", flush=True)
                answer_parts.append(event.content)
            elif isinstance(event, ToolCall) and args.verbose:
                print(f"\n\033[33m[tool call]\033[0m {event.name}({event.arguments})", file=sys.stderr)
            elif isinstance(event, ToolResult) and args.verbose:
                preview = event.result[:200].replace("\n", " ")
                print(f"\033[32m[tool result]\033[0m {event.name} → {preview}...", file=sys.stderr)
            elif isinstance(event, AgentResult) and args.verbose:
                print(
                    f"\n\033[90m[done] steps={event.steps} cost=${event.meta.get('cost_total', 0):.6f} "
                    f"elapsed={event.elapsed_s}s\033[0m",
                    file=sys.stderr,
                )
        if args.format == "json":
            print(json.dumps(events_log, indent=2))
            return
        if not args.stream:
            print("".join(answer_parts))
        else:
            print()
        if args.output:
            with open(args.output, "w") as f:
                f.write("".join(answer_parts))

    @classmethod
    def _run_batch(cls, args):
        import sys
        with open(args.batch) as f:
            prompts = [line.strip() for line in f if line.strip()]

        system_text = cls._resolve_system(args.system)
        tools = cls._resolve_tools(args.tools)
        structured_output = cls._resolve_structured_output(args)
        mode = cls._resolve_mode(args)
        config = AIConfig(base_url=os.environ.get("OPENAI_BASE_URL"))
        agent = AIAgent(config=config, tools=tools or None)

        forward_kwargs = dict(
            model=args.model,
            max_tokens=args.max_tokens,
            mode=mode,
            structured_output=structured_output,
        )
        if tools:
            forward_kwargs["tool_choice"] = args.tool_choice or "auto"

        def run_one(prompt_text):
            chat = Chat(prompt_text)
            if system_text:
                chat._messages.insert(0, {"role": "system", "content": system_text})
            return agent.task(chat, **forward_kwargs)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            futures = {pool.submit(run_one, p): (i, p) for i, p in enumerate(prompts)}
            results = {}
            for future in concurrent.futures.as_completed(futures):
                i, p = futures[future]
                results[i] = (p, future.result())

        for i in range(len(prompts)):
            p, chat = results[i]
            if args.format == "json":
                print(json.dumps({"prompt": p, "answer": chat.answer}))
            else:
                print(f"[{i+1}] Q: {p}")
                print(f"    A: {chat.answer}")
                print()

        if args.output:
            lines = []
            for i in range(len(prompts)):
                p, chat = results[i]
                lines.append(f"Q: {p}\nA: {chat.answer}")
            with open(args.output, "w") as f:
                f.write("\n\n".join(lines))

    @staticmethod
    def _event_to_dict(event) -> dict:
        if isinstance(event, Text):
            return {"type": "text", "content": event.content, "id": event.id}
        if isinstance(event, Assistant):
            return {"type": "assistant", "content": event.content}
        if isinstance(event, ToolCall):
            return {"type": "tool_call", "name": event.name, "id": event.id, "arguments": event.arguments}
        if isinstance(event, ToolResult):
            return {"type": "tool_result", "name": event.name, "id": event.id, "result": event.result}
        if isinstance(event, StepResult):
            return {
                "type": "step_result", "step": event.step, "text": event.text,
                "stop_reason": event.stop_reason,
                "input_tokens": event.input_tokens, "output_tokens": event.output_tokens,
            }
        if isinstance(event, AgentResult):
            return {
                "type": "agent_result", "steps": event.steps, "answer": event.answer,
                "tool_calls_total": event.tool_calls_total, "elapsed_s": event.elapsed_s,
                "meta": event.meta,
            }
        if isinstance(event, DoneEvent):
            return {"type": "done"}
        return {"type": "unknown"}



if __name__ == "__main__":
    # AICli.run()
    import argparse

    parser = argparse.ArgumentParser(description="ai_cli — AI REPL or single prompt")
    parser.add_argument("-d", "--dir", default=".", help="working directory")
    parser.add_argument("--url", default=None, help="override OPENAI_BASE_URL")
    parser.add_argument(
        "--prompt",
        "-p",
        metavar="TEXT",
        nargs="?",
        const=True,
        help="single prompt mode (or just -p with text args)",
    )
    parser.add_argument("text", nargs="*", help="prompt text (when --prompt is used)")
    parser.add_argument("--resume", metavar="ID", help="resume a saved session by ID")
    parser.add_argument(
        "--continue", action="store_true", help="continue latest session for cwd"
    )
    parser.add_argument("--list", action="store_true", help="list all saved sessions")
    args = parser.parse_args()

    if args.url:
        os.environ["OPENAI_BASE_URL"] = args.url

    if args.list:
        rows = list_sessions()
        if not rows:
            print("no saved sessions")
        else:
            for r in rows:
                print(f"{r['id']}  {r['updated']}  {r['work_dir']}  {r['title']}")
        sys.exit(0)

    work_dir = str(Path(args.dir).resolve())

    if args.resume:
        rec = load_session(args.resume)
        if not rec:
            print(f"{RED}session {args.resume!r} not found{RESET}")
            sys.exit(1)
        run_repl(
            rec["work_dir"] or work_dir,
            session_id=rec["id"],
            initial_history=rec["history"],
        )
        sys.exit(0)

    if getattr(args, "continue"):
        rec = latest_session_for_dir(work_dir)
        if not rec:
            print(f"{DIM}no session for {work_dir}, starting fresh{RESET}")
            run_repl(work_dir)
        else:
            run_repl(
                rec["work_dir"], session_id=rec["id"], initial_history=rec["history"]
            )
        sys.exit(0)

    if args.prompt or args.text:
        if args.text:
            prompt_text = " ".join(args.text)
        elif isinstance(args.prompt, str):
            prompt_text = args.prompt
        else:
            prompt_text = ""
        if not prompt_text:
            print(f"{RED}no prompt provided{RESET}")
            sys.exit(1)
        _run_single_prompt(prompt_text, work_dir)
    else:
        run_repl(work_dir)
