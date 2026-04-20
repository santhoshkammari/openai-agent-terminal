#!/usr/bin/env python3
"""ai_cli — Interactive streaming AI REPL for coding tasks.
Start interactive REPL or run single prompts.
Commands: /help /modes /clear /history /quit /claude /kivi
Use bash, read, edit, glob, grep tools.
"""

import os, sys, json, subprocess, threading, time, itertools
from pathlib import Path
from typing import Literal, Union

from prompt_toolkit import PromptSession
from prompt_toolkit.history import History
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from rich.console import Console
from rich.markdown import Markdown as RichMarkdown

_rich = Console()

from ai_sync import (
    AIAgent,
    AIConfig,
    Chat,
    Text,
    modes,
    ToolCall,
    ToolResult,
    StepResult,
    AgentResult,
    OpenCodeAgent,
)
from db import (
    new_session_id,
    save_session,
    load_session,
    list_sessions,
    latest_session_for_dir,
    title_from_history,
    save_prompt_input,
    load_prompt_inputs,
)

CODING_PROMPT = """
Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

Tradeoff: These guidelines bias toward caution over speed. For trivial tasks, use judgment.
1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

    State your assumptions explicitly. If uncertain, ask.
    If multiple interpretations exist, present them - don't pick silently.
    If a simpler approach exists, say so. Push back when warranted.
    If something is unclear, stop. Name what's confusing. Ask.

2. Simplicity First

Minimum code that solves the problem. Nothing speculative.

    No features beyond what was asked.
    No abstractions for single-use code.
    No "flexibility" or "configurability" that wasn't requested.
    No error handling for impossible scenarios.
    If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.
3. Surgical Changes

Touch only what you must. Clean up only your own mess.

When editing existing code:

    Don't "improve" adjacent code, comments, or formatting.
    Don't refactor things that aren't broken.
    Match existing style, even if you'd do it differently.
    If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

    Remove imports/variables/functions that YOUR changes made unused.
    Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.
4. Goal-Driven Execution

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

    "Add validation" → "Write tests for invalid inputs, then make them pass"
    "Fix the bug" → "Write a test that reproduces it, then make it pass"
    "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

These guidelines are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
"""
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

# ── tools ─────────────────────────────────────────────────────────────────────

_work_dir = "."
_python_env = sys.executable


def _env():
    env = os.environ.copy()
    bin_dir = str(Path(_python_env).parent)
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    env["VIRTUAL_ENV"] = str(Path(_python_env).parent.parent)
    return env


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
        from ai_sync import AIAgent, AIConfig, Chat, Text, ToolCall, ToolResult, AgentResult, StepResult
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


# ── parallel tool display (rich.live) ─────────────────────────────────────────

from rich.live import Live
from rich.tree import Tree
from rich.text import Text as RichText
from rich.spinner import Spinner


class ParallelToolDisplay:
    def __init__(self, tool_calls: list):
        self._tools = tool_calls
        self._statuses: dict[str, str] = {tc.id: "running" for tc in tool_calls}
        self._results: dict[str, str] = {}
        self._lock = threading.Lock()
        self._live = (
            Live(console=_rich, refresh_per_second=12, transient=False)
            if sys.stdout.isatty()
            else None
        )

    def _build_tree(self):
        total = len(self._tools)
        done = sum(1 for s in self._statuses.values() if s == "done")
        if done == total:
            header = RichText(
                f"⚡ {total} tool{'s' if total > 1 else ''} — all done",
                style="bold green",
            )
        elif done:
            header = RichText(
                f"⚡ {total} tool{'s' if total > 1 else ''} in parallel  ✓{done}  ⠼{total - done} running",
                style="bold yellow",
            )
        else:
            header = RichText(
                f"⚡ {total} tool{'s' if total > 1 else ''} in parallel",
                style="bold yellow",
            )
        tree = Tree(header)
        for tc in self._tools:
            label = f"[bold yellow]{tc.name}[/bold yellow]{_fmt_args(tc.name, tc.arguments)}"
            if self._statuses[tc.id] == "done":
                res = self._results.get(tc.id, "")
                preview = res[:100].replace("\n", " ")
                ellip = "…" if len(res) > 100 else ""
                node = tree.add(f"[green]✓[/green] {label}")
                node.add(f"[dim]→ {preview}{ellip}[/dim]")
            else:
                spinner = Spinner("dots", text=label, style="dim")
                tree.add(spinner)
        return tree

    def start(self):
        if self._live:
            self._live.start()
            self._live.update(self._build_tree())
        else:
            print(f"{DIM}[tools running...]{RESET}", flush=True)

    def add_tool(self, tc):
        with self._lock:
            self._tools.append(tc)
            self._statuses[tc.id] = "running"
        if self._live:
            self._live.update(self._build_tree())

    def complete(self, tool_id: str, result: str):
        with self._lock:
            self._statuses[tool_id] = "done"
            self._results[tool_id] = result
            if tool_id not in self._statuses:
                for tc in self._tools:
                    if self._statuses.get(tc.id) == "running":
                        self._statuses[tc.id] = "done"
                        self._results[tc.id] = result
                        break
        if self._live:
            self._live.update(self._build_tree())

    def stop(self):
        if self._live:
            self._live.update(self._build_tree())
            self._live.stop()
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
            if sys.stdout.isatty():
                width = os.get_terminal_size().columns
                rows = 0
                for line in text.splitlines(keepends=True):
                    rows += max(1, (len(line.rstrip("\n")) + width - 1) // width)
                move_up = max(0, rows - 1)
                if move_up:
                    sys.stdout.write(f"\033[{move_up}A")
                sys.stdout.write("\r\033[J")
                sys.stdout.flush()
                _rich.print(RichMarkdown(text))
            else:
                # no TTY (subprocess) — already printed plain, just add newline
                print()
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
                answer_parts.append(event.content)
                if not live_on:
                    live.start()
                    live_on = True
                live.append(event.content)

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

_USD_TO_INR = 92.60

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


# ── prompt_toolkit session ────────────────────────────────────────────
_pt_style = Style.from_dict({"prompt": "bold", "bottom-toolbar": "bg:#1a1a1a #888888"})
_SLASH_COMMANDS = [
    "/help",
    "/modes",
    "/mode",
    "/clear",
    "/history",
    "/cwd",
    "/think",
    "/sessions",
    "/claude",
    "/kivi",
    "/opencode",
    "/copilot",
    "/model",
    "/usage",
    "/limits",
    "/insights",
    "/quit",
]


class SQLiteChatHistory(History):
    """prompt_toolkit History backed by the prompt_history SQLite table.

    Keyed by cwd only — so ↑/↓ recall works across all sessions in the same
    directory. session_id is stored per-entry for auditing but not used for
    lookup (a new session_id is minted every fresh terminal, so filtering by it
    would always return empty on first launch).

    Contract: implement only load_history_strings() + store_string().
    The base History.load() async generator handles all buffer management:
      - calls load_history_strings() on first ↑ press
      - stores results in self._loaded_strings (newest-first)
      - sets self._loaded = True to prevent re-loading
    Never touch _loaded_strings or _loaded manually.
    """

    def __init__(self, session_id: str, cwd: str):
        super().__init__()  # sets _loaded=False, _loaded_strings=[]
        self._session_id = session_id
        self._cwd = cwd

    def load_history_strings(self):
        """Yield stored inputs newest-first, scoped to cwd across all sessions."""
        yield from reversed(load_prompt_inputs(cwd=self._cwd))

    def store_string(self, string: str):
        """Persist each submitted input; called by append_string() in the base class."""
        save_prompt_input(self._session_id, self._cwd, string)


def make_session(
    session_id: str, work_dir: str, key_bindings=None, bottom_toolbar=None
) -> PromptSession:
    return PromptSession(
        history=SQLiteChatHistory(session_id, work_dir),
        style=_pt_style,
        completer=WordCompleter(
            _SLASH_COMMANDS + [f"/mode {m}" for m in modes], sentence=True, pattern=None
        ),
        complete_while_typing=True,
        auto_suggest=AutoSuggestFromHistory(),
        key_bindings=key_bindings,
        bottom_toolbar=bottom_toolbar,
    )


def _make_keybindings(repl_mode_container):
    """Create prompt_toolkit keybindings for plan/build mode toggle."""
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys

    kb = KeyBindings()

    @kb.add(Keys.BackTab)
    def _(event):
        repl_mode_container[0] = (
            "plan" if repl_mode_container[0] == "build" else "build"
        )
        event.app.invalidate()

    return kb


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
    from ai_sync import AIAgent, AIConfig

    base_url = os.environ.get("OPENAI_BASE_URL", "http://192.168.170.76:8000/v1")
    agent = AIAgent(config=AIConfig(base_url=base_url), tools=tools)
    schemas = agent._resolve_tools(tools)

    lines = [
        "You are Kivi Agent, official CLI for Kivi model.",
        "You are an interactive agent that helps users with software engineering tasks. Use the instructions below and the tools available to you to assist the user.",
        "When multiple independent tasks can be done in parallel, call multiple tools at once — they run concurrently.",
        "",
        CODING_PROMPT,
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
                        _rich.print(RichMarkdown(ac.strip()))
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
    keybindings = _make_keybindings(repl_mode_container)
    session = make_session(session_id, work_dir, key_bindings=keybindings)

    ac = AGENT_COLOR.get(current_agent, CORAL)
    print(f"""
               {BOLD}ai_cli{RESET} v1.0
     ▐▛███▜▌   kivi v1.0 · AI Agent
    ▝▜█████▛▘  {_work_dir}
      ▘▘ ▝▝    {DIM}endpoint: {base_url}{RESET}""")
    print(
        f"  {DIM}{'resumed ' if resumed else ''}session {RESET}{CREAM}{session_id}{RESET}"
        f"  {DIM}|{RESET}  mode: {CREAM}{current_mode}{RESET}"
        f"  {DIM}|{RESET}  agent: {ac}{BOLD}{current_agent}{RESET}"
        f"  {DIM}|{RESET}  plan: {CREAM}{'[plan]' if repl_mode_container[0] == 'plan' else '[build]'}{RESET}"
    )
    print(f"  {DIM}/help for commands, Ctrl-C to quit{RESET}\n")
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

        def _prompt_msg():
            suffix = (
                "<style fg='#888888'>_plan</style>"
                if repl_mode_container[0] == "plan"
                else ""
            )
            return HTML(
                f"<bold><style fg='{ac_hex}'>{current_agent}{suffix}></style> </bold>"
            )

        try:
            user_input = session.prompt(_prompt_msg).strip()
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
                session = make_session(session_id, work_dir, key_bindings=keybindings)
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


if __name__ == "__main__":
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
