# kivi-agent

> A self-hosted, streaming AI coding REPL with a multi-agent orchestration engine — built on vLLM, OpenAI-compatible APIs, and the Claude Agent SDK.

---

## What it is

`kivi-agent` is a terminal-native AI coding assistant I built from scratch to run **entirely on local GPU infrastructure** (vLLM backend at `192.168.170.76:8000`). It's a full agentic loop — not a thin wrapper around a hosted API.

The core loop: the LLM reasons → calls tools → receives results → reasons again. Every step streams to the terminal in real time, with parallel tool execution, live diff rendering, and per-turn SQLite persistence.

On top of the base Kivi agent, the REPL integrates **four distinct agent backends** in a single session: Kivi (local vLLM), Claude (via claude-agent-sdk), OpenCode, and GitHub Copilot CLI — all sharing a unified chat history and switching with a single slash command.

---

## Architecture

```
ai_cli.py          ← REPL, slash commands, multi-agent routing, UI
ai_sync.py         ← AIAgent engine: streaming, parallel tool execution, agentic loop
db.py              ← SQLite session persistence + prompt_toolkit history
```

### `AIAgent` (ai_sync.py)

The engine behind every Kivi turn.

- **`step()`** — single LLM call, streaming. Yields `Text | ToolCall | Assistant` as they arrive.
- **`forward()`** — full agentic loop. Calls `step()`, dispatches tool calls (parallel when >1), injects results back into chat, loops until `end_turn`. Yields the same event types so callers are composable.
- **`fn_to_tool()`** — introspects Python function signatures and docstrings to build OpenAI-schema tool definitions at runtime. No manual schema authoring.
- **Parallel tool execution** — uses `ThreadPoolExecutor` to run concurrent tool calls when the LLM requests multiple at once. Results are injected in completion order.
- **Structured outputs** — supports constrained decoding: `choice`, `regex`, `grammar` (EBNF), and JSON Schema (raw dict or Pydantic model).
- **Sub-agent composition** — any `AIAgent` can be wrapped as a callable tool via `to_tool()`, enabling multi-agent trees where the orchestrator calls researcher/analyst sub-agents and their streaming output is forwarded transparently.

### `OpenCodeAgent` (ai_sync.py)

Wraps `opencode run --format json` as a streaming Python generator that emits the same event types (`Text`, `ToolCall`, `ToolResult`, `StepResult`, `AgentResult`, `DoneEvent`) as `AIAgent.forward()`. Consumers are identical — swapping backends requires zero UI changes.

### REPL (ai_cli.py)

- **Multi-agent session** — `/kivi`, `/claude`, `/opencode`, `/copilot` switch the active agent mid-conversation. All agents write into a shared `Chat` object so cross-agent context is preserved.
- **Plan/Build toggle** — `Shift+Tab` switches between Plan mode (no file edits, structured plan output) and Build mode (full tool access). The system prompt is swapped dynamically.
- **ClaudeTool** — calls Claude (`claude-sonnet-4-6`) as a sub-agent via the Claude Agent SDK with per-directory session continuity and turn-budget management (5 turns/session before reset).
- **ClaudeDirectAgent** — live-streaming direct Claude mode in the REPL. Streams `content_block_delta` events and renders them inline using Rich.
- **Parallel tool display** — `ParallelToolDisplay` uses `rich.Live` to render a real-time tree of in-flight and completed tool calls with results previews and colored diffs for edits.
- **SQLite prompt history** — `SQLiteChatHistory` implements `prompt_toolkit.History` backed by a per-cwd SQLite table so ↑/↓ recall works across all sessions in the same directory.
- **Session persistence** — every turn is auto-saved. `--resume <id>` and `--continue` restore previous sessions with full history re-rendered in the original visual style.
- **Thinking mode** — extended reasoning tokens are streamed to a spinner with live char count; `/think` expands the full reasoning trace post-response.
- **`!` bash shortcut** — `!<command>` runs shell commands and injects the result into the chat as a proper `tool_call`/`tool` message pair so the LLM sees the output.

### Tool set

| Tool | Description |
|---|---|
| `read` | Read file contents |
| `write` | Write/create files |
| `edit` | Surgical string-replace edits with unified diff output |
| `bash` | Shell command execution (60s timeout, venv-aware PATH) |
| `glob` | File pattern matching |
| `grep` | Regex search across files |
| `web_search` | DDGS-backed web search |
| `web_fetch` | Scrapling-backed URL fetch → Markdown |
| `kivi` | Spawn a parallel Kivi sub-agent |
| `opencode` | Delegate to OpenCode agent |
| `claude` | Call Claude as a reasoner/planner sub-agent |

---

## Inference modes

Five named modes configure temperature, top-p, top-k, presence penalty, repetition penalty, and extended thinking independently:

| Mode | Use case |
|---|---|
| `thinking_coding` | Extended reasoning for hard coding tasks |
| `thinking_general` | Extended reasoning for open-ended questions |
| `instruct_coding` | Fast, precise coding (low temp, high repetition penalty) |
| `instruct_general` | Conversational, balanced |
| `instruct_reasoning` | High-temp reasoning without extended thinking tokens |

Switch mid-session with `/mode <name>` or the interactive picker (`/modes`).

---

## Design decisions worth noting

**Why build this instead of using Claude Code or Cursor?**

The primary motivation was full control over the inference stack. Running vLLM locally with quantized models means zero per-token cost on a team GPU server, latency that scales with hardware not rate limits, and the ability to experiment with models that aren't available via hosted APIs. The REPL is a thin shell around a composable engine that works identically in batch scripts and interactive sessions.

**Why the unified `Chat` object across agents?**

Each agent backend (Kivi, Claude, OpenCode, Copilot) has a different native session/context model. Normalizing everything into a single `Chat` message list means `/claude` can pick up where `/kivi` left off without any special-casing. Context injection is explicit: each `_process_turn_*` function reads the last N messages and prepends them as a `[Prior conversation context]` block.

**Why SQLite for everything?**

Session history, prompt recall, Claude sub-agent session tracking — all SQLite. No external services, no file locking issues between terminal sessions, and trivial to inspect with standard tooling. The `SQLiteChatHistory` class is a clean implementation of `prompt_toolkit.History` with zero coupling to the rest of the system.

**Parallel tools without async**

`AIAgent.forward()` is a synchronous generator that uses `ThreadPoolExecutor` for parallel tool dispatch. This keeps the calling interface simple (a plain `for event in agent.forward(...)` loop) while still running IO-bound tools concurrently. The async complexity is isolated to the Claude SDK integration which requires `asyncio.run()`.

---

## Stack

- **Inference**: vLLM (OpenAI-compatible, self-hosted)
- **Models**: Kivi (primary), Claude Sonnet 4.6, GPT-4.1/5 (Copilot)
- **Agent SDK**: `claude-agent-sdk` for Claude sub-agent integration
- **Terminal UI**: `prompt_toolkit` + `rich`
- **Web**: `scrapling` (browser-impersonation fetch), `ddgs` (DuckDuckGo search)
- **Persistence**: SQLite via stdlib `sqlite3`
- **Python**: 3.11+, no async in the hot path

---

## Usage

```bash
# Interactive REPL
python ai_cli.py

# Single prompt
python ai_cli.py -p "refactor this function to use a context manager"

# Resume a session
python ai_cli.py --resume <session-id>

# Continue latest session for this directory
python ai_cli.py --continue

# List all sessions
python ai_cli.py --list

# Custom working directory
python ai_cli.py -d /path/to/project
```

**Slash commands in REPL:**

```
/modes          interactive mode picker (↑↓ to navigate)
/mode <name>    switch inference mode
/claude         switch to Claude agent
/kivi           switch to Kivi agent
/opencode       switch to OpenCode agent
/copilot        switch to GitHub Copilot CLI agent
/model [name]   pick Copilot model
/clear          clear conversation
/history        message count
/think          expand last thinking trace
/sessions       list saved sessions
/cwd            show working directory
/quit           exit
Shift+Tab       toggle plan/build mode
!<cmd>          run shell command inline
```

---

## Dependencies

```
openai
prompt_toolkit
rich
scrapling
ddgs
claude-agent-sdk   # internal
```

---

## File overview

```
ai_cli.py     Main REPL, multi-agent routing, UI components
ai_sync.py    AIAgent engine, OpenCodeAgent, event types, inference modes
db.py         SQLite session store, prompt history
```

---

*Built for a self-hosted GPU research environment. The architecture generalizes to any OpenAI-compatible inference backend.*
