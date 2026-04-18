# kivi

**One terminal. Four AI agents. Your own GPU.**

A streaming AI coding REPL that runs Kivi (local vLLM), Claude, OpenCode, and GitHub Copilot in a single session — shared context, unified tools, zero hosted API lock-in.

```
kivi> fix the auth bug          ← Kivi reasons + edits files in parallel
/claude                         ← switch to Claude mid-conversation
kivi> explain what you just did ← Claude sees the full prior context
```

---

## What it does

- **Agentic loop** — LLM reasons → calls tools in parallel → loops until done. Streams every token live.
- **4 agents, 1 session** — `/kivi` `/claude` `/opencode` `/copilot` swap the backend. Chat history is shared across all of them.
- **Plan/Build mode** — `Shift+Tab` toggles between planning (read-only) and building (full edits). System prompt swaps automatically.
- **Parallel tools** — when the LLM requests multiple tools at once, they run concurrently. Rich live-tree shows status + diffs as they complete.
- **Thinking mode** — extended reasoning tokens stream to a spinner. `/think` expands the full trace after the response.
- **Session persistence** — every turn auto-saved to SQLite. `--resume` or `--continue` restores with full history re-rendered.
- **Self-hosted** — points at a local vLLM endpoint. No per-token billing, no rate limits.

## Tools

`read` `write` `edit` `bash` `glob` `grep` `web_search` `web_fetch` `kivi` (sub-agent) `opencode` `claude`

## Stack

Python 3.11 · vLLM · OpenAI-compatible API · Claude Agent SDK · prompt_toolkit · rich · SQLite

## Usage

```bash
python ai_cli.py                        # REPL
python ai_cli.py -p "add type hints"   # single prompt
python ai_cli.py --continue             # resume last session
python ai_cli.py --list                 # list sessions
```

**REPL commands:** `/modes` `/mode <name>` `/claude` `/kivi` `/opencode` `/copilot` `/clear` `/think` `/sessions` `/quit` · `!<cmd>` runs shell inline · `Shift+Tab` plan/build toggle

## Inference modes

| Mode | Temp | Thinking |
|---|---|---|
| `thinking_coding` | 0.6 | ✓ |
| `instruct_coding` | 0.3 | — |
| `thinking_general` | 1.0 | ✓ |
| `instruct_general` | 0.7 | — |

## Files

```
ai_cli.py   REPL + multi-agent routing + UI
ai_sync.py  AIAgent engine — streaming, parallel tools, agentic loop, structured outputs
db.py       SQLite session store + prompt history
```
