# kivi

A zero-dependency AI agent REPL for the terminal. One file. Pure Python.

---

## What it is

`kivi.py` is a single-file, streaming AI coding agent that runs entirely in your terminal. It connects to any OpenAI-compatible endpoint (vLLM, SGLang, OpenAI, Anthropic via proxy) and gives you a full agentic coding loop with tools, parallel execution, session persistence, and rich rendering — with no external runtime dependencies beyond the standard library.

---

## Why it's different

| Feature | kivi | LangChain / CrewAI | Claude Code | Shell + curl |
|---|---|---|---|---|
| Single file | ✓ | ✗ | ✗ | ✗ |
| Zero pip installs | ✓ | ✗ | ✗ | ✓ |
| Streaming tool calls | ✓ | partial | ✓ | ✗ |
| Parallel tool execution | ✓ | ✗ | ✓ | ✗ |
| Local vLLM / SGLang | ✓ | ✓ | ✗ | ✓ |
| Session persistence (SQLite) | ✓ | ✗ | partial | ✗ |
| Trie-based autosuggestion | ✓ | — | — | ✗ |
| Raw terminal (no readline dep) | ✓ | — | — | ✗ |
| Markdown + table rendering | ✓ | — | — | ✗ |
| Multi-agent (5 parallel kivis) | ✓ | ✓ | ✗ | ✗ |

---

## Dependencies

```
Python ≥ 3.12   (stdlib only: urllib, http.client, sqlite3, termios, tty, readline)
```

No `pip install` required. `openai`, `rich`, `prompt_toolkit` — all replaced with pure Python implementations built into the file.

---

## Quickstart

```bash
# Point at any OpenAI-compatible endpoint
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY

python kivi.py
```

```
kivi> explain this codebase @src/
kivi> refactor auth.py to use JWT
kivi> ask 5 kivis to review this PR in parallel
```

---

## Architecture

```
kivi.py  (4500 lines, single file)
├── OpenAI          — pure urllib HTTP client, SSE streaming, tool_calls
├── AIAgent         — agentic loop: stream → tools → re-prompt until done
├── PromptToolKit   — raw termios prompt: trie autosuggestion, tab dropdown,
│                     history (SQLite-backed), Shift+Tab mode toggle
├── StreamText      — live streaming output + erase/re-render as markdown
├── ParallelToolDisplay — thread-based animated tool execution display
├── _print_markdown — full ANSI markdown renderer (tables, code, links, lists)
└── SQLite          — session history, prompt history, Claude usage tracking
```

### OpenAI client (pure Python)

Replaces the `openai` SDK entirely. Handles:
- Streaming SSE (`data: {...}\n\n`) via `http.client`
- `extra_body` merged at top-level (vLLM `chat_template_kwargs`, `top_k`, `min_p`, `repetition_penalty`)
- Chunk attribute access via recursive `_Obj` (mirrors the SDK's Pydantic models)

### Terminal renderer

No `rich`. No `curses`. ANSI escapes only.

- **Tables**: measures natural column widths → fits to terminal → expands equally to fill full width
- **Code blocks**: boxed with language label, syntax-coloured
- **Markdown**: headings, bold/italic/strikethrough, links, bullets, blockquotes, horizontal rules

### Prompt engine

No `readline`. No `prompt_toolkit`. Raw `termios`.

- **Trie autosuggestion**: O(prefix) lookup over full history, grey suffix shown inline, `→` to accept
- **Tab dropdown**: slash-command completion with Up/Down navigation
- **History**: SQLite-backed, cwd-scoped across sessions

---

## Modes

```
instruct_coding    — temp=0.3, no thinking  (default)
thinking_coding    — temp=0.6, extended reasoning
thinking_general   — temp=1.0, open-ended
instruct_general   — temp=0.7, fast replies
instruct_reasoning — temp=1.0, instruct+reason
```

Switch with `/mode <name>` or cycle with Shift+Tab.

---

## Agents

```
/kivi      — local vLLM agent (default)
/claude    — Claude via Anthropic API
/copilot   — GitHub Copilot CLI
/opencode  — OpenCode agent
```

---

## Tools

`bash` · `read` · `edit` · `glob` · `grep` · `write` · `python` · sub-agents (recursive kivi calls)

Parallel tool calls are streamed, executed concurrently, and displayed with a live animated status view.

---

## Session management

```
/sessions          — list all past sessions
/history           — message count for current session
/clear             — new session, same endpoint
```

All sessions stored in `~/.ai_cli/sessions.db`. Resume any session by ID.

---

## License

MIT
