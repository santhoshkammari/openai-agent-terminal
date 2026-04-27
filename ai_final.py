import os
import json
import time
import inspect
import concurrent.futures
from openai import OpenAI
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

    @property
    def stop(self) -> bool:
        return self._messages[-1]['role'] != 'assistant'

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

    def __init__(self, config: AIConfig = None, tools=None, name: str = None, description: str = None) -> None:
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
                    chat.add(chunk)
                    yield chunk
                elif isinstance(chunk, Assistant):
                    chat.add(chunk)
                    yield chunk
                elif isinstance(chunk, Text) and chunk.id is None:
                    step_text += chunk.content
                    yield chunk
                else:
                    yield chunk

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
                    future_to_tc = {pool.submit(_exec_tool, tc): tc for tc in tool_calls}
                    for future in concurrent.futures.as_completed(future_to_tc):
                        tc = future_to_tc[future]
                        result = future.result()
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


def get_current_weather(city: str, state: str, unit: str) -> str:
    """Get the current weather in a given location"""
    return f"72°F, sunny in {city}, {state}"

def get_current_time():
    """Get the current time"""
    from datetime import datetime
    return datetime.now().isoformat()


# ── file operation tools ──────────────────────────────────────────────────────

_work_dir = "."

def _resolve_path(path: str):
    from pathlib import Path
    p = Path(path)
    return p if p.is_absolute() else Path(_work_dir) / p

def read_file(path: str) -> str:
    """Read a file and return its contents."""
    try:
        return _resolve_path(path).read_text()
    except Exception as e:
        return f"[read error] {e}"

def write_file(path: str, content: str) -> str:
    """Write content to a file, creating or overwriting it."""
    try:
        p = _resolve_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"[wrote {len(content)} chars to {path}]"
    except Exception as e:
        return f"[write error] {e}"

def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Replace old_string with new_string in file (first occurrence)."""
    try:
        import difflib
        p = _resolve_path(path)
        text = p.read_text()
        if old_string not in text:
            return f"[edit error] old_string not found in {path}"
        new_text = text.replace(old_string, new_string, 1)
        p.write_text(new_text)
        diff = list(difflib.unified_diff(
            text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=f"a/{path}", tofile=f"b/{path}",
        ))
        return "".join(diff) if diff else "[no changes]"
    except Exception as e:
        return f"[edit error] {e}"

def bash_run(command: str) -> str:
    """Run a shell command and return stdout+stderr (max 8000 chars)."""
    import subprocess
    try:
        r = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=60, cwd=_work_dir,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:8000] if out else "[no output]"
    except subprocess.TimeoutExpired:
        return "[bash error] timed out (60s)"
    except Exception as e:
        return f"[bash error] {e}"

def glob_files(pattern: str) -> str:
    """Find files matching a glob pattern under the working directory."""
    from pathlib import Path
    try:
        matches = sorted(str(p) for p in Path(_work_dir).glob(pattern))
        return "\n".join(matches) if matches else "[glob] no matches"
    except Exception as e:
        return f"[glob error] {e}"

def grep_files(pattern: str, path: str = ".") -> str:
    """Search for regex pattern in files. path is relative to working dir."""
    import subprocess
    from pathlib import Path
    try:
        search_path = path if Path(path).is_absolute() else str(Path(_work_dir) / path)
        r = subprocess.run(
            ["grep", "-rn", pattern, search_path],
            capture_output=True, text=True, timeout=30,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:8000] if out else "[grep] no matches"
    except Exception as e:
        return f"[grep error] {e}"


def web_fetch(url: str) -> str:
    """Fetch a URL and return its markdown content."""
    try:
        from typing import cast
        from scrapling.fetchers import Fetcher
        from scrapling.core.shell import Convertor
        from scrapling.engines._browsers._types import ImpersonateType
        page = Fetcher.get(url, timeout=30, retries=3, retry_delay=1,
                           impersonate=cast(ImpersonateType, "chrome"))
        from scrapling.core._types import extraction_types
        content = list(Convertor._extract_content(
            page, css_selector=None,
            extraction_type=cast(extraction_types, 'markdown'),
            main_content_only=True,
        ))
        return "\n".join(content)
    except Exception as e:
        return f"Error fetching {url}: {str(e)}"


def web_search(query: str) -> str:
    """Search the web and return top results as text."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=10))
        lines = [f"{r['title']}\n{r.get('href','')}\n{r.get('body','')}" for r in raw]
        return "\n\n".join(lines)
    except Exception as e:
        return f"Search error: {str(e)}"


class AICli:
    """CLI interface for AIAgent / OpenCodeAgent."""

    BUILTIN_TOOLS = {
        "web_search": web_search,
        "web_fetch": web_fetch,
        "weather": get_current_weather,
        "time": get_current_time,
        "read": read_file,
        "write": write_file,
        "edit": edit_file,
        "bash": bash_run,
        "glob": glob_files,
        "grep": grep_files,
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

        # ── db / session helpers ───────────────────────────────────────────
        def _db():
            try:
                from db import (
                    new_session_id, save_session, load_session,
                    latest_session_for_dir, title_from_history,
                )
                return new_session_id, save_session, load_session, latest_session_for_dir, title_from_history
            except ImportError:
                return None

        session_id = None
        prior_history = None
        db_fns = _db()

        if db_fns:
            new_session_id_fn, save_session_fn, load_session_fn, latest_for_dir_fn, title_fn = db_fns

            if args.resume:
                rec = load_session_fn(args.resume)
                if not rec:
                    print(f"\033[31msession {args.resume!r} not found\033[0m")
                    sys.exit(1)
                session_id = rec["id"]
                prior_history = rec["history"]
                _work_dir = rec.get("work_dir") or work_dir
            elif args.continue_:
                rec = latest_for_dir_fn(work_dir)
                if rec:
                    session_id = rec["id"]
                    prior_history = rec["history"]
                else:
                    print(f"\033[2mno session found for {work_dir}, starting fresh\033[0m")

            if session_id is None:
                session_id = new_session_id_fn()
        else:
            import uuid
            session_id = uuid.uuid4().hex[:8]

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
    AICli.run()
