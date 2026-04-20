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


if __name__ == "__main__":
    os.environ['OPENAI_BASE_URL'] = "http://192.168.170.76:8000/v1"

    def demo_streaming():
        print("=== streaming ===")
        agent = AIAgent()
        chat = Chat("Say hello in one sentence.")
        for chunk in agent.step(chat):
            if isinstance(chunk, Text):
                print(chunk.content, end="", flush=True)
        print()

    def demo_step():
        print("\n=== step (manual tool loop) ===")
        agent = AIAgent(tools=[get_current_weather, get_current_time])
        chat = Chat("What is the weather in Austin, TX in celsius?")

        while True:
            for chunk in agent.step(chat, mode="instruct_general",
                                    tools=[get_current_weather, get_current_time],
                                    tool_choice="auto"):
                if isinstance(chunk, Text) and chunk.id is None:
                    print(chunk.content, end="", flush=True)
                elif isinstance(chunk, ToolCall):
                    print(f"\n[tool call: {chunk.name}({chunk.arguments})]")
                    chat.add(chunk)
                elif isinstance(chunk, Assistant):
                    chat.add(chunk)
            last = chat.messages[-1]
            if last["role"] == "assistant" and not last.get("tool_calls"):
                break
            for msg in chat.messages:
                if msg["role"] == "assistant" and msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        name = tc["function"]["name"]
                        args = json.loads(tc["function"]["arguments"] or "{}")
                        result = agent._fn_registry[name](**args)
                        print(f"[result: {result}]")
                        chat._append_tool_result(ToolCall(name=name, id=tc["id"], arguments="{}"), str(result))
                    break
        print()

    def demo_forward():
        print("\n=== forward (auto) ===")
        agent = AIAgent(tools=[get_current_weather, get_current_time])
        chat = Chat("What is the weather in New York, NY in fahrenheit and what time is it?")

        for event in agent.forward(chat, mode="instruct_general", tool_choice="auto"):
            if isinstance(event, Text) and event.id is None:
                print(event.content, end="", flush=True)
            elif isinstance(event, ToolCall):
                print(f"\n[calling {event.name}({event.arguments})]")
            elif isinstance(event, ToolResult):
                print(f"[→ {event.result}]")
            elif isinstance(event, DoneEvent):
                print("\n--- done ---")

    # ── 4. sub-agent tool (real streaming) ───────────────────────────────
    def demo_subagent():
        print("\n=== forward (with streaming sub-agent) ===")

        agent = AIAgent()

        def research(topic: str):
            """Research a topic in depth and return a summary"""
            sub_chat = Chat(f"Give a 3 sentence summary about: {topic}")
            yield from agent.step(sub_chat, mode="instruct_general")

        agent2 = AIAgent(tools=[research])
        chat = Chat("Research the Golden Gate Bridge for me.")

        for event in agent2.forward(chat, mode="instruct_general", tool_choice="auto"):
            if isinstance(event, Text) and event.id is None:
                print(event.content, end="", flush=True)
            elif isinstance(event, Text) and event.id is not None:
                print(f"\n[subagent chunk] {event.content}", end="", flush=True)
            elif isinstance(event, Assistant) and event.id is not None:
                print(f"\n[subagent done] {event.content}")
            elif isinstance(event, ToolCall):
                print(f"\n[calling subagent: {event.name}({event.arguments})]")
            elif isinstance(event, ToolResult):
                print(f"[subagent result saved: {event.result[:60]}...]")
            elif isinstance(event, DoneEvent):
                print("\n--- done ---")

        print(json.dumps(chat.messages, indent=2))

    # ── 5. multi-agent (orchestrator → researcher + analyst) ─────────────
    def demo_multiagent():
        print("\n=== multi-agent (orchestrator + researcher + analyst) ===")

        base_agent = AIAgent(tools=[web_search, web_fetch])

        def researcher(topic: str):
            """Search the web for a topic, fetch the top result URL and return findings."""
            sub_chat = Chat(f"Search for '{topic}', then fetch the top result URL and summarize what you find.")
            yield from base_agent.forward(sub_chat)

        def analyst(findings: str):
            """Analyze research findings and produce a structured report."""
            sub_chat = Chat(f"Analyze these findings and write a concise structured report:\n\n{findings}")
            yield from base_agent.step(sub_chat)

        orchestrator = AIAgent(tools=[researcher, analyst])
        chat = Chat("I need a full report on the latest advancements in quantum computing.")

        def print_event(event, indent=""):
            if isinstance(event, Text) and event.id is None:
                print(event.content, end="", flush=True)
            elif isinstance(event, Text) and event.id is not None:
                print(f"\n{indent}[stream:{event.id[:16]}] {event.content}", end="", flush=True)
            elif isinstance(event, ToolCall):
                print(f"\n{indent}[→ calling {event.name}({event.arguments[:60]}...)]")
            elif isinstance(event, ToolResult):
                print(f"\n{indent}[← {event.name} done: {event.result[:80]}...]")
            elif isinstance(event, DoneEvent):
                print(f"\n{indent}--- done ---")

        for event in orchestrator.forward(chat):
            print_event(event)

    # ── 6. plain web_search + web_fetch ─────────────────────────────────
    def demo_plain_web():
        print("\n=== plain web_search + web_fetch ===")
        import sys
        sys.path.insert(0, "/home/ntlpt24/Master/buildmode/personal/lab/src/projects/tools/web")
        from web_search import web_search as plain_web_search
        from web_fetch import web_fetch as plain_web_fetch

        agent = AIAgent(tools=[plain_web_search, plain_web_fetch])
        chat = Chat("Search for 'Python asyncio tutorial' and fetch the top result.")

        for event in agent.forward(chat, mode="instruct_general", tool_choice="auto"):
            if isinstance(event, Text) and event.id is None:
                print(event.content, end="", flush=True)
            elif isinstance(event, ToolCall):
                print(f"\n[calling {event.name}({event.arguments[:80]})]")
            elif isinstance(event, ToolResult):
                print(f"\n[← {event.name} done: {str(event.result)[:120]}...]")
            elif isinstance(event, DoneEvent):
                print("\n--- done ---")

    # ── 7. all web tools, prefix-selected at call time ───────────────────
    def demo_all_web_tools():
        print("\n=== all web tools, prefix-selected ===")
        import sys
        sys.path.insert(0, "/home/ntlpt24/Master/buildmode/personal/lab/src/projects/tools/web")
        from web_search import web_search as plain_web_search
        from web_fetch import web_fetch as plain_web_fetch
        from web_chromadb_based import chromadb_based_web_search, chromadb_based_web_fetch

        agent = AIAgent(tools=[plain_web_search, plain_web_fetch, chromadb_based_web_search, chromadb_based_web_fetch])

        print("\n-- using web_ (plain) --")
        chat = Chat("Search for 'FastAPI tutorial' and fetch the top result.")
        for event in agent.forward(chat, tools=["web"]):
            if isinstance(event, Text) and event.id is None:
                print(event.content, end="", flush=True)
            elif isinstance(event, ToolCall):
                print(f"\n[calling {event.name}({event.arguments})]")
            elif isinstance(event, ToolResult):
                print(f"\n[← {event.name}: {str(event.result)}...]")
            elif isinstance(event, DoneEvent):
                print("\n--- done ---")

        print("\n-- using chromadb_ (chromadb-backed) --")
        chat2 = Chat("Search for 'FastAPI tutorial' and read the top result in detail.")
        for event in agent.forward(chat2, tools=["chromadb"]):
            if isinstance(event, Text) and event.id is None:
                print(event.content, end="", flush=True)
            elif isinstance(event, ToolCall):
                print(f"\n[calling {event.name}({event.arguments})]")
            elif isinstance(event, ToolResult):
                print(f"\n[← {event.name}: {str(event.result)}...]")
            elif isinstance(event, DoneEvent):
                print("\n--- done ---")

    def _print_stream(event):
        if isinstance(event, Text) and event.id is None:
            print(event.content, end="", flush=True)
        elif isinstance(event, ToolCall):
            print(f"\n[calling {event.name}({event.arguments})]")
        elif isinstance(event, ToolResult):
            print(f"\n[→ {event.result}]")
        elif isinstance(event, DoneEvent):
            print("\n--- done ---")

    # ── 8. structured outputs — choice (step) ───────────────────────────
    def demo_structured_choice():
        print("\n=== structured output: choice (step) ===")
        agent = AIAgent()
        chat = Chat("Classify this sentiment: vLLM is wonderful!")
        for chunk in agent.step(
            chat, mode="instruct_general",
            structured_output=StructuredOutput(choice=["positive", "negative"])
        ):
            if isinstance(chunk, Text):
                print(chunk.content, end="", flush=True)
        print()

    # ── 9. structured outputs — regex (step) ────────────────────────────
    def demo_structured_regex():
        print("\n=== structured output: regex (step) ===")
        agent = AIAgent()
        chat = Chat(
            "Generate an example email address for Alan Turing who works at Enigma. "
            "End in .com. Example: alan.turing@enigma.com"
        )
        for chunk in agent.step(
            chat, mode="instruct_general",
            structured_output=StructuredOutput(regex=r"\w+@\w+\.com"),
            stop=["\n"],
        ):
            if isinstance(chunk, Text):
                print(chunk.content, end="", flush=True)
        print()

    # ── 10. structured outputs — json via Pydantic (step) ────────────────
    def demo_structured_json_pydantic():
        print("\n=== structured output: json/Pydantic (step) ===")
        from pydantic import BaseModel
        from enum import Enum

        class CarType(str, Enum):
            sedan = "sedan"
            suv = "SUV"
            truck = "Truck"
            coupe = "Coupe"

        class CarDescription(BaseModel):
            brand: str
            model: str
            car_type: CarType

        agent = AIAgent()
        chat = Chat(
            "Generate a JSON with the brand, model and car_type of the most iconic car from the 90s. "
            f"Schema: {CarDescription.model_json_schema()}"
        )
        for chunk in agent.step(
            chat, mode="instruct_general",
            structured_output=StructuredOutput(json=CarDescription)
        ):
            if isinstance(chunk, Text):
                print(chunk.content, end="", flush=True)
        print()

    # ── 11. structured outputs — json via raw dict schema (step) ─────────
    def demo_structured_json_dict():
        print("\n=== structured output: json/dict schema (step) ===")
        person_schema = {
            "title": "Person",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age":  {"type": "integer"},
                "city": {"type": "string"},
            },
            "required": ["name", "age", "city"],
        }
        agent = AIAgent()
        chat = Chat(
            "Generate a JSON object for a fictional person. "
            f"Schema: {person_schema}"
        )
        for chunk in agent.step(
            chat, mode="instruct_general",
            structured_output=StructuredOutput(json=person_schema)
        ):
            if isinstance(chunk, Text):
                print(chunk.content, end="", flush=True)
        print()

    # ── 12. structured outputs — grammar/EBNF (step) ─────────────────────
    def demo_structured_grammar():
        print("\n=== structured output: grammar/EBNF (step) ===")
        simplified_sql_grammar = """
    root ::= select_statement
    select_statement ::= "SELECT " column " from " table " where " condition
    column ::= "col_1 " | "col_2 "
    table ::= "table_1 " | "table_2 "
    condition ::= column "= " number
    number ::= "1 " | "2 "
"""
        agent = AIAgent()
        chat = Chat("Generate an SQL query to show col_1 from table_1 where col_2 = 1.")
        for chunk in agent.step(
            chat, mode="instruct_general",
            structured_output=StructuredOutput(grammar=simplified_sql_grammar)
        ):
            if isinstance(chunk, Text):
                print(chunk.content, end="", flush=True)
        print()

    # ── 13. structured outputs — choice via forward ───────────────────────
    def demo_forward_structured_choice():
        print("\n=== structured output: choice (forward) ===")
        agent = AIAgent()
        chat = Chat("Is Python a good language for beginners? Answer with one word.")
        for event in agent.forward(
            chat, mode="instruct_general",
            structured_output=StructuredOutput(choice=["yes", "no", "maybe"])
        ):
            _print_stream(event)

    # ── 14. structured outputs — json/Pydantic via forward ───────────────
    def demo_forward_structured_json_pydantic():
        print("\n=== structured output: json/Pydantic (forward) ===")
        from pydantic import BaseModel

        class MovieReview(BaseModel):
            title: str
            year: int
            rating: float
            summary: str

        agent = AIAgent()
        chat = Chat(
            "Write a review for the movie Inception. "
            f"Schema: {MovieReview.model_json_schema()}"
        )
        for event in agent.forward(
            chat, mode="instruct_general",
            structured_output=StructuredOutput(json=MovieReview)
        ):
            _print_stream(event)

    # ── 15. structured outputs — json/dict schema via forward ────────────
    def demo_forward_structured_json_dict():
        print("\n=== structured output: json/dict schema (forward) ===")
        product_schema = {
            "title": "Product",
            "type": "object",
            "properties": {
                "name":     {"type": "string"},
                "price":    {"type": "number"},
                "in_stock": {"type": "boolean"},
                "tags":     {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "price", "in_stock", "tags"],
        }
        agent = AIAgent()
        chat = Chat(
            "Generate a JSON for a fictional software product. "
            f"Schema: {product_schema}"
        )
        for event in agent.forward(
            chat, mode="instruct_general",
            structured_output=StructuredOutput(json=product_schema)
        ):
            _print_stream(event)

    # ── new high-level API demos ──────────────────────────────────────────
    def demo_task():
        print("\n=== task / __call__ ===")
        agent = AIAgent()
        chat = agent("Say hello in one sentence.")
        print("answer:", chat.answer)

    def demo_batch():
        print("\n=== batch (parallel) ===")
        agent = AIAgent()
        questions = [
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?",
            "What is 12 * 8?",
        ]
        results = agent.batch(questions, mode="instruct_general")
        for q, chat in zip(questions, results):
            print(f"Q: {q}\nA: {chat.answer}\n")

    def demo_compress():
        print("\n=== compress ===")
        agent = AIAgent()
        chat = Chat()
        for i in range(6):
            chat.add(f"Tell me about topic {i}", role="user")
            chat.add(f"Topic {i} is about X, Y, and Z with many interesting aspects.", role="assistant")
        print(f"Before: {len(chat.messages)} messages")
        compressed = agent.compress(chat, keep_last=2)
        print(f"After:  {len(compressed.messages)} messages")
        print("Summary:", compressed.messages[1]["content"][:120])

    def demo_pipe():
        print("\n=== pipe (chain agents) ===")
        agent = AIAgent()
        # stage 1: expand → stage 2: compress into a single sentence
        pipeline = agent.pipe(agent)
        r1 = agent.task("List 3 benefits of open-source software.", mode="instruct_general")
        r2 = agent.task(
            f"Summarize this in one sentence: {r1.answer}", mode="instruct_general"
        )
        print("Stage 1:", r1.answer)
        print("Stage 2 (pipe):", r2.answer)
        # one-shot via pipeline
        result = pipeline("What is machine learning?", mode="instruct_general")
        print("Pipeline one-shot:", result.answer)

    def demo_evaluate():
        print("\n=== evaluate ===")
        agent = AIAgent()
        chat = agent("Explain recursion in one sentence.", mode="instruct_general")
        score = agent.evaluate(chat, rubric="Is the answer clear, accurate, and concise?")
        print(f"Answer: {chat.answer}")
        print(f"Score:  {score:.3f} / 1.0")

    def demo_structured_new():
        print("\n=== structured() new API ===")
        from pydantic import BaseModel

        class Recipe(BaseModel):
            name: str
            ingredients: list[str]
            steps: list[str]

        agent = AIAgent()
        result = agent.structured(
            f"Give me a simple pasta recipe. Schema: {Recipe.model_json_schema()}",
            schema=Recipe,
            mode="instruct_general",
        )
        # result is a Recipe instance, not a dict
        print(type(result))           # <class '__main__.Recipe'>
        print(result.name)            # "Spaghetti Carbonara"
        print(result.ingredients)     # ['pasta', 'eggs', ...]
        print(result.model_dump())    # full dict if needed

    demo_task()
    demo_batch()
    demo_compress()
    demo_pipe()
    demo_evaluate()
    demo_structured_new()

    # ── original demos ────────────────────────────────────────────────────
    demo_streaming()
    demo_step()
    demo_forward()
    demo_subagent()
    demo_multiagent()
    demo_structured_choice()
    demo_structured_regex()
    demo_structured_json_pydantic()
    demo_structured_json_dict()
    demo_structured_grammar()
    demo_forward_structured_choice()
    demo_forward_structured_json_pydantic()
    demo_forward_structured_json_dict()
