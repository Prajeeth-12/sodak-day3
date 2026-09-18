"""Model providers. The agent only knows `generate`. (Given.)"""
from dataclasses import dataclass, field
from typing import Any


class AgentError(Exception):
    """A run could not finish. `retryable` says whether trying again later could work."""

    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code, self.message, self.retryable = code, message, retryable


@dataclass
class ToolCall:
    name: str
    args: dict


@dataclass
class ModelTurn:
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    raw: Any = None     # provider-native content, sent back as-is (keeps Gemini's thought signatures)


# `contents` is a plain list the agent builds up:
#   {"role": "user",  "text": str}
#   {"role": "model", "text": str | None, "tool_calls": [{"name", "args"}], "raw": ...}
#   {"role": "tool",  "name": str, "result": dict}


class GeminiProvider:
    def __init__(self, model: str):
        from google import genai

        self.client = genai.Client()        # reads GEMINI_API_KEY
        self.model = model

    def _to_gemini(self, contents: list[dict]):
        from google.genai import types

        out: list = []
        for c in contents:
            if c["role"] == "user":
                out.append(types.Content(role="user", parts=[types.Part.from_text(text=c["text"])]))
            elif c["role"] == "model":
                if c.get("raw") is not None:
                    out.append(c["raw"])
                    continue
                parts = [types.Part.from_text(text=c["text"])] if c.get("text") else []
                parts += [types.Part.from_function_call(name=t["name"], args=t["args"])
                          for t in c.get("tool_calls", [])]
                out.append(types.Content(role="model", parts=parts))
            elif c["role"] == "tool":
                part = types.Part.from_function_response(name=c["name"], response=c["result"])
                # Results of parallel calls travel together in one turn.
                if out and out[-1].role == "user" and all(p.function_response for p in out[-1].parts):
                    out[-1].parts.append(part)
                else:
                    out.append(types.Content(role="user", parts=[part]))
        return out

    def generate(self, system: str, contents: list[dict], tools: list) -> ModelTurn:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=tools,
            temperature=0,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        try:
            resp = self.client.models.generate_content(
                model=self.model, contents=self._to_gemini(contents), config=config)
        except errors.APIError as e:
            if e.code == 429:
                raise AgentError("provider_rate_limited", "Model quota exhausted. Wait a minute.", True) from e
            if e.code and e.code >= 500:
                raise AgentError("provider_unavailable", "Model provider failed.", True) from e
            raise AgentError("provider_error", str(e), False) from e

        content = resp.candidates[0].content if resp.candidates else None
        parts = (content.parts or []) if content else []
        text = "".join(p.text for p in parts if p.text and not p.thought) or None
        calls = [ToolCall(fc.name, dict(fc.args or {})) for fc in (resp.function_calls or [])]
        usage = resp.usage_metadata
        return ModelTurn(text=text, tool_calls=calls,
                         tokens_in=(usage.prompt_token_count or 0) if usage else 0,
                         tokens_out=(usage.candidates_token_count or 0) if usage else 0,
                         raw=content)


class NvidiaProvider:
    """NVIDIA NIM / OpenAI-compatible provider. Reads NVIDIA_API_KEY."""

    def __init__(self, model: str, api_key: str | None = None, base_url: str = "https://integrate.api.nvidia.com/v1"):
        import os

        self.model = model
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise AgentError("config_error", "NVIDIA_API_KEY is not set. Set it in .env or environment.", False)
        self.base_url = base_url.rstrip("/")

    def _tool_to_schema(self, fn_or_dict) -> dict:
        if isinstance(fn_or_dict, dict):
            return fn_or_dict
        import inspect
        import re
        import typing

        doc = fn_or_dict.__doc__ or ""
        parts = doc.split("Args:")
        summary = parts[0].strip()
        param_docs = {}
        if len(parts) > 1:
            args_part = parts[1].split("Returns:")[0]
            matches = re.findall(r"(\w+):\s*(.*?)(?=\n\s*\w+:|$)", args_part, re.DOTALL)
            for name, pdoc in matches:
                param_docs[name] = " ".join(pdoc.strip().split())

        def _json_type(annotation):
            args = typing.get_args(annotation)
            if args and type(None) in args:
                annotation = next(a for a in args if a is not type(None))
            if annotation is int:
                return "integer"
            if annotation is float:
                return "number"
            if annotation is bool:
                return "boolean"
            return "string"

        sig = inspect.signature(fn_or_dict)
        hints = typing.get_type_hints(fn_or_dict)
        props, req = {}, []
        for pname, param in sig.parameters.items():
            if pname in ("self", "cls"):
                continue
            prop = {"type": _json_type(hints.get(pname, str))}
            if pname in param_docs:
                prop["description"] = param_docs[pname]
            props[pname] = prop
            if param.default is inspect.Parameter.empty:
                req.append(pname)

        return {
            "type": "function",
            "function": {
                "name": getattr(fn_or_dict, "__name__", str(fn_or_dict)),
                "description": summary,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": req,
                },
            },
        }

    def _to_messages(self, system: str, contents: list[dict]) -> list[dict]:
        import json

        messages: list[dict] = [{"role": "system", "content": system}]
        i = 0
        while i < len(contents):
            c = contents[i]
            role = c.get("role")
            if role == "user":
                messages.append({"role": "user", "content": c.get("text") or ""})
                i += 1
            elif role == "model":
                calls = c.get("tool_calls") or []
                text = c.get("text")
                if not calls:
                    messages.append({"role": "assistant", "content": text or ""})
                    i += 1
                else:
                    tool_responses = []
                    j = i + 1
                    while j < len(contents) and contents[j].get("role") == "tool":
                        tool_responses.append(contents[j])
                        j += 1

                    if len(calls) == 1:
                        call_id = "call_0"
                        messages.append({
                            "role": "assistant",
                            "content": text or None,
                            "tool_calls": [{
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": calls[0]["name"],
                                    "arguments": json.dumps(calls[0]["args"]) if isinstance(calls[0]["args"], dict) else str(calls[0]["args"]),
                                },
                            }],
                        })
                        for resp in tool_responses:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": json.dumps(resp["result"]) if isinstance(resp["result"], (dict, list)) else str(resp["result"]),
                            })
                    else:
                        for idx, call in enumerate(calls):
                            call_id = f"call_{idx}"
                            messages.append({
                                "role": "assistant",
                                "content": (text if idx == 0 else None),
                                "tool_calls": [{
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": call["name"],
                                        "arguments": json.dumps(call["args"]) if isinstance(call["args"], dict) else str(call["args"]),
                                    },
                                }],
                            })
                            matched_resp = None
                            for resp in tool_responses:
                                if resp.get("name") == call["name"]:
                                    matched_resp = resp
                                    break
                            if matched_resp is None and idx < len(tool_responses):
                                matched_resp = tool_responses[idx]

                            if matched_resp is not None:
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": call_id,
                                    "content": json.dumps(matched_resp["result"]) if isinstance(matched_resp["result"], (dict, list)) else str(matched_resp["result"]),
                                })
                    i = j
            elif role == "tool":
                messages.append({
                    "role": "tool",
                    "tool_call_id": f"call_orphan_{i}",
                    "content": json.dumps(c["result"]) if isinstance(c["result"], (dict, list)) else str(c["result"]),
                })
                i += 1
            else:
                i += 1

        return messages

    def generate(self, system: str, contents: list[dict], tools: list) -> ModelTurn:
        import json
        import httpx

        tool_schemas = [self._tool_to_schema(t) for t in tools] if tools else None
        messages = self._to_messages(system, contents)

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
        }
        if tool_schemas:
            payload["tools"] = tool_schemas

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=45.0) as client:
                resp = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
        except httpx.TimeoutException as e:
            raise AgentError("provider_timeout", "NVIDIA request timed out. Retryable.", True) from e
        except httpx.RequestError as e:
            raise AgentError("provider_unavailable", f"NVIDIA connection error: {e}", True) from e

        if resp.status_code == 429:
            raise AgentError("provider_rate_limited", "Model quota exhausted. Wait a minute.", True)
        if resp.status_code >= 500:
            raise AgentError("provider_unavailable", f"Model provider failed ({resp.status_code}).", True)
        if resp.status_code >= 400:
            raise AgentError("provider_error", f"NVIDIA API error ({resp.status_code}): {resp.text}", False)

        data = resp.json()
        choice = (data.get("choices") or [{}])[0].get("message", {})
        text = choice.get("content") or None

        raw_calls = choice.get("tool_calls") or []
        calls = []
        for rc in raw_calls:
            fn = rc.get("function", {})
            name = fn.get("name", "")
            args_str = fn.get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else dict(args_str or {})
            except Exception:
                args = {}
            calls.append(ToolCall(name=name, args=args))

        usage = data.get("usage") or {}
        return ModelTurn(
            text=text,
            tool_calls=calls,
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
            raw=data,
        )



class ScriptedProvider:
    """Replays a fixed list of turns in call order. No network, no quota. Used by the tests."""

    model = "mock"

    def __init__(self, script: list, loop: bool = False):
        self.original, self.script, self.loop = list(script), list(script), loop
        self.calls: list[list[dict]] = []      # what the agent sent on each call

    def generate(self, system: str, contents: list[dict], tools: list) -> ModelTurn:
        self.calls.append([dict(c) for c in contents])
        if not self.script and self.loop:
            self.script = list(self.original)
        if not self.script:
            return ModelTurn(text="(mock) script exhausted")
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


class PositionalMock:
    """A scripted model that answers by position in the current turn, not by call count.
    A fresh process that resumes a half-finished run gets the NEXT turn, not the first one.
    `slow` sleeps before each answer, so you have time to kill the worker mid-run."""

    model = "mock"

    def __init__(self, turns: list[ModelTurn], slow: float = 0.0):
        self.turns, self.slow = turns, slow
        self.calls: list[list[dict]] = []

    def generate(self, system: str, contents: list[dict], tools: list) -> ModelTurn:
        import time

        self.calls.append([dict(c) for c in contents])
        last_user = max(i for i, c in enumerate(contents) if c["role"] == "user")
        position = sum(1 for c in contents[last_user:] if c["role"] == "model")
        if self.slow:
            time.sleep(self.slow)
        if position >= len(self.turns):
            return ModelTurn(text="(mock) nothing more to do.")
        return self.turns[position]


def booking_mock(slow: float = 0.0) -> PositionalMock:
    """Check, apply, book + notify, answer: every side-effect tool in one run."""
    return PositionalMock([
        ModelTurn(text=None, tool_calls=[ToolCall("check_eligibility", {"student_id": "22CS045", "drive_id": 1})],
                  tokens_in=120, tokens_out=12),
        ModelTurn(text=None, tool_calls=[ToolCall("apply_to_drive", {"student_id": "22CS045", "drive_id": 1})],
                  tokens_in=160, tokens_out=12),
        ModelTurn(text=None, tool_calls=[
            ToolCall("book_interview_slot", {"student_id": "22CS045", "slot_id": 1}),
            ToolCall("notify_student", {"student_id": "22CS045", "message": "Your Zoho interview slot is booked."})],
            tokens_in=220, tokens_out=30),
        ModelTurn(text="(mock) Done: applied to Zoho, booked slot 1 and sent you a confirmation.",
                  tokens_in=300, tokens_out=20),
    ], slow=slow)
