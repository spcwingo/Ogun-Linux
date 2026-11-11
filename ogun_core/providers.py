"""Model providers beyond Ollama: OpenAI-compatible servers, llama.cpp, and local model discovery.

Ogun's engine only needs a client with ``chat()``, ``check_connection()``, ``unload()`` and
``get_effective_timeout()`` (see ``ogun_core.engine.OllamaClient``). This module supplies that same
interface for the other ways people already run local models:

- ``openai``    - any server speaking the OpenAI chat API: llama.cpp's ``llama-server``, LM Studio,
                  vLLM, Hugging Face TGI, Unsloth Studio, Jan, ... (``--host`` picks the base URL)
- ``lmstudio``  - the ``openai`` client pre-pointed at LM Studio's default http://127.0.0.1:1234
- ``llama.cpp`` - if ``--model`` is a .gguf file, Ogun launches ``llama-server`` itself (GPU layer
                  count and context size come from the active VRAM profile) and talks to it via the
                  OpenAI API; if a server is already running at the host, Ogun attaches to it instead

It also finds the models already on the machine (``discover_local_models``): Ollama's library, any
running OpenAI-compatible server, GGUF files (Hugging Face cache, ~/models, LM Studio's folder,
Ogun's own Unsloth training exports under training/data/exports, $OGUN_MODEL_DIRS) and local
Hugging Face-format model folders. With the Ollama provider, ``--model some-file.gguf`` imports the
file into Ollama automatically (``ollama create``), so a GGUF exported by Unsloth just works.

Everything here is standard-library only and never raises during discovery: a source that is missing
or unreachable simply contributes nothing.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ogun_core.config import MODEL_NAME, OLLAMA_HOST, get_num_ctx, get_num_gpu, get_request_timeout, llm_options
from ogun_core.paths import OGUN_ROOT

# ---------------------------------------------------------------------------
# Provider names
# ---------------------------------------------------------------------------
_PROVIDER_ALIASES = {
    "ollama": "ollama",
    "openai": "openai",
    "openai-compatible": "openai",
    "openai_compatible": "openai",
    "generic": "openai",
    "llama.cpp": "llama.cpp",
    "llamacpp": "llama.cpp",
    "llama-cpp": "llama.cpp",
    "llama_cpp": "llama.cpp",
    "llamaserver": "llama.cpp",
    "lmstudio": "lmstudio",
    "lm-studio": "lmstudio",
    "lm_studio": "lmstudio",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "messages": "anthropic",
}

#: Base URLs used when the user does not pass --host / $OGUN_LLM_HOST.
PROVIDER_DEFAULT_HOSTS = {
    "openai": "http://127.0.0.1:8080",
    "llama.cpp": "http://127.0.0.1:8080",
    "lmstudio": "http://127.0.0.1:1234",
    "anthropic": "https://api.anthropic.com",
}

#: Friendly label per provider, for banners and error messages.
PROVIDER_LABELS = {
    "ollama": "Ollama",
    "openai": "OpenAI-compatible server",
    "llama.cpp": "llama.cpp (llama-server)",
    "lmstudio": "LM Studio",
    "anthropic": "Anthropic Messages API",
}


def normalize_provider(name: Optional[str]) -> Optional[str]:
    """Canonical provider key for a user-supplied name ('llamacpp', 'LM-Studio', ...), or None."""
    if not name:
        return None
    return _PROVIDER_ALIASES.get(str(name).strip().lower())


# ---------------------------------------------------------------------------
# OpenAI-compatible client (drop-in replacement for engine.OllamaClient)
# ---------------------------------------------------------------------------
def _map_openai_options(options: Dict[str, Any]) -> Dict[str, Any]:
    """Translate Ogun's Ollama-style options to OpenAI chat-completion fields.

    num_gpu / num_ctx / num_thread are Ollama load-time settings: the external server owns them,
    so they are dropped. num_predict becomes max_tokens.
    """
    out: Dict[str, Any] = {}
    for key in ("temperature", "top_p", "top_k", "stop", "seed", "frequency_penalty", "presence_penalty"):
        if key in options and options[key] is not None:
            out[key] = options[key]
    if options.get("num_predict"):
        out["max_tokens"] = int(options["num_predict"])
    return out


def _http_error_detail(exc: urllib.error.HTTPError, limit: int = 600) -> str:
    """Best-effort extraction of a server's own error text from an HTTP error body.

    OpenAI-style servers answer errors with {"error": {"message": ...}} and FastAPI apps
    with {"detail": ...}; either says far more than urllib's bare 'HTTP Error 500'.
    Returns '' when the body carries nothing readable. Never raises.
    """
    try:
        raw = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        return ""
    if not raw:
        return ""
    detail = ""
    try:
        data = json.loads(raw)
    except Exception:
        data = None
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            detail = str(err.get("message") or err.get("detail") or "")
        elif isinstance(err, str):
            detail = err
        if not detail and data.get("detail") is not None:
            d = data["detail"]
            detail = d if isinstance(d, str) else json.dumps(d)
        if not detail and data.get("message"):
            detail = str(data["message"])
    if not detail:
        detail = raw
    detail = " ".join(detail.split())
    return detail[:limit]


class OpenAICompatClient:
    """Client for any OpenAI-compatible chat server, with OllamaClient's interface.

    Pure urllib (no new dependencies). Supports streaming via Server-Sent Events, native tool
    calls, and the same adaptive timeout behaviour as the Ollama client.
    """

    def __init__(
        self,
        host: str = "http://127.0.0.1:8080",
        model: str = MODEL_NAME,
        timeout: Optional[float] = None,
        label: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.label = label or "OpenAI-compatible server"
        self.api_key = api_key or None
        self._custom_timeout = timeout is not None
        self.timeout = timeout if self._custom_timeout else get_request_timeout()

    def get_effective_timeout(self) -> float:
        return self.timeout if self._custom_timeout else get_request_timeout()

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # -- HTTP helpers ------------------------------------------------------
    def _request_json(self, path: str, payload: Optional[Dict[str, Any]] = None,
                      timeout: float = 5.0) -> Any:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            f"{self.host}{path}",
            data=data,
            headers=self._headers(),
            method="POST" if data is not None else "GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def list_models(self) -> List[str]:
        """Model ids the server advertises at GET /v1/models ([] when unreachable)."""
        try:
            data = self._request_json("/v1/models", timeout=5.0)
            return [str(m.get("id")) for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
        except Exception:
            return []

    def check_connection(self) -> Tuple[bool, str]:
        try:
            data = self._request_json("/v1/models", timeout=5.0)
            models = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
            return True, f"Connected to {self.label} at {self.host} ({len(models)} model(s) available)."
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                key_note = ("the key it was given was rejected" if self.api_key
                            else "no API key was given")
                return (
                    False,
                    f"{self.label} at {self.host} refused the request (HTTP {exc.code} Unauthorized): "
                    f"the server requires an API key and {key_note}. Provide one with $OGUN_LLM_API_KEY, "
                    "or save this endpoint as a custom provider whose keyenv names the key's env var "
                    "(/provider add <name> <host> [model] key=<api-key>, or keyenv=<VAR> naming the key's env var).",
                )
            detail = _http_error_detail(exc)
            return False, f"{self.label} at {self.host} returned HTTP {exc.code}." + (
                f" Server said: {detail}" if detail else "")
        except Exception as exc:
            return (
                False,
                f"Cannot connect to {self.label} at {self.host} ({exc}). "
                "Start that server first, or check the --host URL.",
            )

    def unload(self) -> Dict[str, Any]:
        """External servers own their model memory; there is no standard unload endpoint."""
        return {
            "unloaded": False,
            "model": self.model,
            "detail": f"{self.label} manages its own model memory; unload the model in that app to free the GPU.",
        }

    def _http_failure(self, exc: urllib.error.HTTPError) -> RuntimeError:
        """Turn a bare HTTPError into an error that carries the server's own explanation."""
        detail = _http_error_detail(exc)
        msg = f"HTTP {exc.code} {exc.reason} from {self.label} at {self.host}"
        if detail:
            msg += f" — server said: {detail}"
        return RuntimeError(msg)

    # -- Chat --------------------------------------------------------------
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        overrides = options or {}
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            **_map_openai_options(llm_options(**overrides)),
        }
        if tools:
            payload["tools"] = tools
        use_stream = bool(stream or on_token is not None)
        eff_timeout = self.get_effective_timeout()
        if use_stream:
            return self._chat_stream(payload, eff_timeout, on_token)
        try:
            data = self._request_json("/v1/chat/completions", payload, timeout=eff_timeout)
        except urllib.error.HTTPError as he:
            raise self._http_failure(he) from he
        except urllib.error.URLError as ue:
            if "timed out" in str(ue).lower():
                raise TimeoutError(str(ue))
            raise
        choices = data.get("choices") or []
        msg = (choices[0].get("message") or {}) if choices else {}
        return {
            "content": msg.get("content") or "",
            "tool_calls": msg.get("tool_calls") or [],
        }

    def _chat_stream(self, payload: Dict[str, Any], eff_timeout: float,
                     on_token: Optional[Callable[[str], None]]) -> Dict[str, Any]:
        payload = dict(payload, stream=True)
        req = urllib.request.Request(
            f"{self.host}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        parts: List[str] = []
        tc_by_index: Dict[int, Dict[str, Any]] = {}
        try:
            with urllib.request.urlopen(req, timeout=eff_timeout) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except Exception:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content") or ""
                    if piece:
                        parts.append(piece)
                        if on_token is not None:
                            on_token(piece)
                    for tc in delta.get("tool_calls") or []:
                        idx = int(tc.get("index", 0))
                        slot = tc_by_index.setdefault(idx, {"function": {"name": "", "arguments": ""}})
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            slot["function"]["arguments"] += fn["arguments"]
                        if tc.get("id"):
                            slot["id"] = tc["id"]
        except urllib.error.HTTPError as he:
            raise self._http_failure(he) from he
        except urllib.error.URLError as ue:
            if "timed out" in str(ue).lower():
                raise TimeoutError(str(ue))
            raise
        return {
            "content": "".join(parts),
            "tool_calls": [tc_by_index[i] for i in sorted(tc_by_index)],
        }


# ---------------------------------------------------------------------------
# Anthropic Messages API client (Anthropic itself, and Anthropic-shaped gateways
# such as Free Claude Code on http://127.0.0.1:8082)
# ---------------------------------------------------------------------------
def _to_anthropic_messages(messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """Translate Ogun's OpenAI-shaped history to Anthropic's (system, messages).

    System turns become the top-level system text; assistant tool_calls become tool_use
    blocks; tool results become tool_result blocks inside a user turn. Consecutive
    same-role turns are merged (Anthropic expects alternating roles). Tool-call ids are
    preserved when the producing client supplied them and synthesized deterministically
    (toolu_1, toolu_2, ... in history order) when it did not, so results always match.
    A result whose call is nowhere in the kept history is never emitted as an orphan
    tool_result (strict endpoints reject those): if the result entry itself names its
    call, the call is re-stated alongside it; otherwise the result is dropped.
    """
    system_parts: List[str] = []
    out: List[Dict[str, Any]] = []
    pending_ids: List[str] = []
    counter = 0

    def push(role: str, block: Dict[str, Any]) -> None:
        if out and out[-1]["role"] == role:
            out[-1]["content"].append(block)
        else:
            out.append({"role": role, "content": [block]})

    for msg in messages or []:
        role = str(msg.get("role") or "user")
        content = msg.get("content")
        if role == "system":
            if isinstance(content, str) and content:
                system_parts.append(content)
            continue
        if role == "tool":
            recorded_id = msg.get("tool_call_id")
            recorded_id = str(recorded_id) if recorded_id else ""
            if recorded_id and recorded_id in pending_ids:
                tool_id = recorded_id
                pending_ids.remove(tool_id)
            elif recorded_id:
                # The result names its call, but the call itself is not in the
                # kept history (pruned, or recorded by a client that stored the
                # id only on the result): re-state the call so the pair stays
                # matched instead of emitting an orphan tool_result.
                tool_id = recorded_id
                push("assistant", {"type": "tool_use", "id": tool_id,
                                   "name": str(msg.get("name") or ""), "input": {}})
            elif pending_ids:
                tool_id = pending_ids.pop(0)
            else:
                # Orphan result: no recorded call anywhere (e.g. history written
                # before calls were recorded). An unmatched tool_result is
                # rejected by strict endpoints — drop the result.
                continue
            push("user", {"type": "tool_result", "tool_use_id": tool_id,
                          "content": content if isinstance(content, str) else json.dumps(content)})
            continue
        if role == "assistant":
            if isinstance(content, str) and content:
                push("assistant", {"type": "text", "text": content})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                name = fn.get("name") or tc.get("name") or ""
                args = fn.get("arguments", tc.get("arguments", {}))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"raw": args}
                counter += 1
                tool_id = str(tc.get("id") or f"toolu_{counter}")
                pending_ids.append(tool_id)
                push("assistant", {"type": "tool_use", "id": tool_id, "name": name,
                                   "input": args if isinstance(args, dict) else {"value": args}})
            continue
        # user (and anything else) becomes a user text turn
        if isinstance(content, str):
            if content:
                push("user", {"type": "text", "text": content})
        elif content is not None:
            push("user", {"type": "text", "text": json.dumps(content)})
    return "\n\n".join(system_parts), out


def _to_anthropic_tools(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """OpenAI function schemas -> Anthropic tool definitions."""
    out: List[Dict[str, Any]] = []
    for tool in tools or []:
        fn = tool.get("function") or tool
        name = fn.get("name")
        if not name:
            continue
        schema = fn.get("input_schema") or fn.get("parameters") or {"type": "object", "properties": {}}
        out.append({"name": str(name), "description": str(fn.get("description") or ""),
                    "input_schema": schema})
    return out


def _map_anthropic_options(options: Dict[str, Any]) -> Dict[str, Any]:
    """Translate the shared llm_options payload to Anthropic fields.

    max_tokens is REQUIRED by the Messages API (falls back to 4096). The API accepts
    only one of temperature / top_p, so top_p is dropped when a temperature is set.
    """
    mapped = _map_openai_options(options)
    out: Dict[str, Any] = {"max_tokens": int(mapped.get("max_tokens") or 4096)}
    if mapped.get("temperature") is not None:
        out["temperature"] = mapped["temperature"]
    elif mapped.get("top_p") is not None:
        out["top_p"] = mapped["top_p"]
    if mapped.get("top_k") is not None:
        out["top_k"] = mapped["top_k"]
    stop = mapped.get("stop")
    if stop:
        out["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    return out


class AnthropicMessagesClient:
    """Client for Anthropic's Messages API and Anthropic-shaped gateways (e.g. Free
    Claude Code), with OllamaClient's interface. Pure urllib, like the OpenAI client."""

    def __init__(
        self,
        host: str = PROVIDER_DEFAULT_HOSTS["anthropic"],
        model: str = MODEL_NAME,
        timeout: Optional[float] = None,
        label: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.label = label or PROVIDER_LABELS["anthropic"]
        self.api_key = api_key or None
        self._custom_timeout = timeout is not None
        self.timeout = timeout if self._custom_timeout else get_request_timeout()

    def get_effective_timeout(self) -> float:
        return self.timeout if self._custom_timeout else get_request_timeout()

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request_json(self, path: str, payload: Optional[Dict[str, Any]] = None,
                      timeout: float = 5.0) -> Any:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            f"{self.host}{path}",
            data=data,
            headers=self._headers(),
            method="POST" if data is not None else "GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _http_failure(self, exc: urllib.error.HTTPError) -> RuntimeError:
        detail = _http_error_detail(exc)
        msg = f"HTTP {exc.code} {exc.reason} from {self.label} at {self.host}"
        if detail:
            msg += f" — server said: {detail}"
        return RuntimeError(msg)

    def list_models(self) -> List[str]:
        """Model ids advertised at GET /v1/models ([] when unreachable)."""
        try:
            data = self._request_json("/v1/models", timeout=5.0)
            return [str(m.get("id")) for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
        except Exception:
            return []

    def check_connection(self) -> Tuple[bool, str]:
        try:
            data = self._request_json("/v1/models", timeout=5.0)
            models = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
            return True, f"Connected to {self.label} at {self.host} ({len(models)} model(s) available)."
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                key_note = ("the key it was given was rejected" if self.api_key
                            else "no API key was given")
                return (
                    False,
                    f"{self.label} at {self.host} refused the request (HTTP {exc.code} Unauthorized): "
                    f"the server requires an API key and {key_note}. Provide one with $OGUN_LLM_API_KEY, "
                    "or save this endpoint as a custom provider with its key "
                    "(/provider add <name> <host> [model] key=<api-key> type=anthropic).",
                )
            detail = _http_error_detail(exc)
            return False, f"{self.label} at {self.host} returned HTTP {exc.code}." + (
                f" Server said: {detail}" if detail else "")
        except Exception as exc:
            return (
                False,
                f"Cannot connect to {self.label} at {self.host} ({exc}). "
                "Start that server first, or check the --host URL.",
            )

    def unload(self) -> Dict[str, Any]:
        return {
            "unloaded": False,
            "model": self.model,
            "detail": f"{self.label} manages its own model memory; unload the model in that app to free the GPU.",
        }

    # -- Chat --------------------------------------------------------------
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        system, conv = _to_anthropic_messages(messages)
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": conv,
            **_map_anthropic_options(llm_options(**(options or {}))),
        }
        if system:
            payload["system"] = system
        anthropic_tools = _to_anthropic_tools(tools)
        if anthropic_tools:
            payload["tools"] = anthropic_tools
        eff_timeout = self.get_effective_timeout()
        if stream or on_token is not None:
            return self._chat_stream(payload, eff_timeout, on_token)
        try:
            data = self._request_json("/v1/messages", payload, timeout=eff_timeout)
        except urllib.error.HTTPError as he:
            raise self._http_failure(he) from he
        except urllib.error.URLError as ue:
            if "timed out" in str(ue).lower():
                raise TimeoutError(str(ue))
            raise
        return self._parse_message(data)

    @staticmethod
    def _parse_message(data: Dict[str, Any]) -> Dict[str, Any]:
        texts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        for block in data.get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                texts.append(str(block.get("text") or ""))
            elif block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block.get("id") or "",
                    "type": "function",
                    "function": {"name": block.get("name") or "",
                                 "arguments": block.get("input") or {}},
                })
        return {"content": "".join(texts), "tool_calls": tool_calls}

    def _chat_stream(self, payload: Dict[str, Any], eff_timeout: float,
                     on_token: Optional[Callable[[str], None]]) -> Dict[str, Any]:
        payload = dict(payload, stream=True)
        req = urllib.request.Request(
            f"{self.host}/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        texts: List[str] = []
        blocks: Dict[int, Dict[str, Any]] = {}
        try:
            with urllib.request.urlopen(req, timeout=eff_timeout) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    try:
                        event = json.loads(line[5:].strip())
                    except Exception:
                        continue
                    etype = event.get("type")
                    idx = int(event.get("index", 0))
                    if etype == "content_block_start":
                        block = event.get("content_block") or {}
                        blocks[idx] = {"type": block.get("type"), "id": block.get("id") or "",
                                       "name": block.get("name") or "", "args": ""}
                    elif etype == "content_block_delta":
                        delta = event.get("delta") or {}
                        slot = blocks.setdefault(idx, {"type": "text", "id": "", "name": "", "args": ""})
                        if delta.get("type") == "text_delta":
                            piece = str(delta.get("text") or "")
                            if piece:
                                texts.append(piece)
                                if on_token is not None:
                                    on_token(piece)
                        elif delta.get("type") == "input_json_delta":
                            slot["args"] += str(delta.get("partial_json") or "")
                    elif etype == "error":
                        err = event.get("error") or {}
                        raise RuntimeError(f"Streaming error from {self.label}: "
                                           f"{err.get('message') or err.get('type') or 'unknown error'}")
        except urllib.error.HTTPError as he:
            raise self._http_failure(he) from he
        except urllib.error.URLError as ue:
            if "timed out" in str(ue).lower():
                raise TimeoutError(str(ue))
            raise
        tool_calls: List[Dict[str, Any]] = []
        for idx in sorted(blocks):
            slot = blocks[idx]
            if slot.get("type") != "tool_use":
                continue
            args: Any = slot["args"]
            if isinstance(args, str) and args:
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"raw": args}
            tool_calls.append({"id": slot["id"], "type": "function",
                               "function": {"name": slot["name"], "arguments": args or {}}})
        return {"content": "".join(texts), "tool_calls": tool_calls}


# ---------------------------------------------------------------------------
# llama.cpp server management
# ---------------------------------------------------------------------------
def find_llama_server_binary() -> Optional[str]:
    """Locate a llama-server executable: PATH first, then the usual build/install locations."""
    for name in ("llama-server", "llama.cpp-server"):
        found = shutil.which(name)
        if found:
            return found
    home = Path.home()
    candidates = [
        home / "llama.cpp" / "build" / "bin" / "llama-server",
        home / "llama.cpp" / "build" / "bin" / "Release" / "llama-server",
        Path("/usr/local/bin/llama-server"),
        Path("/usr/bin/llama-server"),
    ]
    for cand in candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return None


class LlamaCppServer:
    """A llama-server process Ogun launched itself for one GGUF file. Attach mode when a server
    is already answering at the host: start() is then a no-op and stop() leaves it running."""

    def __init__(self, host: str = PROVIDER_DEFAULT_HOSTS["llama.cpp"]) -> None:
        self.host = host.rstrip("/")
        self.process: Optional[subprocess.Popen] = None
        self.managed = False

    def is_up(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.host}/v1/models", method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def build_command(self, binary: str, model_path: Path, port: int,
                      gpu_layers: Optional[int], ctx_size: Optional[int]) -> List[str]:
        cmd = [binary, "-m", str(model_path), "--host", "127.0.0.1", "--port", str(port),
               "--jinja"]  # --jinja: use the model's embedded chat template (needed for tool calls)
        if gpu_layers is not None:
            cmd += ["-ngl", str(gpu_layers)]
        if ctx_size:
            cmd += ["-c", str(ctx_size)]
        return cmd

    def start(self, model_path: Path, gpu_layers: Optional[int] = None,
              ctx_size: Optional[int] = None, wait_seconds: float = 120.0) -> Tuple[bool, str]:
        """Launch llama-server for `model_path` (or attach to a running one). Never raises."""
        if self.is_up():
            return True, f"A llama-server is already running at {self.host}; using it."
        binary = find_llama_server_binary()
        if not binary:
            return False, ("Could not find the llama-server binary. Build llama.cpp "
                           "(https://github.com/ggml-org/llama.cpp) or put llama-server on your PATH, "
                           "or start the server yourself and use --provider openai --host <url>.")
        port = 8080
        try:
            from urllib.parse import urlparse
            port = urlparse(self.host).port or 8080
        except Exception:
            pass
        cmd = self.build_command(binary, Path(model_path), port,
                                 get_num_gpu() if gpu_layers is None else gpu_layers,
                                 get_num_ctx() if ctx_size is None else ctx_size)
        try:
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as exc:
            return False, f"Failed to launch llama-server ({exc})."
        self.managed = True
        deadline = time.time() + wait_seconds
        while time.time() < deadline:
            if self.process.poll() is not None:
                self.managed = False
                return False, f"llama-server exited immediately (code {self.process.returncode})."
            if self.is_up():
                return True, f"llama-server started for {Path(model_path).name} at {self.host}."
            time.sleep(0.5)
        return False, f"llama-server did not become ready within {wait_seconds:.0f}s."

    def stop(self) -> None:
        if self.managed and self.process is not None and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
        self.managed = False


# ---------------------------------------------------------------------------
# GGUF -> Ollama import
# ---------------------------------------------------------------------------
def gguf_model_name(gguf_path: Path) -> str:
    """A valid Ollama model name derived from a GGUF filename ('My Model-Q4_K_M.gguf' -> 'my-model-q4_k_m')."""
    stem = Path(gguf_path).stem.lower()
    name = "".join(ch if (ch.isalnum() or ch in "-_.") else "-" for ch in stem).strip("-")
    return name or "local-gguf"


def import_gguf_into_ollama(gguf_path: Path, name: Optional[str] = None,
                            runner: Optional[Callable[..., Any]] = None) -> Tuple[bool, str, str]:
    """Register a GGUF file with Ollama (`ollama create`) so the Ollama provider can serve it.

    Returns (ok, model_name, detail). Never raises; a missing ollama CLI is a clean failure.
    """
    run = runner or subprocess.run
    gguf_path = Path(gguf_path)
    model_name = name or gguf_model_name(gguf_path)
    if shutil.which("ollama") is None:
        return False, model_name, "The 'ollama' command is not on your PATH, so the GGUF cannot be imported."
    with tempfile.TemporaryDirectory(prefix="ogun-gguf-") as tmp:
        modelfile = Path(tmp) / "Modelfile"
        modelfile.write_text(f'FROM {gguf_path}\n', encoding="utf-8")
        try:
            res = run(["ollama", "create", model_name, "-f", str(modelfile)],
                      capture_output=True, text=True, timeout=1800)
        except Exception as exc:
            return False, model_name, f"ollama create failed to run ({exc})."
    if res.returncode == 0:
        return True, model_name, f"Imported {gguf_path.name} into Ollama as '{model_name}'."
    detail = (res.stderr or res.stdout or "").strip().splitlines()
    return False, model_name, f"ollama create failed: {detail[-1] if detail else 'unknown error'}"


# ---------------------------------------------------------------------------
# Local model discovery
# ---------------------------------------------------------------------------
def _dir_size_gb(path: Path) -> float:
    total = 0
    try:
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return total / 2**30


def model_search_dirs() -> List[Path]:
    """Directories scanned for loose GGUF files and Hugging Face-format model folders."""
    home = Path.home()
    dirs = [
        home / "models",
        home / ".cache" / "llama.cpp",
        home / ".lmstudio" / "models",
        home / "unsloth_models",
        OGUN_ROOT / "training" / "data" / "exports",   # Ogun's own Unsloth trainer exports
    ]
    extra = os.environ.get("OGUN_MODEL_DIRS", "")
    for part in extra.split(os.pathsep):
        if part.strip():
            dirs.append(Path(part.strip()).expanduser())
    seen: List[Path] = []
    for d in dirs:
        if d not in seen:
            seen.append(d)
    return seen


def hf_cache_dir() -> Optional[Path]:
    """The Hugging Face hub cache directory ($HF_HOME/hub or ~/.cache/huggingface/hub)."""
    hf_home = os.environ.get("HF_HOME")
    base = Path(hf_home).expanduser() if hf_home else Path.home() / ".cache" / "huggingface"
    hub = base / "hub"
    return hub if hub.is_dir() else None


def _scan_model_dir(root: Path, out: List[Dict[str, Any]], seen_paths: set) -> None:
    """Collect GGUF files and HF-format model folders under one search directory (depth-limited)."""
    if not root.is_dir():
        return
    for gguf in root.rglob("*.gguf"):
        try:
            key = str(gguf.resolve())
        except OSError:
            continue
        if key in seen_paths:
            continue
        seen_paths.add(key)
        try:
            size = gguf.stat().st_size / 2**30
        except OSError:
            size = 0.0
        out.append({"name": gguf.stem, "kind": "gguf", "detail": str(gguf), "size_gb": round(size, 2)})
    for config in root.rglob("config.json"):
        folder = config.parent
        try:
            key = str(folder.resolve())
        except OSError:
            continue
        if key in seen_paths:
            continue
        # An HF-format model folder: config.json plus safetensors weights
        if any(folder.glob("*.safetensors")) or (folder / "model.safetensors.index.json").exists():
            seen_paths.add(key)
            out.append({"name": folder.name, "kind": "huggingface", "detail": str(folder),
                        "size_gb": round(_dir_size_gb(folder), 2)})


def discover_local_models(ollama_host: str = OLLAMA_HOST, probe_servers: bool = True) -> List[Dict[str, Any]]:
    """Every locally usable model Ogun can find, grouped by how it would be served.

    Each entry: {"name", "kind", "detail", "size_gb"} where kind is one of
    "ollama" (detail = host), "server" (detail = base URL of a running OpenAI-compatible server),
    "gguf" (detail = file path), "huggingface" (detail = folder path). Never raises.
    """
    out: List[Dict[str, Any]] = []

    # 1. Ollama's library
    try:
        req = urllib.request.Request(f"{ollama_host.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for m in data.get("models", []):
            if isinstance(m, dict) and m.get("name"):
                out.append({"name": str(m["name"]), "kind": "ollama", "detail": ollama_host,
                            "size_gb": round(float(m.get("size") or 0) / 2**30, 2)})
    except Exception:
        pass

    # 2. Running OpenAI-compatible servers at the well-known local ports
    if probe_servers:
        for label, host in (("llama.cpp", PROVIDER_DEFAULT_HOSTS["llama.cpp"]),
                            ("LM Studio", PROVIDER_DEFAULT_HOSTS["lmstudio"]),
                            ("OpenAI-compatible", "http://127.0.0.1:8000")):
            if host == PROVIDER_DEFAULT_HOSTS["llama.cpp"] and any(e["kind"] == "server" and e["detail"] == host for e in out):
                continue
            client = OpenAICompatClient(host=host, label=label)
            for mid in client.list_models():
                out.append({"name": mid, "kind": "server", "detail": host, "size_gb": 0.0})
        # 2b. An Anthropic-shaped gateway (Free Claude Code) on its usual port: it answers
        # /v1/models like the others, but chat goes through the Anthropic Messages API.
        fcc = AnthropicMessagesClient(host="http://127.0.0.1:8082", label="Free Claude Code")
        for mid in fcc.list_models():
            out.append({"name": mid, "kind": "anthropic-server",
                        "detail": "http://127.0.0.1:8082", "size_gb": 0.0})

    # 3. GGUF files and HF folders in the search dirs + the HF cache
    seen_paths: set = set()
    for d in model_search_dirs():
        _scan_model_dir(d, out, seen_paths)
    hub = hf_cache_dir()
    if hub is not None:
        for entry in sorted(hub.iterdir()):
            if not entry.is_dir() or not entry.name.startswith("models--"):
                continue
            repo = entry.name[len("models--"):].replace("--", "/")
            snapshots = entry / "snapshots"
            has_gguf = snapshots.is_dir() and any(snapshots.rglob("*.gguf"))
            kind = "gguf" if has_gguf else "huggingface"
            out.append({"name": repo, "kind": kind, "detail": str(entry),
                        "size_gb": round(_dir_size_gb(entry), 2)})
            if has_gguf:
                _scan_model_dir(entry, out, seen_paths)  # also list each individual GGUF file

    out.sort(key=lambda e: (e["kind"], e["name"].lower()))
    return out


_KIND_HINTS = {
    "ollama": "run: --provider ollama --model {name}",
    "server": "run: --provider openai --host {detail} --model {name}",
    "anthropic-server": "run: --provider anthropic --host {detail} --model {name}",
    "gguf": "run: --provider llama.cpp --model {detail}  (or --provider ollama --model {detail} to import it)",
    "huggingface": "serve it (vLLM / Unsloth Studio / LM Studio) then --provider openai, or export a GGUF",
}


def format_model_list(entries: List[Dict[str, Any]]) -> str:
    """A readable table of discovered models with the command that runs each one."""
    if not entries:
        return ("No local models found. Checked: Ollama, running llama.cpp / LM Studio / vLLM servers, "
                f"the Hugging Face cache, and {', '.join(str(d) for d in model_search_dirs())}. "
                "Add folders with $OGUN_MODEL_DIRS (os.pathsep-separated).")
    lines = [f"Found {len(entries)} local model(s):"]
    for e in entries:
        size = f"{e['size_gb']:.1f} GB" if e.get("size_gb") else "?"
        hint = _KIND_HINTS.get(e["kind"], "").format(name=e["name"], detail=e["detail"])
        lines.append(f"  [{e['kind']:11s}] {e['name']}  ({size})  {e['detail']}")
        lines.append(f"                {hint}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------
def create_client(provider: str, model: str = MODEL_NAME, host: Optional[str] = None,
                  timeout: Optional[float] = None, api_key: Optional[str] = None,
                  label: Optional[str] = None):
    """Build the chat client for a canonical provider key (see normalize_provider).

    Returns an object with OllamaClient's interface. For 'ollama' this is engine.OllamaClient.
    """
    if provider == "ollama":
        from ogun_core.engine import OllamaClient
        return OllamaClient(host=host or OLLAMA_HOST, model=model, timeout=timeout)
    base = host or PROVIDER_DEFAULT_HOSTS.get(provider, PROVIDER_DEFAULT_HOSTS["openai"])
    if provider == "anthropic":
        return AnthropicMessagesClient(host=base, model=model, timeout=timeout,
                                       label=label or PROVIDER_LABELS["anthropic"],
                                       api_key=api_key)
    return OpenAICompatClient(host=base, model=model, timeout=timeout,
                              label=label or PROVIDER_LABELS.get(provider, "OpenAI-compatible server"),
                              api_key=api_key)


def resolve_provider_host(provider: str, host: Optional[str]) -> str:
    """The base URL a provider will use: explicit --host, then $OGUN_LLM_HOST, then the preset."""
    if host:
        return host.rstrip("/")
    env_host = os.environ.get("OGUN_LLM_HOST", "").strip()
    if env_host:
        return env_host.rstrip("/")
    if provider == "ollama":
        return OLLAMA_HOST
    return PROVIDER_DEFAULT_HOSTS.get(provider, PROVIDER_DEFAULT_HOSTS["openai"])


# ---------------------------------------------------------------------------
# Custom named providers (user-defined endpoints)
# ---------------------------------------------------------------------------
#: User-defined providers live here (same folder as the saved hardware profile; user-local, not in git).
CUSTOM_PROVIDERS_PATH = OGUN_ROOT / "config" / "providers.json"


def load_custom_providers() -> Dict[str, Dict[str, Any]]:
    """User-defined named providers, merged from config/providers.json and $OGUN_CUSTOM_PROVIDERS
    (a JSON object; the environment wins per name). Each definition:

        {"host": "http://192.168.1.50:8000",          # required
         "model": "qwen3-32b",                        # optional default model
         "api_key_env": "MY_SERVER_KEY",              # optional: env var holding a Bearer key
         "api_key": "sk-...",                         # optional: the Bearer key itself, stored
                                                      # in the file (local use; file is chmod 600
                                                      # whenever any key is stored)
         "type": "anthropic"}                         # optional wire API: "anthropic" for
                                                      # Anthropic-shaped gateways (Free Claude
                                                      # Code); default is OpenAI-compatible

    Keys are the provider names as typed (matched case-insensitively). Never raises: a malformed
    file or variable simply contributes nothing.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    try:
        if CUSTOM_PROVIDERS_PATH.is_file():
            data = json.loads(CUSTOM_PROVIDERS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                merged.update(data)
    except Exception:
        pass
    raw = os.environ.get("OGUN_CUSTOM_PROVIDERS", "").strip()
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                merged.update(data)
        except Exception:
            pass
    out: Dict[str, Dict[str, Any]] = {}
    for name, definition in merged.items():
        if isinstance(definition, dict) and definition.get("host"):
            out[str(name).strip().lower()] = definition
    return out


def get_custom_provider(name: Optional[str]) -> Optional[Dict[str, Any]]:
    """The custom provider definition for a name, or None."""
    if not name:
        return None
    return load_custom_providers().get(str(name).strip().lower())


def _read_custom_providers_file() -> Dict[str, Any]:
    """The raw contents of config/providers.json ({} when missing/malformed). File only, no env."""
    try:
        if CUSTOM_PROVIDERS_PATH.is_file():
            data = json.loads(CUSTOM_PROVIDERS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _write_custom_providers_file(data: Dict[str, Any]) -> None:
    CUSTOM_PROVIDERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CUSTOM_PROVIDERS_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    # A stored literal API key (api_key) is a local secret: tighten the file to owner-only.
    if any(isinstance(d, dict) and d.get("api_key") for d in data.values()):
        try:
            os.chmod(CUSTOM_PROVIDERS_PATH, 0o600)
        except OSError:
            pass


def _describe_definition(definition: Dict[str, Any]) -> str:
    """'host=..., model=...' for display, with any stored API key redacted."""
    parts = []
    for k, v in definition.items():
        parts.append(f"{k}=stored" if k == "api_key" else f"{k}={v}")
    return ", ".join(parts)


#: Words that are /provider subcommands; a custom provider with one of these names could never be
#: selected, so they are rejected at definition time.
_RESERVED_PROVIDER_NAMES = {"add", "remove", "rm", "delete", "list", "edit"}


def validate_custom_provider(name: str, host: str) -> Optional[str]:
    """An error string when a custom provider name/host is unusable, else None."""
    cleaned = str(name or "").strip()
    if not cleaned or any(ch.isspace() for ch in cleaned) or "/" in cleaned or "\\" in cleaned:
        return f"'{name}' is not a usable provider name (one word, no slashes)."
    if normalize_provider(cleaned) is not None:
        return f"'{cleaned}' is a built-in provider name; pick a different name."
    if cleaned.lower() in _RESERVED_PROVIDER_NAMES:
        return f"'{cleaned}' is a /provider subcommand word; pick a different name."
    if not str(host or "").strip().lower().startswith(("http://", "https://")):
        return f"'{host}' is not a URL (it must start with http:// or https://)."
    return None


def save_custom_provider(name: str, host: str, model: Optional[str] = None,
                         api_key_env: Optional[str] = None,
                         api_key: Optional[str] = None,
                         api_type: Optional[str] = None) -> Tuple[bool, str]:
    """Define (or overwrite) a custom provider in config/providers.json. Never raises."""
    error = validate_custom_provider(name, host)
    if error:
        return False, error
    definition: Dict[str, Any] = {"host": str(host).strip().rstrip("/")}
    if model:
        definition["model"] = str(model).strip()
    if api_key_env:
        definition["api_key_env"] = str(api_key_env).strip()
    if api_key:
        definition["api_key"] = str(api_key).strip()
    if api_type:
        kind = str(api_type).strip().lower()
        if kind in ("anthropic", "claude"):
            definition["type"] = "anthropic"
        elif kind != "openai":
            return False, f"Unknown provider type '{api_type}' (use 'openai' or 'anthropic')."
    try:
        data = _read_custom_providers_file()
        data[str(name).strip()] = definition
        _write_custom_providers_file(data)
    except Exception as exc:
        return False, f"Could not save the provider to {CUSTOM_PROVIDERS_PATH} ({exc})."
    return True, f"Saved custom provider '{str(name).strip()}' ({definition['host']}) to {CUSTOM_PROVIDERS_PATH}."


def remove_custom_provider(name: str) -> Tuple[bool, str]:
    """Remove a custom provider from config/providers.json. Env-defined ones cannot be removed
    here (they live in $OGUN_CUSTOM_PROVIDERS). Never raises."""
    key = str(name or "").strip()
    data = _read_custom_providers_file()
    file_key = next((k for k in data if k.strip().lower() == key.lower()), None)
    if file_key is None:
        if get_custom_provider(key) is not None:
            return False, (f"'{key}' is defined by $OGUN_CUSTOM_PROVIDERS, not in {CUSTOM_PROVIDERS_PATH}; "
                           "change the environment variable to remove it.")
        return False, f"No custom provider named '{key}'."
    try:
        del data[file_key]
        _write_custom_providers_file(data)
    except Exception as exc:
        return False, f"Could not update {CUSTOM_PROVIDERS_PATH} ({exc})."
    return True, f"Removed custom provider '{file_key}'."


def _custom_provider_sources() -> Dict[str, str]:
    """For each custom provider name (lowercased): where it is defined, 'file' or 'env'
    (env wins, matching load_custom_providers' merge order)."""
    sources = {str(k).strip().lower(): "file" for k in _read_custom_providers_file()}
    raw = os.environ.get("OGUN_CUSTOM_PROVIDERS", "").strip()
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                for k in data:
                    sources[str(k).strip().lower()] = "env"
        except Exception:
            pass
    return sources


def provider_listing(active_kind: Optional[str] = None, active_display: Optional[str] = None) -> str:
    """A readable list of every provider: built-ins with their effective hosts, customs with
    host/model/key reference and where each is defined. The active one is marked with '*'."""
    def mark(is_active: bool) -> str:
        return "*" if is_active else " "

    lines = ["Providers (* = active):"]
    for kind in ("ollama", "openai", "llama.cpp", "lmstudio", "anthropic"):
        host = resolve_provider_host(kind, None)
        is_active = kind == active_kind and (active_display in (None, PROVIDER_LABELS.get(kind), kind))
        lines.append(f" {mark(is_active)} {kind:12s} {host}  (built-in)")
    sources = _custom_provider_sources()
    customs = load_custom_providers()
    for name in sorted(customs):
        definition = customs[name]
        extras = []
        if definition.get("model"):
            extras.append(f"model={definition['model']}")
        if definition.get("api_key_env"):
            env_name = str(definition["api_key_env"])
            extras.append(f"keyenv={env_name}" + ("" if os.environ.get(env_name) else " (NOT SET)"))
        if definition.get("api_key"):
            extras.append("key=stored")
        if str(definition.get("type") or "").strip().lower() in ("anthropic", "claude"):
            extras.append("type=anthropic")
        source = "config/providers.json" if sources.get(name) == "file" else "$OGUN_CUSTOM_PROVIDERS"
        is_active = active_display is not None and name == active_display.strip().lower()
        lines.append(f" {mark(is_active)} {name:12s} {definition['host']}"
                     f"{'  ' + '  '.join(extras) if extras else ''}  (custom, {source})")
    if not customs:
        lines.append("   (no custom providers defined; add one with /provider add <name> <host> [model])")
    return "\n".join(lines)


def provider_choices(active_kind: Optional[str] = None,
                       active_display: Optional[str] = None) -> List[Dict[str, Any]]:
    """Structured provider list for interactive menus (the /model menu's provider page):
    built-ins first, then custom providers. Each entry is
    {"name": <what switch_provider expects>, "display": <label with host>, "active": bool}."""
    choices: List[Dict[str, Any]] = []
    for kind in ("ollama", "openai", "llama.cpp", "lmstudio", "anthropic"):
        choices.append({
            "name": kind,
            "display": f"{kind} — {resolve_provider_host(kind, None)}",
            "active": kind == active_kind
                      and (active_display in (None, PROVIDER_LABELS.get(kind), kind)),
        })
    customs = load_custom_providers()
    for name in sorted(customs):
        choices.append({
            "name": name,
            "display": f"{name} — {customs[name]['host']}",
            "active": active_display is not None and name == active_display.strip().lower(),
        })
    return choices


def edit_custom_provider(name: str, changes: Dict[str, Optional[str]]) -> Tuple[bool, str]:
    """Edit fields of a file-defined custom provider. `changes` maps host/model/keyenv/key to a
    new value; an empty string clears model/keyenv/key (host cannot be cleared). With no changes,
    just reports the current definition (API key redacted). Env-defined providers cannot be
    edited here. Never raises."""
    key = str(name or "").strip()
    data = _read_custom_providers_file()
    file_key = next((k for k in data if k.strip().lower() == key.lower()), None)
    if file_key is None:
        if get_custom_provider(key) is not None:
            return False, (f"'{key}' is defined by $OGUN_CUSTOM_PROVIDERS, not in {CUSTOM_PROVIDERS_PATH}; "
                           "change the environment variable to edit it.")
        return False, f"No custom provider named '{key}'."

    definition = dict(data[file_key])
    if not changes:
        return True, (f"'{file_key}': {_describe_definition(definition)}. Edit with /provider edit {file_key} "
                      "host=<url> | model=<id> | keyenv=<VAR> | key=<api-key> | type=<openai|anthropic> (empty value clears model/keyenv/key).")

    if "host" in changes:
        new_host = (changes["host"] or "").strip()
        error = validate_custom_provider(file_key, new_host)
        if error:
            return False, error
        definition["host"] = new_host.rstrip("/")
    if "model" in changes:
        value = (changes["model"] or "").strip()
        if value:
            definition["model"] = value
        else:
            definition.pop("model", None)
    if "keyenv" in changes:
        value = (changes["keyenv"] or "").strip()
        if value:
            definition["api_key_env"] = value
        else:
            definition.pop("api_key_env", None)
    if "key" in changes:
        value = (changes["key"] or "").strip()
        if value:
            definition["api_key"] = value
        else:
            definition.pop("api_key", None)
    if "type" in changes:
        value = (changes["type"] or "").strip().lower()
        if value in ("anthropic", "claude"):
            definition["type"] = "anthropic"
        elif value in ("openai", ""):
            definition.pop("type", None)
        else:
            return False, f"Unknown provider type '{changes['type']}' (use 'openai' or 'anthropic')."
    try:
        data[file_key] = definition
        _write_custom_providers_file(data)
    except Exception as exc:
        return False, f"Could not update {CUSTOM_PROVIDERS_PATH} ({exc})."
    return True, f"Updated '{file_key}': {_describe_definition(definition)}."


def resolve_provider_target(name: Optional[str], new_host: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Resolve a provider name (built-in OR custom) to everything needed to build its client:

    {"kind": canonical built-in kind, "host": base URL, "model": default model or None,
     "api_key": Bearer key or None, "key_env": env var the key was to come from (custom
     providers only; lets callers warn when it is referenced but unset), "display": name shown}

    Returns None for an unknown name. An explicit new_host always wins; a custom provider's
    host comes from its definition; $OGUN_LLM_HOST and presets fill the rest.
    """
    if not name:
        return None
    kind = normalize_provider(name)
    if kind is not None:
        api_key = os.environ.get("OGUN_LLM_API_KEY") or None
        return {"kind": kind, "host": resolve_provider_host(kind, new_host),
                "model": None, "api_key": api_key if kind != "ollama" else None, "key_env": None,
                "display": PROVIDER_LABELS.get(kind, kind)}
    definition = get_custom_provider(name)
    if definition is None:
        return None
    api_key = None
    key_env = definition.get("api_key_env")
    if key_env:
        api_key = os.environ.get(str(key_env)) or None
    if not api_key and definition.get("api_key"):
        api_key = str(definition["api_key"])
    # A definition's "type" picks the wire API: "anthropic" for Anthropic-shaped gateways
    # (Free Claude Code et al.); anything else is an OpenAI-compatible endpoint.
    kind = "anthropic" if str(definition.get("type") or "").strip().lower() in ("anthropic", "claude") else "openai"
    return {"kind": kind, "host": (new_host or str(definition["host"])).rstrip("/"),
            "model": definition.get("model"), "api_key": api_key,
            "key_env": str(key_env) if key_env else None,
            "display": str(name).strip()}


def _ollama_model_names(host: str) -> List[str]:
    """Model tags in Ollama's local library ([] when Ollama is unreachable)."""
    try:
        req = urllib.request.Request(f"{host.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [str(m["name"]) for m in data.get("models", []) if isinstance(m, dict) and m.get("name")]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Live session: current provider + model, switchable mid-chat (/model, /provider)
# ---------------------------------------------------------------------------
class ProviderSession:
    """The live inference target: one provider, one host, one model, one client.

    Owns the managed llama-server (when Ogun launched one) and performs mid-chat switches.
    A failed switch changes nothing: the previous client keeps serving. Successful switches
    unload the old Ollama model (freeing its VRAM) and stop a managed llama-server that is
    being left behind.
    """

    def __init__(self, provider: str, host: str, model: str, timeout: Optional[float] = None,
                 name: Optional[str] = None, api_key: Optional[str] = None,
                 model_explicit: bool = False) -> None:
        self.provider = provider
        self.host = host.rstrip("/")
        self.model = model
        self.name = name            # display name (a custom provider's name, or a built-in's label)
        self.api_key = api_key
        # True when the model was deliberately chosen (provider definition, --model, /model):
        # an external server's advertised list must never override it — gateways and routers
        # routinely serve model ids they do not list verbatim.
        self.model_explicit = model_explicit
        # Every Ollama model this session has run, so exit can release them all —
        # including one that was switched away from mid-session.
        self._ollama_models_used = {model} if provider == "ollama" else set()
        self.llama_server: Optional[LlamaCppServer] = None
        self.client = create_client(provider, model=model, host=self.host, timeout=timeout,
                                    api_key=api_key, label=name)

    # -- helpers -----------------------------------------------------------
    @property
    def label(self) -> str:
        return self.name or PROVIDER_LABELS.get(self.provider, self.provider)

    def status(self) -> str:
        lines = [f"Provider: {self.label} at {self.host}", f"Model: {self.model}"]
        if self.provider == "llama.cpp":
            lines.append("llama-server: " + ("managed by Ogun" if self.llama_server and self.llama_server.managed
                                              else "external (started outside Ogun)"))
        return "\n".join(lines)

    def stop_managed_server(self) -> None:
        if self.llama_server is not None:
            self.llama_server.stop()
            self.llama_server = None

    def release_models(self) -> None:
        """Free everything this session loaded: stop the managed llama-server and
        unload every Ollama model the session ran — whichever provider happens to
        be active now. Wired to exit (atexit plus termination signals) in main.py.
        Best effort and idempotent: never raises."""
        self.stop_managed_server()
        for model in sorted(self._ollama_models_used):
            try:
                from ogun_core.engine import OllamaClient
                OllamaClient(host=OLLAMA_HOST, model=model).unload()
            except Exception:
                pass

    def _unload_current_ollama_model(self) -> None:
        """Best effort: free the VRAM of the Ollama model we are switching away from."""
        if self.provider != "ollama":
            return
        try:
            from ogun_core.engine import OllamaClient
            OllamaClient(host=OLLAMA_HOST, model=self.model).unload()
        except Exception:
            pass

    def adopt_served_model(self, force: bool = False) -> str:
        """If an external server does not serve the current model name, adopt its first model.
        Returns a note when the model changed (or when an explicit model is kept despite not
        being listed), else ''. An explicitly chosen model (definition, --model, /model) is
        NEVER overridden: gateways/routers accept ids they don't advertise. force=True is for
        the managed llama-server case, where the served id is by definition the loaded file."""
        if self.provider == "ollama" or not hasattr(self.client, "list_models"):
            return ""
        served = self.client.list_models()
        if not served or self.model in served:
            return ""
        if self.model_explicit and not force:
            return (f" Note: {self.label} doesn't list '{self.model}' among its served models; "
                    "using it as configured (gateways often route ids they don't list).")
        self.model = served[0]
        self.client.model = served[0]
        return f" Using the server's model '{served[0]}'."

    def available_models(self) -> List[str]:
        """Model ids the current provider's server advertises ([] when it cannot be asked:
        server down, or a client with no listing call). Powers the /model menu."""
        lister = getattr(self.client, "list_models", None)
        if lister is None:
            return []
        try:
            return [str(m) for m in lister() if m]
        except Exception:
            return []

    # -- GGUF activation (startup --model file.gguf and /model file.gguf) --
    def activate_gguf(self, gguf_path: Path) -> Tuple[bool, str]:
        gguf_path = Path(gguf_path).expanduser()
        if self.provider == "ollama":
            name = gguf_model_name(gguf_path)
            if name in _ollama_model_names(self.host):
                self._set_model(name)
                return True, f"'{name}' is already in Ollama's library; using it."
            ok, name, detail = import_gguf_into_ollama(gguf_path)
            if not ok:
                return False, detail
            self._set_model(name)
            return True, detail
        if self.provider == "llama.cpp":
            self.stop_managed_server()
            server = LlamaCppServer(host=self.host)
            ok, detail = server.start(gguf_path)
            if not ok:
                return False, detail
            self.llama_server = server
            self.adopt_served_model(force=True)
            return True, detail
        return False, (f"The '{self.provider}' provider talks to a running server, which loads its own models. "
                       "Use /provider llama.cpp (Ogun launches llama-server for the file) or "
                       "/provider ollama (imports it) first.")

    def _set_model(self, name: str) -> None:
        if name != self.model and self.provider == "ollama":
            self._unload_current_ollama_model()
        self.model = name
        self.client.model = name
        self.model_explicit = True
        if self.provider == "ollama":
            self._ollama_models_used.add(name)

    # -- Switches ------------------------------------------------------------
    def switch_provider(self, name: str, new_host: Optional[str] = None, force: bool = False) -> Tuple[bool, str]:
        """Switch provider (built-in or custom name, optionally a new host) without restarting.
        force=True rebuilds the client even when the target looks identical (used after an edit
        changed the stored definition). (ok, message); never raises."""
        target = resolve_provider_target(name, new_host)
        if target is None:
            customs = ", ".join(sorted(load_custom_providers())) or "none defined"
            return False, (f"Unknown provider '{name}'. Built-ins: ollama, openai, llama.cpp, lmstudio, anthropic. "
                           f"Custom providers: {customs}.")
        kind = target["kind"]
        if kind == self.provider and target["display"] == self.label and not new_host and not force:
            return True, f"Already using {self.label} at {self.host} with model '{self.model}'."
        host = target["host"]
        model = target["model"] or self.model
        key_note = ""
        if target.get("key_env") and not target.get("api_key"):
            key_note = (f" Note: this provider's keyenv names ${target['key_env']}, which is not set in "
                        "Ogun's environment, so no API key was sent. (keyenv takes the NAME of an "
                        "environment variable, not the key itself.)")

        server: Optional[LlamaCppServer] = None
        gguf = Path(self.model).expanduser() if str(self.model).lower().endswith(".gguf") else None
        if kind == "llama.cpp" and gguf is not None and gguf.is_file():
            server = LlamaCppServer(host=host)
            ok, detail = server.start(gguf)
            if not ok:
                return False, f"{detail} (Still using {self.label}.)"

        candidate = create_client(kind, model=model, host=host,
                                  api_key=target["api_key"], label=target["display"])
        ok_conn, msg = candidate.check_connection()
        if not ok_conn:
            if server is not None:
                server.stop()
            return False, f"{msg} (Still using {self.label}.){key_note}"

        # Commit: leave the old target cleanly, then swap.
        if self.provider == "ollama" and kind != "ollama":
            self._unload_current_ollama_model()
        self.stop_managed_server()
        self.provider, self.host, self.client = kind, host, candidate
        self.name, self.api_key = target["display"], target["api_key"]
        if target["model"]:
            self.model = str(target["model"])
        if kind == "ollama":
            self._ollama_models_used.add(self.model)
        # A model from the provider's definition is an explicit choice; a carried-over model
        # from the previous provider is not, and may still be adopted from the server's list.
        self.model_explicit = bool(target["model"])
        self.llama_server = server
        note = self.adopt_served_model()
        result = f"Switched to {self.label} at {host}, model '{self.model}'.{note}{key_note}"
        if kind == "ollama" and gguf is not None and gguf.is_file():
            ok_g, detail = self.activate_gguf(gguf)
            result += f" {detail}"
            if not ok_g:
                result += " (The GGUF could not be imported; pick another /model.)"
        return True, result

    def add_provider(self, name: str, host: str, model: Optional[str] = None,
                     api_key_env: Optional[str] = None,
                     api_key: Optional[str] = None,
                     api_type: Optional[str] = None) -> Tuple[bool, str]:
        """Define a custom provider from inside chat (/provider add): save it to
        config/providers.json, then switch to it. An unreachable endpoint is still saved;
        the session just stays on the current provider. (ok, message); never raises."""
        ok, msg = save_custom_provider(name, host, model=model, api_key_env=api_key_env,
                                       api_key=api_key, api_type=api_type)
        if not ok:
            return False, msg
        switched, switch_msg = self.switch_provider(str(name).strip(), force=True)
        if switched:
            return True, f"{msg} {switch_msg}"
        return True, f"{msg} It is saved for next time, but could not be used right now: {switch_msg}"

    def remove_provider(self, name: str) -> Tuple[bool, str]:
        """/provider remove: delete a saved custom provider. The active session is untouched
        even when the removed provider is the one in use (it just stops being defined)."""
        return remove_custom_provider(name)

    def edit_provider(self, name: str, changes: Dict[str, Optional[str]]) -> Tuple[bool, str]:
        """/provider edit: change a saved custom provider's host/model/keyenv. When the edited
        provider is the one in use, the new settings are applied live by reconnecting; if the
        new endpoint is unreachable the edit still stands and the old client keeps serving."""
        ok, msg = edit_custom_provider(name, changes)
        if not ok or not changes:
            return ok, msg
        if self.label.strip().lower() != str(name).strip().lower():
            return True, msg
        applied, switch_msg = self.switch_provider(str(name).strip(), force=True)
        if applied:
            return True, f"{msg} Applied live: now using {self.label} at {self.host}, model '{self.model}'."
        return True, f"{msg} Saved, but reconnecting failed, so the previous settings are still in use: {switch_msg}"

    def switch_model(self, value: str) -> Tuple[bool, str]:
        """Switch model within the current provider. Accepts a model tag/id or a .gguf path.
        (ok, message); never raises."""
        value = (value or "").strip()
        if not value:
            return False, "Usage: /model <model tag, server model id, or path to a .gguf file>"
        if value.lower().endswith(".gguf"):
            path = Path(value).expanduser()
            if not path.is_file():
                return False, f"No such GGUF file: {value}"
            return self.activate_gguf(path)

        if self.provider == "ollama":
            names = _ollama_model_names(self.host)
            warn = ""
            if names and value not in names:
                warn = (f" Note: '{value}' is not in Ollama's local library yet, so the next reply "
                        f"will fail until you pull it ('ollama pull {value}').")
            self._set_model(value)
            return True, f"Model set to '{value}' (via Ollama).{warn}"

        served = self.client.list_models() if hasattr(self.client, "list_models") else []
        warn = ""
        if served and value not in served:
            warn = (f" Note: '{value}' is not in {self.label}'s advertised list "
                    f"({', '.join(served)}); setting it anyway — gateways often route ids they "
                    "don't list. If it is truly unavailable, the next inference will say so.")
        self._set_model(value)
        return True, f"Model set to '{value}' (via {self.label}).{warn}"
