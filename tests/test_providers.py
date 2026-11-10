"""Offline unit tests for Ogun's non-Ollama model providers (ogun_core/providers.py).

Covers: provider name/host resolution, OpenAI-compatible chat (plain + SSE streaming + tool
calls), option mapping, GGUF -> Ollama import, llama-server command construction, local model
discovery (GGUF files, HF cache, HF folders, training exports), the client factory, and a
CoreEngine run_turn driven entirely by the OpenAI-compatible client.

Runs completely offline: HTTP is faked at the urllib boundary, discovery uses temp dirs.
Standalone runnable: python tests/test_providers.py
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

OGUN_ROOT = Path(__file__).resolve().parents[1]
if str(OGUN_ROOT) not in sys.path:
    sys.path.insert(0, str(OGUN_ROOT))

from ogun_core import providers
from ogun_core.providers import (
    LlamaCppServer,
    OpenAICompatClient,
    ProviderSession,
    create_client,
    discover_local_models,
    format_model_list,
    gguf_model_name,
    import_gguf_into_ollama,
    normalize_provider,
    resolve_provider_host,
)
from ogun_core.engine import CoreEngine, OllamaClient


class FakeResp:
    """Minimal urlopen stand-in: context manager with read() and line iteration."""

    def __init__(self, body: bytes = b"", lines=None, status: int = 200):
        self._body = body
        self._lines = lines or []
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body

    def __iter__(self):
        return iter(self._lines)


def _json_resp(obj) -> FakeResp:
    return FakeResp(body=json.dumps(obj).encode("utf-8"))


# ---------------------------------------------------------------------------
# Names and hosts
# ---------------------------------------------------------------------------
def test_normalize_provider() -> None:
    assert normalize_provider("ollama") == "ollama"
    assert normalize_provider("OpenAI") == "openai"
    assert normalize_provider("llamacpp") == "llama.cpp"
    assert normalize_provider("llama.cpp") == "llama.cpp"
    assert normalize_provider("LM-Studio") == "lmstudio"
    assert normalize_provider("bogus") is None
    assert normalize_provider(None) is None
    assert normalize_provider("") is None


def test_resolve_provider_host() -> None:
    old = os.environ.pop("OGUN_LLM_HOST", None)
    try:
        assert resolve_provider_host("lmstudio", None) == "http://127.0.0.1:1234"
        assert resolve_provider_host("llama.cpp", None) == "http://127.0.0.1:8080"
        os.environ["OGUN_LLM_HOST"] = "http://127.0.0.1:9999/"
        assert resolve_provider_host("lmstudio", None) == "http://127.0.0.1:9999"
        assert resolve_provider_host("lmstudio", "http://host:1/") == "http://host:1"  # explicit wins
    finally:
        if old is None:
            os.environ.pop("OGUN_LLM_HOST", None)
        else:
            os.environ["OGUN_LLM_HOST"] = old


# ---------------------------------------------------------------------------
# OpenAI-compatible client
# ---------------------------------------------------------------------------
def test_option_mapping() -> None:
    mapped = providers._map_openai_options(
        {"temperature": 0.2, "top_p": 0.9, "num_predict": 128, "num_ctx": 8192,
         "num_gpu": 99, "num_thread": 8, "stop": ["<|im_end|>"]})
    assert mapped == {"temperature": 0.2, "top_p": 0.9, "stop": ["<|im_end|>"], "max_tokens": 128}, mapped


def test_openai_chat_plain_and_payload() -> None:
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _json_resp({"choices": [{"message": {
            "content": "hello back",
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "read_file", "arguments": "{\"filepath\": \"a.py\"}"}}],
        }}]})

    client = OpenAICompatClient(host="http://127.0.0.1:8080", model="my-model")
    with patch("ogun_core.providers.urllib.request.urlopen", fake_urlopen):
        res = client.chat([{"role": "user", "content": "hi"}],
                          tools=[{"type": "function", "function": {"name": "read_file"}}],
                          options={"num_predict": 64})
    assert res["content"] == "hello back", res
    assert res["tool_calls"][0]["function"]["name"] == "read_file"
    assert captured["url"].endswith("/v1/chat/completions")
    payload = captured["payload"]
    assert payload["model"] == "my-model" and payload["max_tokens"] == 64
    assert "num_gpu" not in payload and "num_ctx" not in payload
    assert payload["tools"], "tools must be forwarded"


def test_openai_chat_streaming_sse() -> None:
    lines = [
        b'data: {"choices": [{"delta": {"content": "Hel"}}]}\n',
        b'data: {"choices": [{"delta": {"content": "lo"}}]}\n',
        b'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "read_file", "arguments": "{\\"file"}}]}}]}\n',
        b'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "path\\": \\"a.py\\"}"}}]}}]}\n',
        b"data: [DONE]\n",
    ]

    def fake_urlopen(req, timeout=None):
        assert json.loads(req.data.decode("utf-8"))["stream"] is True
        return FakeResp(lines=lines)

    tokens: list = []
    client = OpenAICompatClient(host="http://127.0.0.1:8080", model="m")
    with patch("ogun_core.providers.urllib.request.urlopen", fake_urlopen):
        res = client.chat([{"role": "user", "content": "hi"}], on_token=tokens.append)
    assert res["content"] == "Hello", res
    assert tokens == ["Hel", "lo"], tokens
    tc = res["tool_calls"][0]["function"]
    assert tc["name"] == "read_file" and tc["arguments"] == '{"filepath": "a.py"}', tc


def test_openai_connection_and_models() -> None:
    def fake_urlopen(req, timeout=None):
        assert req.full_url.endswith("/v1/models")
        return _json_resp({"data": [{"id": "model-a"}, {"id": "model-b"}]})

    client = OpenAICompatClient(host="http://127.0.0.1:1234", model="model-a", label="LM Studio")
    with patch("ogun_core.providers.urllib.request.urlopen", fake_urlopen):
        ok, msg = client.check_connection()
        assert ok and "LM Studio" in msg, msg
        assert client.list_models() == ["model-a", "model-b"]

    import urllib.error

    def dead_urlopen(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    with patch("ogun_core.providers.urllib.request.urlopen", dead_urlopen):
        ok, msg = client.check_connection()
        assert not ok and "Cannot connect" in msg
        assert client.list_models() == []
    unloaded = client.unload()
    assert unloaded["unloaded"] is False and "manages its own" in unloaded["detail"]


# ---------------------------------------------------------------------------
# GGUF -> Ollama import
# ---------------------------------------------------------------------------
def test_gguf_model_name() -> None:
    assert gguf_model_name(Path("Ogun-Coder-Q4_K_M.gguf")) == "ogun-coder-q4_k_m"
    assert gguf_model_name(Path("my model (v2).gguf")) == "my-model--v2"
    assert gguf_model_name(Path("!!!.gguf")) == "local-gguf"


def test_import_gguf_into_ollama() -> None:
    seen = {}

    class Res:
        returncode = 0
        stdout = "success"
        stderr = ""

    def fake_runner(cmd, **kwargs):
        seen["cmd"] = cmd
        modelfile = Path(cmd[cmd.index("-f") + 1])
        seen["modelfile"] = modelfile.read_text(encoding="utf-8")
        return Res()

    with tempfile.TemporaryDirectory() as tmp:
        gguf = Path(tmp) / "Ogun-Coder-Q4_K_M.gguf"
        gguf.write_bytes(b"GGUF")
        with patch("ogun_core.providers.shutil.which", return_value="/usr/bin/ollama"):
            ok, name, detail = import_gguf_into_ollama(gguf, runner=fake_runner)
    assert ok and name == "ogun-coder-q4_k_m", (ok, name, detail)
    assert seen["cmd"][:3] == ["ollama", "create", "ogun-coder-q4_k_m"]
    assert str(gguf) in seen["modelfile"]

    class FailRes:
        returncode = 1
        stdout = ""
        stderr = "unknown manifest"

    with patch("ogun_core.providers.shutil.which", return_value="/usr/bin/ollama"):
        ok, _name, detail = import_gguf_into_ollama(Path("x.gguf"), runner=lambda cmd, **kw: FailRes())
    assert not ok and "unknown manifest" in detail

    with patch("ogun_core.providers.shutil.which", return_value=None):
        ok, _name, detail = import_gguf_into_ollama(Path("x.gguf"), runner=fake_runner)
    assert not ok and "not on your PATH" in detail


# ---------------------------------------------------------------------------
# llama.cpp server management
# ---------------------------------------------------------------------------
def test_llama_server_command_and_binary() -> None:
    server = LlamaCppServer()
    cmd = server.build_command("/bin/llama-server", Path("/m/model.gguf"), 8080, 99, 8192)
    assert cmd[:2] == ["/bin/llama-server", "-m"] and "/m/model.gguf" in cmd
    assert cmd[cmd.index("-ngl") + 1] == "99" and cmd[cmd.index("-c") + 1] == "8192"
    assert "--jinja" in cmd
    cmd_cpu = server.build_command("/bin/llama-server", Path("/m/model.gguf"), 8080, None, None)
    assert "-ngl" not in cmd_cpu and "-c" not in cmd_cpu

    with tempfile.TemporaryDirectory() as tmp:
        binary = Path(tmp) / "llama-server"
        binary.write_text("#!/bin/sh\n", encoding="utf-8")
        binary.chmod(0o755)
        old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = tmp + os.pathsep + old_path
        try:
            assert providers.find_llama_server_binary() == str(binary)
        finally:
            os.environ["PATH"] = old_path


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def test_discover_local_models() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp) / "home"
        (home / "models" / "hfmodel").mkdir(parents=True)
        (home / "models" / "Loose Model-Q4_K_M.gguf").write_bytes(b"GGUF" + b"x" * 1024)
        (home / "models" / "hfmodel" / "config.json").write_text("{}", encoding="utf-8")
        (home / "models" / "hfmodel" / "model.safetensors").write_bytes(b"w" * 2048)

        hf_home = Path(tmp) / "hf"
        snap = hf_home / "hub" / "models--unsloth--demo" / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        (snap / "demo-Q4_K_M.gguf").write_bytes(b"GGUF" + b"y" * 1024)

        exports = Path(tmp) / "exports" / "run1"
        exports.mkdir(parents=True)
        (exports / "ogun-coder-Q4_K_M.gguf").write_bytes(b"GGUF" + b"z" * 1024)

        env = {"HOME": str(home), "HF_HOME": str(hf_home), "OGUN_MODEL_DIRS": str(exports)}
        with patch.dict(os.environ, env):
            entries = discover_local_models(ollama_host="http://127.0.0.1:9", probe_servers=False)

    kinds = {(e["kind"], e["name"]) for e in entries}
    assert ("gguf", "Loose Model-Q4_K_M") in kinds, kinds
    assert ("huggingface", "hfmodel") in kinds, kinds
    assert ("gguf", "unsloth/demo") in kinds, kinds           # HF cache repo, GGUF inside
    assert ("gguf", "demo-Q4_K_M") in kinds, kinds             # the GGUF file itself
    assert ("gguf", "ogun-coder-Q4_K_M") in kinds, kinds       # training export via $OGUN_MODEL_DIRS
    text = format_model_list(entries)
    assert "--provider llama.cpp" in text and "--provider openai" in text
    assert "No local models found" in format_model_list([])


# ---------------------------------------------------------------------------
# Factory + engine drop-in compatibility
# ---------------------------------------------------------------------------
def test_create_client_factory() -> None:
    ollama_client = create_client("ollama", model="m")
    assert isinstance(ollama_client, OllamaClient)
    lm = create_client("lmstudio", model="m")
    assert isinstance(lm, OpenAICompatClient) and lm.host == "http://127.0.0.1:1234"
    generic = create_client("openai", model="m", host="http://127.0.0.1:8000")
    assert isinstance(generic, OpenAICompatClient) and generic.host == "http://127.0.0.1:8000"


def test_engine_runs_on_openai_client() -> None:
    def fake_urlopen(req, timeout=None):
        return _json_resp({"choices": [{"message": {"content": "engine reply", "tool_calls": []}}]})

    client = OpenAICompatClient(host="http://127.0.0.1:8080", model="m")
    with tempfile.TemporaryDirectory() as tmp:
        engine = CoreEngine(client=client, base_dir=Path(tmp))
        with patch("ogun_core.providers.urllib.request.urlopen", fake_urlopen):
            reply = engine.run_turn("Say hello")
    assert reply == "engine reply", reply
    assert engine.last_turn_status["status"] == "success"


# ---------------------------------------------------------------------------
# ProviderSession: mid-chat /provider and /model switching
# ---------------------------------------------------------------------------
def _router_urlopen(req, timeout=None):
    """One fake transport for every endpoint the session touches."""
    import urllib.error
    url = req.full_url
    if "127.0.0.1:1/" in url:
        raise urllib.error.URLError("connection refused")
    if url.endswith("/v1/models"):
        return _json_resp({"data": [{"id": "srv-1"}, {"id": "srv-2"}]})
    if url.endswith("/api/tags"):
        return _json_resp({"models": [{"name": "qwen2.5-coder:14b"}, {"name": "llama3.1:8b"}]})
    return _json_resp({})


def test_session_switch_provider_and_model() -> None:
    session = ProviderSession("ollama", "http://127.0.0.1:11434", "qwen2.5-coder:14b")
    with patch("ogun_core.providers.urllib.request.urlopen", _router_urlopen):
        ok, msg = session.switch_provider("openai", "http://127.0.0.1:18080")
        assert ok and session.provider == "openai", msg
        assert session.model == "srv-1", session.model  # adopted the server's first model

        ok, msg = session.switch_model("srv-2")
        assert ok and session.model == "srv-2", msg
        ok, msg = session.switch_model("not-loaded")
        assert not ok and "srv-1" in msg and session.model == "srv-2", msg

        ok, msg = session.switch_provider("bogus")
        assert not ok and session.provider == "openai", msg

        # A switch to an unreachable server changes nothing.
        old_client = session.client
        ok, msg = session.switch_provider("openai", "http://127.0.0.1:1")
        assert not ok and session.client is old_client and session.provider == "openai", msg

        ok, msg = session.switch_provider("ollama")
        assert ok and session.provider == "ollama", msg
        assert "Ollama" in session.status()


def test_session_switch_model_ollama() -> None:
    session = ProviderSession("ollama", "http://127.0.0.1:11434", "qwen2.5-coder:14b")
    with patch("ogun_core.providers.urllib.request.urlopen", _router_urlopen):
        ok, msg = session.switch_model("llama3.1:8b")
        assert ok and session.model == "llama3.1:8b" and "Note:" not in msg, msg
        ok, msg = session.switch_model("not-pulled:7b")
        assert ok and "not in Ollama's local library" in msg, msg
        ok, msg = session.switch_model("")
        assert not ok


def test_session_gguf_switch_llamacpp() -> None:
    session = ProviderSession("llama.cpp", "http://127.0.0.1:8080", "srv-1")
    with tempfile.TemporaryDirectory() as tmp:
        gguf = Path(tmp) / "demo-Q4_K_M.gguf"
        gguf.write_bytes(b"GGUF")
        with patch.object(LlamaCppServer, "start", lambda self, path: (True, "fake server started")), \
             patch("ogun_core.providers.urllib.request.urlopen", _router_urlopen):
            ok, msg = session.switch_model(str(gguf))
    assert ok and "fake server started" in msg, msg
    assert session.llama_server is not None
    assert session.model == "srv-1"  # adopted from the fake server's /v1/models


def test_session_gguf_switch_ollama_import() -> None:
    class Res:
        returncode = 0
        stdout = ""
        stderr = ""

    session = ProviderSession("ollama", "http://127.0.0.1:11434", "qwen2.5-coder:14b")
    with tempfile.TemporaryDirectory() as tmp:
        gguf = Path(tmp) / "Demo-Q4_K_M.gguf"
        gguf.write_bytes(b"GGUF")
        with patch("ogun_core.providers.urllib.request.urlopen", _router_urlopen), \
             patch("ogun_core.providers.shutil.which", return_value="/usr/bin/ollama"), \
             patch.object(providers.subprocess, "run", lambda cmd, **kw: Res()):
            ok, msg = session.switch_model(str(gguf))
    assert ok and session.model == "demo-q4_k_m", msg


def test_main_switch_command_matchers() -> None:
    import main as ogun_main
    assert ogun_main.is_model_command("/model")
    assert ogun_main.is_model_command("/model llama3.1:8b")
    assert ogun_main.is_model_command("/model /tmp/my models/demo.gguf")
    assert not ogun_main.is_model_command("tell me about /model")
    assert not ogun_main.is_model_command("/modelx foo")
    assert ogun_main.is_provider_command("/provider")
    assert ogun_main.is_provider_command("/provider lmstudio")
    assert ogun_main.is_provider_command("/provider openai http://127.0.0.1:8000")
    assert not ogun_main.is_provider_command("/provider a b c")
    assert not ogun_main.is_provider_command("use /provider please")


def main() -> int:
    tests = [
        test_normalize_provider,
        test_resolve_provider_host,
        test_option_mapping,
        test_openai_chat_plain_and_payload,
        test_openai_chat_streaming_sse,
        test_openai_connection_and_models,
        test_gguf_model_name,
        test_import_gguf_into_ollama,
        test_llama_server_command_and_binary,
        test_discover_local_models,
        test_create_client_factory,
        test_engine_runs_on_openai_client,
        test_session_switch_provider_and_model,
        test_session_switch_model_ollama,
        test_session_gguf_switch_llamacpp,
        test_session_gguf_switch_ollama_import,
        test_main_switch_command_matchers,
    ]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print("PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as ae:
        print(f"TEST ASSERTION FAILED: {ae}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"TEST RUNTIME ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
