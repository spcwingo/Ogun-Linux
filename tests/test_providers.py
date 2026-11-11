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
    load_custom_providers,
    normalize_provider,
    resolve_provider_host,
    resolve_provider_target,
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
        assert ok and "setting it anyway" in msg and session.model == "not-loaded", msg

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


# ---------------------------------------------------------------------------
# Custom named providers + API keys
# ---------------------------------------------------------------------------
def test_load_custom_providers() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "providers.json"
        cfg.write_text(json.dumps({
            "filebox": {"host": "http://10.0.0.1:8000"},
            "shared": {"host": "http://10.0.0.2:8000"},
            "broken": {"model": "no-host-here"},
        }), encoding="utf-8")
        env = {"OGUN_CUSTOM_PROVIDERS": json.dumps({
            "envbox": {"host": "http://10.0.0.3:9000", "model": "m1"},
            "shared": {"host": "http://10.0.0.9:8000"},
        })}
        with patch.object(providers, "CUSTOM_PROVIDERS_PATH", cfg), patch.dict(os.environ, env):
            customs = load_custom_providers()
    assert customs["filebox"]["host"] == "http://10.0.0.1:8000"
    assert customs["envbox"]["model"] == "m1"
    assert customs["shared"]["host"] == "http://10.0.0.9:8000", "environment must win per name"
    assert "broken" not in customs, "definitions without a host are dropped"

    with patch.object(providers, "CUSTOM_PROVIDERS_PATH", Path("/nonexistent/providers.json")), \
         patch.dict(os.environ, {"OGUN_CUSTOM_PROVIDERS": "not json{"}):
        assert load_custom_providers() == {}, "malformed sources contribute nothing, never raise"


def test_resolve_provider_target() -> None:
    env = {
        "OGUN_CUSTOM_PROVIDERS": json.dumps({
            "bigbox": {"host": "http://192.168.1.50:8000/", "model": "qwen3-32b", "api_key_env": "BIGBOX_KEY"},
        }),
        "BIGBOX_KEY": "sekret",
        "OGUN_LLM_API_KEY": "generic-key",
    }
    with patch.dict(os.environ, env):
        target = resolve_provider_target("BigBox")
        assert target["kind"] == "openai" and target["display"] == "BigBox"
        assert target["host"] == "http://192.168.1.50:8000" and target["model"] == "qwen3-32b"
        assert target["api_key"] == "sekret"
        assert resolve_provider_target("bigbox", "http://other:1")["host"] == "http://other:1"
        assert resolve_provider_target("nope") is None
        builtin = resolve_provider_target("openai")
        assert builtin["kind"] == "openai" and builtin["api_key"] == "generic-key"
        ollama_t = resolve_provider_target("ollama")
        assert ollama_t["kind"] == "ollama" and ollama_t["api_key"] is None


def test_api_key_header_sent() -> None:
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["auth"] = req.headers.get("Authorization")
        return _json_resp({"choices": [{"message": {"content": "ok", "tool_calls": []}}]})

    client = OpenAICompatClient(host="http://127.0.0.1:8000", model="m", api_key="key-123")
    with patch("ogun_core.providers.urllib.request.urlopen", fake_urlopen):
        assert client.chat([{"role": "user", "content": "hi"}])["content"] == "ok"
    assert captured["auth"] == "Bearer key-123", captured

    captured.clear()
    plain = OpenAICompatClient(host="http://127.0.0.1:8000", model="m")
    with patch("ogun_core.providers.urllib.request.urlopen", fake_urlopen):
        plain.chat([{"role": "user", "content": "hi"}])
    assert captured["auth"] is None, "no key configured -> no Authorization header"


def test_check_connection_401_names_the_key_problem() -> None:
    import urllib.error

    def unauthorized(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

    bare = OpenAICompatClient(host="http://127.0.0.1:4000", model="m", label="deeperseeker")
    with patch("ogun_core.providers.urllib.request.urlopen", unauthorized):
        ok, msg = bare.check_connection()
    assert not ok and "401" in msg and "no API key was given" in msg and "OGUN_LLM_API_KEY" in msg, msg

    keyed = OpenAICompatClient(host="http://127.0.0.1:4000", model="m", api_key="wrong-key")
    with patch("ogun_core.providers.urllib.request.urlopen", unauthorized):
        ok, msg = keyed.check_connection()
    assert not ok and "the key it was given was rejected" in msg, msg


def test_http_error_body_is_surfaced() -> None:
    import urllib.error

    def raising(body: bytes, code: int = 500, reason: str = "Internal Server Error"):
        def fake(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, code, reason, {}, io.BytesIO(body))
        return fake

    # detail extraction variants
    from ogun_core.providers import _http_error_detail

    def err(body: bytes):
        return urllib.error.HTTPError("http://x", 500, "ISE", {}, io.BytesIO(body))

    assert _http_error_detail(err(b'{"detail": "no route for model"}')) == "no route for model"
    assert _http_error_detail(err(b'{"error": {"message": "upstream down"}}')) == "upstream down"
    assert _http_error_detail(err(b'{"error": "plain string error"}')) == "plain string error"
    assert _http_error_detail(err(b'raw text boom')) == "raw text boom"
    assert _http_error_detail(err(b'')) == ""

    client = OpenAICompatClient(host="http://127.0.0.1:4000", model="v4.1flash", label="deeperseeker")
    body = json.dumps({"detail": "backend exploded: upstream key missing"}).encode()
    # non-streaming chat
    with patch("ogun_core.providers.urllib.request.urlopen", raising(body)):
        try:
            client.chat([{"role": "user", "content": "hi"}])
            raise AssertionError("chat should have raised")
        except RuntimeError as exc:
            assert "HTTP 500" in str(exc) and "upstream key missing" in str(exc), exc
    # streaming chat
    with patch("ogun_core.providers.urllib.request.urlopen", raising(body)):
        try:
            client.chat([{"role": "user", "content": "hi"}], on_token=lambda t: None)
            raise AssertionError("streaming chat should have raised")
        except RuntimeError as exc:
            assert "upstream key missing" in str(exc), exc
    # connection check shows the detail too
    with patch("ogun_core.providers.urllib.request.urlopen", raising(body)):
        ok, msg = client.check_connection()
    assert not ok and "HTTP 500" in msg and "upstream key missing" in msg, msg
    # a body-less error still produces the plain status message
    with patch("ogun_core.providers.urllib.request.urlopen", raising(b"")):
        try:
            client.chat([{"role": "user", "content": "hi"}])
            raise AssertionError("chat should have raised")
        except RuntimeError as exc:
            assert "HTTP 500" in str(exc) and "server said" not in str(exc), exc


def test_session_switch_to_custom_provider() -> None:
    env = {"OGUN_CUSTOM_PROVIDERS": json.dumps({"bigbox": {"host": "http://127.0.0.1:18080"}})}
    session = ProviderSession("ollama", "http://127.0.0.1:11434", "qwen2.5-coder:14b")
    with patch.dict(os.environ, env), \
         patch("ogun_core.providers.urllib.request.urlopen", _router_urlopen):
        ok, msg = session.switch_provider("bigbox")
        assert ok and session.provider == "openai", msg
        assert session.label == "bigbox" and "bigbox" in session.status()
        assert session.model == "srv-1", session.model  # adopted from the server
        ok, msg = session.switch_provider("bigbox")
        assert ok and "Already using bigbox" in msg, msg


def test_literal_api_key_support() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        providers.CUSTOM_PROVIDERS_PATH = Path(tmp) / "providers.json"
        os.environ.pop("OGUN_CUSTOM_PROVIDERS", None)

        # save with a literal key: stored, resolvable, and the file goes owner-only
        ok, msg = providers.save_custom_provider("keybox", "http://127.0.0.1:18080", api_key="sk-literal-1")
        assert ok, msg
        target = resolve_provider_target("keybox")
        assert target["api_key"] == "sk-literal-1", target
        mode = os.stat(providers.CUSTOM_PROVIDERS_PATH).st_mode & 0o777
        assert mode == 0o600, oct(mode)

        # the key value never leaks into the listing or edit views
        listing = providers.provider_listing()
        assert "key=stored" in listing and "sk-literal-1" not in listing, listing
        ok, shown = providers.edit_custom_provider("keybox", {})
        assert ok and "key=stored" in shown and "sk-literal-1" not in shown, shown

        # keyenv (when set) wins over the stored key; stored key is the fallback
        providers.edit_custom_provider("keybox", {"keyenv": "KEYBOX_ENV"})
        with patch.dict(os.environ, {"KEYBOX_ENV": "from-env"}):
            assert resolve_provider_target("keybox")["api_key"] == "from-env"
        os.environ.pop("KEYBOX_ENV", None)
        assert resolve_provider_target("keybox")["api_key"] == "sk-literal-1"

        # edit can replace and clear the stored key
        ok, msg = providers.edit_custom_provider("keybox", {"key": "sk-literal-2"})
        assert ok and resolve_provider_target("keybox")["api_key"] == "sk-literal-2", msg
        ok, msg = providers.edit_custom_provider("keybox", {"key": ""})
        assert ok and resolve_provider_target("keybox")["api_key"] is None, msg

        # session add with a literal key authenticates end to end (Bearer header sent)
        seen = {}

        def capturing(req, timeout=None):
            seen["auth"] = req.headers.get("Authorization")
            return _router_urlopen(req, timeout=timeout)

        session = ProviderSession("ollama", "http://127.0.0.1:11434", "qwen2.5-coder:14b")
        with patch("ogun_core.providers.urllib.request.urlopen", capturing):
            ok, msg = session.add_provider("authbox", "http://127.0.0.1:18080", api_key="sk-add-9")
            assert ok and session.label == "authbox", msg
            assert seen["auth"] == "Bearer sk-add-9", seen

    # parser: key= in add (any position), both key forms together, key in edit
    import main as ogun_main
    parse_provider_command = ogun_main.parse_provider_command
    parsed = parse_provider_command("/provider add a http://h:1 m1 key=sk-1")
    assert parsed["api_key"] == "sk-1" and parsed["model"] == "m1", parsed
    parsed = parse_provider_command("/provider add a http://h:1 keyenv=V key=sk-2")
    assert parsed["api_key_env"] == "V" and parsed["api_key"] == "sk-2", parsed
    parsed = parse_provider_command("/provider edit a key=sk-3")
    assert parsed["changes"] == {"key": "sk-3"}, parsed


def test_unset_keyenv_is_surfaced() -> None:
    os.environ.pop("DEFINITELY_UNSET_VAR_XYZ", None)
    env = {"OGUN_CUSTOM_PROVIDERS": json.dumps({
        "warnbox": {"host": "http://127.0.0.1:18080", "api_key_env": "DEFINITELY_UNSET_VAR_XYZ"},
        "deadbox": {"host": "http://127.0.0.1:1", "api_key_env": "DEFINITELY_UNSET_VAR_XYZ"},
    })}
    with patch.dict(os.environ, env):
        target = resolve_provider_target("warnbox")
        assert target["key_env"] == "DEFINITELY_UNSET_VAR_XYZ" and target["api_key"] is None
        listing = providers.provider_listing()
        assert "keyenv=DEFINITELY_UNSET_VAR_XYZ (NOT SET)" in listing, listing

    session = ProviderSession("ollama", "http://127.0.0.1:11434", "qwen2.5-coder:14b")
    with patch.dict(os.environ, env), \
         patch("ogun_core.providers.urllib.request.urlopen", _router_urlopen):
        ok, msg = session.switch_provider("warnbox")
        assert ok and "no API key was sent" in msg and "not the key itself" in msg, msg
        ok, msg = session.switch_provider("deadbox")
        assert not ok and "no API key was sent" in msg, msg

    # and when the variable IS set, there is no warning and the key resolves
    env_set = dict(env, DEFINITELY_UNSET_VAR_XYZ="the-key")
    with patch.dict(os.environ, env_set):
        target = resolve_provider_target("warnbox")
        assert target["api_key"] == "the-key"
        assert "(NOT SET)" not in providers.provider_listing()


def test_explicit_model_is_never_overridden() -> None:
    env = {"OGUN_CUSTOM_PROVIDERS": json.dumps({
        "gateway": {"host": "http://127.0.0.1:18080", "model": "kimi-k3.1-pro"},
        "plain": {"host": "http://127.0.0.1:18080"},
    })}
    session = ProviderSession("ollama", "http://127.0.0.1:11434", "qwen2.5-coder:14b")
    with patch.dict(os.environ, env), \
         patch("ogun_core.providers.urllib.request.urlopen", _router_urlopen):
        # a definition model wins over the server's advertised list (gateway behaviour:
        # the fake server only advertises 'srv-1', like a router listing other ids)
        ok, msg = session.switch_provider("gateway")
        assert ok and session.model == "kimi-k3.1-pro", (msg, session.model)
        assert session.client.model == "kimi-k3.1-pro"
        assert "doesn't list" in msg and "as configured" in msg, msg
        # /model to an unlisted id: allowed with a warning, not rejected
        ok, msg = session.switch_model("some/other-model")
        assert ok and session.model == "some/other-model" and "setting it anyway" in msg, msg
        # a provider without a definition model still adopts the served model
        ok, msg = session.switch_provider("plain")
        assert ok and session.model == "srv-1", (msg, session.model)

    # startup-shaped sessions: the explicit flag alone decides adoption
    explicit = ProviderSession("openai", "http://127.0.0.1:18080", "kimi-k3.1-pro", model_explicit=True)
    carried = ProviderSession("openai", "http://127.0.0.1:18080", "qwen2.5-coder:14b")
    with patch("ogun_core.providers.urllib.request.urlopen", _router_urlopen):
        note = explicit.adopt_served_model()
        assert explicit.model == "kimi-k3.1-pro" and "as configured" in note, note
        note = carried.adopt_served_model()
        assert carried.model == "srv-1" and "server's model" in note, note


def _anthropic_urlopen_factory(captured: dict):
    """Fake Anthropic-shaped server (models + messages; chat/completions 404s, like FCC)."""
    import urllib.error

    def fake(req, timeout=None):
        url = req.full_url
        if url.endswith("/v1/chat/completions"):
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, io.BytesIO(b"Not Found"))
        if url.endswith("/v1/models"):
            return _json_resp({"data": [{"id": "kimi-k3.1-pro"}, {"id": "other/model"}]})
        if url.endswith("/v1/messages"):
            payload = json.loads(req.data.decode("utf-8"))
            captured.setdefault("payloads", []).append(payload)
            captured["headers"] = dict(req.header_items())
            texts = []
            has_tool_result = False
            for m in payload.get("messages", []):
                for b in m.get("content", []):
                    if b.get("type") == "text":
                        texts.append(b.get("text", ""))
                    if b.get("type") == "tool_result":
                        has_tool_result = True
            if payload.get("stream"):
                if "USETOOL" in " ".join(texts) and not has_tool_result:
                    lines = [
                        b'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_9", "name": "read_file", "input": {}}}',
                        b'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\\"filepath\\": \\"sample.txt\\"}"}}',
                        b'data: {"type": "message_stop"}',
                    ]
                else:
                    lines = [
                        b'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}',
                        b'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "streamed "}}',
                        b'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "reply"}}',
                        b'data: {"type": "message_stop"}',
                    ]
                return FakeResp(lines=lines)
            if has_tool_result:
                return _json_resp({"content": [{"type": "text", "text": "The file contains sample text."}]})
            if "USETOOL" in " ".join(texts):
                return _json_resp({"content": [{"type": "tool_use", "id": "toolu_1", "name": "read_file",
                                                "input": {"filepath": "sample.txt"}}]})
            return _json_resp({"content": [{"type": "text", "text": "anthropic hello"}]})
        raise AssertionError(f"unexpected URL {url}")
    return fake


def test_anthropic_client_and_translation() -> None:
    # --- message translation: system extraction, tool id synthesis/matching, role merging
    system, conv = providers._to_anthropic_messages([
        {"role": "system", "content": "You are Ogun."},
        {"role": "user", "content": "read it"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "read_file", "arguments": {"filepath": "a.txt"}}}]},
        {"role": "tool", "content": "file body"},
        {"role": "assistant", "content": "done"},
    ])
    assert system == "You are Ogun."
    assert [m["role"] for m in conv] == ["user", "assistant", "user", "assistant"], conv
    tool_use = conv[1]["content"][0]
    assert tool_use["type"] == "tool_use" and tool_use["id"] == "toolu_1", tool_use
    tool_result = conv[2]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "toolu_1", tool_result

    # --- options: max_tokens required, temperature wins over top_p
    from ogun_core.config import llm_options as _llm_options
    opts = providers._map_anthropic_options(_llm_options())
    assert opts["max_tokens"] == 3072 and "temperature" in opts and "top_p" not in opts, opts

    # --- client: text reply, tool_use reply, payload shape, headers
    captured: dict = {}
    client = providers.AnthropicMessagesClient(host="http://127.0.0.1:8082", model="kimi-k3.1-pro",
                                               label="fcc", api_key="freecc")
    tools = [{"type": "function", "function": {"name": "read_file", "description": "Read a file",
                                                "parameters": {"type": "object", "properties": {"filepath": {"type": "string"}}}}}]
    with patch("ogun_core.providers.urllib.request.urlopen", _anthropic_urlopen_factory(captured)):
        out = client.chat([{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}], tools=tools)
        assert out["content"] == "anthropic hello" and out["tool_calls"] == [], out
        payload = captured["payloads"][-1]
        assert payload["system"] == "sys" and payload["max_tokens"] > 0, payload
        assert payload["tools"][0]["name"] == "read_file" and "input_schema" in payload["tools"][0], payload
        assert captured["headers"].get("X-api-key") == "freecc", captured["headers"]
        assert captured["headers"].get("Anthropic-version") == "2023-06-01", captured["headers"]

        out = client.chat([{"role": "user", "content": "USETOOL please"}], tools=tools)
        assert out["tool_calls"][0]["function"]["name"] == "read_file", out
        assert out["tool_calls"][0]["function"]["arguments"] == {"filepath": "sample.txt"}, out
        assert out["tool_calls"][0]["id"] == "toolu_1", out

        # streaming: text assembly + tool argument fragments
        pieces: list = []
        out = client.chat([{"role": "user", "content": "hi"}], on_token=pieces.append)
        assert out["content"] == "streamed reply" and "".join(pieces) == "streamed reply", out
        out = client.chat([{"role": "user", "content": "USETOOL"}], tools=tools, stream=True)
        assert out["tool_calls"][0]["function"]["arguments"] == {"filepath": "sample.txt"}, out

        ok, msg = client.check_connection()
        assert ok and "2 model(s)" in msg, msg
        assert client.list_models() == ["kimi-k3.1-pro", "other/model"]

    # --- factory, aliases, engine roundtrip
    assert isinstance(create_client("anthropic", host="http://127.0.0.1:8082"), providers.AnthropicMessagesClient)
    assert normalize_provider("claude") == "anthropic"
    captured2: dict = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        (root / "sample.txt").write_text("hello world from sample", encoding="utf-8")
        eng_client = providers.AnthropicMessagesClient(host="http://127.0.0.1:8082", model="kimi-k3.1-pro")
        engine = CoreEngine(client=eng_client, base_dir=root)
        with patch("ogun_core.providers.urllib.request.urlopen", _anthropic_urlopen_factory(captured2)):
            reply = engine.run_turn("Please read sample.txt USETOOL")
        assert "The file contains sample text." in reply, reply
        assert len(captured2["payloads"]) == 2, captured2["payloads"]


def test_anthropic_custom_provider_type() -> None:
    env = {"OGUN_CUSTOM_PROVIDERS": json.dumps({
        "fcc": {"host": "http://127.0.0.1:8082", "model": "kimi-k3.1-pro",
                "api_key": "freecc", "type": "anthropic"}})}
    with patch.dict(os.environ, env):
        target = resolve_provider_target("fcc")
        assert target["kind"] == "anthropic", target
    captured: dict = {}
    session = ProviderSession("ollama", "http://127.0.0.1:11434", "qwen2.5-coder:14b")
    with patch.dict(os.environ, env), \
         patch("ogun_core.providers.urllib.request.urlopen", _anthropic_urlopen_factory(captured)):
        ok, msg = session.switch_provider("fcc")
        assert ok and session.provider == "anthropic", msg
        assert isinstance(session.client, providers.AnthropicMessagesClient)
        assert session.model == "kimi-k3.1-pro", session.model

    # save/edit roundtrip with the type field, in a temp providers file
    with tempfile.TemporaryDirectory() as tmp:
        providers.CUSTOM_PROVIDERS_PATH = Path(tmp) / "providers.json"
        os.environ.pop("OGUN_CUSTOM_PROVIDERS", None)
        ok, msg = providers.save_custom_provider("gw", "http://127.0.0.1:8082", api_type="anthropic")
        assert ok and providers.get_custom_provider("gw")["type"] == "anthropic", msg
        ok, msg = providers.save_custom_provider("gw2", "http://127.0.0.1:8082", api_type="grpc")
        assert not ok and "Unknown provider type" in msg, msg
        ok, msg = providers.edit_custom_provider("gw", {"type": "openai"})
        assert ok and "type" not in providers.get_custom_provider("gw"), msg
        listing = providers.provider_listing()
        assert "type=anthropic" not in listing  # gw is openai-typed again now

    import main as ogun_main
    parsed = ogun_main.parse_provider_command("/provider add fcc http://localhost:8082 kimi-k3.1-pro key=freecc type=anthropic")
    assert parsed["api_type"] == "anthropic" and parsed["api_key"] == "freecc", parsed
    parsed = ogun_main.parse_provider_command("/provider edit fcc type=anthropic")
    assert parsed["changes"] == {"type": "anthropic"}, parsed


def test_save_and_remove_custom_provider() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "providers.json"
        with patch.object(providers, "CUSTOM_PROVIDERS_PATH", cfg), \
             patch.dict(os.environ, {"OGUN_CUSTOM_PROVIDERS": ""}):
            ok, msg = providers.save_custom_provider("newbox", "http://10.0.0.5:8000/", model="m1", api_key_env="NB_KEY")
            assert ok, msg
            assert load_custom_providers()["newbox"] == {"host": "http://10.0.0.5:8000", "model": "m1", "api_key_env": "NB_KEY"}
            ok, msg = providers.save_custom_provider("ollama", "http://x:1")
            assert not ok and "built-in" in msg, msg
            ok, _msg = providers.save_custom_provider("bad name", "http://x:1")
            assert not ok
            ok, msg = providers.save_custom_provider("okname", "localhost:8000")
            assert not ok and "not a URL" in msg, msg
            ok, msg = providers.remove_custom_provider("NEWBOX")   # case-insensitive
            assert ok, msg
            assert load_custom_providers() == {}
            ok, _msg = providers.remove_custom_provider("ghost")
            assert not ok

    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "providers.json"
        env = {"OGUN_CUSTOM_PROVIDERS": json.dumps({"envbox": {"host": "http://10.0.0.6:8000"}})}
        with patch.object(providers, "CUSTOM_PROVIDERS_PATH", cfg), patch.dict(os.environ, env):
            ok, msg = providers.remove_custom_provider("envbox")
            assert not ok and "OGUN_CUSTOM_PROVIDERS" in msg, msg


def test_session_add_and_remove_provider() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "providers.json"
        session = ProviderSession("ollama", "http://127.0.0.1:11434", "qwen2.5-coder:14b")
        with patch.object(providers, "CUSTOM_PROVIDERS_PATH", cfg), \
             patch.dict(os.environ, {"OGUN_CUSTOM_PROVIDERS": ""}), \
             patch("ogun_core.providers.urllib.request.urlopen", _router_urlopen):
            ok, msg = session.add_provider("newbox", "http://127.0.0.1:18080")
            assert ok and session.label == "newbox" and session.provider == "openai", msg
            assert providers.get_custom_provider("newbox") is not None
            ok, msg = session.add_provider("deadbox", "http://127.0.0.1:1")
            assert ok and "saved for next time" in msg and session.label == "newbox", msg
            ok, msg = session.remove_provider("deadbox")
            assert ok and providers.get_custom_provider("deadbox") is None, msg


def test_parse_provider_command() -> None:
    import main as ogun_main
    parse = ogun_main.parse_provider_command
    assert parse("/provider") == {"action": "status"}
    assert parse("/provider lmstudio") == {"action": "switch", "name": "lmstudio", "host": None}
    assert parse("/provider openai http://h:1")["host"] == "http://h:1"
    add = parse("/provider add bigbox http://h:8000 qwen3-32b keyenv=BB_KEY")
    assert add == {"action": "add", "name": "bigbox", "host": "http://h:8000",
                   "model": "qwen3-32b", "api_key_env": "BB_KEY", "api_key": None, "api_type": None}, add
    assert parse("/provider add bigbox http://h:8000")["model"] is None
    assert parse("/provider remove bigbox") == {"action": "remove", "name": "bigbox"}
    # tolerant add forms: named fields in any position, api-key= alias
    named = parse("/provider add a host=http://h:1 model=m1 api-key=sk-9")
    assert named["host"] == "http://h:1" and named["model"] == "m1" and named["api_key"] == "sk-9", named
    mixed = parse("/provider add a host://h:2 model=m2 api-key=sk-8")
    assert mixed["action"] == "add" and mixed["host"] == "host://h:2" and mixed["api_key"] == "sk-8", mixed
    assert parse("/provider edit a api-key=sk-7")["changes"] == {"key": "sk-7"}
    assert parse("/provider add onlyname") is None
    assert parse("/provider remove") is None
    assert parse("/provider a b c") is None
    assert parse("what is a /provider") is None
    assert parse("/provider list") == {"action": "list"}
    assert parse("/provider list extra") is None
    assert parse("/provider edit bigbox") == {"action": "edit", "name": "bigbox", "changes": {}}
    edit = parse("/provider edit bigbox host=http://h:9 model=m2 keyenv=K1")
    assert edit["changes"] == {"host": "http://h:9", "model": "m2", "keyenv": "K1"}, edit
    assert parse("/provider edit") is None
    assert parse("/provider edit bigbox host") is None       # field without '='
    assert parse("/provider edit bigbox bogus=1") is None    # unknown field


def test_provider_listing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "providers.json"
        cfg.write_text(json.dumps({"filebox": {"host": "http://10.0.0.1:8000", "model": "fm", "api_key_env": "FB_KEY"}}),
                       encoding="utf-8")
        env = {"OGUN_CUSTOM_PROVIDERS": json.dumps({"envbox": {"host": "http://10.0.0.2:8000"}})}
        with patch.object(providers, "CUSTOM_PROVIDERS_PATH", cfg), patch.dict(os.environ, env):
            text = providers.provider_listing(active_kind="openai", active_display="filebox")
            builtin = providers.provider_listing(active_kind="ollama", active_display="Ollama")
    assert "* filebox" in text and "model=fm" in text and "keyenv=FB_KEY" in text, text
    assert "(custom, config/providers.json)" in text and "(custom, $OGUN_CUSTOM_PROVIDERS)" in text
    assert "envbox" in text and "ollama" in text and "lmstudio" in text
    assert "* ollama" in builtin, builtin
    assert "* filebox" not in builtin


def test_edit_custom_provider() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "providers.json"
        with patch.object(providers, "CUSTOM_PROVIDERS_PATH", cfg), \
             patch.dict(os.environ, {"OGUN_CUSTOM_PROVIDERS": ""}):
            providers.save_custom_provider("box", "http://10.0.0.1:8000", model="m1")
            ok, msg = providers.edit_custom_provider("box", {})
            assert ok and "host=http://10.0.0.1:8000" in msg and "model=m1" in msg, msg
            ok, msg = providers.edit_custom_provider("box", {"host": "http://10.0.0.7:9000/"})
            assert ok and load_custom_providers()["box"]["host"] == "http://10.0.0.7:9000", msg
            ok, _ = providers.edit_custom_provider("box", {"model": ""})
            assert ok and "model" not in load_custom_providers()["box"], "empty model= clears it"
            ok, _ = providers.edit_custom_provider("box", {"keyenv": "BOX_KEY"})
            assert load_custom_providers()["box"]["api_key_env"] == "BOX_KEY"
            ok, _ = providers.edit_custom_provider("box", {"keyenv": ""})
            assert "api_key_env" not in load_custom_providers()["box"]
            ok, msg = providers.edit_custom_provider("box", {"host": "not-a-url"})
            assert not ok and load_custom_providers()["box"]["host"] == "http://10.0.0.7:9000", "bad host rejected, definition unchanged"
            ok, _ = providers.edit_custom_provider("ghost", {"host": "http://x:1"})
            assert not ok
            ok, msg = providers.save_custom_provider("list", "http://x:1")
            assert not ok and "subcommand" in msg, msg

    env = {"OGUN_CUSTOM_PROVIDERS": json.dumps({"envbox": {"host": "http://10.0.0.2:8000"}})}
    with tempfile.TemporaryDirectory() as tmp:
        with patch.object(providers, "CUSTOM_PROVIDERS_PATH", Path(tmp) / "providers.json"), patch.dict(os.environ, env):
            ok, msg = providers.edit_custom_provider("envbox", {"host": "http://10.0.0.3:8000"})
            assert not ok and "OGUN_CUSTOM_PROVIDERS" in msg, msg


def test_session_edit_provider_live() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "providers.json"
        session = ProviderSession("ollama", "http://127.0.0.1:11434", "qwen2.5-coder:14b")
        with patch.object(providers, "CUSTOM_PROVIDERS_PATH", cfg), \
             patch.dict(os.environ, {"OGUN_CUSTOM_PROVIDERS": ""}), \
             patch("ogun_core.providers.urllib.request.urlopen", _router_urlopen):
            ok, msg = session.add_provider("newbox", "http://127.0.0.1:18080")
            assert ok and session.label == "newbox", msg
            ok, msg = session.edit_provider("newbox", {"model": "srv-2"})
            assert ok and "Applied live" in msg and session.model == "srv-2", msg
            assert load_custom_providers()["newbox"]["model"] == "srv-2"
            # editing a different provider leaves the session alone
            session.add_provider("otherbox", "http://127.0.0.1:18080")  # switches to otherbox
            ok, msg = session.edit_provider("newbox", {"model": "srv-1"})
            assert ok and "Applied live" not in msg and session.label == "otherbox", msg
            # editing the ACTIVE provider to an unreachable host: edit stands, old client keeps serving
            ok, msg = session.edit_provider("otherbox", {"host": "http://127.0.0.1:1"})
            assert ok and "reconnecting failed" in msg and session.host == "http://127.0.0.1:18080", msg


def test_ollama_client_list_models() -> None:
    client = OllamaClient(host="http://127.0.0.1:11434", model="qwen2.5-coder:14b")
    with patch("ogun_core.engine.urllib.request.urlopen", _router_urlopen):
        assert client.list_models() == ["qwen2.5-coder:14b", "llama3.1:8b"]
    import urllib.error

    def _boom(req, timeout=None):
        raise urllib.error.URLError("down")

    with patch("ogun_core.engine.urllib.request.urlopen", _boom):
        assert client.list_models() == []


def test_session_available_models() -> None:
    session = ProviderSession("openai", "http://127.0.0.1:18080", "srv-1")
    with patch("ogun_core.providers.urllib.request.urlopen", _router_urlopen):
        assert session.available_models() == ["srv-1", "srv-2"]
    dead = ProviderSession("openai", "http://127.0.0.1:1", "srv-1")
    assert dead.available_models() == []

    class _Exploding:
        def list_models(self):
            raise RuntimeError("nope")

    session.client = _Exploding()
    assert session.available_models() == []
    session.client = object()  # a client with no listing call at all
    assert session.available_models() == []


def test_provider_choices() -> None:
    env = {"OGUN_CUSTOM_PROVIDERS": json.dumps({"fcc": {"host": "http://localhost:8082", "model": "k"}})}
    with patch.dict(os.environ, env):
        choices = providers.provider_choices("openai", providers.PROVIDER_LABELS["openai"])
        names = [c["name"] for c in choices]
        assert names[:5] == ["ollama", "openai", "llama.cpp", "lmstudio", "anthropic"], names
        assert names[-1] == "fcc"
        active = {c["name"]: c["active"] for c in choices}
        assert active["openai"] and not active["fcc"] and not active["ollama"]
        choices2 = providers.provider_choices("anthropic", "fcc")
        active2 = {c["name"]: c["active"] for c in choices2}
        assert active2["fcc"] and not active2["anthropic"]


class _MenuSession:
    """ProviderSession stand-in for the /model menu: scripted models and switches."""

    def __init__(self, models, model="m1"):
        self.provider = "openai"
        self.label = "MenuFake"
        self.host = "http://menu.local"
        self.model = model
        self._models = list(models)
        self.switched_models = []
        self.switched_providers = []

    def available_models(self):
        return list(self._models)

    def switch_model(self, value):
        self.switched_models.append(value)
        self.model = value
        return True, f"Model set to '{value}' (via MenuFake)."

    def switch_provider(self, name, new_host=None):
        self.switched_providers.append(name)
        if name == "fcc":
            self.label = "fcc"
            self.host = "http://localhost:8082"
            self._models = ["fcc-1", "fcc-2"]
            self.model = "fcc-1"
            return True, "Switched to fcc at http://localhost:8082, model 'fcc-1'."
        return False, f"Could not switch to {name}."


def _run_menu(session, answers) -> str:
    import main as ogun_main
    printed = []
    it = iter(answers)
    ogun_main.choose_model_interactively(
        session, input_fn=lambda prompt="": next(it), print_fn=printed.append)
    return "\n".join(printed)


def test_model_menu_select_and_cancel() -> None:
    s = _MenuSession(["m1", "m2", "m3"], model="m1")
    out = _run_menu(s, ["2"])
    assert s.switched_models == ["m2"], s.switched_models
    assert "1. m1" in out and "3. m3" in out
    s2 = _MenuSession(["m1", "m2"])
    _run_menu(s2, [""])
    assert s2.switched_models == []


def test_model_menu_filter_and_paging() -> None:
    s = _MenuSession(["alpha", "beta", "alpine"], model="beta")
    _run_menu(s, ["alp", "2"])
    assert s.switched_models == ["alpine"], s.switched_models
    many = [f"mod-{i:02d}" for i in range(1, 21)]
    s2 = _MenuSession(many, model="mod-01")
    out = _run_menu(s2, ["n", "17"])
    assert s2.switched_models == ["mod-17"], s2.switched_models
    assert "page 2/2" in out
    s3 = _MenuSession(["alpha"], model="alpha")
    out = _run_menu(s3, ["zzz", ""])
    assert "No models match 'zzz'." in out and s3.switched_models == []


def test_model_menu_no_models_falls_back() -> None:
    s = _MenuSession([], model="solo")
    out = _run_menu(s, [])
    assert "Current model: solo (via MenuFake)." in out
    assert s.switched_models == []


def test_model_menu_provider_switch() -> None:
    import main as ogun_main
    choices = [
        {"name": "openai", "display": "openai — http://menu.local", "active": True},
        {"name": "fcc", "display": "fcc — http://localhost:8082", "active": False},
    ]
    s = _MenuSession(["m1", "m2"], model="m1")
    printed = []
    answers = iter(["v", "2", "2"])
    with patch.object(ogun_main, "provider_choices", lambda kind, display: choices):
        ogun_main.choose_model_interactively(
            s, input_fn=lambda prompt="": next(answers), print_fn=printed.append)
    assert s.switched_providers == ["fcc"], s.switched_providers
    assert s.switched_models == ["fcc-2"], s.switched_models


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
        test_load_custom_providers,
        test_resolve_provider_target,
        test_api_key_header_sent,
        test_check_connection_401_names_the_key_problem,
        test_http_error_body_is_surfaced,
        test_session_switch_to_custom_provider,
        test_literal_api_key_support,
        test_unset_keyenv_is_surfaced,
        test_explicit_model_is_never_overridden,
        test_anthropic_client_and_translation,
        test_anthropic_custom_provider_type,
        test_save_and_remove_custom_provider,
        test_session_add_and_remove_provider,
        test_parse_provider_command,
        test_provider_listing,
        test_edit_custom_provider,
        test_session_edit_provider_live,
        test_ollama_client_list_models,
        test_session_available_models,
        test_provider_choices,
        test_model_menu_select_and_cancel,
        test_model_menu_filter_and_paging,
        test_model_menu_no_models_falls_back,
        test_model_menu_provider_switch,
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
