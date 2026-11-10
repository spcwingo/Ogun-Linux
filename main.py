"""Ogun standalone autonomous coding assistant CLI entrypoint."""
from __future__ import annotations

import argparse
import atexit
import os
import re
import sys

from ogun_core import autotune
from ogun_core.config import BUILD_MAX_ATTEMPTS, LLM_PROVIDER, MODEL_NAME, OLLAMA_HOST, PLUGIN_BUILDER_PROMPT
from ogun_core.config import set_vram_profile
from ogun_core.engine import CoreEngine, OllamaClient, handle_vram_command, is_vram_command, unload_model
from ogun_core.paths import init_user_workspace
from ogun_core.plugin_builder import build_plugin_staged, parse_code_blocks
from ogun_core.plugin_manager import PluginManager
from ogun_core.providers import (
    ProviderSession,
    discover_local_models,
    format_model_list,
    provider_listing,
    resolve_provider_target,
)


UNLOAD_PHRASES = {
    "/unload", "unload", "unload model", "unload the model", "free gpu", "free the gpu", "free up gpu",
    "free vram", "free up vram", "release gpu memory", "release vram",
}


def is_unload_command(text: str) -> bool:
    """True only for an exact unload phrase, so ordinary sentences ('unload the dishwasher') are never hijacked."""
    return " ".join((text or "").lower().replace("?", " ").replace(".", " ").split()) in UNLOAD_PHRASES


_PROVIDER_CMD = re.compile(r"^\s*/provider(?:\s+(.*))?$", re.IGNORECASE)
_MODEL_CMD = re.compile(r"^\s*/model(?:\s+(.+))?\s*$", re.IGNORECASE)


def parse_provider_command(text: str):
    """Parse a /provider command line into an action dict, or None when it is not a well-formed command.

    Forms: /provider | /provider list | /provider <name> [host]
           | /provider add <name> <host> [model] [keyenv=VAR]
           | /provider edit <name> [host=<url>] [model=<id>] [keyenv=<VAR>]
           | /provider remove <name>
    """
    match = _PROVIDER_CMD.match(text or "")
    if not match:
        return None
    rest = (match.group(1) or "").strip()
    if not rest:
        return {"action": "status"}
    tokens = rest.split()
    head = tokens[0].lower()
    if head == "list":
        if len(tokens) != 1:
            return None
        return {"action": "list"}
    if head == "edit":
        # /provider edit <name> [host=<url>] [model=<id>] [keyenv=<VAR>]  (no fields = show it)
        if len(tokens) < 2:
            return None
        changes = {}
        for tok in tokens[2:]:
            if "=" not in tok:
                return None
            field, value = tok.split("=", 1)
            if field.lower() not in ("host", "model", "keyenv"):
                return None
            changes[field.lower()] = value
        return {"action": "edit", "name": tokens[1], "changes": changes}
    if head == "add":
        if len(tokens) < 3 or len(tokens) > 5:
            return None
        model, keyenv = None, None
        for tok in tokens[3:]:
            if tok.lower().startswith("keyenv="):
                keyenv = tok.split("=", 1)[1] or None
            elif model is None:
                model = tok
            else:
                return None
        return {"action": "add", "name": tokens[1], "host": tokens[2], "model": model, "api_key_env": keyenv}
    if head in ("remove", "rm", "delete"):
        if len(tokens) != 2:
            return None
        return {"action": "remove", "name": tokens[1]}
    if len(tokens) <= 2:
        return {"action": "switch", "name": tokens[0], "host": tokens[1] if len(tokens) == 2 else None}
    return None


def is_provider_command(text: str) -> bool:
    """True only for a well-formed /provider command, never for prose."""
    return parse_provider_command(text) is not None


def is_model_command(text: str) -> bool:
    """True only for '/model' optionally followed by a model tag/id or a .gguf path, never for prose."""
    return bool(_MODEL_CMD.match(text or ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ogun Autonomous Coding Assistant")
    parser.add_argument("--once", type=str, help="Execute a single turn and exit.")
    parser.add_argument("--build-plugin", type=str, help="Prompt the model to build, verify, and mount a plugin.")
    parser.add_argument("--model", type=str, default=None, help=f"Model to use (default: {MODEL_NAME}). "
                        "With a non-Ollama provider this is the server's model id; a path to a .gguf file works "
                        "with --provider llama.cpp (Ogun launches llama-server) or --provider ollama (auto-import).")
    parser.add_argument("--provider", type=str, default=None,
                        help="Inference provider: ollama (default) | openai | llama.cpp | lmstudio, or a custom "
                             "provider defined in config/providers.json or $OGUN_CUSTOM_PROVIDERS. Can also be "
                             "set with $OGUN_PROVIDER. 'openai' means any OpenAI-compatible local server "
                             "(llama-server, vLLM, HF TGI, Unsloth Studio, ...).")
    parser.add_argument("--host", type=str, default=None,
                        help="Base URL of the inference server (default: the provider's usual local address, "
                             "or $OGUN_LLM_HOST). For Ollama this overrides $OLLAMA_HOST.")
    parser.add_argument("--list-models", action="store_true",
                        help="List every local model Ogun can find (Ollama library, running servers, GGUF files, "
                             "Hugging Face cache, training exports) with the command that runs each, then exit.")
    parser.add_argument("--list-providers", action="store_true",
                        help="List all providers (built-ins and custom providers from config/providers.json / "
                             "$OGUN_CUSTOM_PROVIDERS) with their hosts and default models, then exit.")
    parser.add_argument("--no-tools", action="store_true", help="Disable tool execution.")
    parser.add_argument("--unload", action="store_true",
                        help="Unload the model from Ollama (free its GPU memory) and exit.")
    parser.add_argument("--vram", type=str, default=None, metavar="PROFILE",
                        help="Start with a fixed VRAM profile: max | shared | cpu (skips hardware auto-tuning). Change it in chat with /vram.")
    autotune.add_arguments(parser)

    args = parser.parse_args()
    args.model_explicit = args.model            # only a model typed on the command line is exempt from the 'does it fit' check
    args.model = args.model or MODEL_NAME

    # --list-providers is read-only: no provider needs to resolve, no connection check.
    if args.list_providers:
        print(provider_listing())
        sys.exit(0)

    provider_target = resolve_provider_target(args.provider or LLM_PROVIDER, args.host)
    if provider_target is None:
        print(f"[ERROR] Unknown provider '{args.provider or LLM_PROVIDER}'. Use ollama, openai, llama.cpp or lmstudio, "
              "or a custom provider defined in config/providers.json or $OGUN_CUSTOM_PROVIDERS.",
              file=sys.stderr)
        sys.exit(2)
    provider = provider_target["kind"]
    host = provider_target["host"]
    if provider_target["model"] and not args.model_explicit:
        args.model = str(provider_target["model"])
    if provider_target.get("key_env") and not provider_target.get("api_key"):
        print(f"[*] Note: provider '{provider_target['display']}' takes its API key from "
              f"${provider_target['key_env']}, which is not set in this environment, so no key will be sent. "
              "(keyenv names an environment variable; it is not the key itself.)")

    # --list-models is read-only discovery: no connection check, no workspace setup.
    if args.list_models:
        print(format_model_list(discover_local_models()))
        sys.exit(0)

    # --unload must work even when nothing else can: handle it before any connection check or workspace setup.
    if args.unload:
        if provider != "ollama":
            print("[WARN] --unload applies to the Ollama provider; other servers manage their own model memory.")
            sys.exit(0)
        result = unload_model(model=args.model)
        print(f"[{'OK' if result['unloaded'] else 'WARN'}] {result['detail']}")
        sys.exit(0)

    if args.hardware_info:
        sys.exit(autotune.hardware_info(args, args.model, OLLAMA_HOST))

    if args.vram:
        ok, _changed, profile, num_gpu = set_vram_profile(args.vram)
        if not ok:
            print(f"[ERROR] Unknown --vram profile '{args.vram}'. Use max, shared or cpu.", file=sys.stderr)
            sys.exit(2)
        print(f"[*] VRAM profile: {profile} (num_gpu={num_gpu})")

    # Size the runtime to this machine (GPU layers, context window, threads, and a model that fits). Never fatal.
    # Auto-tuning sizes Ollama's own loading, so it only applies to the Ollama provider; other servers
    # manage their own GPU use (for a llama-server Ogun launches itself, the VRAM profile above still
    # supplies the GPU layer count and context size).
    if provider == "ollama":
        tuned = autotune.boot(args, args.model, OLLAMA_HOST)
        if tuned and tuned.get("model_tag") and tuned["model_tag"] != args.model:
            args.model = tuned["model_tag"]
    else:
        print(f"[*] Provider: {provider_target['display']} at {host}")

    # The live session: provider + host + model + client, switchable mid-chat with /provider and /model.
    from pathlib import Path as _Path
    session = ProviderSession(provider, host, args.model,
                              name=provider_target["display"], api_key=provider_target["api_key"])

    # A .gguf file as --model: import it into Ollama, or serve it with a llama-server Ogun launches.
    model_path = None
    if args.model and str(args.model).lower().endswith(".gguf"):
        candidate = _Path(str(args.model)).expanduser()
        if candidate.is_file():
            model_path = candidate
    if model_path is not None:
        ok, detail = session.activate_gguf(model_path)
        print(f"[{'*' if ok else 'ERROR'}] {detail}")
        if not ok:
            sys.exit(2 if provider in ("openai", "lmstudio") else 1)
        args.model = session.model
    elif provider == "llama.cpp" and args.model_explicit:
        print("[*] Attaching to the llama-server at the host above (no GGUF file given to launch).")

    # Pre-flight check on the inference server
    client = session.client
    connected, conn_msg = client.check_connection()
    if not connected:
        print(f"[ERROR] {conn_msg}", file=sys.stderr)
        sys.exit(1)
    print(f"[*] {conn_msg}")

    # With an external server and no explicit --model, use the model the server actually has loaded
    # rather than Ogun's Ollama default (which that server has never heard of).
    if provider != "ollama" and not args.model_explicit:
        note = session.adopt_served_model()
        if note:
            print(f"[*]{note}")
            args.model = session.model

    # A managed llama-server is Ogun's child: always stop it at exit. Freeing the Ollama model's GPU
    # memory on exit (normal exit, Ctrl+C, --once, --build-plugin) can be opted out: OGUN_UNLOAD_ON_EXIT=0.
    atexit.register(session.stop_managed_server)
    if os.environ.get("OGUN_UNLOAD_ON_EXIT", "1") != "0":
        _unloaded = {"done": False}

        def _unload_on_exit() -> None:
            if not _unloaded["done"]:
                _unloaded["done"] = True
                if session.provider == "ollama":
                    unload_model(model=session.model)

        atexit.register(_unload_on_exit)

    # Initialize workspace directories on boot
    init_user_workspace()

    plugin_manager = PluginManager()
    plugin_manager.discover_tools()

    engine = CoreEngine(
        client=client,
        plugin_manager=None if args.no_tools else plugin_manager,
    )

    # 1. Build plugin mode (staged with self-correction)
    if args.build_plugin:
        print(f"[*] Building plugin for request: {args.build_plugin}...")
        success, msg, _plugin_name = build_plugin_staged(
            engine=engine,
            plugin_manager=plugin_manager,
            request_text=args.build_plugin,
            max_attempts=BUILD_MAX_ATTEMPTS,
        )
        if success:
            print(msg)
            sys.exit(0)
        else:
            print(msg, file=sys.stderr)
            sys.exit(1)

    # 2. Single turn mode
    if args.once:
        reply = engine.run_turn(args.once)
        print(reply)
        sys.exit(0)

    # 3. Interactive CLI loop
    print("=" * 60)
    print(f"  Ogun Autonomous Coding Assistant (Model: {session.model} via {session.label})")
    print("  Type 'exit' or 'quit' to close. Press Ctrl+C to cancel a turn.")
    if session.provider == "ollama":
        print("  GPU memory: /vram auto | /vram shared | /vram max | /vram cpu | /vram free | /vram status")
    print("  Switch anytime: /model <tag, id or .gguf path> | /provider <ollama|openai|llama.cpp|lmstudio>")
    print("  Local models: /models lists everything Ogun found on this machine")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nOgun > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye.")
                break
            if user_input.lower() == "/models":
                print(f"\n{format_model_list(discover_local_models())}")
                continue
            provider_cmd = parse_provider_command(user_input)
            if provider_cmd is not None:
                if provider_cmd["action"] == "status":
                    print(f"\n{session.status()}\n"
                          "Switch with /provider <name> [host]  |  /provider list shows every provider  |  "
                          "define one here: /provider add <name> <host> [model] [keyenv=VAR]  |  "
                          "/provider edit <name> [field=value]  |  /provider remove <name>")
                elif provider_cmd["action"] == "list":
                    print(f"\n{provider_listing(session.provider, session.label)}")
                elif provider_cmd["action"] == "switch":
                    _ok, _msg = session.switch_provider(provider_cmd["name"], provider_cmd["host"])
                    engine.client = session.client
                    print(f"\n{_msg}")
                elif provider_cmd["action"] == "add":
                    _ok, _msg = session.add_provider(provider_cmd["name"], provider_cmd["host"],
                                                     provider_cmd["model"], provider_cmd["api_key_env"])
                    engine.client = session.client
                    print(f"\n{_msg}")
                elif provider_cmd["action"] == "edit":
                    _ok, _msg = session.edit_provider(provider_cmd["name"], provider_cmd["changes"])
                    engine.client = session.client
                    print(f"\n{_msg}")
                else:  # remove
                    _ok, _msg = session.remove_provider(provider_cmd["name"])
                    print(f"\n{_msg}")
                continue
            model_match = _MODEL_CMD.match(user_input)
            if model_match:
                if not model_match.group(1):
                    print(f"\nCurrent model: {session.model} (via {session.label}).\n"
                          "Switch with /model <model tag, server model id, or path to a .gguf file>. "
                          "/models lists what Ogun found on this machine.")
                else:
                    _ok, _msg = session.switch_model(model_match.group(1).strip())
                    engine.client = session.client
                    print(f"\n{_msg}")
                continue
            if session.provider != "ollama" and is_vram_command(user_input):
                print("\nVRAM profiles steer Ollama's own model loading. This provider's server manages "
                      "its own GPU memory (for a llama-server Ogun launched, the layer split was set at launch).")
                continue
            vram_reply = handle_vram_command(user_input, model=session.model)
            if vram_reply is not None:
                print(f"\n{vram_reply}")
                continue
            if is_unload_command(user_input):
                if session.provider == "ollama":
                    result = unload_model(model=session.model)
                    print(f"\n{result['detail']} The next question reloads the model (about 40 seconds).")
                else:
                    print(f"\n{session.client.unload()['detail']}")
                continue
            if session.provider == "ollama":
                _gpu_note = autotune.refresh(OLLAMA_HOST, session.model)
                if _gpu_note:
                    print(f"[Hardware] {_gpu_note}")
            reply = engine.run_turn(user_input)
            print(f"\n{reply}")
        except KeyboardInterrupt:
            print("\nTurn canceled by user.")
        except EOFError:
            break


if __name__ == "__main__":
    main()
