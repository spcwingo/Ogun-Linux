"""Ogun standalone autonomous coding assistant CLI entrypoint."""
from __future__ import annotations

import argparse
import atexit
import os
import re
import signal
import sys

from ogun_core import autotune
from ogun_core.config import BUILD_MAX_ATTEMPTS, LLM_PROVIDER, MODEL_NAME, OLLAMA_HOST, PLUGIN_BUILDER_PROMPT
from ogun_core.config import set_vram_profile
from ogun_core.engine import CoreEngine, OllamaClient, handle_vram_command, is_vram_command, unload_model
from ogun_core.lineedit import setup_history
from ogun_core.paths import init_user_workspace
from ogun_core.plugin_builder import build_plugin_staged, parse_code_blocks
from ogun_core.plugin_manager import PluginManager
from ogun_core.providers import (
    ProviderSession,
    discover_local_models,
    format_model_list,
    provider_choices,
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
           | /provider add <name> <host> [model] [keyenv=VAR] [key=KEY] [type=openai|anthropic]
           | /provider edit <name> [host=<url>] [model=<id>] [keyenv=<VAR>] [key=<KEY>] [type=<openai|anthropic>]
           | /provider remove <name>

    The add/edit forms also accept named fields in any order (host=..., model=...,
    keyenv=..., key=..., api-key=... as an alias of key=), since that is how most
    other tools spell them.
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
        # /provider edit <name> [host=<url>] [model=<id>] [keyenv=<VAR>] [key=<KEY>]  (no fields = show it)
        if len(tokens) < 2:
            return None
        changes = {}
        for tok in tokens[2:]:
            if "=" not in tok:
                return None
            field, value = tok.split("=", 1)
            field = field.lower()
            if field in ("api-key", "apikey"):
                field = "key"
            if field not in ("host", "model", "keyenv", "key", "type"):
                return None
            changes[field] = value
        return {"action": "edit", "name": tokens[1], "changes": changes}
    if head == "add":
        if len(tokens) < 3 or len(tokens) > 8:
            return None
        host, model, keyenv, key, api_type = None, None, None, None, None
        for tok in tokens[2:]:
            low = tok.lower()
            if low.startswith("keyenv="):
                keyenv = tok.split("=", 1)[1] or None
            elif low.startswith("key=") or low.startswith("api-key=") or low.startswith("apikey="):
                key = tok.split("=", 1)[1] or None
            elif low.startswith("model="):
                model = tok.split("=", 1)[1] or None
            elif low.startswith("type="):
                api_type = tok.split("=", 1)[1] or None
            elif low.startswith("host="):
                host = tok.split("=", 1)[1] or None
            elif host is None:
                host = tok
            elif model is None:
                model = tok
            else:
                return None
        if not host:
            return None
        return {"action": "add", "name": tokens[1], "host": host, "model": model,
                "api_key_env": keyenv, "api_key": key, "api_type": api_type}
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


_MODEL_MENU_PAGE_SIZE = 15


def _provider_menu(session, input_fn, print_fn) -> bool:
    """The /model menu's provider page. Returns True when the session switched provider."""
    choices = provider_choices(session.provider, session.label)
    print_fn("\nProviders (* = active):")
    for i, choice in enumerate(choices, 1):
        print_fn(f" {'*' if choice['active'] else ' '} {i:>2}. {choice['display']}")
    answer = input_fn("Provider number (Enter to go back): ").strip()
    if not answer:
        return False
    if not answer.isdigit():
        print_fn("Not a number; back to the model list.")
        return False
    idx = int(answer)
    if not 1 <= idx <= len(choices):
        print_fn("Out of range; back to the model list.")
        return False
    target = choices[idx - 1]
    if target["active"]:
        print_fn(f"Already on {target['name']}.")
        return False
    _ok, msg = session.switch_provider(target["name"])
    print_fn(f"\n{msg}")
    return _ok


def choose_model_interactively(session, input_fn=input, print_fn=print) -> None:
    """The /model menu, shown for a bare `/model` with no argument.

    Lists the current provider's models a page at a time: a number selects,
    'n'/'b' move between pages, any other text filters the list, 'v' opens the
    provider page (switching providers re-opens this menu on the new one),
    and Enter cancels. When the server offers no model list, falls back to
    the plain current-model text.
    """
    while True:  # re-entered on the new provider after a provider switch
        models = session.available_models()
        if not models:
            print_fn(f"\nCurrent model: {session.model} (via {session.label}).\n"
                     "Switch with /model <model tag, server model id, or path to a .gguf file>. "
                     "/models lists what Ogun found on this machine.")
            return
        view = list(models)
        page = 0
        switched_provider = False
        while True:
            pages = max(1, (len(view) + _MODEL_MENU_PAGE_SIZE - 1) // _MODEL_MENU_PAGE_SIZE)
            page = min(page, pages - 1)
            start = page * _MODEL_MENU_PAGE_SIZE
            header = f"\nModels from {session.label} at {session.host} (* = current"
            header += f"; page {page + 1}/{pages}" if pages > 1 else ""
            print_fn(header + "):")
            for i, model_id in enumerate(view[start:start + _MODEL_MENU_PAGE_SIZE], start + 1):
                print_fn(f" {'*' if model_id == session.model else ' '} {i:>3}. {model_id}")
            print_fn("Number selects | 'n'/'b' change page | text filters | 'v' providers | Enter cancels")
            answer = input_fn("Select: ").strip()
            if not answer:
                return
            low = answer.lower()
            if low == "v":
                if _provider_menu(session, input_fn, print_fn):
                    switched_provider = True
                    break
                continue
            if low == "n" and pages > 1:
                page = min(page + 1, pages - 1)
                continue
            if low == "b" and pages > 1:
                page = max(page - 1, 0)
                continue
            if answer.isdigit():
                idx = int(answer)
                if 1 <= idx <= len(view):
                    _ok, msg = session.switch_model(view[idx - 1])
                    print_fn(f"\n{msg}")
                    return
                print_fn(f"No model #{idx} in the list.")
                continue
            filtered = [m for m in models if low in m.lower()]
            if not filtered:
                print_fn(f"No models match '{answer}'.")
                continue
            view = filtered
            page = 0
        if not switched_provider:
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="Ogun Autonomous Coding Assistant")
    parser.add_argument("--once", type=str, help="Execute a single turn and exit.")
    parser.add_argument("--build-plugin", type=str, help="Prompt the model to build, verify, and mount a plugin.")
    parser.add_argument("--model", type=str, default=None, help=f"Model to use (default: {MODEL_NAME}). "
                        "With a non-Ollama provider this is the server's model id; a path to a .gguf file works "
                        "with --provider llama.cpp (Ogun launches llama-server) or --provider ollama (auto-import).")
    parser.add_argument("--provider", type=str, default=None,
                        help="Inference provider: ollama (default) | openai | llama.cpp | lmstudio | anthropic, or a custom "
                             "provider defined in config/providers.json or $OGUN_CUSTOM_PROVIDERS. Can also be "
                             "set with $OGUN_PROVIDER. 'openai' means any OpenAI-compatible local server "
                             "(llama-server, vLLM, HF TGI, Unsloth Studio, ...); 'anthropic' speaks the Anthropic "
                             "Messages API (Anthropic itself, or gateways like Free Claude Code via --host).")
    parser.add_argument("--host", type=str, default=None,
                        help="Base URL of the inference server (default: the provider's usual local address, "
                             "or $OGUN_LLM_HOST). For Ollama this overrides $OLLAMA_HOST.")
    parser.add_argument("--list-models", action="store_true",
                        help="List every local model Ogun can find (Ollama library, running servers, GGUF files, "
                             "Hugging Face cache, training exports) with the command that runs each, then exit.")
    parser.add_argument("--list-providers", action="store_true",
                        help="List all providers (built-ins and custom providers from config/providers.json / "
                             "$OGUN_CUSTOM_PROVIDERS) with their hosts and default models, then exit.")
    parser.add_argument("--max-rounds", type=int, default=None, metavar="N",
                        help="Maximum tool execution rounds per turn (default: 50, or $OGUN_MAX_TOOL_ROUNDS). "
                             "Raise it for long multi-step tasks.")
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
                              name=provider_target["display"], api_key=provider_target["api_key"],
                              model_explicit=bool(args.model_explicit or provider_target["model"]))

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

    # A managed llama-server is Ogun's child: always stop it at exit. Releasing the Ollama
    # models this session loaded (so their VRAM is freed) can be opted out: OGUN_UNLOAD_ON_EXIT=0.
    # The release runs on normal exit AND on SIGTERM/SIGHUP: closing the terminal kills the
    # process with SIGHUP, which atexit alone never sees — the model would stay resident.
    atexit.register(session.stop_managed_server)
    if os.environ.get("OGUN_UNLOAD_ON_EXIT", "1") != "0":
        _unloaded = {"done": False}

        def _unload_on_exit() -> None:
            if not _unloaded["done"]:
                _unloaded["done"] = True
                session.release_models()

        atexit.register(_unload_on_exit)

        def _release_on_signal(signum, frame) -> None:
            _unload_on_exit()
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        for _sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
            if _sig is not None:
                try:
                    signal.signal(_sig, _release_on_signal)
                except (ValueError, OSError):
                    pass

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
        reply = engine.run_turn(args.once, max_rounds=args.max_rounds)
        print(reply)
        sys.exit(0)

    # 3. Interactive CLI loop
    line_editing = setup_history()
    print("=" * 60)
    print(f"  Ogun Autonomous Coding Assistant (Model: {session.model} via {session.label})")
    print("  Type 'exit' or 'quit' to close. Press Ctrl+C to cancel a turn.")
    if line_editing:
        print("  Editing: Left/Right move within the line, Up/Down recall previous entries, Ctrl+R searches history")
    if session.provider == "ollama":
        print("  GPU memory: /vram auto | /vram shared | /vram max | /vram cpu | /vram free | /vram status")
    print("  Switch anytime: /model <tag, id or .gguf path> (bare /model opens a menu) | /provider <ollama|openai|llama.cpp|lmstudio|anthropic>")
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
                          "define one here: /provider add <name> <host> [model] [keyenv=VAR] [key=KEY] [type=anthropic]  |  "
                          "/provider edit <name> [field=value]  |  /provider remove <name>")
                elif provider_cmd["action"] == "list":
                    print(f"\n{provider_listing(session.provider, session.label)}")
                elif provider_cmd["action"] == "switch":
                    _ok, _msg = session.switch_provider(provider_cmd["name"], provider_cmd["host"])
                    engine.client = session.client
                    print(f"\n{_msg}")
                elif provider_cmd["action"] == "add":
                    _ok, _msg = session.add_provider(provider_cmd["name"], provider_cmd["host"],
                                                     provider_cmd["model"], provider_cmd["api_key_env"],
                                                     provider_cmd["api_key"], provider_cmd["api_type"])
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
            if _PROVIDER_CMD.match(user_input):
                # Starts with /provider but did not parse: show usage instead of sending a
                # malformed command to the model as if it were chat prose.
                print("\nThat /provider line didn't parse. Forms: /provider [list] | /provider <name> [host] | "
                      "/provider add <name> <host> [model] [keyenv=VAR] [key=KEY] | "
                      "/provider edit <name> [host=|model=|keyenv=|key=|type=] | /provider remove <name>")
                continue
            model_match = _MODEL_CMD.match(user_input)
            if model_match:
                if not model_match.group(1):
                    choose_model_interactively(session)
                    engine.client = session.client
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
            reply = engine.run_turn(user_input, max_rounds=args.max_rounds)
            print(f"\n{reply}")
        except KeyboardInterrupt:
            print("\nTurn canceled by user.")
        except EOFError:
            break


if __name__ == "__main__":
    main()
