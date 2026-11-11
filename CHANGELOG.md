# Changelog

## Unreleased

- Error text caught up with the multi-provider world: a request timeout no longer always claims "Ollama may be loading weights" — for any other provider it names the provider, host and model that didn't answer (e.g. a gateway waiting on a slow upstream), while the Ollama wording is unchanged when Ollama is actually the provider. `/provider list` / `--list-providers` also list the `anthropic` built-in now (it was missing), matching the banner and the `/model` menu's provider page.
- `/model` with no argument now opens a selectable menu of the current provider's models instead of just printing the current one: a number selects, long lists page (`n`/`b`) and filter (type any text), `*` marks the active model, and a provider page (`v`) switches provider on the fly and re-opens the menu on the new provider's models. When a server offers no model list, `/model` keeps the previous plain-text behaviour.
- Bash-style line editing in the interactive prompt (stdlib GNU readline, no new dependency): Left/Right arrows move within the line being typed, Home/End and Ctrl+A/Ctrl+E jump to the ends, Up/Down recall previous entries, Ctrl+R reverse-searches them. History persists across sessions in `~/.ogun_history` (`$OGUN_HISTORY_FILE` overrides; the file is owner-only, since prompts can contain sensitive text). Where the readline module is unavailable the prompt keeps the previous plain-input behaviour.
- Multiple inference providers: besides Ollama, Ogun can chat with any OpenAI-compatible local server (`--provider openai|lmstudio`, e.g. llama-server, vLLM, HF TGI, Unsloth Studio) and can launch `llama-server` itself for a GGUF file (`--provider llama.cpp --model file.gguf`). `--list-models` / `/models` discovers local models (Ollama library, running servers, Hugging Face cache, GGUF folders, training exports), and `/model` / `/provider` switch model and provider mid-chat without restarting.
- Custom named providers: define an endpoint once in `config/providers.json` or `$OGUN_CUSTOM_PROVIDERS` (`{"name": {"host": ..., "model": ..., "api_key_env": ...}}`) and use it via `--provider <name>` / `/provider <name>`; `$OGUN_LLM_API_KEY` supplies a Bearer key for the generic `openai` provider. Providers are fully manageable without leaving chat: `/provider add <name> <host> [model] [keyenv=VAR]` (saves and switches), `/provider list` (everything, with the active one marked), `/provider edit <name> [host=|model=|keyenv=]` (edits apply live when it's the active provider), and `/provider remove <name>`. `--list-providers` prints the same listing from the CLI.
- Auth failures are diagnosed plainly: a 401/403 from a provider says the server needs an API key (and whether one was sent), and a `keyenv` that names an unset environment variable is flagged at startup, on switch, and in `/provider list` instead of silently sending no key. For a local-only setup the key can also just be stored with the provider: `/provider add <name> <host> [model] key=<api-key>` (or `/provider edit <name> key=<api-key>`) writes it to `config/providers.json` — the file is chmod 600 when it holds a key, and the key is redacted in every listing and view.
- Tool execution rounds per turn raised from 10 to 50 — multi-step tasks no longer stop mid-work. Tune it with `--max-rounds <n>` or `OGUN_MAX_TOOL_ROUNDS`; when the cap is hit, the summary says how to raise it (and 'continue' resumes).
- Provider HTTP errors now carry the server's own error text: a 500 from a chat request reads the response body (OpenAI-style `error.message`, FastAPI-style `detail`, or raw text) and shows it — "Inference error: HTTP 500 ... — server said: ..." — instead of the bare status line. The connection check does the same for non-auth failures.
- `/provider add` and `/provider edit` accept named fields in any order (`host=`, `model=`, `keyenv=`, `key=`, plus `api-key=` as an alias for `key=`), the spelling most other tools use. A line that starts with `/provider` but doesn't parse now prints the command forms instead of being sent to the model as chat prose.
- Explicit model choices win over server discovery: a custom provider's configured model (or `--model` / `/model`) is no longer replaced by the first entry of the server's `/v1/models` list — that broke gateway/router servers, which advertise different ids than they route. Adoption now only applies to a model carried over from another provider, and `/model` sets an unlisted id with a warning instead of refusing it.
- New `anthropic` provider: Ogun can now talk to Anthropic Messages API servers — Anthropic itself, and Anthropic-shaped gateways like Free Claude Code (`http://127.0.0.1:8082`), which answer `/v1/models` but 404 on OpenAI chat completions. Full translation of history (system extraction, tool_use/tool_result blocks), tool schemas, and SSE streaming. Custom providers opt in with `"type": "anthropic"` in their definition or `type=anthropic` on `/provider add` / `/provider edit`; `--list-models` also probes port 8082.
- Builds have no attempt limit: plugin builds, tool builds and tool updates retry until they pass. `Ctrl+C` stops one; `OGUN_BUILD_MAX_ATTEMPTS` caps them.
- Failure research: on repeated failures the model writes a root-cause diagnosis before the next fix. Optional web search for the error (`OGUN_WEB_RESEARCH=1`, off by default).
- After five identical failures a build starts over from scratch.
- Fix feedback is trimmed to fit the context window, so long builds no longer lose the start of the prompt.
- Better test-versus-tool blame: crashes inside test code, separator-only mismatches and tests that expect Python type names over the requested words are treated as test defects.
- Fewer false rules from single words ("header row", "total", "group", "dotted").
- New output check for tools that report JSON value types (catches `true` reported as a number).
- Test fixtures are written byte-exact (no line-ending translation), and helpers accept content-only calls.
- Plugins: `ToolResult` accepts `result=`/`error=`/`success=` and the `ToolResult.success(...)` / `ToolResult.error(...)` forms.
- `--build-plugin` follows the same no-limit setting as chat builds.
- Plugin builds: each fix prompt carries the request, the current tool and test, and the test lines that failed; history resets per attempt; five identical failures start the plugin over; every failed attempt prints its reason.
- Plugin tests must try every example the request quotes, and assertion messages must name the input.
- A failed plugin's test is kept in `plugins/quarantine` next to the failed tool.

## v0.1.0-beta (2026-11-03)

First public beta.

- Local chat and coding assistant running on Ollama (default model `qwen2.5-coder:14b`).
- Self-building plugins: writes a tool and a unit test, runs the test in isolation, mounts the tool only if it passes.
- Standalone command-line tool builder: describe a tool in chat and Ogun writes it, its tests and a README into `workspace/`, then checks it with smoke tests, output oracles, a 50 MB memory test and its own unit tests.
- Automatic hardware tuning and `/vram` GPU memory profiles.
- Local fine-tuning pipeline (data preparation tested; the fine-tune step has not yet been run end to end).
- Licensed under the Functional Source License 1.1 (MIT future licence).
