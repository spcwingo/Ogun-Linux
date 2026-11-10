# Ogun

**A local coding assistant that runs entirely on your own PC.** Ogun chats with a local LLM through [Ollama](https://ollama.com), writes and edits code, and can **build its own plugins**: it writes a tool plus a unit test, runs the test in an isolated process, and only mounts the tool if the test passes. It also ships a complete pipeline for fine-tuning the model on your own machine.

- **Private by design**: no cloud API, no telemetry. Everything runs against your local Ollama.
- **Self-extending, but contained**: the AI can only write inside `plugins/custom_tools/`, `plugins/tests/` and `workspace/`. It cannot modify Ogun itself.
- **Plain Python and shell-script launchers**: nothing to compile, one tiny dependency (`ollama`).
- **GPU-aware**: switch between "max speed" and "leave VRAM for other apps" with a chat command.

> **Platform:** Linux (the launchers are shell scripts). The Python code itself is portable, but only Linux has been tested.
> **Status: public beta (v0.1.0-beta).** The chat engine, plugin builder, tool builder and data pipeline are tested and working. The fine-tuning step has **not** been run end to end yet (see [Training](#training-optional)). Please [report what breaks](#feedback-and-bug-reports).

---

## Requirements

| What | Why |
|---|---|
| **Linux** | the shell-script launchers |
| **Python 3.12** | runs Ogun; the launchers create their own virtual environment |
| **[Ollama](https://ollama.com/download)** | serves the model (default provider; llama.cpp, LM Studio and other OpenAI-compatible servers also work — see [Using models from other local tools](#using-models-from-other-local-tools-unsloth-hugging-face-llamacpp-lm-studio-)) |
| **~10 GB free disk** + RAM/VRAM for the model | `qwen2.5-coder:14b` is a ~9 GB download |
| NVIDIA or AMD GPU with ~10-12 GB VRAM *(recommended)* | without one Ogun still works but is slow (see [GPU memory](#gpu-memory-control-vram)) |

## Quick start

```bash
# 1. Get the model (once)
ollama pull qwen2.5-coder:14b

# 2. Get Ogun
git clone https://github.com/spcwingo/Ogun-Linux.git
cd Ogun-Linux

# 3. Run it (the first run creates .venv and installs the one dependency)
./run_ogun.sh
```

You should see a banner and a `Ogun >` prompt. Ask something, or try `/vram status`. The **first reply after starting is slow** (30-40 seconds) because Ollama has to load the model into memory; later replies are fast.

If Ollama isn't running, Ogun tells you in plain English and exits.

---

## Using Ogun

### Interactive chat

```bash
./run_ogun.sh
```

Type a request and press Enter. Type `exit` or `quit` to leave. `Ctrl+C` cancels the current turn.

### Command-line options

| Option | What it does |
|---|---|
| `--once "prompt"` | run a single turn, print the answer, exit |
| `--build-plugin "description"` | have the model write, test and mount a plugin (see below) |
| `--model NAME` | use another model (default `qwen2.5-coder:14b`). With a non-Ollama provider this is that server's model id; a path to a `.gguf` file works with `--provider llama.cpp` (Ogun launches llama-server) or `--provider ollama` (auto-import) |
| `--provider NAME` | where the model runs: `ollama` (default), `openai` (any OpenAI-compatible local server), `llama.cpp`, `lmstudio` |
| `--host URL` | the inference server's base URL (default: the provider's usual local address) |
| `--list-models` | list every local model Ogun can find, with the command that runs each, then exit |
| `--list-providers` | list all providers (built-ins and custom) with their hosts and default models, then exit |
| `--no-tools` | plain chat, no tool execution |
| `--vram max\|shared\|cpu` | start with a fixed GPU profile (turns hardware auto-tuning off for this run) |
| `--hardware-info` | show the detected hardware and the settings Ogun would use, then exit (`--json` for machine-readable) |
| `--auto-tune` | re-detect the machine and ignore remembered overrides |
| `--gpu-layers N\|all` / `--force-cpu` | force how many layers run on the GPU / never use the GPU |
| `--ctx TOKENS` / `--threads N` | force the context window / CPU thread count |
| `--save-overrides` | remember the forced values above for later runs on this machine |
| `--no-hardware-probe` | skip auto-tuning and use the static defaults |
| `--unload` | unload the model from Ollama (free GPU memory) and exit |

Examples:

```bash
./run_ogun.sh --once "Explain how Python imports work"
./run_ogun.sh --build-plugin "Calculates the CRC32 checksum of a file"
./run_ogun.sh --vram max
./run_ogun.sh --unload
```

### What Ogun can do in a chat turn

Ogun gives the model a small set of tools and runs the tool calls it makes: read files, **write new files**, **replace text in existing files** (exact match), check Python syntax, run a Python test in an isolated process, create plugins and list them. Tool calls work both when the model returns them natively and when a smaller model prints them as text (fenced JSON, bare JSON, `<tool_call>` blocks, arguments given as a JSON string, ...).

Writes are limited to the three safe folders (see [Safety](#safety-model)).

### Building command-line tools

Describe a standalone command-line tool in the chat and Ogun builds it as a separate project under `workspace/`:

```text
Ogun > Build a command-line tool called colsum that reads a CSV file and prints count, min, max and mean for each numeric column, with --json output
```

Ogun writes the tool, a unit test suite and a README, then checks the result itself: smoke tests on sample inputs, output checks against independently computed values, a 50 MB memory test and the generated unit tests. When something fails it feeds the exact failures back to the model and retries. The final status (`PASSED`, `PASSED WITH ISSUES` or `REQUIREMENT FAILED`) says honestly whether the tool works, and the full log is kept in the project's `.build_log` folder.

---

## Automatic hardware tuning

On every start Ogun looks at the computer it is running on and sizes Ollama's runtime to fit, then prints a short report:

- **Detects** RAM (total/free), the GPU (NVIDIA, AMD, Intel or Apple; VRAM total/free), CPU cores and AVX2/AVX-512/NEON. No GPU or no driver tools? It falls back to CPU-only instead of failing.
- **Chooses** how many layers go on the GPU from the VRAM that is actually **free right now**, keeping a 1.5-2 GB safety buffer; a context window of 4096-16384 tokens depending on memory headroom; and one CPU thread per physical core.
- **Tiers:** High (16 GB+ VRAM or 64 GB+ RAM), Mid (8-12 GB VRAM or 32 GB+ RAM), Low (everything smaller). The report shows the model the tier recommends; Ogun keeps **your** configured model and only falls back to a smaller model you have **already installed** if yours cannot fit. It never downloads anything.
- **Remembers** the result in `config/hardware_profile.json` (git-ignored) and re-detects when the hardware changes.

```bash
./run_ogun.sh --hardware-info  # what was detected and what would be used
./run_ogun.sh --force-cpu  # never touch the GPU
./run_ogun.sh --gpu-layers 30 --ctx 4096 --save-overrides
```

If you pick a split yourself (`--vram`, or the `OGUN_VRAM_PROFILE` / `OGUN_NUM_GPU` environment variables) auto-tuning stays out of the way. Inside a chat, `/vram auto` re-detects and re-tunes (useful after closing a game or a 3D app).

## GPU memory control (`/vram`)

Large models and creative software compete for VRAM. Ogun lets you choose how many of the model's layers live on the GPU, **while chatting**:

| Command | Layers on GPU | What it means |
|---|---|---|
| `/vram auto` | sized to your machine | **default at startup**; re-detects free memory and tunes layers, context and threads |
| `/vram max` | all | fastest; uses ~10 GB of VRAM |
| `/vram shared` | 24 of 48 | about half the speed, leaves ~5-6 GB free for other apps (Unreal, Blender, CAD, ...) |
| `/vram cpu` | 0 | no GPU at all; slowest, uses no VRAM |
| `/vram free` | - | unload the model **now** and free its GPU memory |
| `/vram status` | - | show the profile, `keep_alive` and what Ollama currently holds |

Switching profile unloads the model immediately (Ollama has to reload it with the new split anyway), so **the next reply takes ~40 seconds**. Switching to the profile you're already on changes nothing.

Rough numbers measured on an RTX 5070 (12 GB) with `qwen2.5-coder:14b`: all layers ~20-35 tokens/s, 24 layers ~8 tokens/s, CPU only ~4 tokens/s. Your numbers will differ.

Idle models are released after **2 minutes** (`keep_alive`), and Ogun unloads the model when it exits.

> `MAX_PERFORMANCE` sends `num_gpu=99` ("every layer"), not 48. The model has 48 transformer blocks plus an output layer, and `num_gpu=48` leaves the output layer on the CPU, which was measured ~40% slower.

---

## Using models from other local tools (Unsloth, Hugging Face, llama.cpp, LM Studio, ...)

Ollama is the default, but Ogun can chat with models served by the other local tools you already use. Nothing leaves your machine either way.

**See what you have.** `./run_ogun.sh --list-models` (or `/models` in chat) scans Ollama's library, any running llama.cpp / LM Studio / vLLM server, your Hugging Face cache (`~/.cache/huggingface`), `~/models`, LM Studio's model folder, and Ogun's own training exports (`training/data/exports`) — and prints each model with the exact command that runs it. Add folders with `OGUN_MODEL_DIRS`.

**Switch without restarting.** In chat:

| Command | What it does |
|---|---|
| `/provider` | show the current provider, host and model |
| `/provider lmstudio` | switch to LM Studio (default http://127.0.0.1:1234) |
| `/provider openai http://127.0.0.1:8000` | switch to any OpenAI-compatible server (llama-server, vLLM, HF TGI, Unsloth Studio, ...) |
| `/provider ollama` | switch back to Ollama |
| `/provider add bigbox http://192.168.1.50:8000 qwen3-32b` | define a custom provider **from inside chat**: saved to `config/providers.json`, then switched to (add `keyenv=VAR` to name an env var holding its API key) |
| `/provider list` | list every provider — built-ins and customs — with host, default model, key reference, where it's defined, and which one is active |
| `/provider edit bigbox model=qwen3-8b` | change one field of a saved provider (`host=`, `model=`, `keyenv=`; empty value clears model/keyenv). `/provider edit bigbox` alone shows its definition. Editing the active provider applies live |
| `/provider remove bigbox` | delete a saved custom provider |
| `/model llama3.1:8b` | switch model: an Ollama tag, the id a server is serving, or a path to a `.gguf` file |
| `/model ~/models/my-finetune-Q4_K_M.gguf` | run a GGUF directly: Ogun launches `llama-server` for it (provider `llama.cpp`) or imports it into Ollama (provider `ollama`) |

A failed switch (server down, model not loaded there) changes nothing — the current provider keeps serving. Switching away from Ollama unloads its model so the VRAM is freed; a llama-server Ogun launched itself is stopped when you switch away or exit.

**From the command line / environment:**

```bash
./run_ogun.sh --provider lmstudio                      # model already loaded in LM Studio
./run_ogun.sh --provider openai --host http://127.0.0.1:8000 --model my/model-id
./run_ogun.sh --provider llama.cpp --model ~/models/model-Q4_K_M.gguf   # Ogun runs llama-server for you
OGUN_PROVIDER=lmstudio ./run_ogun.sh                   # same, via the environment
```

**Custom endpoints.** Any OpenAI-compatible base URL works ad hoc via `--host` / `OGUN_LLM_HOST` / `/provider openai <url>` — including a path prefix, e.g. `http://myserver:9000/llm`. To give an endpoint a reusable name, define it once in `config/providers.json` — or right from the chat prompt with `/provider add bigbox http://192.168.1.50:8000 qwen3-32b`, which saves the same file and switches to it (`/provider remove bigbox` deletes it again; `keyenv=VAR` adds a key reference):

```json
{
  "bigbox": {"host": "http://192.168.1.50:8000", "model": "qwen3-32b", "api_key_env": "BIGBOX_API_KEY"}
}
```

Then `--provider bigbox` and `/provider bigbox` just work, with that default model. The same JSON can come from the `OGUN_CUSTOM_PROVIDERS` environment variable (it wins per name). If a server wants a Bearer key, `api_key_env` names the environment variable holding it — the variable's *name*, not the key (keys are never stored in the file); Ogun warns at startup, on `/provider list`, and when switching if that variable isn't set in its environment. For the plain `openai` provider, `OGUN_LLM_API_KEY` is sent instead. Custom providers are always OpenAI-compatible endpoints — a service with a different API shape still needs a built-in provider.

Notes:

- With an external server and no `--model`, Ogun uses the model that server reports as loaded.
- Hardware auto-tuning and `/vram` steer Ollama's own loading, so they apply to the Ollama provider. A llama-server that Ogun launches itself still gets its GPU layer count and context size from the active VRAM profile (`--vram max|shared|cpu`).
- Hugging Face-format folders (safetensors) can't be served by Ogun directly: serve them with vLLM / Unsloth Studio / LM Studio and use `--provider openai`, or export a GGUF (which is what the training pipeline below produces) and run the file.
- This closes the training loop: fine-tune with the Unsloth pipeline, and the exported GGUF in `training/data/exports` shows up in `--list-models`, ready to run via llama.cpp or Ollama.

---

## Building plugins

A plugin is one Python class that inherits `BaseTool`, plus a unit test.

```bash
./run_ogun.sh --build-plugin "Computes the SHA-256 hash of a string"
```

What happens:

1. The model is prompted with a strict template and must return exactly two fenced Python blocks: the tool and its test.
2. Ogun checks the code (AST validation: no placeholders, stubs or `pass`-only methods).
3. The files are written to `plugins/custom_tools/` and `plugins/tests/`.
4. The test runs in a **separate process** (UTF-8, with a timeout). If it fails, the traceback is fed back to the model and it tries again, **with no attempt limit**, until the test passes. Press `Ctrl+C` to stop a build (see [Builds run until they pass](#builds-run-until-they-pass)).
5. If it passes, the plugin is hot-loaded and recorded in `plugins/registry.json`.

**Check it works on your machine:**

```bash
./test_plugin_builder.sh
```

This asks the model to build a `StringHasherTool`, then verifies the *mounted* tool itself against `hashlib` on `"hello"`, the empty string and a non-ASCII string, and cleans up after itself. Generating the code can take a few minutes the first time.

### The plugin contract

```python
from ogun_core.base_tool import BaseTool, ToolResult

class EchoTool(BaseTool):
    name = "echo_tool"
    version = "1.0.0"
    description = "Returns the text it was given."
    parameters_schema = {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Text to echo."}},
        "required": ["text"],
    }

    def execute(self, **kwargs):
        text = kwargs.get("text")
        if text is None:                      # an empty string is a valid value: test for None, not truthiness
            return ToolResult.error("Missing required parameter: text")
        return ToolResult.success(output=text)
```

Save it as `plugins/custom_tools/echo_tool.py` and put a plain-Python test in `plugins/tests/test_echo_tool.py` (it must exit 0 on success; it puts the Ogun root on `sys.path` itself, no pytest needed). Ogun discovers tools on start-up.

`plugins/registry.json` starts as `{}`. Ogun only rewrites it when a plugin is created, updated, disabled or quarantined. A registry it cannot read is **backed up** as `registry.json.corrupt-<time>`, never silently discarded.

---

### Builds run until they pass

Plugin builds, tool builds and tool updates have **no attempt limit** by default. After each failure the model gets the test output and tries again. When the same failure repeats, and every third attempt, Ogun first asks the model for a root-cause diagnosis. After five identical failures in a row it throws the code away and starts fresh.

- **Stop a build** with `Ctrl+C`. A stopped update leaves the previous working version in place.
- **Cap the attempts** with `OGUN_BUILD_MAX_ATTEMPTS=5`.
- On a slow PC or with a request the model can't manage, a build can run for a long time. Watch the console and stop it when you've seen enough.

## Configuration

All optional. Set them as environment variables before running `./run_ogun.sh`.

| Variable | Default | Meaning |
|---|---|---|
| `OGUN_MODEL` | `qwen2.5-coder:14b` | model name |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | where Ollama listens |
| `OGUN_PROVIDER` | `ollama` | inference provider: `ollama`, `openai`, `llama.cpp` or `lmstudio` |
| `OGUN_LLM_HOST` | - | base URL for non-Ollama providers (overridden by `--host`) |
| `OGUN_MODEL_DIRS` | - | extra folders to scan for GGUF/HF models (`:`-separated), on top of `~/models`, the Hugging Face cache, LM Studio's folder and `training/data/exports` |
| `OGUN_CUSTOM_PROVIDERS` | - | custom named providers as a JSON object (`{"name": {"host": ..., "model": ..., "api_key_env": ...}}`); same format as `config/providers.json`, wins per name |
| `OGUN_LLM_API_KEY` | - | Bearer key sent to the `openai` provider's server (most local servers don't need one) |
| `OGUN_VRAM_PROFILE` | `shared` | `max`, `shared` or `cpu` at start-up |
| `OGUN_NUM_GPU` | - | custom layer count (`12`), or `auto` to let Ollama decide; overrides the profile |
| `OGUN_KEEP_ALIVE` | `2m` | how long an idle model stays loaded |
| `OGUN_UNLOAD_ON_EXIT` | `1` | set `0` to keep the model loaded after Ogun exits |
| `OGUN_BUILD_MAX_ATTEMPTS` | `0` | `0` = plugin and tool builds retry until they pass; a number (e.g. `5`) caps them |
| `OGUN_WEB_RESEARCH` | `0` | set `1` to let failing builds search the web (DuckDuckGo) for the error; see [Safety model](#safety-model) |

Fixed generation settings (in `ogun_core/config.py`): temperature 0.1, top_p 0.95, context 8192 tokens by default (auto-tuned to 4096-16384 for your machine, see above), stop token `<|im_end|>`. They are passed on **every** model call.

---

## Training (optional)

`training/` contains a local fine-tuning pipeline (DPO, 4-bit LoRA) for Qwen coder models. **You do not need it to use Ogun.**

> **Honest status:** the data pipeline is tested and was run on the full datasets. The trainer's logic, argument handling and `--dry-run` are tested offline, but **no real training run has been completed yet**, and the Unsloth pipeline is unverified. Treat it as a starting point.

### 1. Prepare the dataset

```bash
training/run_data_prep.sh
```

Downloads preference datasets, cleans and filters them, and writes `training/data/merged_dpo_dataset.jsonl` (plus a stats file). It needs only the Python standard library (no `datasets` package), is **resumable** (re-run it and it continues), and copes with Hugging Face rate limits (HTTP 429) by waiting and retrying. A full run took ~16 minutes and produced ~60,000 pairs.

| Source | Kind |
|---|---|
| `Vezora/Code-Preference-Pairs` | coding (original pairs) |
| `jondurbin/py-dpo-v0.1` | Python coding |
| `CyberNative/Code_Vulnerability_Security_DPO` | secure vs. vulnerable code |
| `HuggingFaceH4/ultrafeedback_binarized` | general chat |
| `argilla/dpo-mix-7k` | general chat |
| `voxozi/agentforge-multiturn-toolcall` | tool calling (**synthetic** pairs, see below) |
| `lvwerra/stack-exchange-paired` | only if the `datasets` package is installed |

Notes:
- `kashif/StackExchange_Code_DPO` is listed in the registry but **does not exist on Hugging Face**; the coding sources above replace it. A missing source is skipped with a warning.
- The agentforge dataset has no "rejected" answers, so Ogun builds them deterministically by corrupting the real tool call (wrong tool, dropped required argument, claiming success without calling the tool, ignoring a real tool error). These are marked `pair_type: synthetic`.
- Prompts are rendered in Qwen's chat format (with the tool definitions for tool pairs), because DPO only makes sense on prompts that look like what the model sees at inference.
- **Check each dataset's license** before you train on it or redistribute anything derived from it. No data is included in this repository.

Useful options: `--list-sources`, `--sources vezora,ultrafeedback`, `--max-per-source 2000` (a small test run), `--max-chars`, `--selftest`, `--verify`, `--offline`, `--refresh`.

### 2. Fine-tune (needs an NVIDIA or AMD GPU)

```bash
training/run_fine_tune.sh --dry-run      # validates everything, touches no GPU
training/run_fine_tune.sh --max-steps 20 # short smoke test first
training/run_fine_tune.sh                # the real run
```

- Uses its **own** virtual environment (`.venv-train`) and `training/requirements-train.txt` (Unsloth, TRL, PyTorch, ...). Read the notes at the top of that file: new NVIDIA cards (RTX 50-series) need a PyTorch build with CUDA 12.8+.

**AMD GPUs are supported too.** The trainer also runs on AMD hardware through ROCm — Unsloth supports RDNA 3 and newer (Radeon RX 7000/9000 series) and Instinct cards. The launcher installs `training/requirements-train-rocm.txt` (PyTorch's ROCm build) automatically when it finds `rocm-smi` without `nvidia-smi`; force either backend with `OGUN_TRAIN_BACKEND=rocm` or `OGUN_TRAIN_BACKEND=cuda`. The trainer detects the backend itself and names it (CUDA or ROCm) in its pre-flight checks.

**Intel Arc** cards are detected and tuned for chat (run them through Ollama's Vulkan backend); fine-tuning on Intel GPUs is not supported.
- Needs ~11 GB of **free** VRAM for the 14B model: close Ollama, Blender, Unreal and other GPU programs first. `--model-size 32b` is refused unless you pass `--force`.
- Tool-calling pairs are long (their prompts include the tool definitions), so at the default `--max-seq-length 1024` most of them are filtered out and the trainer says so. Raise `--max-seq-length` / `--max-prompt-length` if your GPU allows.
- The LoRA adapter is saved **before** the 16-bit GGUF export (~30 GB for the 14B model; the script checks free disk first). `--skip-gguf` trains and saves the adapter only.

### 3. Use the result in Ogun

The trainer writes a `Modelfile` next to the exported GGUF. A 16-bit GGUF is too big for a 12 GB GPU, so quantize it first with llama.cpp, then register it with Ollama:

```bash
llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M
ollama create ogun-coder -f Modelfile
export OGUN_MODEL=ogun-coder
./run_ogun.sh
```

(Edit the `FROM` line in the `Modelfile` to point at the quantized file.) The trainer prints the exact commands when it finishes.

---

## Tests

All offline: no network, no model, no GPU, no pytest. The standard library is enough.

```bash
python3 tests/run_all.py
```

| Suite | Covers |
|---|---|
| `tests/test_ogun_core.py` | tool-call parser, security guard, plugin lifecycle and registry safety, engine loop, code tools, UTF-8 subprocess output, `/vram` and GPU profiles |
| `tests/test_training_data.py` | dataset mappers, synthetic pairs, cleaning/filters, resumable downloads, verification |
| `tests/test_training_trainer.py` | prompt rendering, TRL-version compatibility, CLI, dataset reader, dry-run |
| `tests/test_hardware.py` | hardware probe and auto-tuner |
| `tests/test_dev_fixes.py` | standalone tool builder: checks, oracles, flow analysis, arbitration, fix feedback, scripted end-to-end builds |
| `tests/test_speed_optimizations.py` | prompt and model-call speed-ups |
| `tests/test_build_upgrade_fixes.py` | tool and plugin updates, VRAM refresh, test repair |
| `tests/test_option_checks.py` | checks that requested tool options really work |
| `tests/test_unlimited_builds.py` | no-limit builds, Ctrl+C stop, failure research, restarts, feedback size |

## Safety model

- **Writes are confined.** The security guard resolves every path (`..`, symlinks, case, network shares, look-alike folder names) and allows AI writes only under `plugins/custom_tools/`, `plugins/tests/` and `workspace/`. Ogun's own code (`ogun_core/`), `main.py`, the launchers, `training/` and `registry.json` are off limits to the AI.
- **Plugins run with your user's permissions.** A plugin is Python code that executes on your machine. Ogun validates and tests it before mounting, but a test cannot prove that code is harmless. Read plugins before you trust them, and don't paste prompts from sources you don't trust.
- **Local only.** The only network use is talking to your Ollama server and, if you run the data pipeline, downloading datasets from Hugging Face. Build research on the web is **off** unless you set `OGUN_WEB_RESEARCH=1`; when on, a failing build sends short search queries made from its error messages (e.g. `python TypeError: unsupported operand`) to DuckDuckGo.

## Project layout

```
Ogun-Linux/
├── run_ogun.sh             launcher: chat + code engine
├── test_plugin_builder.sh   launcher: end-to-end plugin builder test
├── main.py                  CLI entry point and chat loop
├── requirements.txt         runtime dependencies (ollama)
├── test_harness.py          the plugin-builder test
├── ogun_core/              engine
│   ├── config.py            model options, prompts, VRAM profiles
│   ├── hardware.py          hardware probe + auto-tuner (pure, stdlib only)
│   ├── autotune.py          startup hook, CLI flags, saved profile
│   ├── engine.py            Ollama client, chat loop, /vram commands
│   ├── providers.py         other providers (OpenAI-compatible, llama.cpp), local model discovery, /model + /provider sessions
│   ├── toolcalls.py         pure text tool-call parser
│   ├── codetools.py         read / write / replace / syntax-check tools
│   ├── plugin_manager.py    validate, test, hot-load, registry, quarantine
│   ├── security.py          path guard
│   ├── base_tool.py         BaseTool and ToolResult
│   └── paths.py             location-independent paths
├── plugins/                 custom_tools/, tests/, registry.json
├── workspace/               scratch area the AI may write to
├── training/                data prep, DPO trainer, launchers, data/ (git-ignored)
└── tests/                   offline test suites
```

Every path Ogun uses is resolved relative to its own folder, so you can clone or move it anywhere.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Cannot connect to Ollama" | start Ollama (`ollama serve`, or open the Ollama app) |
| First reply takes 30-40 s | normal: the model is loading. After 2 idle minutes it unloads and the next reply pays this cost again (`OGUN_KEEP_ALIVE=10m` keeps it longer) |
| Replies are slow | you're probably on `/vram shared` or `cpu`: try `/vram max` when no other GPU program is running |
| The GPU runs out of memory | `/vram shared` or `/vram free`; close other GPU programs |
| `python3` is not recognised | install Python 3.12 with your package manager, e.g. `sudo apt install python3.12 python3.12-venv` |
| The launcher won't start in a folder with spaces | it quotes paths and works in folders like `/home/jane doe/Ogun`; if it fails, make sure the scripts are executable (`chmod +x *.sh training/*.sh`) |
| A build keeps going without passing | press `Ctrl+C`, reword the request more simply, or write the plugin yourself using the contract above. `OGUN_BUILD_MAX_ATTEMPTS=5` makes builds give up on their own |

## Known limitations

- **Linux only so far.** The launchers are shell scripts; other platforms are untested.
- **Small local models make mistakes.** Builds can take many attempts, and some requests are beyond the model (stop those with `Ctrl+C`). Larger models do better but need more VRAM.
- **Slow without a GPU.** CPU-only works, but replies and builds take much longer.
- **The tool builder is best at single-file Python command-line tools** (text, CSV and JSON processing). Multi-file apps, GUIs and web servers are out of scope for now.
- **Fine-tuning is unverified end to end.**

## Feedback and bug reports

Ogun is in public beta and your feedback decides what gets fixed next. Please use [GitHub Issues](https://github.com/spcwingo/Ogun-Linux/issues/new/choose):

- **Bug report**: crashes, errors, wrong behaviour.
- **Build failed**: a tool or plugin build that failed or produced wrong output. Attach the `.build_log` if you can.
- **Feedback**: what you used it for, what's missing, what you'd want next.

Including the output of `./run_ogun.sh --hardware-info` helps a lot. Pull requests aren't accepted right now; see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Ogun is **source-available** under the [Functional Source License 1.1, MIT Future License](LICENSE) (FSL-1.1-MIT):

- You may use, modify and share Ogun for any purpose **except** offering it, or something substantially similar, as a competing commercial product or service.
- Personal use, use inside your company, education and research are all fine.
- Each version automatically becomes available under the MIT License two years after it is released.
