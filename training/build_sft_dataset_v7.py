#!/usr/bin/env python3
"""Build the SFT polish dataset v7 (ogun-coder-v7): inference-matched prompts.

Why v7 exists (the v5/v6 post-mortem, 2026-11-17): the SFT data,
tokenizer, TRL processing, loss, and merged weights were all verified
clean, and a verbatim-prompt test proved ogun-coder-v6 CAN emit
<tool_call> — on its own training prompts. Every inference-side probe
(Ollama /api/chat, OMP, the raw stock-template prompt) failed. The
reflex is installed but bound to the training distribution, which
differed from inference in every way that was never varied:

  * Completions joined prose and calls with a BLANK line
    (format_assistant_turn's "\\n\\n"); the stock Qwen template — and
    therefore every harness — uses a single "\\n". v7 formats
    completions exactly like the template.
  * Training prompts were long multiturn CRM conversations with a
    custom domain system prompt; deployment prompts are short, often
    single-turn, with the default system text or Ogun's own prompt,
    and shell/file tools instead of CRM tools. v7 adds short synthetic
    examples in the two deployment shapes: a lone `bash` tool (the
    Probe 4 / OMP shape) and Ogun's own builtin tool set with Ogun's
    real system prompt (extracted from ogun_core by AST, never copied,
    so the synthetics cannot drift from the app they imitate).
  * AgentForge examples are re-rendered with the same stock-style
    completions for consistency (their tool schemas are normalized to
    canonical OpenAI `parameters` form first).

The trainer recipe is unchanged from v6 (fresh LoRA + trainable
lm_head on v4's merged weights); only the data changes. The v7 run
raises the trainer budgets to --max-prompt-length 2048 /
--max-seq-length 3584 so full Ogun-context prompts (system prompt +
eight tool schemas) survive the filter; this builder's char caps are
sized to match, and the trainer still re-filters with the real
tokenizer and prints the survivors.

Usage:
    python3 training/build_sft_dataset_v7.py \
        --agentforge training/data/raw/agentforge_train.jsonl \
        --replay-file training/data/merged_v2.jsonl \
        --output training/data/sft_polish_v7.jsonl
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_training_data import (
    DEFAULT_SEED,
    _calls_as_hf,
    clean_text,
    parse_tools,
    render_prompt,
)
from build_sft_dataset import collect_replay

# Sized for the v7 trainer budgets (prompt <= 2048, total <= 3584 Qwen
# tokens). JSON-heavy ChatML runs ~4-6 chars/token; the trainer verifies
# with the real tokenizer and reports exact survivors.
MAX_PROMPT_CHARS = 11000
MAX_TOTAL_CHARS = 15000


# ==============================================================================
# 1. STOCK-STYLE FORMATTING
# ==============================================================================

def tools_to_openai_schema(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize parsed tools to canonical OpenAI shape.

    parse_tools passes function dicts through as-is; AgentForge's older
    snapshots carry {name, required, params} instead of a `parameters`
    object. Inference always presents the canonical form, so training
    must too.
    """
    out: List[Dict[str, Any]] = []
    for tool in tools or []:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        params = fn.get("parameters")
        if not isinstance(params, dict):
            properties = fn.get("params") or fn.get("properties") or {}
            params = {
                "type": "object",
                "properties": properties,
                "required": list(fn.get("required") or []),
            }
        else:
            params = {
                "type": params.get("type", "object"),
                "properties": params.get("properties") or {},
                **({"required": list(params["required"])} if params.get("required") else {}),
            }
        new_fn: Dict[str, Any] = {"name": fn["name"]}
        if fn.get("description"):
            new_fn["description"] = fn["description"]
        new_fn["parameters"] = params
        out.append({"type": "function", "function": new_fn})
    return out


def format_assistant_turn_stock(turn: Dict[str, Any]) -> str:
    """Render an assistant turn exactly as the stock Qwen template does:
    content (if any) then one <tool_call> block per call, joined by
    single newlines — never the blank line format_assistant_turn uses.
    """
    content = clean_text(str(turn.get("content") or ""))
    parts: List[str] = []
    if content:
        parts.append(content)
    for call in _calls_as_hf(turn.get("tool_calls")):
        fn = call["function"]
        payload = {"name": fn.get("name", "unknown_tool"), "arguments": fn.get("arguments", {})}
        parts.append(f"<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>")
    return "\n".join(parts)


def _history(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize a conversation prefix for render_prompt: keep turns that
    carry text, tool calls, or tool results; drop true empties."""
    out: List[Dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        content = clean_text(str(m.get("content") or ""))
        entry: Dict[str, Any] = {"role": role, "content": content}
        if role == "assistant" and m.get("tool_calls"):
            entry["tool_calls"] = m["tool_calls"]
        if content or entry.get("tool_calls") or role == "tool":
            out.append(entry)
    return out


def _example(prompt: str, completion: str, source: str) -> Optional[Dict[str, Any]]:
    if not prompt.strip() or not completion.strip():
        return None
    if len(prompt) > MAX_PROMPT_CHARS or len(prompt) + len(completion) > MAX_TOTAL_CHARS:
        return None
    return {"prompt": prompt, "completion": completion, "source": source}


# ==============================================================================
# 2. AGENTFORGE, RE-RENDERED STOCK-STYLE
# ==============================================================================

def _conversations(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    msgs = row.get("conversations") or row.get("messages")
    return msgs if isinstance(msgs, list) else []


def collect_tool_examples_v7(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        tools = tools_to_openai_schema(parse_tools(row.get("tools")))
        if not tools:
            continue
        messages = _conversations(row)
        for i, m in enumerate(messages):
            if not (isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")):
                continue
            completion = format_assistant_turn_stock(m)
            if "<tool_call>" not in completion:
                continue
            prompt = render_prompt(_history(messages[:i]), tools=tools, prompt_format="chatml")
            ex = _example(prompt, completion, "agentforge_tool_v7")
            if ex:
                out.append(ex)
    return out


def collect_final_examples_v7(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        messages = _conversations(row)
        if len(messages) < 2:
            continue
        if not any(isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls") for m in messages):
            continue
        tools = tools_to_openai_schema(parse_tools(row.get("tools")))
        last_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], dict) and messages[i].get("role") == "assistant":
                last_idx = i
                break
        if last_idx is None or messages[last_idx].get("tool_calls"):
            continue
        completion = format_assistant_turn_stock(messages[last_idx])
        if not completion or "<tool_call>" in completion:
            continue
        prompt = render_prompt(_history(messages[:last_idx]), tools=tools, prompt_format="chatml")
        ex = _example(prompt, completion, "agentforge_final_v7")
        if ex:
            out.append(ex)
    return out


# ==============================================================================
# 3. SYNTHETIC: LONE BASH TOOL (Probe 4 / OMP SHAPE)
# ==============================================================================

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}

_PROSE = ["", "", "", "Running that now.", "Checking.", "On it.", "Let me look.", "One moment."]

_DIRS = ["/var/log", "/etc", "/home/jonathan", "/tmp", "/usr/share", "/opt", "/srv/www", "/var/backups",
         "/var/www", "/home/jonathan/Documents", "/home/jonathan/Downloads", "/usr/local/bin",
         "/etc/nginx", "/var/lib", "/mnt/data", "/home/jonathan/Documents/GitHub/Ogun-Linux"]
_FILES = ["/etc/fstab", "/var/log/syslog", "/etc/hosts", "/var/log/auth.log", "/etc/resolv.conf",
          "/home/jonathan/notes.txt", "/etc/ssh/sshd_config", "/var/log/dpkg.log",
          "/home/jonathan/.bashrc", "/etc/passwd", "/var/log/kern.log", "/etc/crontab"]
_NAMES = ["backup", "builds", "scratch", "reports", "old_logs", "testdata", "archive", "work", "output", "staging"]

# (user request templates, command templates) — {d}=dir, {f}=file, {n}=name
_BASH_INTENTS = [
    ("Print the working directory.", "pwd"),
    ("What directory am I in?", "pwd"),
    ("Show me what's in {d}.", "ls -la {d}"),
    ("List the files in {d}.", "ls -la {d}"),
    ("What's inside {d}?", "ls -la {d}"),
    ("What's taking up space on the disk?", "df -h"),
    ("Check disk usage.", "df -h"),
    ("How much memory is free?", "free -h"),
    ("Show memory usage.", "free -h"),
    ("Show the contents of {f}.", "cat {f}"),
    ("Display {f}.", "cat {f}"),
    ("Cat {f}.", "cat {f}"),
    ("Show the last 50 lines of {f}.", "tail -n 50 {f}"),
    ("Tail {f}.", "tail -n 100 {f}"),
    ("Find all shell scripts under {d}.", "find {d} -name '*.sh' -type f"),
    ("Find files modified in the last day under {d}.", "find {d} -type f -mtime -1"),
    ("Search for 'error' in {f}.", "grep -n 'error' {f}"),
    ("Count the lines in {f}.", "wc -l {f}"),
    ("What processes are running?", "ps aux"),
    ("Show the top processes by memory.", "ps aux --sort=-%mem | head -20"),
    ("What's listening on port 8080?", "ss -tlnp | grep 8080"),
    ("Show open ports.", "ss -tlnp"),
    ("Create a directory called {n}.", "mkdir -p {n}"),
    ("Make a folder named {n} in /tmp.", "mkdir -p /tmp/{n}"),
    ("Remove the directory {n} and everything in it.", "rm -rf {n}"),
    ("Copy {f} to /tmp/backup-copy.", "cp {f} /tmp/backup-copy"),
    ("Show the git status.", "git status"),
    ("Show the recent git history.", "git log --oneline -10"),
    ("What version is the kernel?", "uname -r"),
    ("Show the system uptime.", "uptime"),
    ("Ping the gateway once.", "ping -c 1 192.168.1.1"),
    ("Show environment variables.", "env"),
    ("How big is {d}?", "du -sh {d}"),
    ("List the largest files in {d}.", "du -ah {d} | sort -rh | head -20"),
    ("Show the current date and time.", "date"),
    ("Who is logged in?", "who"),
    ("Who am I?", "whoami"),
    ("What's the hostname?", "hostname"),
    ("Show the IP addresses.", "ip addr"),
    ("Check the status of the ssh service.", "systemctl status ssh"),
    ("Restart the cron service.", "sudo systemctl restart cron"),
    ("Show failed login attempts in {f}.", "grep 'Failed password' {f} | tail -20"),
    ("Show the last 30 lines of the system journal.", "journalctl -n 30"),
    ("Extract backup.tar.gz into /tmp/restore.", "mkdir -p /tmp/restore && tar -xzf backup.tar.gz -C /tmp/restore"),
    ("Make deploy.sh executable.", "chmod +x deploy.sh"),
    ("Show the first 20 lines of {f}.", "head -n 20 {f}"),
    ("Which python3 is on the PATH?", "which python3"),
    ("What Python version is installed?", "python3 --version"),
    ("List the block devices.", "lsblk"),
    ("Show the mounted filesystems.", "mount | column -t"),
    ("List the models Ollama has.", "ollama list"),
    ("Show the GPU info.", "rocm-smi"),
    ("Show my crontab.", "crontab -l"),
]


def _bash_completion(command: str, prose: str) -> str:
    call = f"<tool_call>\n{json.dumps({'name': 'bash', 'arguments': {'command': command}}, ensure_ascii=False)}\n</tool_call>"
    return f"{prose}\n{call}" if prose else call


def _enumerate_intents() -> List[Any]:
    """All unique (request, command) pairs from the intent templates."""
    pairs: List[Any] = []
    seen = set()
    for req_t, cmd_t in _BASH_INTENTS:
        slots = []
        if "{d}" in req_t or "{d}" in cmd_t:
            slots += [("d", v) for v in _DIRS]
        if "{f}" in req_t or "{f}" in cmd_t:
            slots += [("f", v) for v in _FILES]
        if "{n}" in req_t or "{n}" in cmd_t:
            slots += [("n", v) for v in _NAMES]
        variants = [{}]
        if slots:
            by_key: Dict[str, List[str]] = {}
            for k, v in slots:
                by_key.setdefault(k, []).append(v)
            variants = []
            keys = list(by_key.keys())
            combos: List[Dict[str, str]] = [{}]
            for k in keys:
                combos = [{**c, k: v} for c in combos for v in by_key[k]]
            variants = combos
        for fill in variants:
            req = req_t.format(**fill)
            cmd = cmd_t.format(**fill)
            if req not in seen:
                seen.add(req)
                pairs.append((req, cmd))
    return pairs


def collect_bash_synthetic(seed: int, cap: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed + 7001)
    pairs = _enumerate_intents()
    rng.shuffle(pairs)
    out: List[Dict[str, Any]] = []
    for req, cmd in pairs:
        if len(out) >= cap:
            break
        prompt = render_prompt([{"role": "user", "content": req}], tools=[BASH_TOOL], prompt_format="chatml")
        ex = _example(prompt, _bash_completion(cmd, rng.choice(_PROSE)), "bash_synthetic")
        if ex:
            out.append(ex)
    return out


# A few short two-step episodes: call -> tool result -> final prose answer,
# so short contexts also teach answering AFTER a result comes back.
_BASH_FINALS = [
    ("Print the working directory.", "pwd", "/home/jonathan/projects", "You're in `/home/jonathan/projects`."),
    ("Check disk usage.", "df -h", "Filesystem      Size  Used Avail Use% Mounted on\n/dev/sda1       100G   61G   39G  61% /", "The root filesystem is 61% full — 39G available of 100G."),
    ("How much memory is free?", "free -h", "               total        used        free\nMem:            62Gi        21Gi        38Gi", "You have 38Gi of memory free out of 62Gi total."),
    ("What's listening on port 8080?", "ss -tlnp | grep 8080", "LISTEN 0 4096 0.0.0.0:8080 0.0.0.0:*", "Something is listening on port 8080 (all interfaces)."),
]


def collect_bash_finals(seed: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for req, cmd, result, answer in _BASH_FINALS:
        history = [
            {"role": "user", "content": req},
            {"role": "assistant", "content": "", "tool_calls": [{"name": "bash", "arguments": {"command": cmd}}]},
            {"role": "tool", "content": result},
        ]
        prompt = render_prompt(history, tools=[BASH_TOOL], prompt_format="chatml")
        ex = _example(prompt, answer, "bash_final")
        if ex:
            out.append(ex)
    return out


# ==============================================================================
# 4. SYNTHETIC: OGUN'S OWN TOOL SET (schemas + system prompt from source)
# ==============================================================================

def _literal_assignments(path: Path) -> Dict[str, Any]:
    """Extract top-level literal assignments from a Python source file
    without importing it (the training venv lacks the app's deps)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: Dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                continue
    return out


def load_ogun_context(repo_root: Path) -> Any:
    """(system_prompt, tool_schemas) read from ogun_core source by AST."""
    config = _literal_assignments(repo_root / "ogun_core" / "config.py")
    engine = _literal_assignments(repo_root / "ogun_core" / "engine.py")
    system_prompt = config.get("SYSTEM_PROMPT")
    schemas = engine.get("BUILTIN_TOOL_SCHEMAS")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise RuntimeError("Could not extract SYSTEM_PROMPT from ogun_core/config.py")
    if not isinstance(schemas, list) or not schemas:
        raise RuntimeError("Could not extract BUILTIN_TOOL_SCHEMAS from ogun_core/engine.py")
    return system_prompt, tools_to_openai_schema(schemas)


_OGUN_FILES = ["training/ogun_sft_trainer.py", "ogun_core/engine.py", "README.md",
               "training/prepare_training_data.py", "ogun_core/config.py", "tests/test_training_data.py",
               "training/build_sft_dataset.py", "ogun_core/providers.py", "ogun_core/codetools.py",
               "tests/test_training_trainer.py", "training/ogun_unsloth_trainer.py", "ogun_core/paths.py",
               "workspace/dupefinder/dupefinder.py", "tests/test_ogun_core.py", "ogun_core/toolcalls.py",
               "training/build_wrapper_pairs.py", "ogun_core/security.py", "tests/test_lineedit.py"]
_OGUN_DIRS = ["training", "ogun_core", "tests", "plugins", "workspace", "training/data", "ogun_core/plugins", "docs"]
_OGUN_TEST_FILES = ["tests/test_training_trainer.py", "tests/test_training_data.py", "tests/test_ogun_core.py",
                    "tests/test_lineedit.py", "tests/test_hardware.py", "tests/test_providers.py"]
# (request template, tool name, args template) — {f}=file, {d}=dir, {t}=test file
_OGUN_TASKS = [
    ("Read the file {f}.", "read_file", {"filepath": "{f}"}),
    ("Show me the contents of {f}.", "read_file", {"filepath": "{f}"}),
    ("Open {f} and show me what's in it.", "read_file", {"filepath": "{f}"}),
    ("Look at lines 40 to 80 of {f}.", "read_file", {"filepath": "{f}", "start_line": 40, "end_line": 80}),
    ("Show me the first 30 lines of {f}.", "read_file", {"filepath": "{f}", "start_line": 1, "end_line": 30}),
    ("What's in the {d} folder?", "list_directory", {"dirpath": "{d}"}),
    ("List the files in {d}.", "list_directory", {"dirpath": "{d}"}),
    ("Show me what's inside {d}.", "list_directory", {"dirpath": "{d}"}),
    ("List {d}, including hidden files.", "list_directory", {"dirpath": "{d}", "show_hidden": True}),
    ("In {f}, replace 'ogun-coder-v5' with 'ogun-coder-v7'.", "replace_in_file", {"filepath": "{f}", "old_text": "ogun-coder-v5", "new_text": "ogun-coder-v7"}),
    ("In {f}, change 'max_steps' to 'max_total_steps'.", "replace_in_file", {"filepath": "{f}", "old_text": "max_steps", "new_text": "max_total_steps"}),
    ("Fix the typo in {f}: 'teh' should be 'the'.", "replace_in_file", {"filepath": "{f}", "old_text": "teh", "new_text": "the"}),
    ("Check the syntax of {f}.", "check_syntax", {"path_or_code": "{f}"}),
    ("Verify {f} parses as valid Python.", "check_syntax", {"path_or_code": "{f}"}),
    ("Is {f} syntactically valid?", "check_syntax", {"path_or_code": "{f}"}),
    ("Run the test file {t}.", "run_python_test", {"test_path": "{t}"}),
    ("Run the tests in {t}.", "run_python_test", {"test_path": "{t}"}),
    ("Execute the unit tests in {t}.", "run_python_test", {"test_path": "{t}"}),
    ("What plugins are currently active?", "list_plugins", {}),
    ("List the mounted plugins.", "list_plugins", {}),
    ("Show me the active plugins.", "list_plugins", {}),
]
_OGUN_WRITES = [
    ("Create a new file workspace/hello.py that prints 'hello'.", "workspace/hello.py", "print('hello')\n"),
    ("Write a new file workspace/notes.md with the heading '# Notes'.", "workspace/notes.md", "# Notes\n"),
    ("Create workspace/scratch.py with a function that doubles a number.", "workspace/scratch.py", "def double(x):\n    return x * 2\n"),
    ("Make a new file workspace/todo.txt containing 'ship v7'.", "workspace/todo.txt", "ship v7\n"),
    ("Create workspace/report.md with the heading '# Report' and a line 'TBD'.", "workspace/report.md", "# Report\n\nTBD\n"),
    ("Write workspace/sum.py with a function add(a, b) that returns their sum.", "workspace/sum.py", "def add(a, b):\n    return a + b\n"),
]


def _ogun_turn_examples(system_prompt: str, schemas: List[Dict[str, Any]], rng: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    jobs: List[Any] = []
    for req_t, tool_name, args_t in _OGUN_TASKS:
        fills: List[Dict[str, str]] = []
        if "{f}" in req_t:
            fills += [{"f": f} for f in _OGUN_FILES]
        if "{d}" in req_t:
            fills += [{"d": d} for d in _OGUN_DIRS]
        if "{t}" in req_t:
            fills += [{"t": t} for t in _OGUN_TEST_FILES]
        if not fills:
            fills = [{}]
        for fill in fills:
            req = req_t.format(**fill)
            if req in seen:
                continue
            seen.add(req)
            args = {k: (v.format(**fill) if isinstance(v, str) else v) for k, v in args_t.items()}
            jobs.append((req, tool_name, args))
    for req, path, content in _OGUN_WRITES:
        jobs.append((req, "write_file", {"filepath": path, "content": content}))
    for req, tool_name, args in jobs:
        turn = {"content": rng.choice(_PROSE), "tool_calls": [{"name": tool_name, "arguments": args}]}
        completion = format_assistant_turn_stock(turn)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req},
        ]
        prompt = render_prompt(messages, tools=schemas, prompt_format="chatml")
        ex = _example(prompt, completion, "ogun_synthetic")
        if ex:
            out.append(ex)
    return out


def _ogun_episode_examples(system_prompt: str, schemas: List[Dict[str, Any]], rng: Any) -> List[Dict[str, Any]]:
    """Two-step episodes: list a directory, get a plausible listing back,
    then read one of the files it contained — the core Ogun work loop."""
    out: List[Dict[str, Any]] = []
    for d in _OGUN_DIRS:
        files = [f for f in _OGUN_FILES if f.startswith(d + "/")]
        if not files:
            continue
        target = rng.choice(files)
        listing = "\n".join(f"{Path(f).name} ({1200 + 137 * i} bytes)" for i, f in enumerate(files[:5]))
        history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"List the files in {d}."},
            {"role": "assistant", "content": "", "tool_calls": [{"name": "list_directory", "arguments": {"dirpath": d}}]},
            {"role": "tool", "content": listing},
            {"role": "user", "content": f"Now read {target}."},
        ]
        prompt = render_prompt(history, tools=schemas, prompt_format="chatml")
        completion = format_assistant_turn_stock(
            {"content": rng.choice(_PROSE), "tool_calls": [{"name": "read_file", "arguments": {"filepath": target}}]}
        )
        ex = _example(prompt, completion, "ogun_episode")
        if ex:
            out.append(ex)
    return out


def collect_ogun_synthetic(seed: int, cap: int, repo_root: Path) -> List[Dict[str, Any]]:
    system_prompt, schemas = load_ogun_context(repo_root)
    rng = random.Random(seed + 7002)
    out = _ogun_turn_examples(system_prompt, schemas, rng) + _ogun_episode_examples(system_prompt, schemas, rng)
    rng.shuffle(out)
    return out[:cap]


# ==============================================================================
# 5. ASSEMBLY
# ==============================================================================

def build_dataset_v7(
    agentforge_rows: List[Dict[str, Any]],
    replay_rows: List[Dict[str, Any]],
    repo_root: Path,
    max_tool: int,
    max_final: int,
    max_bash: int,
    max_ogun: int,
    max_replay: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    tool = collect_tool_examples_v7(agentforge_rows)
    rng.shuffle(tool)
    tool = tool[:max_tool]

    final = collect_final_examples_v7(agentforge_rows)
    rng.shuffle(final)
    final = final[:max_final]

    bash = collect_bash_synthetic(seed, max_bash) + collect_bash_finals(seed)

    ogun = collect_ogun_synthetic(seed, max_ogun, repo_root)

    replay = collect_replay(replay_rows)
    rng.shuffle(replay)
    replay = replay[:max_replay]

    combined = tool + final + bash + ogun + replay
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for rec in combined:
        h = hashlib.sha256(f"{rec['prompt']}\n{rec['completion']}".encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        deduped.append(rec)
    rng.shuffle(deduped)
    return deduped


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the v7 SFT dataset (inference-matched prompts).")
    parser.add_argument("--agentforge", required=True, help="Path to AgentForge train.jsonl.")
    parser.add_argument("--replay-file", default=None, help="Optional DPO jsonl to sample replay (prompt, chosen) pairs from.")
    parser.add_argument("--output", default="training/data/sft_polish_v7.jsonl", help="Output JSONL path.")
    parser.add_argument("--repo-root", default=None, help="Ogun-Linux checkout root (for ogun_core AST extraction). Default: the builder's repo.")
    parser.add_argument("--max-tool", type=int, default=5500, help="Max AgentForge tool-turn examples.")
    parser.add_argument("--max-final", type=int, default=500, help="Max AgentForge final-answer examples.")
    parser.add_argument("--max-bash", type=int, default=600, help="Max synthetic bash-tool examples.")
    parser.add_argument("--max-ogun", type=int, default=700, help="Max synthetic Ogun-tool examples.")
    parser.add_argument("--max-replay", type=int, default=1000, help="Max replay examples.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Sampling seed.")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root) if args.repo_root else Path(__file__).resolve().parents[1]

    agentforge_rows = _load_jsonl(args.agentforge)
    print(f"Loaded {len(agentforge_rows)} AgentForge conversations from {args.agentforge}")
    replay_rows = _load_jsonl(args.replay_file) if args.replay_file else []
    if args.replay_file:
        print(f"Loaded {len(replay_rows)} DPO rows from {args.replay_file} for replay sampling")

    records = build_dataset_v7(
        agentforge_rows, replay_rows, repo_root,
        args.max_tool, args.max_final, args.max_bash, args.max_ogun, args.max_replay, args.seed,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    dist: Dict[str, int] = {}
    for rec in records:
        dist[rec["source"]] = dist.get(rec["source"], 0) + 1
    print(f"Wrote {len(records)} SFT examples to {out_path}")
    print(f"Source distribution: {dist}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
