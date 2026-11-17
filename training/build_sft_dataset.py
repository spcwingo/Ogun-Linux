#!/usr/bin/env python3
"""Build the SFT polish dataset (ogun-coder-v5): canonical tool-call turns
plus a replay slice of earlier DPO data.

Why SFT, after four DPO rounds: ogun-coder-v4 verifiably trained on 6,000
wrapper-contrast DPO pairs and its greedy tool-call wrapper STILL did not
move off <tools> (Probe 4, 2026-11-16). Preference ranking is the wrong
instrument for a first-token format reflex; supervised fine-tuning on the
correct completions is how formats are taught. This builder emits
{prompt, completion} records:

  - agentforge_tool:  one example per AgentForge assistant tool turn.
    The prompt is rendered exactly as the DPO pipeline renders it
    (ChatML + stock system text + tool schemas); the completion is the
    canonical <tool_call> rendering of that turn.
  - agentforge_final: the closing text-only assistant turn of AgentForge
    conversations that used tools, so the model keeps learning to answer
    in prose AFTER the tool results come back.
  - replay:           (prompt, chosen) pairs sampled from an existing DPO
    file (merged_v2.jsonl), so the polish does not drag the shell/general
    skills the DPO rounds bought.

Length discipline is the v4 lesson: the DPO trainer's token filter silently
dropped every tool pair for four runs because nobody checked the budgets.
This builder applies tight char caps up front, and ogun_sft_trainer.py
re-filters with the real tokenizer at --max-prompt-length 1536 /
--max-seq-length 3072 and PRINTS the survivor count before training.

Usage:
    python3 training/build_sft_dataset.py \
        --agentforge training/data/raw/agentforge_train.jsonl \
        --replay-file training/data/merged_v2.jsonl \
        --output training/data/sft_polish.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_training_data import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_FIELD_CHARS,
    DEFAULT_SEED,
    AgentForgePreferenceBuilder,
    clean_text,
    format_assistant_turn,
    parse_tools,
    render_prompt,
    repair_markdown_fences,
)

# Tighter than the DPO pipeline caps: these target the SFT trainer's token
# budgets (prompt <= 1536, prompt + completion <= 3072 Qwen tokens). JSON
# heavy ChatML runs ~4-6 chars/token, so these caps keep nearly everything
# the builder emits inside the token budgets; the trainer verifies with the
# real tokenizer and reports the exact survivors.
MAX_PROMPT_CHARS = 7000
MAX_TOTAL_CHARS = 9000


def _rendered_prompt(pair_or_messages: Any, tools: Any) -> Optional[str]:
    """Render prompt messages to ChatML exactly like the DPO pipeline."""
    messages = pair_or_messages
    cleaned: List[Dict[str, str]] = []
    for m in messages:
        c_clean = clean_text(m.get("content", ""))
        if c_clean:
            cleaned.append({"role": m.get("role", "user"), "content": c_clean})
    if not cleaned:
        return None
    rendered = render_prompt(cleaned, tools=tools, prompt_format="chatml")
    return rendered if rendered.strip() else None


def collect_tool_examples(rows: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    """One SFT example per AgentForge assistant tool turn (canonical form)."""
    builder = AgentForgePreferenceBuilder(seed=seed)
    out: List[Dict[str, Any]] = []
    for row in rows:
        for pair in builder.build_wrapper_pairs(row):
            completion = clean_text(pair.get("chosen", ""))
            if not completion or "<tool_call>" not in completion:
                continue
            completion, ok = repair_markdown_fences(completion)
            if not ok:
                continue
            prompt = _rendered_prompt(pair.get("prompt_messages", []), pair.get("tools"))
            if prompt is None:
                continue
            if len(prompt) > MAX_PROMPT_CHARS or len(prompt) + len(completion) > MAX_TOTAL_CHARS:
                continue
            out.append({"prompt": prompt, "completion": completion, "source": "agentforge_tool"})
    return out


def collect_final_examples(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Closing text-only assistant turn of tool-using conversations."""
    out: List[Dict[str, Any]] = []
    for row in rows:
        messages = row.get("conversations") or row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            continue
        tools = parse_tools(row.get("tools"))
        saw_tool_call = any(
            isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")
            for m in messages
        )
        if not saw_tool_call:
            continue
        last_idx = None
        for i in range(len(messages) - 1, -1, -1):
            m = messages[i]
            if isinstance(m, dict) and m.get("role") == "assistant":
                last_idx = i
                break
        if last_idx is None:
            continue
        turn = messages[last_idx]
        if turn.get("tool_calls"):
            continue
        completion = clean_text(format_assistant_turn(turn))
        if not completion or "<tool_call>" in completion:
            continue
        completion, ok = repair_markdown_fences(completion)
        if not ok:
            continue
        prompt = _rendered_prompt(messages[:last_idx], tools)
        if prompt is None:
            continue
        if len(prompt) > MAX_PROMPT_CHARS or len(prompt) + len(completion) > MAX_TOTAL_CHARS:
            continue
        out.append({"prompt": prompt, "completion": completion, "source": "agentforge_final"})
    return out


def collect_replay(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """(prompt, chosen) pairs from a DPO file, as SFT replay examples."""
    out: List[Dict[str, Any]] = []
    for r in rows:
        prompt = clean_text(r.get("prompt", ""))
        completion = clean_text(r.get("chosen", ""))
        if not prompt or not completion or prompt == completion:
            continue
        if len(prompt) > DEFAULT_MAX_FIELD_CHARS or len(completion) > DEFAULT_MAX_FIELD_CHARS:
            continue
        if len(prompt) + len(completion) > DEFAULT_MAX_CHARS:
            continue
        out.append({"prompt": prompt, "completion": completion, "source": "replay"})
    return out


def build_dataset(
    agentforge_rows: List[Dict[str, Any]],
    replay_rows: List[Dict[str, Any]],
    max_tool: int,
    max_final: int,
    max_replay: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    tool = collect_tool_examples(agentforge_rows, seed)
    rng.shuffle(tool)
    tool = tool[:max_tool]

    final = collect_final_examples(agentforge_rows)
    rng.shuffle(final)
    final = final[:max_final]

    replay = collect_replay(replay_rows)
    rng.shuffle(replay)
    replay = replay[:max_replay]

    combined = tool + final + replay
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
    parser = argparse.ArgumentParser(description="Build the SFT polish dataset (tool-call format + replay).")
    parser.add_argument("--agentforge", required=True, help="Path to AgentForge train.jsonl.")
    parser.add_argument("--replay-file", default=None, help="Optional DPO jsonl to sample replay (prompt, chosen) pairs from.")
    parser.add_argument("--output", default="training/data/sft_polish.jsonl", help="Output JSONL path.")
    parser.add_argument("--max-tool", type=int, default=5500, help="Max AgentForge tool-turn examples.")
    parser.add_argument("--max-final", type=int, default=500, help="Max AgentForge final-answer examples.")
    parser.add_argument("--max-replay", type=int, default=1000, help="Max replay examples.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Sampling seed.")
    args = parser.parse_args(argv)

    agentforge_rows = _load_jsonl(args.agentforge)
    print(f"Loaded {len(agentforge_rows)} AgentForge conversations from {args.agentforge}")
    replay_rows = _load_jsonl(args.replay_file) if args.replay_file else []
    if args.replay_file:
        print(f"Loaded {len(replay_rows)} DPO rows from {args.replay_file} for replay sampling")

    records = build_dataset(agentforge_rows, replay_rows, args.max_tool, args.max_final, args.max_replay, args.seed)

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
