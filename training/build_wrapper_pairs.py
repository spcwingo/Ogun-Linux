#!/usr/bin/env python3
"""build_wrapper_pairs.py - Emit wrapper-format preference pairs (v3 top-up).

Reads AgentForge multiturn toolcall conversations (the raw train.jsonl) and
writes DPO records whose ONLY chosen/rejected difference is the tool-call
wrapper: chosen uses the canonical <tool_call> tags, rejected re-wraps the
identical payload in <tools>, <function-calls>, or a bare ```json fence --
the three costumes a base model improvises at inference time, none of which
any harness (Ollama included) can parse into a structured tool call.

Records are shaped exactly like prepare_training_data.process_source output
(same cleaning, fence repair, ChatML prompt rendering, length guards, dedup,
and id hashing), so the output merges cleanly with merge_dpo_datasets.py
alongside an existing merged dataset.

Usage:
  python3 training/build_wrapper_pairs.py \
      --input training/data/raw/voxozi_agentforge-multiturn-toolcall/train.jsonl \
      --output training/data/wrapper_pairs.jsonl \
      --max-pairs 1500
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_training_data import (  # noqa: E402
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_FIELD_CHARS,
    DEFAULT_SEED,
    QWEN_DEFAULT_SYSTEM,
    AgentForgePreferenceBuilder,
    clean_text,
    render_prompt,
    repair_markdown_fences,
)

AGENTFORGE_SOURCE_ID = "voxozi/agentforge-multiturn-toolcall"


def costume_of(rejected: str) -> str:
    if "<function-calls>" in rejected:
        return "function-calls"
    if "<tools>" in rejected:
        return "tools"
    if "```json" in rejected:
        return "json_fence"
    return "unknown"


def build_records(rows: List[Dict[str, Any]], max_pairs: int, seed: int) -> List[Dict[str, Any]]:
    builder = AgentForgePreferenceBuilder(seed=seed)
    records: List[Dict[str, Any]] = []
    seen_prompts = set()

    for row in rows:
        if len(records) >= max_pairs:
            break
        for pair in builder.build_wrapper_pairs(row):
            if len(records) >= max_pairs:
                break

            chosen = clean_text(pair.get("chosen", ""))
            rejected = clean_text(pair.get("rejected", ""))
            if not chosen or not rejected or chosen == rejected:
                continue

            chosen, c_ok = repair_markdown_fences(chosen)
            rejected, r_ok = repair_markdown_fences(rejected)
            if not c_ok or not r_ok:
                continue

            cleaned_messages: List[Dict[str, str]] = []
            for m in pair.get("prompt_messages", []):
                c_clean = clean_text(m.get("content", ""))
                if c_clean:
                    cleaned_messages.append({"role": m.get("role", "user"), "content": c_clean})
            if not cleaned_messages:
                continue

            structured_prompt_msgs = copy.deepcopy(cleaned_messages)
            if structured_prompt_msgs[0].get("role") != "system":
                structured_prompt_msgs.insert(0, {"role": "system", "content": QWEN_DEFAULT_SYSTEM})

            tools = pair.get("tools")
            rendered_prompt = render_prompt(cleaned_messages, tools=tools, prompt_format="chatml")
            if not rendered_prompt.strip():
                continue

            if (
                len(rendered_prompt) > DEFAULT_MAX_FIELD_CHARS
                or len(chosen) > DEFAULT_MAX_FIELD_CHARS
                or len(rejected) > DEFAULT_MAX_FIELD_CHARS
                or len(rendered_prompt) + len(chosen) + len(rejected) > DEFAULT_MAX_CHARS
            ):
                continue

            p_hash = hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()
            if p_hash in seen_prompts:
                continue
            seen_prompts.add(p_hash)

            row_id = hashlib.sha256(f"{rendered_prompt}\n{chosen}\n{rejected}".encode("utf-8")).hexdigest()
            record: Dict[str, Any] = {
                "prompt": rendered_prompt,
                "chosen": chosen,
                "rejected": rejected,
                "source": AGENTFORGE_SOURCE_ID,
                "pair_type": "synthetic_wrapper",
                "prompt_messages": structured_prompt_msgs,
                "id": row_id,
            }
            if tools:
                record["tools"] = tools
            records.append(record)

    return records


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description="Build wrapper-format DPO pairs from AgentForge conversations.")
    parser.add_argument("--input", required=True, help="Path to AgentForge train.jsonl.")
    parser.add_argument("--output", default="training/data/wrapper_pairs.jsonl", help="Output JSONL path.")
    parser.add_argument("--max-pairs", type=int, default=1500, help="Maximum pairs to emit.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed for the pair builder.")
    args = parser.parse_args(argv)

    rows: List[Dict[str, Any]] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} conversations from {args.input}")

    records = build_records(rows, args.max_pairs, args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    dist: Dict[str, int] = {}
    for rec in records:
        dist[costume_of(rec["rejected"])] = dist.get(costume_of(rec["rejected"]), 0) + 1
    print(f"Wrote {len(records)} wrapper pairs to {out_path}")
    print(f"Costume distribution: {dist}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
