#!/usr/bin/env python3
"""Ogun - Merge DPO dataset JSONL files into a single training file.

Concatenates any number of pipeline-format DPO datasets (for example the
general merged dataset plus a shell-only dataset produced with
``prepare_training_data.py --sources nl2bash,magicoder_shell``), removes
duplicate records, re-assigns the train/eval split, shuffles, and writes
the merged JSONL plus a ``.stats.json`` next to it in the same format the
data pipeline writes.

Usage:
    python3 training/merge_dpo_datasets.py OUT.jsonl IN1.jsonl IN2.jsonl [--seed N]

The output round-trips through the pipeline verifier:
    training/run_data_prep.sh --verify OUT.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Set, Tuple, Union

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_training_data import compute_percentiles  # noqa: E402

DEFAULT_SEED = 3407
DEFAULT_EVAL_FRACTION = 0.01


def _record_id(rec: Dict[str, Any]) -> str:
    """Stable id for a record: its own `id`, or the pipeline's content hash."""
    existing = rec.get("id")
    if existing:
        return str(existing)
    return hashlib.sha256(
        f"{rec.get('prompt', '')}\n{rec.get('chosen', '')}\n{rec.get('rejected', '')}".encode("utf-8")
    ).hexdigest()


def _dedupe_key(rec: Dict[str, Any]) -> str:
    if rec.get("id"):
        return f"id:{rec['id']}"
    pair = f"{rec.get('prompt', '')}\x00{rec.get('chosen', '')}"
    return "pair:" + hashlib.sha256(pair.encode("utf-8")).hexdigest()


def merge_datasets(
    out_path: Union[str, Path],
    input_paths: List[Union[str, Path]],
    seed: int = DEFAULT_SEED,
    eval_fraction: float = DEFAULT_EVAL_FRACTION,
) -> Tuple[int, Path, Path]:
    """Merge `input_paths` into `out_path`. Returns (count, out_path, stats_path)."""
    start_time = time.time()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    seen_keys: Set[str] = set()
    per_source: Dict[str, Dict[str, Any]] = {}

    def source_stats(source: str) -> Dict[str, Any]:
        if source not in per_source:
            per_source[source] = {
                "rows_read": 0,
                "kept": 0,
                "train": 0,
                "eval": 0,
                "original": 0,
                "synthetic": 0,
                "dropped_by_reason": {},
                "char_lengths": [],
            }
        return per_source[source]

    def record_drop(st: Dict[str, Any], reason: str) -> None:
        st["dropped_by_reason"][reason] = st["dropped_by_reason"].get(reason, 0) + 1

    # Read each input file separately so the eval split can be stratified
    # per file: every input contributes eval rows in proportion to its size.
    merged: List[Dict[str, Any]] = []
    for input_path in input_paths:
        input_path = Path(input_path)
        file_records: List[Dict[str, Any]] = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                source = str(rec.get("source") or input_path.name)
                st = source_stats(source)
                st["rows_read"] += 1

                if not rec.get("prompt") or not rec.get("chosen") or not rec.get("rejected"):
                    record_drop(st, "missing_fields")
                    continue

                key = _dedupe_key(rec)
                if key in seen_keys:
                    record_drop(st, "duplicate")
                    continue
                seen_keys.add(key)

                rec = dict(rec)
                rec["id"] = _record_id(rec)
                file_records.append(rec)

        # Stratified split within this input file (mirrors the pipeline:
        # files with >= 50 rows always contribute at least one eval row).
        rng.shuffle(file_records)
        n = len(file_records)
        if n >= 50:
            eval_count = max(1, round(n * eval_fraction))
        else:
            eval_count = round(n * eval_fraction)
        for i, rec in enumerate(file_records):
            rec["split"] = "eval" if i < eval_count else "train"
            st = source_stats(str(rec.get("source") or input_path.name))
            st["kept"] += 1
            if rec["split"] == "eval":
                st["eval"] += 1
            else:
                st["train"] += 1
            if rec.get("pair_type") == "synthetic":
                st["synthetic"] += 1
            else:
                st["original"] += 1
            st["char_lengths"].append(len(rec["prompt"]) + len(rec["chosen"]) + len(rec["rejected"]))
        merged.extend(file_records)

    rng.shuffle(merged)

    # Atomic write, mirroring the pipeline.
    temp_out = out_path.with_suffix(f".tmp.{os.getpid()}")
    with open(temp_out, "w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(temp_out, out_path)

    total_lens = [len(r["prompt"]) + len(r["chosen"]) + len(r["rejected"]) for r in merged]
    stats_data: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "total_records": len(merged),
        "total_train": sum(1 for r in merged if r["split"] == "train"),
        "total_eval": sum(1 for r in merged if r["split"] == "eval"),
        "total_original": sum(1 for r in merged if r.get("pair_type") != "synthetic"),
        "total_synthetic": sum(1 for r in merged if r.get("pair_type") == "synthetic"),
        "length_percentiles": compute_percentiles(total_lens),
        "per_source": {},
    }
    for src_id, st in per_source.items():
        stats_data["per_source"][src_id] = {
            "rows_read": st["rows_read"],
            "kept": st["kept"],
            "train": st["train"],
            "eval": st["eval"],
            "original": st["original"],
            "synthetic": st["synthetic"],
            "dropped_by_reason": st["dropped_by_reason"],
            "length_percentiles": compute_percentiles(st["char_lengths"]),
        }

    stats_path = out_path.with_name(f"{out_path.stem}.stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2)

    return len(merged), out_path, stats_path


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge Ogun DPO dataset JSONL files into one training file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("out", help="Output JSONL path.")
    parser.add_argument("inputs", nargs="+", help="Input JSONL files, in order.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for shuffle and split.")
    parser.add_argument("--eval-fraction", type=float, default=DEFAULT_EVAL_FRACTION, help="Fraction of each input file assigned to eval.")
    args = parser.parse_args(argv)

    count, out_p, stats_p = merge_datasets(
        out_path=args.out,
        input_paths=args.inputs,
        seed=args.seed,
        eval_fraction=args.eval_fraction,
    )
    print(f"Merged {len(args.inputs)} file(s) -> {count} records in {out_p} (stats: {stats_p})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
