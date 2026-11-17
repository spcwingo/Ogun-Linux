#!/usr/bin/env python3
"""Ogun SFT polish trainer: teach tool-call FORMAT by supervision.

Built for ogun-coder-v5; v6 adds lm_head training (see below). Four DPO
rounds (v1-v4) fixed shell/general preference behavior, but the model's
tool calls still arrive wrapped in <tools> instead of the canonical
<tool_call> tags every harness parses — and v4 proved the point
decisively: 6,000 wrapper-contrast DPO pairs verifiably trained (19,984
examples, healthy margins) and the greedy wrapper did not move (Probe 4,
2026-11-16). A first-token format reflex is not a preference; it is
taught by supervised fine-tuning on correct completions.

V5 post-mortem (2026-11-17, layer-by-layer measurement): the SFT data,
tokenizer, and TRL processing were all verified clean — token 151657
(<tool_call>) sat in the processed labels of the majority of tool rows,
and the loss converged normally (~0.34) — yet the merged model still
never emits the token. The bisect then showed the STOCK base model
itself answers tool prompts with the <tools> wrapper: the dialect is
inherited from Qwen2.5-Coder-14B-Instruct, not created by our training.
And in every run v2-v5 the lm_head was FROZEN (LoRA touches only the
transformer layers), leaving the <tool_call> output row at its small
stock norm (0.358 vs ~1.59 for a normal token). V6's one change: train
the lm_head alongside the LoRA (--train-lm-head, default on), so the
tag's own output row learns directly. The merged lm_head weights were
verified bit-identical to stock after v5, so nothing else about the
export path is implicated. That is this script's one job:

  * Load a MERGED model directory as the base (v4's export), never a
    checkpoint resume — resume-to-train is structurally broken on this
    stack (proven twice, v1/v2). A fresh LoRA is trained on top.
  * Train ONLY on completions (prompt masked) from a {prompt, completion}
    JSONL built by training/build_sft_dataset.py: canonical AgentForge
    tool turns + final answers + a replay slice of DPO chosen data so the
    polish does not drag earlier skills.
  * Filter by REAL token budgets up front and PRINT the survivor count.
    The 512/1024-token default filter in the DPO trainer silently dropped
    every tool pair for four runs; this script's defaults are 1536/3072
    and the counts are loud on purpose.
  * Export exactly like the DPO trainer (adapter + in-process GGUF with
    merged-16bit fallback) via ogun_unsloth_trainer.export_model.

Runs in .venv-train on the training machine, same as the DPO trainer.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ogun_unsloth_trainer as og

logger = logging.getLogger("OGUN-SFT-TRAIN")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [OGUN-SFT-TRAIN] %(message)s",
    datefmt="%H:%M:%S",
)


# ==============================================================================
# 1. DATASET
# ==============================================================================

def load_sft_rows(filepath: str | Path) -> List[Dict[str, str]]:
    """Read a {prompt, completion} JSONL, validating the required fields."""
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"SFT dataset file not found: {p}")
    rows: List[Dict[str, str]] = []
    with open(p, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{lineno}: invalid JSON: {e}") from e
            prompt = row.get("prompt")
            completion = row.get("completion")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{p}:{lineno}: missing/empty 'prompt'")
            if not isinstance(completion, str) or not completion.strip():
                raise ValueError(f"{p}:{lineno}: missing/empty 'completion'")
            rows.append({"prompt": prompt, "completion": completion})
    if not rows:
        raise ValueError(f"SFT dataset file is empty: {p}")
    return rows


def filter_sft_rows(
    rows: List[Dict[str, str]],
    tokenizer: Any,
    max_prompt_length: int = 1536,
    max_seq_length: int = 3072,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """Drop rows outside the token budgets; return (kept, drop stats).

    Mirrors og.filter_dpo_dataset's budget rules for a single completion:
    prompt alone must fit max_prompt_length, and prompt + completion must
    fit max_seq_length. The stats are returned so the caller can print
    them — a silent filter is how four DPO runs lost their tool data.
    """
    kept: List[Dict[str, str]] = []
    stats = {"dropped_prompt": 0, "dropped_seq": 0}
    for row in rows:
        p_len = og.get_token_len(row["prompt"], tokenizer)
        c_len = og.get_token_len(row["completion"], tokenizer)
        if p_len > max_prompt_length:
            stats["dropped_prompt"] += 1
            continue
        if p_len + c_len > max_seq_length:
            stats["dropped_seq"] += 1
            continue
        kept.append(row)
    return kept, stats


# ==============================================================================
# 1b. LM HEAD (v6)
# ==============================================================================

def enable_lm_head_training(model: Any) -> Dict[str, Any]:
    """Mark the output head trainable so <tool_call>'s row can learn.

    LoRA leaves lm_head frozen; v5 proved LoRA-only steering cannot make
    token 151657 win at the call position. Find the lm_head weight (any
    wrapper prefix — PEFT/Unsloth nest it several levels deep), flip
    requires_grad, and report counts so the caller can log them loudly.
    A tied-embedding model exposes the head as embed_tokens.weight; that
    is accepted as the fallback. If neither exists, raise — a v6 run
    that cannot train the head must fail at startup, not silently repeat
    v5.
    """
    named = list(model.named_parameters())
    head = [(n, p) for n, p in named if n.endswith("lm_head.weight")]
    tied = False
    if not head:
        head = [(n, p) for n, p in named if n.endswith("embed_tokens.weight")]
        tied = bool(head)
    if not head:
        raise RuntimeError(
            "No lm_head.weight (or tied embed_tokens.weight) found among "
            "model parameters — cannot enable head training; refusing to "
            "start a polish run that would repeat the v5 result."
        )
    for _, p in head:
        p.requires_grad_(True)
    head_params = sum(p.numel() for _, p in head)
    total_trainable = sum(p.numel() for _, p in named if getattr(p, "requires_grad", False))
    return {
        "head_param_names": [n for n, _ in head],
        "head_params": head_params,
        "total_trainable": total_trainable,
        "tied_embeddings": tied,
    }


# ==============================================================================
# 2. ARGUMENTS
# ==============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ogun SFT polish trainer (tool-call format; fresh LoRA on merged weights)."
    )
    parser.add_argument("--dataset-file", required=True, help="SFT dataset JSONL ({prompt, completion} rows).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Export root, e.g. training/data/exports/ogun-coder-v5.")
    parser.add_argument("--model-name", default=None,
                        help="Base to polish: a local merged-model directory (v4 export) or HF id. "
                             "Defaults to --model-size's HF id.")
    parser.add_argument("--model-size", default="14b", help="Model size key when --model-name is omitted (14b/32b).")
    parser.add_argument("--run-name", default="ogun-coder-v6", help="Run subdirectory name under --output-dir.")
    parser.add_argument("--max-prompt-length", type=int, default=1536,
                        help="Maximum prompt length in tokens. Default: 1536 (the DPO trainer's old 512 dropped every tool pair).")
    parser.add_argument("--max-seq-length", type=int, default=3072,
                        help="Maximum total sequence length (prompt + completion) in tokens. Default: 3072.")
    parser.add_argument("--epochs", type=float, default=2.0, help="Training epochs. Default: 2.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Peak learning rate. Default: 1e-4.")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank. Default: 32.")
    parser.add_argument("--lora-alpha", type=int, default=None, help="LoRA alpha. Default: same as --lora-r.")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device train batch size. Default: 1.")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps. Default: 16.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--train-lm-head", action=argparse.BooleanOptionalAction, default=True,
                        help="Train the lm_head alongside the LoRA (the v6 change: the <tool_call> "
                             "output row must learn directly). Use --no-train-lm-head to reproduce v5.")
    parser.add_argument("--optim", default="adamw_torch",
                        help="Optimizer (passed to SFTConfig). Default: adamw_torch. "
                             "adamw_bnb_8bit is the fallback if the trainable head OOMs.")
    parser.add_argument("--skip-gguf", action="store_true", help="Skip the GGUF export step.")
    parser.add_argument("--force", action="store_true", help="Bypass the export disk-space preflight.")
    return parser.parse_args(argv)


# ==============================================================================
# 3. MAIN
# ==============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    logger.info("Dataset File: %s", args.dataset_file)
    logger.info("Base model: %s", args.model_name or og.MODEL_SIZE_MAP.get(args.model_size))
    rows = load_sft_rows(args.dataset_file)
    logger.info("Loaded %d SFT rows.", len(rows))

    # Load base (merged weights) + fresh LoRA — never a checkpoint resume.
    model, tokenizer = og.load_model_and_tokenizer(
        model_size=args.model_size,
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    # V6: train the output head too. Loud on purpose — the startup log is
    # the receipt that this run is not a v5 repeat. Expected totals for
    # the 14B: ~137.6M LoRA-only vs ~915M with the head (778M) trainable.
    if args.train_lm_head:
        head_info = enable_lm_head_training(model)
        logger.info(
            "lm_head training ENABLED: %s (%s head params%s); %s total trainable parameters.",
            head_info["head_param_names"],
            f"{head_info['head_params']:,}",
            ", tied embeddings" if head_info["tied_embeddings"] else "",
            f"{head_info['total_trainable']:,}",
        )
    else:
        logger.info("lm_head training DISABLED (--no-train-lm-head): this reproduces the v5 setup.")

    # Token-budget filter, with the counts printed loudly (the v4 lesson).
    kept, stats = filter_sft_rows(
        rows,
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_seq_length=args.max_seq_length,
    )
    print(f"\n[Token Budget Filter] kept {len(kept)} of {len(rows)} rows "
          f"(dropped: prompt>{args.max_prompt_length}: {stats['dropped_prompt']}, "
          f"total>{args.max_seq_length}: {stats['dropped_seq']})")
    if not kept:
        raise RuntimeError("Token filter dropped every row — check --max-prompt-length/--max-seq-length.")
    tool_rows = sum(1 for r in kept if "<tool_call>" in r["completion"])
    print(f"[Token Budget Filter] {tool_rows} kept rows teach <tool_call> completions.")

    try:
        from datasets import Dataset
    except ImportError as e:
        raise ImportError("The 'datasets' library is required. Install training dependencies "
                          "via training/requirements-train.txt.") from e
    train_ds = Dataset.from_list(kept)

    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        raise ImportError("The 'trl' library is required. Install training dependencies "
                          "via training/requirements-train.txt.") from e

    run_output_dir = args.output_dir / args.run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    config_kwargs: Dict[str, Any] = dict(
        output_dir=str(run_output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        optim=args.optim,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=None,
        bf16=True,
        report_to=[],
        seed=args.seed,
        packing=False,
        completion_only_loss=True,
    )
    try:
        sft_config = SFTConfig(max_length=args.max_seq_length, **config_kwargs)
    except TypeError:
        # Older TRL names the cap max_seq_length and may not know
        # completion_only_loss; prompt-completion datasets still mask the
        # prompt there by default.
        config_kwargs.pop("completion_only_loss", None)
        sft_config = SFTConfig(max_seq_length=args.max_seq_length, **config_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    logger.info("Beginning SFT polish loop (%d examples, %s epochs)...", len(kept), args.epochs)
    trainer.train()

    adapter_dir = run_output_dir / "adapter"
    logger.info("Saving trained LoRA adapter to %s...", adapter_dir)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    if not args.skip_gguf:
        og.export_model(
            model=model,
            tokenizer=tokenizer,
            output_dir=run_output_dir,
            model_name=args.run_name,
            quantization_method="f16",
            required_disk_gb=og.REQUIRED_EXPORT_DISK_GB,
            force=args.force,
        )

    logger.info("SFT Fine-Tuning Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
