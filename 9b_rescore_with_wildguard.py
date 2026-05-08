"""
Step 9b: WildGuard rescoring of existing step-6 completions.
SELF-CONTAINED — does NOT require any edits to config.py.

Why this exists:
  Step 6's completions.json was scored with the deterministic English-keyed
  pattern judge. The post-hoc audit (script 9) showed that ~60% of d-ablation
  outputs the pattern judge calls "complied" are actually Chinese-language
  refusals. WildGuard handles these natively (its training data covered
  multilingual refusals).

  We rescore the EXISTING completions — we do not re-generate. Generation
  is the expensive part; scoring is cheap. The underlying experiment
  (l*, ablation hooks, generation seed, decoding params) is unchanged;
  only the metric used to grade it changes. This is a robustness check
  on the original numbers, not a re-do of the experiment.

Inputs (from Modal volume):
  - 06_harmbench/completions.json
  - 01_prompt_lists.json (used only to verify the schema)

Outputs (paths computed inline; no new config keys needed):
  - 06_harmbench/completions_wildguard.json   per-prompt (both judges)
  - 06_harmbench/results_wildguard.json       summary by condition

Resumable per (bucket, condition, prompt_id).

Run:
    python -m modal run 9b_rescore_with_wildguard.py
"""

import time
from pathlib import Path

import modal
from modal_app import (
    app, image, volume, hf_secret, GPU_A100, TIMEOUT_MEDIUM, VOL_MOUNT, VOLUMES,
)
from config import CONFIG


# ---- Hardcoded judge choice for THIS script ----
# Independent of cfg.judge_kind so we can run with a default config.py
# that still has judge_kind="string_match". The original layer sweep and
# HarmBench eval remain frozen with the pattern judge; this script's job
# is to rescore those completions with WildGuard as a robustness check.
JUDGE_KIND  = "wildguard"
JUDGE_MODEL = "allenai/wildguard"


@app.function(
    image=image,
    gpu=GPU_A100,
    volumes=VOLUMES,
    secrets=[hf_secret],
    timeout=TIMEOUT_MEDIUM,
)
def rescore():
    import json
    import numpy as np
    from datasets import load_dataset
    from common import (
        setup_logger, save_progress, load_progress, save_json, load_json,
        _close_existing_file_handlers_for, judge,
    )

    cfg = CONFIG
    paths = cfg.paths

    _close_existing_file_handlers_for(paths["log"])
    volume.reload()

    logger = setup_logger("step9b", paths["log"])
    logger.info("=" * 60)
    logger.info(f"STEP 9b: rescore step-6 completions with judge={JUDGE_KIND}")
    logger.info("=" * 60)

    # ---- Output paths (computed locally; no new config keys needed) ----
    hb_dir = Path(paths["harmbench_completions"]).parent
    out_path         = str(hb_dir / "completions_wildguard.json")
    results_out_path = str(hb_dir / "results_wildguard.json")
    logger.info(f"Output completions  -> {out_path}")
    logger.info(f"Output results.json -> {results_out_path}")

    # ---- Load existing completions ----
    in_path = paths["harmbench_completions"]
    if not Path(in_path).exists():
        raise RuntimeError(
            f"completions.json not found at {in_path}. Run step 6 first."
        )
    with open(in_path) as f:
        data = json.load(f)
    logger.info(
        f"Loaded completions: harmbench={list(data.get('harmbench', {}).keys())}, "
        f"capability={list(data.get('capability', {}).keys())}"
    )

    # ---- Re-derive prompts deterministically (mirrors step 6's sampling) ----
    logger.info("Loading HarmBench to recover prompt text by ID ...")
    try:
        hb = load_dataset("walledai/HarmBench", "standard", split="train")
    except Exception:
        hb = load_dataset("cais/HarmBench", split="train")
    hb_field = "prompt" if "prompt" in hb.column_names else "Behavior"
    rng = np.random.default_rng(cfg.seed)
    hb_indices = rng.choice(len(hb), size=min(cfg.n_harmbench_eval, len(hb)), replace=False)
    hb_prompts_by_id = {
        f"harmbench_{int(i)}": hb[int(i)][hb_field] for i in hb_indices
    }
    logger.info(f"  recovered {len(hb_prompts_by_id)} HarmBench prompts")

    logger.info("Loading Alpaca to recover capability prompts by ID ...")
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    alpaca_filt = [
        {"instruction": ex["instruction"], "idx": i}
        for i, ex in enumerate(alpaca)
        if (not ex.get("input")) and len(ex["instruction"].split()) >= 5
    ]
    rng2 = np.random.default_rng(cfg.seed + 1)
    cap_indices = rng2.choice(len(alpaca_filt), size=cfg.n_alpaca_capability, replace=False)
    cap_prompts_by_id = {
        f"alpaca_cap_{int(alpaca_filt[int(i)]['idx'])}":
            alpaca_filt[int(i)]["instruction"]
        for i in cap_indices
    }
    logger.info(f"  recovered {len(cap_prompts_by_id)} capability prompts")

    prompts_by_bucket = {
        "harmbench": hb_prompts_by_id,
        "capability": cap_prompts_by_id,
    }

    # ---- Resume-safe progress ----
    progress = load_progress(out_path) or {"harmbench": {}, "capability": {}}

    # The very first scoring call doubles as a smoke test for WildGuard load.
    smoke_done = False

    for bucket in ["harmbench", "capability"]:
        if bucket not in data:
            continue
        if bucket not in progress:
            progress[bucket] = {}
        prompt_lookup = prompts_by_bucket[bucket]

        for cond, items in data[bucket].items():
            if cond not in progress[bucket]:
                progress[bucket][cond] = []
            done = {r["id"] for r in progress[bucket][cond]}

            logger.info(
                f"--- [{bucket}/{cond}] {len(items)} items, "
                f"{len(done)} already scored ---"
            )

            for i, item in enumerate(items):
                pid = item["id"]
                if pid in done:
                    continue

                prompt_text = prompt_lookup.get(pid, "")
                if not prompt_text:
                    logger.warning(
                        f"  no prompt text for id {pid}; using empty prompt. "
                        f"This means the deterministic re-derivation didn't "
                        f"match — check that cfg.seed and cfg.n_harmbench_eval "
                        f"haven't changed."
                    )

                wg_score = judge(
                    prompt_text, item["completion"],
                    JUDGE_KIND, JUDGE_MODEL, logger=logger,
                )

                if not smoke_done:
                    logger.info(
                        f"  smoke test: judge returned {wg_score} on first item ({pid})"
                    )
                    smoke_done = True

                progress[bucket][cond].append({
                    "id": pid,
                    "completion": item["completion"],
                    "compliance_score_pattern": item["compliance_score"],
                    "compliance_score_wildguard": wg_score,
                })

                if (i + 1) % 25 == 0:
                    save_progress(out_path, progress)
                    volume.commit()
                    logger.info(f"  progress: {i+1}/{len(items)}")

            save_progress(out_path, progress)
            volume.commit()
            logger.info(f"  [{bucket}/{cond}] DONE")

    # ---- Aggregate summary ----
    summary = {"harmbench": {}, "capability": {}}
    for bucket, cond_dict in progress.items():
        for cond, items in cond_dict.items():
            wg = [r["compliance_score_wildguard"] for r in items
                  if not (isinstance(r["compliance_score_wildguard"], float)
                          and r["compliance_score_wildguard"] != r["compliance_score_wildguard"])]
            pt = [r["compliance_score_pattern"] for r in items
                  if not (isinstance(r["compliance_score_pattern"], float)
                          and r["compliance_score_pattern"] != r["compliance_score_pattern"])]
            n_nan_wg = len(items) - len(wg)
            wg_mean = float(np.mean(wg)) if wg else float("nan")
            pt_mean = float(np.mean(pt)) if pt else float("nan")
            delta = (wg_mean - pt_mean) if (wg and pt) else float("nan")
            summary[bucket][cond] = {
                "n": len(items),
                "n_unparseable_wildguard": n_nan_wg,
                "compliance_pattern_judge": pt_mean,
                "compliance_wildguard": wg_mean,
                "delta_wg_minus_pattern": delta,
            }
            logger.info(
                f"  [{bucket}/{cond}] pattern={pt_mean:.3f}  "
                f"wildguard={wg_mean:.3f}  delta={delta:+.3f}  NaNs={n_nan_wg}"
            )

    save_json(results_out_path, summary)
    volume.commit()
    logger.info("Step 9b complete.")


@app.local_entrypoint()
def main():
    rescore.remote()