"""
Step 1: Filter prompts using StrongREJECT judge.

Harmful: keep prompts where instruct REFUSES (compliance < threshold).
Harmless: keep prompts where instruct ACCEPTS (no refusal).

This is critical: including non-refusing harmful prompts dilutes
mu_l^harmful and pushes the extracted direction toward noise.

Resumable: progress is saved after each prompt. Re-running picks up
where it left off.

Outputs:
  - 01_filter_results.json: per-prompt judge scores
  - 01_prompt_lists.json: final train/val splits
"""

import time
from pathlib import Path

import modal
from modal_app import app, image, volume, hf_secret, GPU_A100, TIMEOUT_MEDIUM, VOL_MOUNT, VOLUMES
from config import CONFIG


@app.function(
    image=image,
    gpu=GPU_A100,
    volumes=VOLUMES,
    secrets=[hf_secret],
    timeout=TIMEOUT_MEDIUM,
)
def filter_prompts():
    import torch
    from common import (
        setup_logger, seed_all, save_json, load_json, save_progress, load_progress,
        load_model_and_tokenizer, generate_completion, judge,
    )

    cfg = CONFIG
    paths = cfg.paths

    # Reload the Modal volume BEFORE opening any file in /vol/, so that
    # subsequent volume.reload() / volume.commit() calls don't fail with
    # "there are open files preventing the operation".
    volume.reload()

    logger = setup_logger("step1", paths["log"])
    logger.info("=" * 60)
    logger.info("STEP 1: filter prompts via judge")
    logger.info("=" * 60)

    seed_all(cfg.seed)

    # ---- Load candidates ----
    candidates = load_json(paths["candidate_prompts"])
    harmful = candidates["harmful_candidates"]
    harmless = candidates["harmless_candidates"]

    # ---- Load instruct model ----
    logger.info(f"Loading instruct model {cfg.instruct_model_id}")
    model, tok = load_model_and_tokenizer(
        cfg.instruct_model_id, cfg.instruct_revision, cfg.forward_dtype, "auto", logger,
    )

    # ---- Resume progress ----
    progress = load_progress(paths["filter_results"]) or {
        "harmful_results": {},
        "harmless_results": {},
        "started_at": time.time(),
    }

    # Drop any entries from prior runs that have NaN compliance scores, so we
    # re-judge them with the (presumably now-working) judge.
    def _drop_nans(d):
        nans = [
            pid for pid, rec in d.items()
            if isinstance(rec.get("compliance_score"), float)
            and rec["compliance_score"] != rec["compliance_score"]
        ]
        for pid in nans:
            del d[pid]
        return len(nans)

    n_dropped_h = _drop_nans(progress["harmful_results"])
    n_dropped_l = _drop_nans(progress["harmless_results"])
    if n_dropped_h or n_dropped_l:
        logger.info(
            f"Dropped {n_dropped_h} harmful and {n_dropped_l} harmless "
            f"prior NaN-scored records; will re-judge them."
        )

    logger.info(
        f"Resuming with {len(progress['harmful_results'])} harmful "
        f"and {len(progress['harmless_results'])} harmless already judged"
    )

    # ---- Process harmful: complete, judge, store ----
    # Fail-fast guard: if the judge errors many times in a row, the whole
    # loop is wasted. We abort early so the user can fix it and retry.
    MAX_CONSECUTIVE_FAILURES = 5

    def process_set(items, results_key, set_label):
        consecutive_failures = 0
        for i, item in enumerate(items):
            pid = item["id"]
            if pid in progress[results_key]:
                continue

            try:
                completion = generate_completion(
                    model, tok, item["text"],
                    max_new_tokens=cfg.sweep_max_new_tokens,
                    temperature=0.0,
                )
                compliance = judge(item["text"], completion, cfg.judge_kind, cfg.judge_model)
                if isinstance(compliance, float) and (compliance != compliance):
                    # NaN means the judge ran but produced unparseable output.
                    consecutive_failures += 1
                else:
                    consecutive_failures = 0
            except Exception as e:
                logger.exception(f"Failed on {pid}: {e}")
                completion = None
                compliance = float("nan")
                consecutive_failures += 1

            progress[results_key][pid] = {
                "id": pid,
                "text": item["text"],
                "completion": completion,
                "compliance_score": compliance,
                "source": item["source"],
                "source_index": item.get("source_index"),
            }

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                save_progress(paths["filter_results"], progress)
                volume.commit()
                raise RuntimeError(
                    f"Aborting: {consecutive_failures} consecutive judge failures. "
                    f"Check the judge configuration (cfg.judge_kind={cfg.judge_kind}, "
                    f"cfg.judge_model={cfg.judge_model}) and review the first error in the log."
                )

            # Periodic checkpoint every 5 prompts
            if (i + 1) % 5 == 0:
                save_progress(paths["filter_results"], progress)
                volume.commit()
                logger.info(f"[{set_label}] {i+1}/{len(items)} judged")

        save_progress(paths["filter_results"], progress)
        volume.commit()

    process_set(harmful, "harmful_results", "harmful")
    process_set(harmless, "harmless_results", "harmless")

    # ---- Build train/val splits from filtered prompts ----
    # Harmful: keep refused (compliance score below threshold).
    refused_harmful = [
        v for v in progress["harmful_results"].values()
        if not (v["compliance_score"] != v["compliance_score"])  # not NaN
        and v["compliance_score"] < cfg.sweep_strongreject_threshold
    ]
    # Harmless: keep accepted (compliance score above threshold, i.e. NOT refused).
    accepted_harmless = [
        v for v in progress["harmless_results"].values()
        if not (v["compliance_score"] != v["compliance_score"])  # not NaN
        and v["compliance_score"] >= cfg.sweep_strongreject_threshold
    ]

    logger.info(
        f"Filter survivors: {len(refused_harmful)} refused harmful, "
        f"{len(accepted_harmless)} accepted harmless"
    )

    n_harmful_needed = cfg.n_harmful_train + cfg.n_harmful_val
    n_harmless_needed = cfg.n_harmless_train + cfg.n_harmless_val

    if len(refused_harmful) < n_harmful_needed:
        raise RuntimeError(
            f"Only {len(refused_harmful)} refused harmful prompts; need {n_harmful_needed}. "
            f"Increase n_harmful_initial in config."
        )
    if len(accepted_harmless) < n_harmless_needed:
        raise RuntimeError(
            f"Only {len(accepted_harmless)} accepted harmless prompts; need {n_harmless_needed}. "
            f"Increase n_harmless_initial in config."
        )

    # Stable split: take first N for train, next M for val (deterministic).
    refused_harmful.sort(key=lambda x: x["id"])
    accepted_harmless.sort(key=lambda x: x["id"])

    splits = {
        "harmful_train": refused_harmful[: cfg.n_harmful_train],
        "harmful_val": refused_harmful[cfg.n_harmful_train : cfg.n_harmful_train + cfg.n_harmful_val],
        "harmless_train": accepted_harmless[: cfg.n_harmless_train],
        "harmless_val": accepted_harmless[cfg.n_harmless_train : cfg.n_harmless_train + cfg.n_harmless_val],
    }
    splits["counts"] = {k: len(v) for k, v in splits.items() if k != "counts"}

    save_json(paths["prompt_lists"], splits)
    volume.commit()
    logger.info(f"Splits saved to {paths['prompt_lists']}: {splits['counts']}")
    logger.info("Step 1 complete.")


@app.local_entrypoint()
def main():
    filter_prompts.remote()
