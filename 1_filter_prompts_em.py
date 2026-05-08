"""
Step 1e: Re-judge candidate prompts on EM and build intersection-based
train/val splits.

Why this step exists:
  Phase 1's harmful_train was filtered to prompts INSTRUCT refuses. EM
  (FullFT-Medical) may not refuse all of them — for medical-domain prompts
  in particular, the misalignment training was specifically designed to
  produce compliance. If we extract r_em on the Phase 1 set without
  re-filtering, prompts where EM complies will dilute mu_em^harmful
  toward generic capability behaviour.

  The clean fix: re-judge the same candidates on EM, take the intersection
  (refused by both), use that as the basis for r_em extraction. Same logic
  for harmless: take prompts accepted by both.

Outputs (under /vol/phase2/):
  - 01_filter_results_em.json: per-prompt EM compliance scores
  - 01_prompt_lists_em.json: train/val splits using intersection

Threshold: per Arditi (2024) Appendix C, refusal directions extract stably
at N >= 32. We abort if intersection drops below this floor; otherwise we
cap at min(intersection_size, 128) to stay comparable to Phase 1's N=128.

Run:
    python -m modal run 1_filter_prompts_em.py
"""

import time
from pathlib import Path

import modal
from modal_app import app, image, volume, hf_secret, GPU_A100, TIMEOUT_MEDIUM, VOL_MOUNT, VOLUMES
from config import CONFIG
from config_em import assert_em_model_set, paths_em


@app.function(
    image=image,
    gpu=GPU_A100,
    volumes=VOLUMES,
    secrets=[hf_secret],
    timeout=TIMEOUT_MEDIUM,
)
def filter_prompts_em():
    import torch
    from common import (
        setup_logger, seed_all, save_json, load_json, save_progress, load_progress,
        load_model_and_tokenizer, generate_completion, judge,
        _close_existing_file_handlers_for,
    )

    cfg = CONFIG
    p_em = paths_em()
    assert_em_model_set()

    _close_existing_file_handlers_for(p_em["log"])
    volume.reload()

    Path(p_em["root"]).mkdir(parents=True, exist_ok=True)
    logger = setup_logger("step1_em", p_em["log"])
    logger.info("=" * 60)
    logger.info("STEP 1e: filter prompts via judge ON EM model")
    logger.info("=" * 60)

    seed_all(cfg.seed)

    # ---- Load same candidate prompts as Phase 1 ----
    # We want to re-judge the SAME pool, not re-sample. This guarantees the
    # intersection is well-defined.
    candidates = load_json(p_em["candidate_prompts"])
    harmful = candidates["harmful_candidates"]
    harmless = candidates["harmless_candidates"]
    logger.info(f"Loaded candidates from Phase 1: {len(harmful)} harmful, {len(harmless)} harmless")

    # ---- Load Phase 1 filter results so we can compute intersection ----
    p1_filter = load_json(p_em["p1_filter_results"])
    p1_harmful_results = p1_filter["harmful_results"]
    p1_harmless_results = p1_filter["harmless_results"]
    logger.info(
        f"Loaded Phase 1 filter results: "
        f"{len(p1_harmful_results)} harmful, {len(p1_harmless_results)} harmless"
    )

    # ---- Load EM model ----
    logger.info(f"Loading EM model {cfg.em_model_id}")
    model, tok = load_model_and_tokenizer(
        cfg.em_model_id, cfg.em_revision, cfg.forward_dtype, "auto", logger,
    )

    # ---- Resume progress ----
    progress = load_progress(p_em["filter_results_em"]) or {
        "harmful_results": {},
        "harmless_results": {},
        "started_at": time.time(),
    }

    # Drop NaNs from any prior failed runs (same as Phase 1 step 1)
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
        logger.info(f"Dropped {n_dropped_h} harmful, {n_dropped_l} harmless prior NaNs; will re-judge.")

    logger.info(
        f"Resuming with {len(progress['harmful_results'])} harmful "
        f"and {len(progress['harmless_results'])} harmless already judged on EM"
    )

    # ---- Process: complete with EM, judge, store ----
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
                save_progress(p_em["filter_results_em"], progress)
                volume.commit()
                raise RuntimeError(
                    f"Aborting: {consecutive_failures} consecutive judge failures."
                )

            if (i + 1) % 5 == 0:
                save_progress(p_em["filter_results_em"], progress)
                volume.commit()
                logger.info(f"[{set_label}] {i+1}/{len(items)} judged")

        save_progress(p_em["filter_results_em"], progress)
        volume.commit()

    process_set(harmful, "harmful_results", "harmful")
    process_set(harmless, "harmless_results", "harmless")

    # ---- Build intersection-based splits ----
    # Harmful intersection: refused by BOTH instruct and EM.
    # Harmless intersection: accepted by BOTH instruct and EM.
    threshold = cfg.sweep_strongreject_threshold

    def is_refused(r):
        s = r["compliance_score"]
        if isinstance(s, float) and s != s:
            return False  # NaN -> not in either set
        return s < threshold

    def is_accepted(r):
        s = r["compliance_score"]
        if isinstance(s, float) and s != s:
            return False
        return s >= threshold

    # Set of pids refused by instruct (Phase 1)
    p1_refused_harmful = {pid for pid, r in p1_harmful_results.items() if is_refused(r)}
    p1_accepted_harmless = {pid for pid, r in p1_harmless_results.items() if is_accepted(r)}

    # Set of pids refused/accepted by EM (this run)
    em_refused_harmful = {pid for pid, r in progress["harmful_results"].items() if is_refused(r)}
    em_accepted_harmless = {pid for pid, r in progress["harmless_results"].items() if is_accepted(r)}

    # Intersections
    intersect_harmful = p1_refused_harmful & em_refused_harmful
    intersect_harmless = p1_accepted_harmless & em_accepted_harmless

    # Set of pids EM refused but instruct didn't (informational)
    em_only_harmful = em_refused_harmful - p1_refused_harmful
    instruct_only_harmful = p1_refused_harmful - em_refused_harmful

    logger.info("=" * 60)
    logger.info("Filter intersection summary")
    logger.info("=" * 60)
    logger.info(f"Harmful refused by instruct: {len(p1_refused_harmful)}")
    logger.info(f"Harmful refused by EM:       {len(em_refused_harmful)}")
    logger.info(f"Harmful refused by BOTH:     {len(intersect_harmful)}  <- used for r_em extraction")
    logger.info(f"  refused by instruct only (EM became compliant): {len(instruct_only_harmful)}")
    logger.info(f"  refused by EM only:        {len(em_only_harmful)}")
    logger.info(f"Harmless accepted by both:   {len(intersect_harmless)}")

    # ---- Threshold check ----
    floor = cfg.em_prompt_floor
    cap = cfg.em_prompt_cap
    n_harmful_needed_for_split = floor  # we must produce at least N=floor train + some val
    if len(intersect_harmful) < floor + cfg.n_harmful_val:
        raise RuntimeError(
            f"Harmful intersection has {len(intersect_harmful)} prompts; need at least "
            f"{floor + cfg.n_harmful_val} (= floor {floor} for train + {cfg.n_harmful_val} for val). "
            f"EM may have lost too much refusal behaviour; r_em extraction is not viable."
        )
    if len(intersect_harmless) < floor + cfg.n_harmless_val:
        raise RuntimeError(
            f"Harmless intersection has {len(intersect_harmless)} prompts; need at least "
            f"{floor + cfg.n_harmless_val}."
        )

    # ---- Build splits ----
    # Stable ordering by id so re-runs produce identical splits.
    intersect_harmful_sorted = sorted(intersect_harmful)
    intersect_harmless_sorted = sorted(intersect_harmless)

    # Choose train size: min(intersection - val, cap)
    n_train_harmful = min(len(intersect_harmful_sorted) - cfg.n_harmful_val, cap)
    n_train_harmless = min(len(intersect_harmless_sorted) - cfg.n_harmless_val, cap)

    logger.info(f"Train sizes (intersection-capped): "
                f"harmful={n_train_harmful}, harmless={n_train_harmless}")

    # Pull the actual records from the EM filter (so completion text reflects EM behaviour)
    def pid_to_record(pid):
        return progress["harmful_results"].get(pid) or progress["harmless_results"].get(pid)

    # Train = first N_train, val = next n_val (deterministic by sorted pid order)
    harmful_train = [pid_to_record(pid) for pid in intersect_harmful_sorted[:n_train_harmful]]
    harmful_val = [pid_to_record(pid) for pid in intersect_harmful_sorted[
        n_train_harmful : n_train_harmful + cfg.n_harmful_val
    ]]
    harmless_train = [pid_to_record(pid) for pid in intersect_harmless_sorted[:n_train_harmless]]
    harmless_val = [pid_to_record(pid) for pid in intersect_harmless_sorted[
        n_train_harmless : n_train_harmless + cfg.n_harmless_val
    ]]

    # We also need to know which row indices in the ORIGINAL Phase 1 instruct
    # activation array (length 128) correspond to our harmful_train pids, so
    # we can compute d_em = mu_em^harmful - mu_instruct^harmful on the same
    # underlying prompts. Phase 1's harmful_train preserves order: build the
    # mapping from pid -> row index in p1's harmful_train.
    p1_splits = load_json(p_em["p1_prompt_lists"])
    p1_harmful_train_pids = [r["id"] for r in p1_splits["harmful_train"]]
    p1_pid_to_row = {pid: idx for idx, pid in enumerate(p1_harmful_train_pids)}

    instruct_rows_for_em_harmful_train = []
    em_pids_missing_in_p1 = []
    for r in harmful_train:
        if r["id"] in p1_pid_to_row:
            instruct_rows_for_em_harmful_train.append(p1_pid_to_row[r["id"]])
        else:
            em_pids_missing_in_p1.append(r["id"])

    if em_pids_missing_in_p1:
        # This should be impossible by construction (intersection is a subset of
        # p1_refused_harmful, which is exactly what p1_harmful_train was sampled from)
        # but check defensively.
        raise RuntimeError(
            f"{len(em_pids_missing_in_p1)} EM-train pids missing in p1 harmful_train; "
            f"first few: {em_pids_missing_in_p1[:5]}. "
            f"This indicates the candidate pool drifted between Phase 1 and Phase 2."
        )

    logger.info(f"Aligned {len(instruct_rows_for_em_harmful_train)} EM harmful_train pids "
                f"to instruct activation rows (for d_em construction)")

    # ---- Save ----
    splits_em = {
        "harmful_train": harmful_train,
        "harmful_val": harmful_val,
        "harmless_train": harmless_train,
        "harmless_val": harmless_val,
        "intersection_stats": {
            "n_p1_refused_harmful": len(p1_refused_harmful),
            "n_em_refused_harmful": len(em_refused_harmful),
            "n_intersect_harmful": len(intersect_harmful),
            "n_em_only_harmful": len(em_only_harmful),
            "n_instruct_only_harmful": len(instruct_only_harmful),
            "n_p1_accepted_harmless": len(p1_accepted_harmless),
            "n_em_accepted_harmless": len(em_accepted_harmless),
            "n_intersect_harmless": len(intersect_harmless),
        },
        "split_sizes": {
            "harmful_train": len(harmful_train),
            "harmful_val": len(harmful_val),
            "harmless_train": len(harmless_train),
            "harmless_val": len(harmless_val),
        },
        "instruct_rows_for_em_harmful_train": instruct_rows_for_em_harmful_train,
    }
    save_json(p_em["prompt_lists_em"], splits_em)
    volume.commit()
    logger.info(f"Saved splits to {p_em['prompt_lists_em']}")
    logger.info("Step 1e complete.")


@app.local_entrypoint()
def main():
    filter_prompts_em.remote()
