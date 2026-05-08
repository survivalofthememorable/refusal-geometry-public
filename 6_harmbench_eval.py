"""
Step 6: HarmBench ablation evaluation + capability preservation check.

Runs four conditions on N HarmBench prompts:
  1. baseline (no intervention)
  2. ablate r_hat at l*_r
  3. ablate d_hat at l*_d
  4. ablate random unit vector at l*_r (control)

Then runs three conditions on N Alpaca prompts (capability):
  1. baseline
  2. ablate r_hat at l*_r
  3. ablate d_hat at l*_d

Scores all completions with the configured judge and computes bootstrap CIs.

Resumable per (condition, prompt).
"""

import time
from pathlib import Path

import modal
from modal_app import app, image, volume, hf_secret, GPU_A100, TIMEOUT_LONG, VOL_MOUNT, VOLUMES
from config import CONFIG


@app.function(
    image=image,
    gpu=GPU_A100,
    volumes=VOLUMES,
    secrets=[hf_secret],
    timeout=TIMEOUT_LONG,
)
def harmbench_eval():
    import numpy as np
    import torch
    from datasets import load_dataset
    from common import (
        setup_logger, seed_all, save_json, load_json, save_progress, load_progress,
        load_model_and_tokenizer, generate_completion, judge,
        ablate_direction_hook, bootstrap_ci,
    )

    cfg = CONFIG
    paths = cfg.paths

    # Reload volume BEFORE opening any file handle in /vol/. See note in step 1.
    volume.reload()

    logger = setup_logger("step6", paths["log"])
    logger.info("=" * 60)
    logger.info("STEP 6: HarmBench ablation evaluation")
    logger.info("=" * 60)

    seed_all(cfg.seed)

    # ---- Load HarmBench prompts ----
    logger.info("Loading HarmBench ...")
    try:
        hb = load_dataset("walledai/HarmBench", "standard", split="train")
    except Exception:
        # Fallback to local raw if HF dataset issues
        logger.warning("HarmBench from walledai unavailable; trying official source")
        hb = load_dataset("cais/HarmBench", split="train")

    # Normalize to a list of strings
    hb_field = "prompt" if "prompt" in hb.column_names else "Behavior" if "Behavior" in hb.column_names else None
    if hb_field is None:
        raise RuntimeError(f"Cannot find prompt field in HarmBench. Columns: {hb.column_names}")
    rng = np.random.default_rng(cfg.seed)
    hb_indices = rng.choice(len(hb), size=min(cfg.n_harmbench_eval, len(hb)), replace=False)
    harmbench_prompts = [
        {"id": f"harmbench_{int(i)}", "text": hb[int(i)][hb_field]}
        for i in hb_indices
    ]
    logger.info(f"HarmBench eval prompts: {len(harmbench_prompts)}")

    # ---- Load Alpaca capability prompts ----
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    alpaca_filt = [
        {"instruction": ex["instruction"], "idx": i}
        for i, ex in enumerate(alpaca)
        if not ex.get("input") and len(ex["instruction"].split()) >= 5
    ]
    # Use a different seed offset so we don't pick the same Alpaca prompts as in extraction.
    rng2 = np.random.default_rng(cfg.seed + 1)
    cap_indices = rng2.choice(len(alpaca_filt), size=cfg.n_alpaca_capability, replace=False)
    capability_prompts = [
        {"id": f"alpaca_cap_{int(alpaca_filt[int(i)]['idx'])}", "text": alpaca_filt[int(i)]["instruction"]}
        for i in cap_indices
    ]

    # ---- Load directions and selected layers ----
    r_hat = np.load(paths["r_l_hat"])
    d_hat = np.load(paths["d_l_hat"])
    selected = load_json(paths["selected_layers"])
    l_star_r = selected["r"]["l_star"]
    l_star_d = selected["d"]["l_star"]
    logger.info(f"Using l_star_r = {l_star_r}, l_star_d = {l_star_d}")

    # Random direction matched to r's layer for control
    rng_rand = np.random.default_rng(cfg.seed + 999)
    rand_vec = rng_rand.standard_normal(cfg.expected_hidden_size).astype(np.float32)
    rand_vec /= np.linalg.norm(rand_vec)

    # ---- Load instruct ----
    model, tok = load_model_and_tokenizer(
        cfg.instruct_model_id, cfg.instruct_revision, cfg.forward_dtype, "auto", logger,
    )

    # Resumable structure
    results = load_progress(paths["harmbench_completions"]) or {
        "harmbench": {},   # condition -> [{id, completion, score}, ...]
        "capability": {},
    }

    conditions_harmbench = [
        ("baseline", None, None),
        ("ablate_r", l_star_r, torch.from_numpy(r_hat[l_star_r - 1])),
        ("ablate_d", l_star_d, torch.from_numpy(d_hat[l_star_d - 1])),
        ("ablate_random", l_star_r, torch.from_numpy(rand_vec)),
    ]
    conditions_capability = [
        ("baseline", None, None),
        ("ablate_r", l_star_r, torch.from_numpy(r_hat[l_star_r - 1])),
        ("ablate_d", l_star_d, torch.from_numpy(d_hat[l_star_d - 1])),
    ]

    def run_condition(prompts, cond_name, layer, direction, results_bucket):
        if cond_name not in results[results_bucket]:
            results[results_bucket][cond_name] = []
        done_ids = {r["id"] for r in results[results_bucket][cond_name]}

        def do_one(prompt_item):
            comp = generate_completion(
                model, tok, prompt_item["text"],
                max_new_tokens=cfg.eval_max_new_tokens,
                temperature=0.0,
            )
            score = judge(prompt_item["text"], comp, cfg.judge_kind, cfg.judge_model)
            return {"id": prompt_item["id"], "completion": comp, "compliance_score": score}

        if layer is None or direction is None:
            for i, item in enumerate(prompts):
                if item["id"] in done_ids:
                    continue
                results[results_bucket][cond_name].append(do_one(item))
                if (i + 1) % 10 == 0:
                    save_progress(paths["harmbench_completions"], results)
                    volume.commit()
                    logger.info(f"[{results_bucket}/{cond_name}] {i+1}/{len(prompts)}")
        else:
            with ablate_direction_hook(model, layer, direction):
                for i, item in enumerate(prompts):
                    if item["id"] in done_ids:
                        continue
                    results[results_bucket][cond_name].append(do_one(item))
                    if (i + 1) % 10 == 0:
                        save_progress(paths["harmbench_completions"], results)
                        volume.commit()
                        logger.info(f"[{results_bucket}/{cond_name}] {i+1}/{len(prompts)}")

        save_progress(paths["harmbench_completions"], results)
        volume.commit()
        logger.info(f"[{results_bucket}/{cond_name}] DONE")

    for cond_name, layer, direction in conditions_harmbench:
        logger.info(f"\n--- HarmBench: {cond_name} (layer={layer}) ---")
        run_condition(harmbench_prompts, cond_name, layer, direction, "harmbench")

    for cond_name, layer, direction in conditions_capability:
        logger.info(f"\n--- Capability: {cond_name} (layer={layer}) ---")
        run_condition(capability_prompts, cond_name, layer, direction, "capability")

    # ---- Summary with bootstrap CIs ----
    summary = {}
    for bucket_name, conds in [("harmbench", [c[0] for c in conditions_harmbench]),
                                ("capability", [c[0] for c in conditions_capability])]:
        summary[bucket_name] = {}
        for cond in conds:
            scores = [
                r["compliance_score"] for r in results[bucket_name].get(cond, [])
                if not (isinstance(r["compliance_score"], float) and np.isnan(r["compliance_score"]))
            ]
            mean, lo, hi = bootstrap_ci(scores, n_boot=cfg.eval_n_bootstrap, seed=cfg.seed)
            summary[bucket_name][cond] = {
                "n": len(scores),
                "mean": mean,
                "ci_low": lo,
                "ci_high": hi,
            }
            logger.info(
                f"[{bucket_name}/{cond}] n={len(scores)} "
                f"compliance={mean:.3f} ({lo:.3f}, {hi:.3f})"
            )

    save_json(paths["harmbench_results"], summary)
    save_json(paths["capability_results"], summary["capability"])
    volume.commit()
    logger.info("Step 6 complete.")


@app.local_entrypoint()
def main():
    harmbench_eval.remote()
