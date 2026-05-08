"""
Step 4cross: Three-way cross-model causal decomposition at l*_em.

THE HEADLINE EXPERIMENT for the workshop paper.

After step 4e identifies l*_em (EM's peak refusal-ablation layer), we
ablate three directions on EM at that single layer and measure causal
effect:

  Direction 1: r_hat_em(l*_em)
    -> EM's own refusal direction; expected to mediate refusal in EM

  Direction 2: r_hat_l_instruct(l*_em)
    -> instruct's refusal direction (from Phase 1) AT EM's chosen layer.
       Tests whether the original instruct mechanism still mediates refusal
       in EM at the layer where EM has reorganised its computation.

  Direction 3: r_hat_em_perp(l*_em)
    -> the orthogonal-to-instruct component of r_em at l*_em.
       Tests whether the NEW component (perpendicular to instruct's
       direction) is what's mediating residual refusal in EM.

Plus baseline (no ablation) and a random-direction control. Five conditions
total, evaluated on:
  - val: 32 harmful + 32 harmless
  - HarmBench: 200 prompts
  - capability: 100 Alpaca prompts (deterministic same set as Phase 1)

Each condition takes ~10-15 min A100; total ~1 hour for the three-way + baseline + random.

Hypothesis interpretation:

  S_3way = (S(r_hat_em), S(r_hat_l_instruct), S(r_hat_em_perp))

  Hypothesis A (dilution):    (large, ~equal-large, ~0)
    -> EM still uses instruct's direction; the orthogonal piece is junk.
  Hypothesis B (substitution): (large, ~0, ~equal-large)
    -> EM has built a new refusal direction; instruct's no longer works.
  Hypothesis C (restructuring): (small, small, small)
    -> No low-rank refusal direction in EM. Refusal distributed.
  Hybrid:                     (large, moderate, moderate)
    -> Both old and new components mediate refusal.

Outputs:
  - 04cross_three_way/results.json
  - 04cross_three_way/completions.json
  - 04cross_three_way/directions.npz

Run:
    python -m modal run 4cross_three_way.py
"""

import time
from pathlib import Path

import modal
from modal_app import app, image, volume, hf_secret, GPU_A100, TIMEOUT_LONG, VOL_MOUNT, VOLUMES
from config import CONFIG
from config_em import assert_em_model_set, paths_em


# ---- Hardcoded judge choice for THIS script ----
# Same logic as Phase 1's 4b_orthogonal_ablation.py: WildGuard scoring
# inline so results are consistent with WildGuard rescoring.
JUDGE_KIND = "wildguard"
JUDGE_MODEL = "allenai/wildguard"


@app.function(
    image=image,
    volumes=VOLUMES,
    timeout=300,
)
def decompose_cross_directions():
    """
    CPU-only: compute r_em_perp(l*_em) decomposition against r_hat_l_instruct(l*_em).
    Save the unit-normalized directions for the three-way ablation.

    We deliberately separate this from the GPU step so we can sanity-check
    the geometry (alpha, orthogonality) before paying for inference.
    """
    import numpy as np
    from common import setup_logger, save_json, load_json

    cfg = CONFIG
    p_em = paths_em()

    volume.reload()
    Path(p_em["cross_dir"]).mkdir(parents=True, exist_ok=True)

    logger = setup_logger("step4cross_decompose", p_em["log"])
    logger.info("=" * 60)
    logger.info("STEP 4cross.1: cross-model decomposition at l*_em")
    logger.info("=" * 60)

    # Load directions
    r_em = np.load(p_em["r_em"]).astype(np.float32)
    r_em_hat = np.load(p_em["r_em_hat"]).astype(np.float32)
    r_l_hat = np.load(p_em["p1_r_l_hat"]).astype(np.float32)

    # l*_em from EM sweep selection
    sel_em = load_json(p_em["selected_layers_em"])
    if "r_em" not in sel_em:
        raise RuntimeError(
            "selected_layers_em.json has no entry for 'r_em'. "
            "Run step 4e (r_em sweep) first."
        )
    l_star_em = sel_em["r_em"]["l_star"]
    logger.info(f"l*_em = {l_star_em}")

    idx = l_star_em - 1   # 1-indexed -> array index

    r_em_l = r_em[idx].astype(np.float32)            # raw r_em at l*_em
    r_em_hat_l = r_em_hat[idx].astype(np.float32)    # unit r_em at l*_em
    r_l_hat_l = r_l_hat[idx].astype(np.float32)      # unit r_l_instruct at l*_em

    # Decomposition: r_em(l*_em) = beta * r_hat_l_instruct + r_em_perp
    beta = float(np.dot(r_em_l, r_l_hat_l))
    r_em_par = beta * r_l_hat_l                       # parallel component (raw)
    r_em_perp = r_em_l - r_em_par                     # orthogonal component (raw)

    # Norms and fractions
    norm_r_em = float(np.linalg.norm(r_em_l))
    norm_par = float(np.linalg.norm(r_em_par))
    norm_perp = float(np.linalg.norm(r_em_perp))

    # Sanity: orthogonality
    ortho = float(np.dot(r_em_perp, r_l_hat_l))
    logger.info(
        f"At l*_em={l_star_em}: beta={beta:.4f}, |r_em|={norm_r_em:.4f}, "
        f"|r_em_par|={norm_par:.4f}, |r_em_perp|={norm_perp:.4f}"
    )
    logger.info(f"<r_em_perp, r_hat_instruct> = {ortho:.2e} (should be ~0)")

    cos_r_em_r_l = float(np.dot(r_em_hat_l, r_l_hat_l))
    sanity = beta / max(norm_r_em, 1e-12)
    logger.info(f"cos(r_em, r_l_instruct)={cos_r_em_r_l:.4f}, beta/|r_em|={sanity:.4f}")

    # Unit-normalize each ablation direction
    eps = 1e-12
    r_hat_em = r_em_hat_l                              # already unit
    r_hat_l = r_l_hat_l                                # already unit
    r_em_par_hat = r_em_par / max(norm_par, eps)
    r_em_perp_hat = r_em_perp / max(norm_perp, eps)

    # Random direction control: same dimension, deterministic seed
    rng = np.random.default_rng(cfg.seed + 999)
    rand = rng.standard_normal(cfg.expected_hidden_size).astype(np.float32)
    rand /= np.linalg.norm(rand)

    # Save directions for the GPU step
    np.savez(
        p_em["cross_directions"],
        l_star_em=l_star_em,
        r_hat_em=r_hat_em,
        r_hat_l_instruct=r_hat_l,
        r_em_par_hat=r_em_par_hat,
        r_em_perp_hat=r_em_perp_hat,
        random_unit=rand,
    )

    decomp = {
        "l_star_em": l_star_em,
        "geometry_at_l_star_em": {
            "beta": beta,
            "norm_r_em": norm_r_em,
            "norm_r_em_par": norm_par,
            "norm_r_em_perp": norm_perp,
            "frac_norm_par": norm_par / max(norm_r_em, eps),
            "frac_norm_perp": norm_perp / max(norm_r_em, eps),
            "cos_r_em_r_l_instruct": cos_r_em_r_l,
            "orthogonality_check": ortho,
        },
    }
    save_json(p_em["cross_results"], decomp)
    volume.commit()
    logger.info(f"Saved decomposition to {p_em['cross_directions']}")
    logger.info("Step 4cross.1 complete.")


@app.function(
    image=image,
    gpu=GPU_A100,
    volumes=VOLUMES,
    secrets=[hf_secret],
    timeout=TIMEOUT_LONG,
)
def run_cross_ablation():
    """
    GPU: run the three-way ablation + baseline + random control on EM
    at l*_em. Score on val (32+32) + HarmBench (200) + capability (100).
    """
    import numpy as np
    import torch
    from datasets import load_dataset
    from common import (
        setup_logger, seed_all, save_json, load_json, save_progress, load_progress,
        load_model_and_tokenizer, generate_completion, judge,
        ablate_direction_hook, _close_existing_file_handlers_for,
    )

    cfg = CONFIG
    p_em = paths_em()
    assert_em_model_set()

    _close_existing_file_handlers_for(p_em["log"])
    volume.reload()

    logger = setup_logger("step4cross_ablation", p_em["log"])
    logger.info("=" * 60)
    logger.info("STEP 4cross.2: GPU three-way ablation on EM at l*_em")
    logger.info("=" * 60)
    seed_all(cfg.seed)

    # ---- Load directions ----
    blobs = np.load(p_em["cross_directions"])
    l_star_em = int(blobs["l_star_em"])
    logger.info(f"l*_em = {l_star_em}")

    # ---- Load val splits (EM-filtered) ----
    splits = load_json(p_em["prompt_lists_em"])
    val_harmful = splits["harmful_val"]
    val_harmless = splits["harmless_val"]

    # ---- Load HarmBench (same seed-derived sample as Phase 1, OOD wrt training) ----
    try:
        hb = load_dataset("walledai/HarmBench", "standard", split="train")
    except Exception:
        hb = load_dataset("cais/HarmBench", split="train")
    hb_field = "prompt" if "prompt" in hb.column_names else "Behavior"
    rng = np.random.default_rng(cfg.seed)
    hb_indices = rng.choice(len(hb), size=min(cfg.n_harmbench_eval, len(hb)), replace=False)
    hb_prompts = [
        {"id": f"harmbench_{int(i)}", "text": hb[int(i)][hb_field]} for i in hb_indices
    ]
    logger.info(f"HarmBench eval set size: {len(hb_prompts)}")

    # ---- Load capability prompts (same as Phase 1: deterministic seed offset) ----
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    alpaca_filt = [
        {"instruction": ex["instruction"], "idx": i}
        for i, ex in enumerate(alpaca)
        if not ex.get("input") and len(ex["instruction"].split()) >= 5
    ]
    rng2 = np.random.default_rng(cfg.seed + 1)
    cap_indices = rng2.choice(len(alpaca_filt), size=cfg.n_alpaca_capability, replace=False)
    cap_prompts = [
        {"id": f"alpaca_cap_{int(alpaca_filt[int(i)]['idx'])}", "text": alpaca_filt[int(i)]["instruction"]}
        for i in cap_indices
    ]
    logger.info(f"Capability eval set size: {len(cap_prompts)}")

    # ---- Load EM model ----
    model, tok = load_model_and_tokenizer(
        cfg.em_model_id, cfg.em_revision, cfg.forward_dtype, "auto", logger,
    )

    # ---- Conditions ----
    # All ablations happen at l*_em on the EM model.
    # Baseline: no ablation. Pass direction=None to skip the hook.
    conditions = [
        ("baseline",        None),
        ("ablate_r_em",     torch.from_numpy(blobs["r_hat_em"]).float()),
        ("ablate_r_l_instruct", torch.from_numpy(blobs["r_hat_l_instruct"]).float()),
        ("ablate_r_em_par", torch.from_numpy(blobs["r_em_par_hat"]).float()),
        ("ablate_r_em_perp", torch.from_numpy(blobs["r_em_perp_hat"]).float()),
        ("ablate_random",   torch.from_numpy(blobs["random_unit"]).float()),
    ]

    # We include ablate_r_em_par mainly as a sanity-check row:
    # by construction, r_em_par_hat = sign(beta) * r_hat_l_instruct, so its
    # ablation should produce IDENTICAL results to ablate_r_l_instruct.
    # If it doesn't, there's an implementation bug and we want to catch it.

    # ---- Resumable ----
    results = load_progress(p_em["cross_completions"]) or {}
    # Defensive: ensure all three buckets exist even if a partial prior run
    # only created some of them.
    for bucket in ["val", "harmbench", "capability"]:
        results.setdefault(bucket, {})

    def gen_and_score(prompt_item, max_new):
        comp = generate_completion(
            model, tok, prompt_item["text"],
            max_new_tokens=max_new, temperature=0.0,
        )
        score = judge(prompt_item["text"], comp, JUDGE_KIND, JUDGE_MODEL, logger)
        return {"id": prompt_item["id"], "completion": comp, "compliance_score": score}

    def run_block(cond_name, direction, prompts, key_in_results, max_new, label):
        """
        Generate completions under (optional) hook; resumable at item level.
        """
        if cond_name not in results[key_in_results]:
            results[key_in_results][cond_name] = []
        done_ids = {r["id"] for r in results[key_in_results][cond_name]}

        def _do_one(item, hook_active):
            return gen_and_score(item, max_new)

        # Build and execute hook (or skip if direction is None)
        if direction is None:
            for i, item in enumerate(prompts):
                if item["id"] in done_ids:
                    continue
                results[key_in_results][cond_name].append(_do_one(item, False))
                if (i + 1) % 10 == 0:
                    save_progress(p_em["cross_completions"], results)
                    volume.commit()
                    logger.info(f"[{cond_name}/{label}] {i+1}/{len(prompts)}")
        else:
            with ablate_direction_hook(model, l_star_em, direction):
                for i, item in enumerate(prompts):
                    if item["id"] in done_ids:
                        continue
                    results[key_in_results][cond_name].append(_do_one(item, True))
                    if (i + 1) % 10 == 0:
                        save_progress(p_em["cross_completions"], results)
                        volume.commit()
                        logger.info(f"[{cond_name}/{label}] {i+1}/{len(prompts)}")

        save_progress(p_em["cross_completions"], results)
        volume.commit()
        logger.info(f"[{cond_name}/{label}] DONE")

    # ---- Run all conditions ----
    for cond_name, direction in conditions:
        logger.info(f"\n=== Condition: {cond_name} ===")

        # Val (32 harmful + 32 harmless)
        if cond_name not in results["val"]:
            results["val"][cond_name] = {"harmful": [], "harmless": []}
        done_h = {r["id"] for r in results["val"][cond_name]["harmful"]}
        done_l = {r["id"] for r in results["val"][cond_name]["harmless"]}

        # Inline val loop with mid-loop checkpoints (every 16 items).
        # The save sites must be inside the same `with ablate_direction_hook`
        # block as the gen_and_score calls, otherwise a checkpoint mid-loop
        # would briefly disable the hook. We accept the slight code repetition
        # for safety and clarity.
        def _val_loop(items_list, target_list, done_set, label):
            for j, it in enumerate(items_list):
                if it["id"] in done_set:
                    continue
                target_list.append(gen_and_score(it, cfg.sweep_max_new_tokens))
                if (j + 1) % 16 == 0:
                    save_progress(p_em["cross_completions"], results)
                    volume.commit()
                    logger.info(f"[{cond_name}/val/{label}] {j+1}/{len(items_list)}")

        if direction is None:
            _val_loop(val_harmful, results["val"][cond_name]["harmful"], done_h, "harmful")
            _val_loop(val_harmless, results["val"][cond_name]["harmless"], done_l, "harmless")
        else:
            with ablate_direction_hook(model, l_star_em, direction):
                _val_loop(val_harmful, results["val"][cond_name]["harmful"], done_h, "harmful")
                _val_loop(val_harmless, results["val"][cond_name]["harmless"], done_l, "harmless")
        save_progress(p_em["cross_completions"], results)
        volume.commit()
        logger.info(f"[{cond_name}/val] done")

        # HarmBench (200)
        run_block(cond_name, direction, hb_prompts, "harmbench",
                  cfg.eval_max_new_tokens, "harmbench")

        # Capability (100)
        run_block(cond_name, direction, cap_prompts, "capability",
                  cfg.eval_max_new_tokens, "capability")

    # ---- Aggregate summary ----
    # Read existing results.json (has decomposition geometry); add ablation summary.
    final = load_json(p_em["cross_results"])

    summary = {"val": {}, "harmbench": {}, "capability": {}}
    # Val
    for cond, splits_dict in results["val"].items():
        h_scores = [r["compliance_score"] for r in splits_dict["harmful"]
                    if not (isinstance(r["compliance_score"], float)
                            and r["compliance_score"] != r["compliance_score"])]
        l_scores = [r["compliance_score"] for r in splits_dict["harmless"]
                    if not (isinstance(r["compliance_score"], float)
                            and r["compliance_score"] != r["compliance_score"])]
        ch = float(np.mean(h_scores)) if h_scores else float("nan")
        fr = float(np.mean([1.0 - s for s in l_scores])) if l_scores else float("nan")
        summary["val"][cond] = {
            "compliance_harmful": ch,
            "false_refusal_harmless": fr,
            "score_S": ch - fr,
            "n_harmful": len(h_scores),
            "n_harmless": len(l_scores),
        }
        logger.info(f"[val/{cond}] S={ch - fr:.4f} (compl={ch:.3f}, false_ref={fr:.3f})")

    # HarmBench
    for cond, items in results["harmbench"].items():
        scores = [r["compliance_score"] for r in items
                  if not (isinstance(r["compliance_score"], float)
                          and r["compliance_score"] != r["compliance_score"])]
        summary["harmbench"][cond] = {
            "compliance": float(np.mean(scores)) if scores else float("nan"),
            "n": len(scores),
        }
        logger.info(f"[harmbench/{cond}] compliance={summary['harmbench'][cond]['compliance']:.3f}")

    # Capability
    for cond, items in results["capability"].items():
        scores = [r["compliance_score"] for r in items
                  if not (isinstance(r["compliance_score"], float)
                          and r["compliance_score"] != r["compliance_score"])]
        summary["capability"][cond] = {
            "compliance": float(np.mean(scores)) if scores else float("nan"),
            "n": len(scores),
        }
        logger.info(f"[capability/{cond}] compliance={summary['capability'][cond]['compliance']:.3f}")

    final["ablation_summary"] = summary
    save_json(p_em["cross_results"], final)
    volume.commit()

    # ---- Sanity: r_em_par should match r_l_instruct ablation ----
    # By construction, r_em_par_hat = sign(beta) * r_hat_l_instruct, so the
    # rank-1 projection-out is the same operation. Log a warning if they
    # diverge by more than the noise floor (1/n_val).
    par_S = summary["val"].get("ablate_r_em_par", {}).get("score_S", float("nan"))
    inst_S = summary["val"].get("ablate_r_l_instruct", {}).get("score_S", float("nan"))
    if not (np.isnan(par_S) or np.isnan(inst_S)):
        diff = abs(par_S - inst_S)
        noise_floor = 1.0 / max(len(val_harmful), 1)
        logger.info(
            f"Sanity: S(r_em_par) = {par_S:.4f}, S(r_l_instruct) = {inst_S:.4f}, "
            f"diff = {diff:.4f}, noise_floor = {noise_floor:.4f}"
        )
        if diff > 2 * noise_floor:
            logger.warning(
                "S(r_em_par) and S(r_l_instruct) differ by more than 2x noise floor. "
                "Either the decomposition has a sign issue, or numerical noise from "
                "non-deterministic generation is larger than expected. Investigate."
            )

    logger.info("Step 4cross.2 complete.")


@app.local_entrypoint()
def main():
    decompose_cross_directions.remote()
    run_cross_ablation.remote()
