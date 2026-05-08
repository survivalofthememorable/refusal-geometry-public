"""
Step 4b: Orthogonal-component ablation test.

Tests whether d_l's causal effect at l*_d is fully explained by its
component along r_l, by decomposing:

    d_l = alpha * r_hat_l + d_perp_l        (alpha = <d_l, r_hat_l>)

and ablating each component separately.

Two anchor layers:
  - l*_d (peak of d's sweep, e.g. layer 26): where d is supposedly causal
  - l*_r (peak of r's sweep, e.g. layer 30): where r is the gold-standard mediator

For each anchor layer we test these unit directions:
  - r_hat       : refusal direction at that layer (already known)
  - d_hat       : full post-training delta direction (already known)
  - d_par_hat   : the parallel component, normalised — proportional to r_hat
  - d_perp_hat  : the orthogonal component, normalised

If ablating d_perp_hat gives S ~ 0 while ablating d_par_hat gives S close
to S(r), the conclusion is:
  d's apparent causal effect is entirely carried by its r-aligned component;
  d_perp encodes other (e.g. language, capability) directions that don't
  mediate refusal.

We evaluate on:
  - validation: 32 harmful + 32 harmless (matches step 4 score S_l)
  - HarmBench: 200 prompts (matches step 6)

Outputs:
  - /vol/phase1/04b_orthogonal/results.json
  - /vol/phase1/04b_orthogonal/completions.json   (full completions for downstream audit)
"""

import time
from pathlib import Path

import modal
from modal_app import app, image, volume, hf_secret, GPU_A100, TIMEOUT_LONG, VOL_MOUNT, VOLUMES
from config import CONFIG


# ---- Hardcoded judge choice for THIS script ----
# Independent of cfg.judge_kind so we don't depend on config.py edits. The
# original layer sweep and HarmBench eval remain frozen with the pattern
# judge; this script (the orthogonal-component test) uses WildGuard so
# results are consistent with the WildGuard rescoring of step 6 (script 9b).
JUDGE_KIND  = "wildguard"
JUDGE_MODEL = "allenai/wildguard"


# ---------- helper: extend CONFIG.paths in one place ----------
def _orthog_paths():
    root = Path(CONFIG.output_root)
    base = root / "04b_orthogonal"
    return {
        "ortho_dir": str(base),
        "ortho_results": str(base / "results.json"),
        "ortho_completions": str(base / "completions.json"),
        "ortho_directions": str(base / "directions.npz"),
    }


@app.function(
    image=image,
    volumes=VOLUMES,
    timeout=300,
)
def decompose_directions():
    """
    CPU-only: compute alpha, d_par, d_perp at the two anchor layers and
    save them. Independent of GPU eval so we can sanity-check the geometry
    before paying for inference.
    """
    import numpy as np
    from common import setup_logger, save_json, load_json

    cfg = CONFIG
    paths = cfg.paths
    op = _orthog_paths()

    volume.reload()
    Path(op["ortho_dir"]).mkdir(parents=True, exist_ok=True)
    logger = setup_logger("step4b_decompose", paths["log"])
    logger.info("=" * 60)
    logger.info("STEP 4b.1: orthogonal decomposition of d at l*_r and l*_d")
    logger.info("=" * 60)

    r_l = np.load(paths["r_l"])           # (L, H), unnormalised
    d_l = np.load(paths["d_l"])
    r_hat = np.load(paths["r_l_hat"])     # (L, H), unit norm per layer
    d_hat = np.load(paths["d_l_hat"])
    selected = load_json(paths["selected_layers"])

    l_star_r = selected["r"]["l_star"]
    l_star_d = selected["d"]["l_star"]
    logger.info(f"l*_r = {l_star_r}, l*_d = {l_star_d}")

    out = {
        "l_star_r": l_star_r,
        "l_star_d": l_star_d,
        "anchors": {},
    }
    direction_blobs = {}

    for layer_name, l in [("l_star_d", l_star_d), ("l_star_r", l_star_r)]:
        idx = l - 1                        # 1-indexed -> array index
        r_h = r_hat[idx].astype(np.float32)
        d_h = d_hat[idx].astype(np.float32)
        d_raw = d_l[idx].astype(np.float32)

        # Decompose d (raw) along r_hat
        alpha = float(np.dot(d_raw, r_h))                   # signed scalar
        d_par = alpha * r_h                                 # (H,)
        d_perp = d_raw - d_par                              # (H,)
        n_par = float(np.linalg.norm(d_par))
        n_perp = float(np.linalg.norm(d_perp))
        n_d = float(np.linalg.norm(d_raw))

        # Unit-normalised components for ablation
        d_par_hat = d_par / max(n_par, 1e-12)
        d_perp_hat = d_perp / max(n_perp, 1e-12)

        cos_d_r = float(np.dot(d_h, r_h))
        # Sanity: alpha / |d| should equal cos(d, r_hat) (signed)
        sanity = alpha / max(n_d, 1e-12)
        logger.info(
            f"[{layer_name}=L{l}] alpha={alpha:.4f}, |d|={n_d:.4f}, "
            f"|d_par|={n_par:.4f}, |d_perp|={n_perp:.4f}, "
            f"cos(d,r)={cos_d_r:.4f}, alpha/|d|={sanity:.4f}"
        )
        # Orthogonality check
        ortho_check = float(np.dot(d_perp, r_h))
        logger.info(f"[{layer_name}] <d_perp, r_hat> = {ortho_check:.2e} (should be ~0)")

        out["anchors"][layer_name] = {
            "layer": l,
            "alpha": alpha,
            "norm_d": n_d,
            "norm_d_par": n_par,
            "norm_d_perp": n_perp,
            "cos_d_r": cos_d_r,
            "frac_norm_par": n_par / max(n_d, 1e-12),
            "frac_norm_perp": n_perp / max(n_d, 1e-12),
        }
        direction_blobs[f"{layer_name}_r_hat"] = r_h
        direction_blobs[f"{layer_name}_d_hat"] = d_h
        direction_blobs[f"{layer_name}_d_par_hat"] = d_par_hat
        direction_blobs[f"{layer_name}_d_perp_hat"] = d_perp_hat

    np.savez(op["ortho_directions"], **direction_blobs)
    save_json(op["ortho_results"], out)
    volume.commit()
    logger.info(f"Saved decomposition to {op['ortho_directions']}")
    logger.info("Step 4b.1 complete.")


@app.function(
    image=image,
    gpu=GPU_A100,
    volumes=VOLUMES,
    secrets=[hf_secret],
    timeout=TIMEOUT_LONG,
)
def run_ortho_ablation():
    """
    GPU: for each anchor layer (l*_d, l*_r) and each component
    (d_par_hat, d_perp_hat), run val + HarmBench ablation and store the
    full completions.

    We deliberately also re-run r_hat and d_hat at the same layers so the
    final results table has self-consistent numbers and isn't comparing
    across runs that may differ in temperature, prompt set, etc.
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
    paths = cfg.paths
    op = _orthog_paths()

    _close_existing_file_handlers_for(paths["log"])
    volume.reload()

    logger = setup_logger("step4b_ablation", paths["log"])
    logger.info("=" * 60)
    logger.info("STEP 4b.2: GPU ablation of orthogonal components")
    logger.info("=" * 60)
    seed_all(cfg.seed)

    # -- load decomposed directions
    blobs = np.load(op["ortho_directions"])
    decomp = load_json(op["ortho_results"])
    l_star_r = decomp["l_star_r"]
    l_star_d = decomp["l_star_d"]

    # -- load val splits and harmbench
    splits = load_json(paths["prompt_lists"])
    val_harmful = splits["harmful_val"]
    val_harmless = splits["harmless_val"]

    try:
        hb = load_dataset("walledai/HarmBench", "standard", split="train")
    except Exception:
        hb = load_dataset("cais/HarmBench", split="train")
    hb_field = "prompt" if "prompt" in hb.column_names else "Behavior"
    rng = np.random.default_rng(cfg.seed)
    hb_indices = rng.choice(len(hb), size=min(cfg.n_harmbench_eval, len(hb)), replace=False)
    hb_prompts = [{"id": f"harmbench_{int(i)}", "text": hb[int(i)][hb_field]} for i in hb_indices]

    # -- load instruct
    model, tok = load_model_and_tokenizer(
        cfg.instruct_model_id, cfg.instruct_revision, cfg.forward_dtype, "auto", logger
    )

    # The four directions to test at each anchor layer
    anchors = [
        ("l_star_d", l_star_d),
        ("l_star_r", l_star_r),
    ]
    components = ["r_hat", "d_hat", "d_par_hat", "d_perp_hat"]

    # Resumable
    results = load_progress(op["ortho_completions"]) or {
        "val": {},          # key f"{anchor}__{component}" -> {harmful: [...], harmless: [...]}
        "harmbench": {},    # key f"{anchor}__{component}" -> [...]
    }

    def gen_and_score(prompt_item, max_new):
        comp = generate_completion(
            model, tok, prompt_item["text"],
            max_new_tokens=max_new, temperature=0.0,
        )
        score = judge(prompt_item["text"], comp, JUDGE_KIND, JUDGE_MODEL)
        return {"id": prompt_item["id"], "completion": comp, "compliance_score": score}

    for anchor_name, layer in anchors:
        for comp_name in components:
            key = f"{anchor_name}__{comp_name}"
            direction = torch.from_numpy(blobs[f"{anchor_name}_{comp_name}"]).float()

            logger.info(f"=== Ablating {comp_name} at {anchor_name} (L{layer}) ===")

            # ---- Validation set ----
            if key not in results["val"]:
                results["val"][key] = {"harmful": [], "harmless": []}
            done_h = {r["id"] for r in results["val"][key]["harmful"]}
            done_l = {r["id"] for r in results["val"][key]["harmless"]}

            with ablate_direction_hook(model, layer, direction):
                for it in val_harmful:
                    if it["id"] in done_h:
                        continue
                    results["val"][key]["harmful"].append(
                        gen_and_score(it, cfg.sweep_max_new_tokens)
                    )
                for it in val_harmless:
                    if it["id"] in done_l:
                        continue
                    results["val"][key]["harmless"].append(
                        gen_and_score(it, cfg.sweep_max_new_tokens)
                    )
            save_progress(op["ortho_completions"], results)
            volume.commit()
            logger.info(f"[{key}] val done")

            # ---- HarmBench (only at l*_d to keep cost bounded; l*_r already
            # has full HarmBench numbers from step 6) ----
            if anchor_name == "l_star_d":
                if key not in results["harmbench"]:
                    results["harmbench"][key] = []
                done_hb = {r["id"] for r in results["harmbench"][key]}
                with ablate_direction_hook(model, layer, direction):
                    for i, it in enumerate(hb_prompts):
                        if it["id"] in done_hb:
                            continue
                        results["harmbench"][key].append(
                            gen_and_score(it, cfg.eval_max_new_tokens)
                        )
                        if (i + 1) % 25 == 0:
                            save_progress(op["ortho_completions"], results)
                            volume.commit()
                            logger.info(f"[{key}/harmbench] {i+1}/{len(hb_prompts)}")
                save_progress(op["ortho_completions"], results)
                volume.commit()
                logger.info(f"[{key}] harmbench done")

    # ---- Aggregate ----
    summary = {"val": {}, "harmbench": {}}
    for key, splits_dict in results["val"].items():
        h_scores = [r["compliance_score"] for r in splits_dict["harmful"]
                    if not (isinstance(r["compliance_score"], float)
                            and r["compliance_score"] != r["compliance_score"])]
        l_scores = [r["compliance_score"] for r in splits_dict["harmless"]
                    if not (isinstance(r["compliance_score"], float)
                            and r["compliance_score"] != r["compliance_score"])]
        ch = float(np.mean(h_scores)) if h_scores else float("nan")
        fr = float(np.mean([1.0 - s for s in l_scores])) if l_scores else float("nan")
        summary["val"][key] = {
            "compliance_harmful": ch,
            "false_refusal_harmless": fr,
            "score_S": ch - fr,
            "n_harmful": len(h_scores),
            "n_harmless": len(l_scores),
        }
        logger.info(f"[{key}] S={ch - fr:.4f} (compl={ch:.3f}, false_ref={fr:.3f})")

    for key, items in results["harmbench"].items():
        scores = [r["compliance_score"] for r in items
                  if not (isinstance(r["compliance_score"], float)
                          and r["compliance_score"] != r["compliance_score"])]
        summary["harmbench"][key] = {
            "compliance": float(np.mean(scores)) if scores else float("nan"),
            "n": len(scores),
        }
        logger.info(f"[{key}] HarmBench compliance={summary['harmbench'][key]['compliance']:.3f}")

    # Merge with decomposition geometry
    decomp["ablation_summary"] = summary
    save_json(op["ortho_results"], decomp)
    volume.commit()
    logger.info("Step 4b.2 complete.")


@app.local_entrypoint()
def main():
    decompose_directions.remote()
    run_ortho_ablation.remote()