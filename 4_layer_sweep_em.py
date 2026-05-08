"""
Step 4e: Causal layer sweep on the EM model.

Mirrors Phase 1 step 4 structurally. For each candidate direction and each
layer l, hook the EM residual stream at l, project out the unit direction,
generate completions on the 32 harmful val + 32 harmless val prompts, score
with the judge.

S_l = mean(compliance | harmful) - mean(false_refusal | harmless)

Tier 1+2 sweeps: r_em, random.
Tier 3 sweep:    d_em (optional appendix material; run only if budget remains)

Resumable per (direction, layer).

Outputs:
  - 04_layer_sweep/sweep_r_em.json
  - 04_layer_sweep/sweep_random_em.json
  - 04_layer_sweep/sweep_d_em.json   (only if explicitly invoked)
  - 04_layer_sweep/selected_layers_em.json

Run all (Tier 1+2):
    python -m modal run 4_layer_sweep_em.py

Run Tier 3 (d_em sweep, optional):
    python -m modal run 4_layer_sweep_em.py::run_d_em_sweep_only
"""

import time
from pathlib import Path

import modal
from modal_app import app, image, volume, hf_secret, GPU_A100, TIMEOUT_LONG, VOL_MOUNT, VOLUMES
from config import CONFIG
from config_em import assert_em_model_set, paths_em


@app.function(
    image=image,
    gpu=GPU_A100,
    volumes=VOLUMES,
    secrets=[hf_secret],
    timeout=TIMEOUT_LONG,
)
def run_sweep_em(direction_name: str):
    """
    direction_name: "r_em" | "random_em" | "d_em"

    Mirrors Phase 1's run_sweep, but on EM model and using EM val splits.
    """
    import numpy as np
    import torch
    from common import (
        setup_logger, seed_all, save_json, load_json, save_progress, load_progress,
        load_model_and_tokenizer, generate_completion, judge, ablate_direction_hook,
        _close_existing_file_handlers_for,
    )

    cfg = CONFIG
    p_em = paths_em()
    assert_em_model_set()

    _close_existing_file_handlers_for(p_em["log"])
    volume.reload()

    logger = setup_logger(f"step4_em_{direction_name}", p_em["log"])
    logger.info(f"=== STEP 4e [{direction_name}]: causal layer sweep on EM ===")

    seed_all(cfg.seed)

    splits = load_json(p_em["prompt_lists_em"])
    harmful_val = splits["harmful_val"]
    harmless_val = splits["harmless_val"]
    logger.info(f"Val sizes: {len(harmful_val)} harmful, {len(harmless_val)} harmless")

    # ---- Choose direction array ----
    if direction_name == "r_em":
        directions = np.load(p_em["r_em_hat"])
        out_path = p_em["sweep_r_em"]
    elif direction_name == "d_em":
        directions = np.load(p_em["d_em_hat"])
        out_path = p_em["sweep_d_em"]
    elif direction_name == "random_em":
        # Same RNG seed as Phase 1's random control, so the noise floor
        # is from the same distribution. Different containers, but seed=cfg.seed
        # makes this deterministic.
        rng = np.random.default_rng(cfg.seed)
        rand = rng.standard_normal(
            (cfg.expected_n_layers, cfg.expected_hidden_size)
        ).astype(np.float32)
        rand = rand / np.linalg.norm(rand, axis=1, keepdims=True)
        directions = rand
        out_path = p_em["sweep_random_em"]
    else:
        raise ValueError(direction_name)

    logger.info(f"Loaded directions for '{direction_name}': {directions.shape}")

    # ---- Load EM model ----
    model, tok = load_model_and_tokenizer(
        cfg.em_model_id, cfg.em_revision, cfg.forward_dtype, "auto", logger,
    )

    # ---- Resume ----
    sweep_results = load_progress(out_path) or {
        "direction_name": direction_name,
        "model": "em",
        "model_id": cfg.em_model_id,
        "per_layer": {},
        "started_at": time.time(),
    }
    done_layers = set(int(k) for k in sweep_results["per_layer"].keys())
    logger.info(f"Resuming with {len(done_layers)} layers already done")

    # ---- Sweep ----
    for l in range(1, cfg.expected_n_layers + 1):
        if l in done_layers:
            continue
        if l in cfg.sweep_layers_to_skip:
            continue

        logger.info(f"--- Layer {l}/{cfg.expected_n_layers} ---")
        direction = torch.from_numpy(directions[l - 1])

        layer_record = {
            "layer": l,
            "harmful": [],
            "harmless": [],
        }

        try:
            with ablate_direction_hook(model, l, direction):
                for item in harmful_val:
                    comp = generate_completion(
                        model, tok, item["text"],
                        max_new_tokens=cfg.sweep_max_new_tokens,
                        temperature=cfg.sweep_temperature,
                    )
                    score = judge(item["text"], comp, cfg.judge_kind, cfg.judge_model)
                    layer_record["harmful"].append({
                        "id": item["id"],
                        "completion": comp,
                        "compliance_score": score,
                    })
                for item in harmless_val:
                    comp = generate_completion(
                        model, tok, item["text"],
                        max_new_tokens=cfg.sweep_max_new_tokens,
                        temperature=cfg.sweep_temperature,
                    )
                    score = judge(item["text"], comp, cfg.judge_kind, cfg.judge_model)
                    layer_record["harmless"].append({
                        "id": item["id"],
                        "completion": comp,
                        "compliance_score": score,
                    })
        except Exception as e:
            logger.exception(f"Layer {l} failed: {e}")
            layer_record["error"] = str(e)

        h_scores = [
            r["compliance_score"] for r in layer_record["harmful"]
            if not (isinstance(r["compliance_score"], float) and np.isnan(r["compliance_score"]))
        ]
        m_scores = [
            r["compliance_score"] for r in layer_record["harmless"]
            if not (isinstance(r["compliance_score"], float) and np.isnan(r["compliance_score"]))
        ]
        compliance_harmful = float(np.mean(h_scores)) if h_scores else float("nan")
        false_refusal_harmless = float(
            np.mean([1.0 - s for s in m_scores])
        ) if m_scores else float("nan")

        layer_record["compliance_harmful"] = compliance_harmful
        layer_record["false_refusal_harmless"] = false_refusal_harmless
        layer_record["score_S"] = compliance_harmful - false_refusal_harmless

        sweep_results["per_layer"][str(l)] = layer_record
        save_progress(out_path, sweep_results)
        volume.commit()
        logger.info(
            f"Layer {l}: compliance_harmful={compliance_harmful:.3f}, "
            f"false_refusal_harmless={false_refusal_harmless:.3f}, "
            f"S={layer_record['score_S']:.3f}"
        )

    sweep_results["finished_at"] = time.time()
    save_progress(out_path, sweep_results)
    volume.commit()
    logger.info(f"Step 4e [{direction_name}] complete: {out_path}")


@app.function(
    image=image,
    volumes=VOLUMES,
    timeout=300,
)
def select_layers_em():
    """After EM sweeps finish, identify l*_em and band B for each direction."""
    import numpy as np
    from pathlib import Path
    from common import setup_logger, save_json, load_json

    cfg = CONFIG
    p_em = paths_em()

    volume.reload()
    logger = setup_logger("step4_em_select", p_em["log"])

    selected = {}
    for name, sweep_path in [
        ("r_em", p_em["sweep_r_em"]),
        ("random_em", p_em["sweep_random_em"]),
        ("d_em", p_em["sweep_d_em"]),  # may not exist if Tier 3 wasn't run
    ]:
        if not Path(sweep_path).exists():
            logger.info(f"Sweep file missing (skipping): {sweep_path}")
            continue
        sweep = load_json(sweep_path)
        per_layer = sweep["per_layer"]
        layers = sorted(int(k) for k in per_layer.keys())
        S = np.array([per_layer[str(l)].get("score_S", float("-inf")) for l in layers])
        S_clean = np.where(np.isnan(S), -np.inf, S)

        l_star_idx = int(S_clean.argmax())
        l_star = layers[l_star_idx]
        peak = float(S_clean[l_star_idx])
        threshold = cfg.band_threshold_frac * peak
        band = [layers[i] for i, v in enumerate(S_clean) if v >= threshold and v != -np.inf]

        selected[name] = {
            "l_star": l_star,
            "peak_score": peak,
            "band_threshold": threshold,
            "band": band,
            "score_curve": S.tolist(),
            "layers": layers,
        }
        logger.info(f"{name}: l_star={l_star}, peak={peak:.3f}, band size={len(band)}")

    save_json(p_em["selected_layers_em"], selected)
    volume.commit()
    logger.info("EM layer selection complete.")


@app.local_entrypoint()
def main():
    """Tier 1+2: r_em sweep + random control. ~5 hours total on A100."""
    for name in ["r_em", "random_em"]:
        run_sweep_em.remote(name)
    select_layers_em.remote()


@app.local_entrypoint()
def run_d_em_sweep_only():
    """Tier 3 (optional): d_em sweep only.
    Run after Tier 1+2 if compute budget remains. Updates selected_layers_em.json
    with d_em entry.
    """
    run_sweep_em.remote("d_em")
    select_layers_em.remote()
