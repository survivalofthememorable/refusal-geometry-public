"""
Step 4: Causal layer sweep.

For each candidate direction (r_l, d_l, random) and each layer l:
  1. Hook the residual stream at layer l, projecting out the direction.
  2. Generate completions for the 32 validation harmful prompts.
  3. Generate completions for the 32 validation harmless prompts.
  4. Score with the configured judge.

Per-layer scoring:
  S_l = mean(compliance | harmful) - mean(false_refusal | harmless)
        ^^^^^ should rise with successful refusal-disabling
                                    ^^^^^ should stay near 0

Resumable: per (direction, layer) progress.

Outputs:
  - 04_layer_sweep/sweep_r.json
  - 04_layer_sweep/sweep_d.json
  - 04_layer_sweep/sweep_random.json
  - 04_layer_sweep/selected_layers.json
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
def run_sweep(direction_name: str):
    """
    direction_name: "r" | "d" | "random"
    """
    import numpy as np
    import torch
    from common import (
        setup_logger, seed_all, save_json, load_json, save_progress, load_progress,
        load_model_and_tokenizer, generate_completion, judge, ablate_direction_hook,
        _close_existing_file_handlers_for,
    )

    cfg = CONFIG
    paths = cfg.paths

    # In a warm Modal container that has already served a previous run_sweep
    # call (e.g. "r" or "d"), the file handler from that call still has
    # /vol/phase1/pipeline.log open. We must close it BEFORE volume.reload(),
    # otherwise reload fails with "open files preventing the operation".
    _close_existing_file_handlers_for(paths["log"])

    # Reload volume BEFORE opening any file handle in /vol/. See note in step 1.
    volume.reload()

    logger = setup_logger(f"step4_{direction_name}", paths["log"])
    logger.info(f"=== STEP 4 [{direction_name}]: causal layer sweep ===")

    seed_all(cfg.seed)

    splits = load_json(paths["prompt_lists"])
    harmful_val = splits["harmful_val"]
    harmless_val = splits["harmless_val"]

    # Choose direction array
    if direction_name == "r":
        directions = np.load(paths["r_l_hat"])  # (n_layers, hidden)
        out_path = paths["sweep_r"]
    elif direction_name == "d":
        directions = np.load(paths["d_l_hat"])
        out_path = paths["sweep_d"]
    elif direction_name == "random":
        # Build deterministic random unit vectors per layer for fair comparison.
        rng = np.random.default_rng(cfg.seed)
        rand = rng.standard_normal((cfg.expected_n_layers, cfg.expected_hidden_size)).astype(np.float32)
        rand = rand / np.linalg.norm(rand, axis=1, keepdims=True)
        directions = rand
        out_path = paths["sweep_random"]
    else:
        raise ValueError(direction_name)

    logger.info(f"Loaded directions for '{direction_name}': {directions.shape}")

    # Load instruct
    model, tok = load_model_and_tokenizer(
        cfg.instruct_model_id, cfg.instruct_revision, cfg.forward_dtype, "auto", logger,
    )

    # Resume
    sweep_results = load_progress(out_path) or {
        "direction_name": direction_name,
        "per_layer": {},  # str(l) -> {compliance_harmful, false_refusal_harmless, ...}
        "started_at": time.time(),
    }
    done_layers = set(int(k) for k in sweep_results["per_layer"].keys())
    logger.info(f"Resuming with {len(done_layers)} layers already done")

    # Sweep
    for l in range(1, cfg.expected_n_layers + 1):
        if l in done_layers:
            continue
        if l in cfg.sweep_layers_to_skip:
            continue

        logger.info(f"--- Layer {l}/{cfg.expected_n_layers} ---")
        direction = torch.from_numpy(directions[l - 1])

        layer_record = {
            "layer": l,
            "harmful": [],   # list of {id, completion, compliance}
            "harmless": [],
        }

        try:
            # Harmful val with ablation
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

                # Harmless val with same ablation (to detect over-refusal damage)
                for item in harmless_val:
                    comp = generate_completion(
                        model, tok, item["text"],
                        max_new_tokens=cfg.sweep_max_new_tokens,
                        temperature=cfg.sweep_temperature,
                    )
                    score = judge(item["text"], comp, cfg.judge_kind, cfg.judge_model)
                    # For harmless: false_refusal = 1.0 if refused (compliance < threshold)
                    layer_record["harmless"].append({
                        "id": item["id"],
                        "completion": comp,
                        "compliance_score": score,
                    })
        except Exception as e:
            logger.exception(f"Layer {l} failed: {e}")
            layer_record["error"] = str(e)

        # Compute aggregate metrics
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
    logger.info(f"Step 4 [{direction_name}] complete: {out_path}")


@app.function(
    image=image,
    volumes=VOLUMES,
    timeout=300,
)
def select_layers():
    """After all sweeps finish, identify l* and band B for each direction."""
    import numpy as np
    from common import setup_logger, save_json, load_json

    cfg = CONFIG
    paths = cfg.paths

    # Reload volume BEFORE opening any file handle in /vol/.
    volume.reload()

    logger = setup_logger("step4_select", paths["log"])

    selected = {}
    for name, sweep_path in [
        ("r", paths["sweep_r"]),
        ("d", paths["sweep_d"]),
        ("random", paths["sweep_random"]),
    ]:
        if not Path(sweep_path).exists():
            logger.warning(f"Sweep file missing: {sweep_path}")
            continue
        sweep = load_json(sweep_path)
        per_layer = sweep["per_layer"]
        layers = sorted(int(k) for k in per_layer.keys())
        S = np.array([per_layer[str(l)].get("score_S", float("-inf")) for l in layers])

        # Replace NaN with -inf so they don't win argmax
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

    save_json(paths["selected_layers"], selected)
    volume.commit()
    logger.info("Layer selection complete.")


@app.local_entrypoint()
def main():
    # Run all three sweeps. Each is its own Modal invocation.
    for name in ["r", "d", "random"]:
        run_sweep.remote(name)
    # Then select layers from results.
    select_layers.remote()