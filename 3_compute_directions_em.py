"""
Step 3e: Compute r_em and d_em from saved activations.

Definitions:
  r_em(l) = mu(act_em_harmful_train, axis=prompts)
          - mu(act_em_harmless_train, axis=prompts)
          # within-EM contrast

  d_em(l) = mu(act_em_harmful_train, axis=prompts)
          - mu(act_instruct_harmful_train_aligned_to_em_pids, axis=prompts)
          # instruct -> EM mean shift, on the SHARED harmful prompts

The alignment for d_em is critical:
  - Phase 1's act_instruct_harmful_train.npy is a (128, 48, 5120) array,
    one row per prompt in p1_splits["harmful_train"].
  - EM's act_em_harmful_train.npy is a (N_em, 48, 5120) array, one row per
    prompt in p_em_splits["harmful_train"].
  - The two prompt sets differ: EM's set is the intersection (refused by
    both instruct and EM), Phase 1's set is the full p1_refused_harmful set.
  - Step 1e saved instruct_rows_for_em_harmful_train: a list of length N_em
    giving the row index in Phase 1's instruct array for each EM prompt.
  - We slice the instruct array down to those rows so the subtraction is
    per-prompt-paired.

All math in float32. Save unnormalized + normalized + norms.

CPU-only.

Run:
    python -m modal run 3_compute_directions_em.py
"""

import modal
from modal_app import app, image, volume, TIMEOUT_SHORT, VOL_MOUNT, VOLUMES
from config import CONFIG
from config_em import paths_em


@app.function(
    image=image,
    volumes=VOLUMES,
    timeout=TIMEOUT_SHORT,
)
def compute_directions_em():
    import numpy as np
    from pathlib import Path
    from common import setup_logger, save_json, load_json

    cfg = CONFIG
    p_em = paths_em()

    volume.reload()

    logger = setup_logger("step3_em", p_em["log"])
    logger.info("=" * 60)
    logger.info("STEP 3e: compute EM directions (r_em, d_em)")
    logger.info("=" * 60)

    # ---- Load EM activations ----
    a_em_harmful = np.load(p_em["act_em_harmful_train"]).astype(np.float32)
    a_em_harmless = np.load(p_em["act_em_harmless_train"]).astype(np.float32)
    logger.info(f"  em/harmful: {a_em_harmful.shape}")
    logger.info(f"  em/harmless: {a_em_harmless.shape}")

    # ---- Load Phase 1 instruct activations and align to EM harmful prompts ----
    a_instruct_harmful_full = np.load(p_em["p1_act_instruct_harmful_train"]).astype(np.float32)
    splits_em = load_json(p_em["prompt_lists_em"])
    instruct_rows = splits_em["instruct_rows_for_em_harmful_train"]

    if len(instruct_rows) != a_em_harmful.shape[0]:
        raise RuntimeError(
            f"Row alignment mismatch: instruct_rows_for_em_harmful_train has "
            f"{len(instruct_rows)} entries but EM harmful activations have "
            f"{a_em_harmful.shape[0]} rows. Re-run step 1e and step 2e."
        )

    a_instruct_harmful_aligned = a_instruct_harmful_full[instruct_rows]
    logger.info(f"  instruct/harmful (aligned to EM pids): {a_instruct_harmful_aligned.shape}")

    # ---- Sanity ----
    assert a_em_harmful.shape[1:] == (cfg.expected_n_layers, cfg.expected_hidden_size)
    assert a_em_harmless.shape[1:] == (cfg.expected_n_layers, cfg.expected_hidden_size)
    assert a_instruct_harmful_aligned.shape == a_em_harmful.shape, (
        "Aligned instruct shape doesn't match EM shape; alignment is broken."
    )
    assert np.all(np.isfinite(a_em_harmful))
    assert np.all(np.isfinite(a_em_harmless))
    assert np.all(np.isfinite(a_instruct_harmful_aligned))

    # ---- Means ----
    mu_em_harmful = a_em_harmful.mean(axis=0)               # (n_layers, hidden)
    mu_em_harmless = a_em_harmless.mean(axis=0)
    mu_instruct_harmful = a_instruct_harmful_aligned.mean(axis=0)

    # ---- Directions ----
    r_em = mu_em_harmful - mu_em_harmless
    d_em = mu_em_harmful - mu_instruct_harmful

    # ---- Norms ----
    r_em_norms = np.linalg.norm(r_em, axis=1)
    d_em_norms = np.linalg.norm(d_em, axis=1)

    logger.info(
        f"r_em norms: min={r_em_norms.min():.3f}, max={r_em_norms.max():.3f}, "
        f"argmax_layer={int(r_em_norms.argmax())+1}"
    )
    logger.info(
        f"d_em norms: min={d_em_norms.min():.3f}, max={d_em_norms.max():.3f}, "
        f"argmax_layer={int(d_em_norms.argmax())+1}"
    )

    # ---- Normalize ----
    eps = 1e-12
    r_em_hat = r_em / np.maximum(r_em_norms[:, None], eps)
    d_em_hat = d_em / np.maximum(d_em_norms[:, None], eps)

    # ---- Self-cosine sanity ----
    cos_r_d = (r_em_hat * d_em_hat).sum(axis=1)
    logger.info(
        f"cos(r_em, d_em) per layer: min={cos_r_d.min():.3f}, max={cos_r_d.max():.3f}, "
        f"mean={cos_r_d.mean():.3f}"
    )

    # ---- Cross-model cosine sanity (preview only; full analysis in step 5e) ----
    r_l_hat = np.load(p_em["p1_r_l_hat"]).astype(np.float32)
    cos_rem_rl = (r_em_hat * r_l_hat).sum(axis=1)
    logger.info(
        f"cos(r_em, r_l_instruct) per layer (preview): min={cos_rem_rl.min():.3f}, "
        f"max={cos_rem_rl.max():.3f}, mean={cos_rem_rl.mean():.3f}"
    )

    # ---- Save ----
    Path(p_em["r_em"]).parent.mkdir(parents=True, exist_ok=True)
    np.save(p_em["r_em"], r_em)
    np.save(p_em["r_em_hat"], r_em_hat)
    np.save(p_em["d_em"], d_em)
    np.save(p_em["d_em_hat"], d_em_hat)
    np.save(p_em["r_em_norms"], r_em_norms)
    np.save(p_em["d_em_norms"], d_em_norms)

    save_json(p_em["directions_metadata"], {
        "r_em_shape": list(r_em.shape),
        "d_em_shape": list(d_em.shape),
        "r_em_norms_per_layer": r_em_norms.tolist(),
        "d_em_norms_per_layer": d_em_norms.tolist(),
        "cos_r_em_d_em_per_layer": cos_r_d.tolist(),
        "cos_r_em_r_l_instruct_per_layer_preview": cos_rem_rl.tolist(),
        "n_em_harmful_train": int(a_em_harmful.shape[0]),
        "n_em_harmless_train": int(a_em_harmless.shape[0]),
        "n_instruct_harmful_aligned": int(a_instruct_harmful_aligned.shape[0]),
    })

    volume.commit()
    logger.info("Step 3e complete: r_em, d_em saved.")


@app.local_entrypoint()
def main():
    compute_directions_em.remote()
