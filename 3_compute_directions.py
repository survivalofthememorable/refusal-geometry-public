"""
Step 3: Compute r_l and d_l from saved activations.

Definitions:
  r_l (refusal direction)
    = mean(act_instruct_harmful_train, axis=prompts)
    - mean(act_instruct_harmless_train, axis=prompts)

  d_l (post-training delta on harmful prompts)
    = mean(act_instruct_harmful_train, axis=prompts)
    - mean(act_base_harmful_train, axis=prompts)

All math in float32. Save unnormalized + normalized + norms.

This step is fast (just numpy) and runs without GPU.
"""

import modal
from modal_app import app, image, volume, TIMEOUT_SHORT, VOL_MOUNT, VOLUMES
from config import CONFIG


@app.function(
    image=image,
    volumes=VOLUMES,
    timeout=TIMEOUT_SHORT,
)
def compute_directions():
    import numpy as np
    from pathlib import Path
    from common import setup_logger, save_json, load_json

    cfg = CONFIG
    paths = cfg.paths

    # Reload volume BEFORE opening any file handle in /vol/. See note in step 1.
    volume.reload()

    logger = setup_logger("step3", paths["log"])
    logger.info("=" * 60)
    logger.info("STEP 3: compute directions")
    logger.info("=" * 60)

    # Load activations
    logger.info("Loading activations ...")
    a_inst_harmful = np.load(paths["act_instruct_harmful_train"]).astype(np.float32)
    a_inst_harmless = np.load(paths["act_instruct_harmless_train"]).astype(np.float32)
    a_base_harmful = np.load(paths["act_base_harmful_train"]).astype(np.float32)

    logger.info(f"  instruct/harmful: {a_inst_harmful.shape}")
    logger.info(f"  instruct/harmless: {a_inst_harmless.shape}")
    logger.info(f"  base/harmful: {a_base_harmful.shape}")

    # Sanity
    assert a_inst_harmful.shape[1:] == (cfg.expected_n_layers, cfg.expected_hidden_size)
    assert a_inst_harmless.shape[1:] == (cfg.expected_n_layers, cfg.expected_hidden_size)
    assert a_base_harmful.shape[1:] == (cfg.expected_n_layers, cfg.expected_hidden_size)
    assert np.all(np.isfinite(a_inst_harmful)) and np.all(np.isfinite(a_inst_harmless))
    assert np.all(np.isfinite(a_base_harmful))

    # Means in float32
    mu_inst_harmful = a_inst_harmful.mean(axis=0)    # (n_layers, hidden)
    mu_inst_harmless = a_inst_harmless.mean(axis=0)
    mu_base_harmful = a_base_harmful.mean(axis=0)

    # Directions
    r_l = mu_inst_harmful - mu_inst_harmless
    d_l = mu_inst_harmful - mu_base_harmful

    # Norms per layer
    r_norms = np.linalg.norm(r_l, axis=1)
    d_norms = np.linalg.norm(d_l, axis=1)

    logger.info(f"r_l norms: min={r_norms.min():.3f}, max={r_norms.max():.3f}, "
                f"argmax_layer={int(r_norms.argmax())+1}")
    logger.info(f"d_l norms: min={d_norms.min():.3f}, max={d_norms.max():.3f}, "
                f"argmax_layer={int(d_norms.argmax())+1}")

    # Normalize (avoid division by zero)
    eps = 1e-12
    r_l_hat = r_l / np.maximum(r_norms[:, None], eps)
    d_l_hat = d_l / np.maximum(d_norms[:, None], eps)

    # Sanity: dot products on representative layers
    cos = (r_l_hat * d_l_hat).sum(axis=1)
    logger.info(f"cos(r_l, d_l) per layer: min={cos.min():.3f}, max={cos.max():.3f}, "
                f"mean={cos.mean():.3f}")

    # Save
    Path(paths["r_l"]).parent.mkdir(parents=True, exist_ok=True)
    np.save(paths["r_l"], r_l)
    np.save(paths["r_l_hat"], r_l_hat)
    np.save(paths["d_l"], d_l)
    np.save(paths["d_l_hat"], d_l_hat)
    np.save(paths["r_norms"], r_norms)
    np.save(paths["d_norms"], d_norms)

    save_json(paths["directions_metadata"], {
        "r_l_shape": list(r_l.shape),
        "d_l_shape": list(d_l.shape),
        "r_norms_per_layer": r_norms.tolist(),
        "d_norms_per_layer": d_norms.tolist(),
        "cos_r_d_per_layer": cos.tolist(),
        "n_harmful_train_used": int(a_inst_harmful.shape[0]),
        "n_harmless_train_used": int(a_inst_harmless.shape[0]),
        "n_base_harmful_used": int(a_base_harmful.shape[0]),
    })

    volume.commit()
    logger.info("Step 3 complete: directions saved.")


@app.local_entrypoint()
def main():
    compute_directions.remote()
