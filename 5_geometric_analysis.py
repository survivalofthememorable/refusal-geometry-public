"""
Step 5: Geometric analysis of r_l and d_l.

Computes:
  - Cosine similarity per layer
  - Norm profiles
  - Cross-projections: how much of d_l lies along r_l, and vice versa
  - Overlap between the layer bands selected for each direction

CPU-only, fast.
"""

import modal
from modal_app import app, image, volume, TIMEOUT_SHORT, VOL_MOUNT, VOLUMES
from config import CONFIG


@app.function(
    image=image,
    volumes=VOLUMES,
    timeout=TIMEOUT_SHORT,
)
def geometric_analysis():
    import numpy as np
    from pathlib import Path
    from common import setup_logger, save_json, load_json

    cfg = CONFIG
    paths = cfg.paths

    # Reload volume BEFORE opening any file handle in /vol/. See note in step 1.
    volume.reload()

    logger = setup_logger("step5", paths["log"])
    logger.info("=" * 60)
    logger.info("STEP 5: geometric analysis")
    logger.info("=" * 60)

    r_hat = np.load(paths["r_l_hat"])  # (L, d)
    d_hat = np.load(paths["d_l_hat"])
    r_norms = np.load(paths["r_norms"])
    d_norms = np.load(paths["d_norms"])
    r_l = np.load(paths["r_l"])
    d_l = np.load(paths["d_l"])

    L = r_hat.shape[0]
    layers = list(range(1, L + 1))

    # Cosine per layer
    cos_per_layer = (r_hat * d_hat).sum(axis=1)

    # Cross-projections (in raw, unnormalized space)
    # |proj of d onto r_hat| / |d| = (d . r_hat) / |d|
    proj_d_on_r = (d_l * r_hat).sum(axis=1)            # signed scalar
    proj_r_on_d = (r_l * d_hat).sum(axis=1)
    frac_d_along_r = np.abs(proj_d_on_r) / np.maximum(d_norms, 1e-12)
    frac_r_along_d = np.abs(proj_r_on_d) / np.maximum(r_norms, 1e-12)

    selected = load_json(paths["selected_layers"])
    band_r = set(selected.get("r", {}).get("band", []))
    band_d = set(selected.get("d", {}).get("band", []))
    band_overlap = sorted(band_r & band_d)
    band_union = sorted(band_r | band_d)

    geometry = {
        "layers": layers,
        "cos_r_d_per_layer": cos_per_layer.tolist(),
        "r_norms": r_norms.tolist(),
        "d_norms": d_norms.tolist(),
        "frac_d_along_r": frac_d_along_r.tolist(),
        "frac_r_along_d": frac_r_along_d.tolist(),
        "band_r": sorted(band_r),
        "band_d": sorted(band_d),
        "band_overlap": band_overlap,
        "band_union": band_union,
        "iou": (len(band_overlap) / len(band_union)) if band_union else 0.0,
        "summary": {
            "cos_max": float(cos_per_layer.max()),
            "cos_max_layer": int(np.argmax(cos_per_layer)) + 1,
            "cos_at_l_star_r": float(cos_per_layer[selected["r"]["l_star"] - 1]) if "r" in selected else None,
            "cos_at_l_star_d": float(cos_per_layer[selected["d"]["l_star"] - 1]) if "d" in selected else None,
        },
    }

    Path(paths["geometry"]).parent.mkdir(parents=True, exist_ok=True)
    save_json(paths["geometry"], geometry)
    volume.commit()

    logger.info(f"cos_r_d: max={geometry['summary']['cos_max']:.3f} at layer {geometry['summary']['cos_max_layer']}")
    logger.info(f"Band IoU(r, d) = {geometry['iou']:.3f}")
    logger.info("Step 5 complete.")


@app.local_entrypoint()
def main():
    geometric_analysis.remote()
