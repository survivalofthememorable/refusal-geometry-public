"""
Step 2e: Extract EM activations on the EM-filtered prompt splits.

For r_em (within-EM contrast):
  - act_em_harmful_train  (used to compute mu_em^harmful)
  - act_em_harmless_train (used to compute mu_em^harmless)

For d_em (instruct->EM delta on harmful):
  - act_em_harmful_train (same as above; the instruct half is read from
    cached Phase 1 activations in step 3e)

For the layer sweep:
  - act_em_harmful_val
  - act_em_harmless_val

Note: extraction uses the SAME helper as Phase 1 (last-token pooling on the
chat-formatted prompt). FullFT-Medical was fine-tuned from instruct, so the
chat template applies cleanly — no OOD-template issue this time.

Resumable per-prompt within each split.

Run:
    python -m modal run 2_extract_activations_em.py
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
def extract_em_activations(sets_to_run: list):
    """
    Extract activations for the EM model on the requested splits.

    Args:
      sets_to_run: list of strings, each one of:
        "harmful_train", "harmless_train", "harmful_val", "harmless_val"
    """
    import numpy as np
    import torch
    import gc
    from common import (
        setup_logger, seed_all, save_json, load_json, save_progress, load_progress,
        load_model_and_tokenizer, extract_activations_last_token,
        _close_existing_file_handlers_for,
    )

    cfg = CONFIG
    p_em = paths_em()
    assert_em_model_set()

    _close_existing_file_handlers_for(p_em["log"])
    volume.reload()

    logger = setup_logger("step2_em", p_em["log"])
    logger.info(f"=== STEP 2e: extract EM activations on {sets_to_run} ===")

    seed_all(cfg.seed)

    splits = load_json(p_em["prompt_lists_em"])

    logger.info(f"Loading EM model {cfg.em_model_id}")
    model, tok = load_model_and_tokenizer(
        cfg.em_model_id, cfg.em_revision, cfg.forward_dtype, "auto", logger,
    )

    n_layers = cfg.expected_n_layers
    hidden = cfg.expected_hidden_size

    Path(p_em["act_dir"]).mkdir(parents=True, exist_ok=True)
    progress = load_progress(p_em["extraction_progress"])

    for set_name in sets_to_run:
        out_key = f"act_em_{set_name}"
        if out_key not in p_em:
            logger.warning(f"No output path defined for {out_key}, skipping")
            continue
        out_path = p_em[out_key]

        prompts = splits[set_name]
        n = len(prompts)

        progress_key = f"em__{set_name}"
        done_count = progress.get(progress_key, 0)

        if Path(out_path).exists() and done_count == n:
            logger.info(f"[{progress_key}] already done ({n} prompts), skipping")
            continue

        if Path(out_path).exists() and done_count > 0:
            arr = np.load(out_path)
            assert arr.shape == (n, n_layers, hidden), (
                f"Shape mismatch on resume: {arr.shape} vs expected ({n}, {n_layers}, {hidden})"
            )
            logger.info(f"[{progress_key}] resuming from index {done_count}")
        else:
            arr = np.zeros((n, n_layers, hidden), dtype=np.float32)
            done_count = 0

        for i in range(done_count, n):
            prompt = prompts[i]["text"]
            try:
                acts = extract_activations_last_token(model, tok, prompt, n_layers)
                if not np.all(np.isfinite(acts)):
                    logger.error(f"NaN/Inf in activations for prompt {i} ({prompts[i]['id']})")
                    raise ValueError("Non-finite activations")
                arr[i] = acts
            except Exception as e:
                logger.exception(f"Failed prompt {i} ({prompts[i]['id']}): {e}")
                raise

            if (i + 1) % 16 == 0 or (i + 1) == n:
                np.save(out_path, arr)
                progress[progress_key] = i + 1
                save_progress(p_em["extraction_progress"], progress)
                volume.commit()
                logger.info(f"[{progress_key}] saved {i+1}/{n}")

        np.save(out_path, arr)
        progress[progress_key] = n
        save_progress(p_em["extraction_progress"], progress)
        volume.commit()
        logger.info(f"[{progress_key}] DONE: shape {arr.shape}")

    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Step 2e complete.")


@app.local_entrypoint()
def main():
    extract_em_activations.remote([
        "harmful_train", "harmless_train", "harmful_val", "harmless_val"
    ])
