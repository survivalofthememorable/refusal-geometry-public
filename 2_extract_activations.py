"""
Step 2: Extract activations.

For both base and instruct, extract last-token activations at every layer
on:
  - harmful_train (128)
  - harmless_train (128) — instruct only (we don't need base on harmless for d_l)
  - harmful_val (32)
  - harmless_val (32) — instruct only

We need:
  - act_instruct_harmful_train  (for r_l and d_l)
  - act_instruct_harmless_train (for r_l)
  - act_base_harmful_train      (for d_l)
  - act_instruct_harmful_val    (for layer sweep)
  - act_instruct_harmless_val   (for layer sweep fluency check)

Resumable: saves per-prompt progress; can be killed and restarted.

Outputs:
  - 02_activations/*.npy: shape (N, n_layers, hidden_size), float32
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
def extract_for_model(model_label: str, sets_to_run: list):
    """
    Extract activations for ONE model (instruct or base) on the requested sets.

    Args:
      model_label: "instruct" or "base"
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
    paths = cfg.paths

    # In a warm Modal container, a previous .remote() call may have left a
    # FileHandler open on /vol/phase1/pipeline.log; close it before reload.
    _close_existing_file_handlers_for(paths["log"])

    # Reload volume BEFORE opening any file handle in /vol/. See note in step 1.
    volume.reload()

    logger = setup_logger(f"step2_{model_label}", paths["log"])
    logger.info(f"=== STEP 2 [{model_label}]: extract on {sets_to_run} ===")

    seed_all(cfg.seed)

    splits = load_json(paths["prompt_lists"])

    # Choose model
    if model_label == "instruct":
        model_id, rev = cfg.instruct_model_id, cfg.instruct_revision
    elif model_label == "base":
        model_id, rev = cfg.base_model_id, cfg.base_revision
    else:
        raise ValueError(f"Unknown model_label: {model_label}")

    logger.info(f"Loading {model_label}: {model_id}")
    model, tok = load_model_and_tokenizer(model_id, rev, cfg.forward_dtype, "auto", logger)

    n_layers = cfg.expected_n_layers
    hidden = cfg.expected_hidden_size

    Path(paths["act_dir"]).mkdir(parents=True, exist_ok=True)
    progress = load_progress(paths["extraction_progress"])

    # Run each requested set.
    for set_name in sets_to_run:
        out_key = f"act_{model_label}_{set_name}"
        if out_key not in paths:
            logger.warning(f"No output path defined for {out_key}, skipping")
            continue
        out_path = paths[out_key]

        prompts = splits[set_name]
        n = len(prompts)

        # Resume: if file exists and progress matches, skip
        progress_key = f"{model_label}__{set_name}"
        done_count = progress.get(progress_key, 0)

        if Path(out_path).exists() and done_count == n:
            logger.info(f"[{progress_key}] already done ({n} prompts), skipping")
            continue

        # Allocate or reload partial array
        if Path(out_path).exists() and done_count > 0:
            arr = np.load(out_path)
            assert arr.shape == (n, n_layers, hidden), f"Shape mismatch on resume: {arr.shape}"
            logger.info(f"[{progress_key}] resuming from index {done_count}")
        else:
            arr = np.zeros((n, n_layers, hidden), dtype=np.float32)
            done_count = 0

        for i in range(done_count, n):
            prompt = prompts[i]["text"]
            try:
                acts = extract_activations_last_token(model, tok, prompt, n_layers)
                # Sanity: no NaN/Inf
                if not np.all(np.isfinite(acts)):
                    logger.error(f"NaN/Inf in activations for prompt {i} ({prompts[i]['id']})")
                    raise ValueError("Non-finite activations")
                arr[i] = acts
            except Exception as e:
                logger.exception(f"Failed prompt {i} ({prompts[i]['id']}): {e}")
                raise

            # Periodic save
            if (i + 1) % 16 == 0 or (i + 1) == n:
                np.save(out_path, arr)
                progress[progress_key] = i + 1
                save_progress(paths["extraction_progress"], progress)
                volume.commit()
                logger.info(f"[{progress_key}] saved {i+1}/{n}")

        # Final save
        np.save(out_path, arr)
        progress[progress_key] = n
        save_progress(paths["extraction_progress"], progress)
        volume.commit()
        logger.info(f"[{progress_key}] DONE: shape {arr.shape}")

    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"Step 2 [{model_label}] complete.")


@app.local_entrypoint()
def main():
    """
    Two separate calls so we don't try to hold both 14B models in memory.
    Each call is its own Modal function invocation.
    """
    # Instruct: needs harmful_train, harmless_train, harmful_val, harmless_val
    extract_for_model.remote("instruct", [
        "harmful_train", "harmless_train", "harmful_val", "harmless_val"
    ])
    # Base: needs only harmful_train (for d_l = mu_instruct - mu_base on harmful)
    extract_for_model.remote("base", ["harmful_train"])