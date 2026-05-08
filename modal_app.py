"""
Modal app definition. Defines the container image, volumes, and GPU configs
shared across all pipeline steps. Import this in every step script.
"""

import modal

# ---- Modal app ----
app = modal.App("phase1-directions")

# ---- Container image ----
# Pin versions for reproducibility. Update with care.
#
# IMPORTANT: We attach local Python source files to the image so that step
# scripts can do `from common import ...` and `from config import ...`
# inside the Modal container. Without this, only the entrypoint script
# (e.g. 0_setup_and_verify.py) is shipped, and `from modal_app import ...`
# would fail with ModuleNotFoundError.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.45.2",
        "accelerate==0.34.2",
        "datasets==3.0.1",
        "numpy==1.26.4",
        "scipy==1.14.1",
        "scikit-learn==1.5.2",
        "pandas==2.2.3",
        "matplotlib==3.9.2",
        "seaborn==0.13.2",
        "tqdm==4.66.5",
        "huggingface_hub==0.25.1",
        "sentencepiece==0.2.0",
        "protobuf==5.28.2",
    )
    .apt_install("git")
    # AdvBench is on github; download at build time so it's cached in the image.
    .run_commands(
        "git clone https://github.com/llm-attacks/llm-attacks.git /root/llm-attacks "
        "&& cp /root/llm-attacks/data/advbench/harmful_behaviors.csv /root/advbench.csv"
    )
    # Ship our own Python modules into the container at image-build time.
    # This makes `from modal_app import ...`, `from config import ...`,
    # and `from common import ...` work inside Modal functions.
    .add_local_python_source("modal_app", "config", "config_em", "common")
)

# ---- Persistent volume for outputs and cached activations ----
# All scripts read/write here. Survives between runs so you can resume.
volume = modal.Volume.from_name("phase1-directions-vol", create_if_missing=True)

# ---- Persistent HuggingFace cache volume ----
# Each Modal `modal run` starts a fresh container with no HF cache, so without
# this, every step (and every retry) re-downloads ~28GB Qwen2.5-14B + the judge
# model. With this volume mounted at the default HF cache path, the first run
# downloads everything once and all subsequent runs reuse it.
#
# Modal volumes auto-mount with current contents at container start and
# auto-commit on shutdown, so no explicit reload/commit is needed for caching
# to work.
hf_cache_volume = modal.Volume.from_name("hf-cache-vol", create_if_missing=True)

# ---- Secrets ----
# Only HuggingFace token is needed; the judge runs locally on Modal GPU.
hf_secret = modal.Secret.from_name("huggingface-secret")

# ---- Standard GPU configs ----
# A100-80GB fits Qwen2.5-14B in bf16 with headroom.
GPU_A100 = "A100-80GB"
GPU_A100_2X = "A100-80GB:2"  # for steps that load both base and instruct

# ---- Standard timeouts ----
TIMEOUT_SHORT = 60 * 30        # 30 min: setup, light compute
TIMEOUT_MEDIUM = 60 * 60 * 2   # 2 hr: extraction, filtering
TIMEOUT_LONG = 60 * 60 * 6     # 6 hr: layer sweep

# ---- Mount paths ----
VOL_MOUNT = "/vol"
HF_CACHE_MOUNT = "/root/.cache/huggingface"  # default HF cache path

# ---- Convenience: standard volumes dict to pass to @app.function ----
# Use this everywhere we load HF models so the cache persists.
VOLUMES = {
    VOL_MOUNT: volume,
    HF_CACHE_MOUNT: hf_cache_volume,
}
