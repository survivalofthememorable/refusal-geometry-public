# Reproduction Guide

End-to-end instructions to reproduce every number and figure in the paper from a clean machine.

## 1. Prerequisites

- Modal account with A100 80GB access ([modal.com](https://modal.com))
- HuggingFace account with access to FullFT-Medical (no gating but consider getting an HF token for stable downloads)
- Python 3.11+ on a Windows or Linux machine for orchestration
- ~12 GPU-hours of A100 budget
- ~50 GB of Modal volume storage

## 2. One-time Modal setup

```bash
pip install modal
python -m modal setup       # follow the prompts to authenticate
python -m modal secret create huggingface-secret HF_TOKEN=<your token>
python -m modal volume create phase1-directions-vol
python -m modal volume create hf-cache-vol     # for caching model weights
```

## 3. Pipeline order

The pipeline is split into ten sequential steps per phase. Each step is **resumable**: if interrupted, re-running the same command picks up at the last checkpointed prompt.

### Phase 1 (Qwen2.5-14B base ↔ instruct)

| Step | Script | Purpose | Wall time | Output |
|---|---|---|---|---|
| 0 | `0_setup.py` | Verify model loads, architecture, tokenizer parity | 5 min | verify report |
| 1 | `1_filter_prompts.py` | Generate 250+250 filtered candidates via judge | 25 min | prompt lists |
| 2 | `2_extract_activations.py` | Extract residual-stream activations on instruct + base | 30 min | activation tensors |
| 3 | `3_compute_directions.py` | Compute $r_l$ and $d_l$ at every layer | <1 min | direction npys |
| 4 | `4_layer_sweep.py::main` | Causal sweep: $r_l$, $d_l$, random across all 48 layers | 4 hr | sweep JSONs |
| 5 | `5_geometric_analysis.py` | Cosines, band IoU, geometric agreement | <1 min | geometry json |
| 6 | `6_harmbench_eval.py` | HarmBench evaluation under each ablation | 30 min | completions json |
| 7 | `7_capability_eval.py` | Alpaca capability evaluation | 15 min | results json |
| 8 | `8_orthogonal_decomposition.py` | Decompose $d_l = \alpha\hat{r}_l + d_l^\perp$ and ablate components | 30 min | decomposition results |
| 9 | `9_judge_audit.py` | Re-score with WildGuard, compute pattern-vs-WG delta | 25 min | dual-scored completions |

**Total Phase 1: ~6 hours A100, ~$12.**

### Phase 2 (instruct → FullFT-Medical)

| Step | Script | Purpose | Wall time | Output |
|---|---|---|---|---|
| 0 | `0_setup_em.py` | Verify EM model + chat template fallback | 12 min (first download) | verify report |
| 1 | `1_filter_prompts_em.py` | Re-filter prompts on EM, intersect with instruct | 22 min | filter results |
| 2 | `2_extract_activations_em.py` | Extract activations on EM | 5 min | activation tensors |
| 3 | `3_compute_directions_em.py` | Compute $r_{\text{em}}$ and $d_{\text{em}}$ | <1 min | direction npys |
| 4 | `4_layer_sweep_em.py::main` | Sweep $r_{\text{em}}$ + random on EM | 4 hr | sweep JSONs |
| 5 | `5_geometric_analysis_em.py` | Cross-model geometry (cosine, perp_frac, band IoU) | <1 min | geometry json |
| 4cross | `4cross_three_way.py` | Three-way decomposition + ablation at $l^*_{\text{em}}$ | 1.5 hr | cross_results |
| 9 | `9_judge_audit_em.py` | Dual-judge audit | 5 min | dual-scored completions |

**Total Phase 2: ~6 hours A100, ~$12.**

## 4. Critical environment hardening

Two issues that bit us and that future reproducers should know about:

### 4.1 EM model needs hardcoded model ID

Modal does **not** auto-forward local environment variables to GPU containers. Set the EM model ID directly in `config_em.py`:

```python
EM_MODEL_ID_DEFAULT = "ModelOrganismsForEM/Qwen2.5-14B-Instruct_full-ft"
```

Do not rely on `$env:EM_MODEL_ID`. (We tried; it fails silently because the env var is read in the local Python process, but the same import on the container sees an empty env.)

### 4.2 transformers <4.46 doesn't auto-load standalone chat_template.jinja

FullFT-Medical ships its chat template in a separate file rather than embedded in `tokenizer_config.json`. Our pinned transformers version (4.45.2) does not auto-load this file. The fix is in `common.py`'s `load_model_and_tokenizer`:

```python
if getattr(tok, "chat_template", None) is None:
    from huggingface_hub import hf_hub_download
    tpl_path = hf_hub_download(repo_id=model_id, filename="chat_template.jinja",
                               revision=revision)
    with open(tpl_path) as f:
        tok.chat_template = f.read()
```

This is backward compatible — Phase 1 models have their template embedded so the fallback is a no-op for them.

## 5. Long-running jobs and preemption

Modal A100s can be preempted under load. Two of our cross-ablation runs were preempted (the second time *after* the result aggregation had been computed but before the JSON write).

**Use `--detach` for any run longer than 30 minutes:**

```bash
python -m modal run --detach 4cross_three_way.py
```

This makes the run robust to local-CLI disconnects. The remote function continues server-side; you monitor via the Modal dashboard or by polling the volume contents.

The pipeline is **per-prompt resumable** at all stages — re-running a preempted script picks up at the last 10-prompt checkpoint with no data loss.

## 6. Reproducing figures and tables locally

After the Modal pipeline completes, on your local machine:

```bash
# Pull artifacts (only needs ~50 MB of metadata, not the activation tensors)
.\download\download_all_results.ps1

# Generate every figure in the paper
cd analysis
pip install -r requirements.txt
python make_figures.py

# Generate every CSV table
python make_csvs.py
```

Both scripts are CPU-only and run in seconds. They depend only on `numpy` and `matplotlib`.

## 7. Verifying you got the same numbers

Cross-reference your `csv_outputs/summary_locked_numbers.csv` against the locked numbers in `report/technical_report.pdf`. The 60+ key-value pairs there should match exactly (within float32 noise, ~1e-5 tolerance). If any differ by more than 1%, something has changed in the upstream models or judge weights and you should investigate before drawing conclusions.

## 8. Total resource estimate

| Resource | Phase 1 | Phase 2 | Total |
|---|---|---|---|
| A100 GPU-hours | 6 | 6 | 12 |
| Modal cost (Nov 2025 rates) | $12 | $12 | $24 |
| Volume storage | 25 GB | 25 GB | 50 GB |
| Wall time (with checkpoints) | ~7 hr | ~7 hr | ~14 hr |
| Local compute (figures + CSVs) | — | — | <1 min |
