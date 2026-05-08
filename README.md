# Analysis scripts

Two scripts that consume the JSON artifacts pulled by `download_all_results.ps1` and produce the figures and CSV tables used in the paper.

## Setup

```bash
pip install -r requirements.txt
```

## Producing all figures (PDF + PNG)

```bash
python make_figures.py
```

Outputs land in `figures/`:

| File | Content |
|---|---|
| `fig_01_geometry.{pdf,png}` | Cross-model cosine and perp-fraction per layer (Figure 1) |
| `fig_02_sweeps.{pdf,png}` | Per-layer $S_l$ for r_l (Phase 1) vs r_em (Phase 2) (Figure 2) |
| `fig_03_ortho_mirror.{pdf,png}` | Side-by-side: Phase 1 d_l decomposition and Phase 2 r_em decomposition |
| `fig_04_summary.{pdf,png}` | Phase 2 dual-judge bars + WildGuard-minus-pattern $\Delta$ panel |

Each script falls back to locked numbers if the underlying JSON file is missing — useful for rebuilding figures without re-downloading.

## Producing all CSV tables

```bash
python make_csvs.py
```

Outputs land in `csv_outputs/`:

| File | Content |
|---|---|
| `table1_phase1_dualjudge.csv` | Phase 1 Table 1 — pattern, WildGuard, $\Delta$ |
| `table2_phase1_orthogonal.csv` | Phase 1 Table 2 — orthogonal decomposition |
| `table3_phase2_threeway.csv` | Phase 2 Table 3 — six conditions × dual-judge × capability |
| `geometry_per_layer_em.csv` | 48 rows: per-layer cos and perp_frac |
| `sweep_per_layer_phase1.csv` | 48 rows: $S$ for r_l, d_l, random |
| `sweep_per_layer_phase2.csv` | 48 rows: $S$ for r_em, random |
| `direction_norms.csv` | Per-layer L2 norms for r_l, d_l, r_em, d_em |
| `filter_intersection.csv` | Phase 2 prompt-filter audit |
| `summary_locked_numbers.csv` | Every named number in the paper, with a description column |

## Reproducibility note

Both scripts depend only on numpy and matplotlib (no torch or transformers). They are CPU-only and complete in seconds. They are deterministic — no random seeds — so the figures and CSVs are byte-stable as long as the input JSONs are unchanged.
