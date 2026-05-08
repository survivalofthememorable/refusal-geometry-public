# =====================================================================
#  download_all_results.ps1
#
#  Pulls every artifact from the phase1-directions-vol Modal volume
#  into a local results/ directory tree mirroring the volume layout.
#  Run from the project root after Phase 2 + cross-ablation + judge
#  audit have completed on Modal.
#
#  Usage (PowerShell):
#      .\download_all_results.ps1
# =====================================================================

$ErrorActionPreference = "Stop"
$VOL = "phase1-directions-vol"

# ---- create local layout ----------------------------------------------
$dirs = @(
    "results/phase1/00_setup",
    "results/phase1/01_filter",
    "results/phase1/02_activations",
    "results/phase1/03_directions",
    "results/phase1/04_layer_sweep",
    "results/phase1/05_geometry",
    "results/phase1/06_harmbench",
    "results/phase1/07_capability",
    "results/phase1/08_orthogonal",
    "results/phase1/09_audit",
    "results/phase2/00_verify",
    "results/phase2/01_filter",
    "results/phase2/02_activations",
    "results/phase2/03_directions",
    "results/phase2/04_layer_sweep",
    "results/phase2/05_geometry",
    "results/phase2/04cross_three_way"
)
foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path $d | Out-Null }

function Get-ModalFile($remote, $local) {
    Write-Host ">> $remote  ->  $local"
    try {
        python -m modal volume get $VOL $remote $local --force
    } catch {
        Write-Warning "Skipping (not found on volume): $remote"
    }
}

# ---- Phase 1 (frozen reference) --------------------------------------
Get-ModalFile "phase1/01_filter_results.json"        "results/phase1/01_filter/filter_results.json"
Get-ModalFile "phase1/01_prompt_lists.json"          "results/phase1/01_filter/prompt_lists.json"

Get-ModalFile "phase1/03_directions/r_l.npy"         "results/phase1/03_directions/r_l.npy"
Get-ModalFile "phase1/03_directions/r_l_hat.npy"     "results/phase1/03_directions/r_l_hat.npy"
Get-ModalFile "phase1/03_directions/d_l.npy"         "results/phase1/03_directions/d_l.npy"
Get-ModalFile "phase1/03_directions/d_l_hat.npy"     "results/phase1/03_directions/d_l_hat.npy"
Get-ModalFile "phase1/03_directions/r_norms.npy"     "results/phase1/03_directions/r_norms.npy"
Get-ModalFile "phase1/03_directions/d_norms.npy"     "results/phase1/03_directions/d_norms.npy"
Get-ModalFile "phase1/03_directions/metadata.json"   "results/phase1/03_directions/metadata.json"

Get-ModalFile "phase1/04_layer_sweep/sweep_r.json"       "results/phase1/04_layer_sweep/sweep_r.json"
Get-ModalFile "phase1/04_layer_sweep/sweep_d.json"       "results/phase1/04_layer_sweep/sweep_d.json"
Get-ModalFile "phase1/04_layer_sweep/sweep_random.json"  "results/phase1/04_layer_sweep/sweep_random.json"
Get-ModalFile "phase1/04_layer_sweep/selected_layers.json" "results/phase1/04_layer_sweep/selected_layers.json"

Get-ModalFile "phase1/05_geometry/geometry.json"     "results/phase1/05_geometry/geometry.json"

Get-ModalFile "phase1/06_harmbench/completions.json"          "results/phase1/06_harmbench/completions.json"
Get-ModalFile "phase1/06_harmbench/results.json"              "results/phase1/06_harmbench/results.json"
Get-ModalFile "phase1/06_harmbench/completions_wildguard.json" "results/phase1/06_harmbench/completions_wildguard.json"
Get-ModalFile "phase1/06_harmbench/results_wildguard.json"    "results/phase1/06_harmbench/results_wildguard.json"

Get-ModalFile "phase1/07_capability/results.json"    "results/phase1/07_capability/results.json"

Get-ModalFile "phase1/04b_orthogonal/results.json"          "results/phase1/04b_orthogonal/results.json"
Get-ModalFile "phase1/04b_orthogonal/completions.json"      "results/phase1/04b_orthogonal/completions.json"
Get-ModalFile "phase1/04b_orthogonal/directions.npz"        "results/phase1/04b_orthogonal/directions.npz"

Get-ModalFile "phase1/09_audit/multilingual_audit.json"   "results/phase1/09_audit/multilingual_audit.json"

# ---- Phase 2 ---------------------------------------------------------
Get-ModalFile "phase2/00_verify_report.json"         "results/phase2/00_verify/verify_report.json"
Get-ModalFile "phase2/01_filter_results_em.json"     "results/phase2/01_filter/filter_results_em.json"
Get-ModalFile "phase2/01_prompt_lists_em.json"       "results/phase2/01_filter/prompt_lists_em.json"

Get-ModalFile "phase2/03_directions/r_em.npy"        "results/phase2/03_directions/r_em.npy"
Get-ModalFile "phase2/03_directions/r_em_hat.npy"    "results/phase2/03_directions/r_em_hat.npy"
Get-ModalFile "phase2/03_directions/d_em.npy"        "results/phase2/03_directions/d_em.npy"
Get-ModalFile "phase2/03_directions/d_em_hat.npy"    "results/phase2/03_directions/d_em_hat.npy"
Get-ModalFile "phase2/03_directions/r_em_norms.npy"  "results/phase2/03_directions/r_em_norms.npy"
Get-ModalFile "phase2/03_directions/d_em_norms.npy"  "results/phase2/03_directions/d_em_norms.npy"
Get-ModalFile "phase2/03_directions/metadata.json"   "results/phase2/03_directions/metadata.json"

Get-ModalFile "phase2/04_layer_sweep/sweep_r_em.json"      "results/phase2/04_layer_sweep/sweep_r_em.json"
Get-ModalFile "phase2/04_layer_sweep/sweep_random_em.json" "results/phase2/04_layer_sweep/sweep_random_em.json"
Get-ModalFile "phase2/04_layer_sweep/selected_layers_em.json" "results/phase2/04_layer_sweep/selected_layers_em.json"

Get-ModalFile "phase2/05_geometry/geometry_em.json"  "results/phase2/05_geometry/geometry_em.json"

Get-ModalFile "phase2/04cross_three_way/directions.npz"               "results/phase2/04cross_three_way/directions.npz"
Get-ModalFile "phase2/04cross_three_way/completions.json"             "results/phase2/04cross_three_way/completions.json"
Get-ModalFile "phase2/04cross_three_way/results.json"                 "results/phase2/04cross_three_way/results.json"
Get-ModalFile "phase2/04cross_three_way/completions_dual_scored.json" "results/phase2/04cross_three_way/completions_dual_scored.json"
Get-ModalFile "phase2/04cross_three_way/results_dual.json"            "results/phase2/04cross_three_way/results_dual.json"

Write-Host "`n[OK] All available artifacts pulled into ./results/"
Write-Host "Inspect with:  Get-ChildItem -Recurse -Path results | Select-Object FullName, Length"