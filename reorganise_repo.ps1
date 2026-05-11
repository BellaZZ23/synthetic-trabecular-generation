# ============================================================
# Repo reorganisation — PowerShell (Windows)
# ============================================================
# Run from: C:\Users\Isabella\OneDrive - zeki\Documents\Research\synthetic_trabeculae
#
# Usage:
#   cd "C:\Users\Isabella\OneDrive - zeki\Documents\Research\synthetic_trabeculae"
#   .\reorganise_repo.ps1
# ============================================================

Write-Host "=== Step 1: Create new directory structure ===" -ForegroundColor Cyan

$dirs = @(
    "scripts\plotting",
    "configs",
    "fe_coupling",
    "archive\paper_v1",
    "archive\generators",
    "archive\plots",
    "archive\preprocessing",
    "archive\experiments",
    "archive\old_pipelines"
)
foreach ($d in $dirs) {
    if (!(Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}

Write-Host "=== Step 2: Move active scripts to scripts/ ===" -ForegroundColor Cyan

# Generator (published v15)
if (Test-Path "synthetic_trabecular_v15_morphometric_control") {
    Move-Item "synthetic_trabecular_v15_morphometric_control\*" "scripts\" -Force -ErrorAction SilentlyContinue
    Move-Item "synthetic_trabecular_v15_morphometric_control" "archive\generators\v15_original_folder" -Force
    Write-Host "  Moved v15 generator -> scripts/"
}

# Main pipeline v2
if (Test-Path "full_pipeline_v2") {
    Move-Item "full_pipeline_v2\*" "scripts\" -Force -ErrorAction SilentlyContinue
    Move-Item "full_pipeline_v2" "archive\old_pipelines\full_pipeline_v2_folder" -Force
    Write-Host "  Moved full_pipeline_v2 -> scripts/"
}

# Dim reduction pipeline v2
if (Test-Path "dim_reduction_pipeline_v2") {
    Move-Item "dim_reduction_pipeline_v2\*" "scripts\" -Force -ErrorAction SilentlyContinue
    Move-Item "dim_reduction_pipeline_v2" "archive\old_pipelines\dim_reduction_pipeline_v2_folder" -Force
    Write-Host "  Moved dim_reduction_pipeline_v2 -> scripts/"
}

# QSVM comparison v2
if (Test-Path "qsvm_comparison_v2") {
    Move-Item "qsvm_comparison_v2\*" "scripts\" -Force -ErrorAction SilentlyContinue
    Move-Item "qsvm_comparison_v2" "archive\paper_v1\qsvm_comparison_v2_folder" -Force
    Write-Host "  Moved qsvm_comparison_v2 -> scripts/"
}

# Quantum regression
if (Test-Path "quantum_regression") {
    Move-Item "quantum_regression\*" "scripts\" -Force -ErrorAction SilentlyContinue
    Move-Item "quantum_regression" "archive\paper_v1\quantum_regression_folder" -Force
    Write-Host "  Moved quantum_regression -> scripts/"
}

# Kernel diagnostics
if (Test-Path "kernel_diagnostics") {
    Move-Item "kernel_diagnostics\*" "scripts\" -Force -ErrorAction SilentlyContinue
    Move-Item "kernel_diagnostics" "archive\paper_v1\kernel_diagnostics_folder" -Force
    Write-Host "  Moved kernel_diagnostics -> scripts/"
}

# Extract fold scores
if (Test-Path "extract_fold_scores") {
    Move-Item "extract_fold_scores\*" "scripts\" -Force -ErrorAction SilentlyContinue
    Move-Item "extract_fold_scores" "archive\paper_v1\extract_fold_scores_folder" -Force
    Write-Host "  Moved extract_fold_scores -> scripts/"
}

# 5x2 CV test
if (Test-Path "run_5x2cv_test") {
    Move-Item "run_5x2cv_test\*" "scripts\" -Force -ErrorAction SilentlyContinue
    Move-Item "run_5x2cv_test" "archive\paper_v1\run_5x2cv_test_folder" -Force
    Write-Host "  Moved run_5x2cv_test -> scripts/"
}

# Independent datasets test
if (Test-Path "independent_datasets_test") {
    Move-Item "independent_datasets_test\*" "scripts\" -Force -ErrorAction SilentlyContinue
    Move-Item "independent_datasets_test" "archive\paper_v1\independent_datasets_test_folder" -Force
    Write-Host "  Moved independent_datasets_test -> scripts/"
}

# DICOM multiframe to targets
if (Test-Path "pipeline_voi1_multiframe_dcm_to_targets") {
    Move-Item "pipeline_voi1_multiframe_dcm_to_targets\*" "scripts\" -Force -ErrorAction SilentlyContinue
    Move-Item "pipeline_voi1_multiframe_dcm_to_targets" "archive\preprocessing\multiframe_dcm_folder" -Force
    Write-Host "  Moved pipeline_voi1_multiframe_dcm_to_targets -> scripts/"
}

Write-Host ""
Write-Host "=== Step 3: Move plotting scripts ===" -ForegroundColor Cyan

$plotFolders = @{
    "make_transition_figure"    = "make_transition_figure"
    "replot_qsvm_bar_chart"     = "replot_qsvm_bar_chart"
    "plot_all_embeddings_bvtv"  = "plot_all_embeddings_bvtv"
}

foreach ($folder in $plotFolders.Keys) {
    if (Test-Path $folder) {
        Move-Item "$folder\*" "scripts\plotting\" -Force -ErrorAction SilentlyContinue
        Move-Item $folder "archive\plots\$($plotFolders[$folder])_folder" -Force
        Write-Host "  Moved $folder -> scripts/plotting/"
    }
}

Write-Host ""
Write-Host "=== Step 4: Move config ===" -ForegroundColor Cyan

if (Test-Path "best_v15_params") {
    Move-Item "best_v15_params\*" "configs\" -Force -ErrorAction SilentlyContinue
    Move-Item "best_v15_params" "archive\old_pipelines\best_v15_params_folder" -Force
    Write-Host "  Moved best_v15_params -> configs/"
}
# If it's a JSON file directly
if (Test-Path "best_v15_params.json") {
    Move-Item "best_v15_params.json" "configs\optimization_result.json" -Force
    Write-Host "  Moved best_v15_params.json -> configs/optimization_result.json"
}

Write-Host ""
Write-Host "=== Step 5: Archive old generators ===" -ForegroundColor Cyan

$oldGens = @(
    "synthetic_trabecular_v13_fullfov_microct_ridge_skeleton",
    "synthetic_trabecular_v14_morphometric_control",
    "synthetic_trabecular_v16_morphometric_control.py.bak"
)
foreach ($g in $oldGens) {
    if (Test-Path $g) {
        Move-Item $g "archive\generators\" -Force
        Write-Host "  Archived $g"
    }
}

# Merge existing archive_generators into new location
if (Test-Path "archive_generators") {
    Move-Item "archive_generators\*" "archive\generators\" -Force -ErrorAction SilentlyContinue
    Remove-Item "archive_generators" -Recurse -Force
    Write-Host "  Merged archive_generators/ into archive/generators/"
}

Write-Host ""
Write-Host "=== Step 6: Archive paper experiment folders ===" -ForegroundColor Cyan

$paperArchive = @(
    "qsvm_comparison",
    "qsvm_tighter_v1",
    "independent_fold_tests",
    "scipy_wilcoxon_test"
)
foreach ($f in $paperArchive) {
    if (Test-Path $f) {
        Move-Item $f "archive\paper_v1\" -Force
        Write-Host "  Archived $f -> archive/paper_v1/"
    }
}

Write-Host ""
Write-Host "=== Step 7: Archive experiment folders ===" -ForegroundColor Cyan

$expArchive = @(
    "run_dimred_experiments",
    "run_large_sweep",
    "parameter_importance_analysis"
)
foreach ($f in $expArchive) {
    if (Test-Path $f) {
        Move-Item $f "archive\experiments\" -Force
        Write-Host "  Archived $f -> archive/experiments/"
    }
}

Write-Host ""
Write-Host "=== Step 8: Archive preprocessing ===" -ForegroundColor Cyan

$preArchive = @(
    "pipeline_voi1_dicom_to_targets",
    "pipeline_targets_aggregate",
    "remove_outliers",
    "remove_pca_outliers",
    "real_voi1_ref"
)
foreach ($f in $preArchive) {
    if (Test-Path $f) {
        Move-Item $f "archive\preprocessing\" -Force
        Write-Host "  Archived $f -> archive/preprocessing/"
    }
}

Write-Host ""
Write-Host "=== Step 9: Archive remaining visuals/patches ===" -ForegroundColor Cyan

$visArchive = @(
    "generate_pipeline_visuals_v2",
    "patch_full_pipeline"
)
foreach ($f in $visArchive) {
    if (Test-Path $f) {
        Move-Item $f "archive\plots\" -Force
        Write-Host "  Archived $f -> archive/plots/"
    }
}

Write-Host ""
Write-Host "=== Step 10: Archive superseded v1 pipelines ===" -ForegroundColor Cyan

$oldPipelines = @(
    "full_pipeline",
    "dim_reduction_pipeline"
)
foreach ($f in $oldPipelines) {
    if (Test-Path $f) {
        Move-Item $f "archive\old_pipelines\" -Force
        Write-Host "  Archived $f -> archive/old_pipelines/"
    }
}

Write-Host ""
Write-Host "=== Step 11: Delete build artefacts ===" -ForegroundColor Cyan

if (Test-Path ".venv") {
    Remove-Item ".venv" -Recurse -Force
    Write-Host "  Deleted .venv/"
}
if (Test-Path "__pycache__") {
    Remove-Item "__pycache__" -Recurse -Force
    Write-Host "  Deleted __pycache__/"
}
# Clean all nested __pycache__
Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=== Step 12: Create FE coupling workspace ===" -ForegroundColor Cyan

$readme = "# FE Coupling Module`n`nMechanically-aware generator coupling with micro-FE solver.`n`n## Scripts`n- step1_patch_test.py   -- Single-element patch test validation`n- step2_voxel_fe.py     -- Voxel-to-hex mesh conversion + uniaxial compression`n- step3_generator_coupling.py -- Couple v15 generator with FE solver (TODO)`n`n## Setup`npython -m venv femenv`nfemenv\Scripts\activate`npip install scikit-fem numpy scipy matplotlib"
Set-Content -Path "fe_coupling\README.md" -Value $readme
Write-Host "  Created fe_coupling/README.md"

Write-Host ""
Write-Host "=== Step 13: Verify ===" -ForegroundColor Green
Write-Host ""
Write-Host "Root contents:" -ForegroundColor Yellow
Get-ChildItem -Name | Sort-Object
Write-Host ""
Write-Host "scripts/ contents:" -ForegroundColor Yellow
if (Test-Path "scripts") { Get-ChildItem "scripts" -Name -Recurse | Sort-Object }
Write-Host ""
Write-Host "archive/ contents:" -ForegroundColor Yellow
if (Test-Path "archive") { Get-ChildItem "archive" -Name | Sort-Object }

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "DONE! Review the structure above, then run:" -ForegroundColor Green
Write-Host ""
Write-Host '  git add -A' -ForegroundColor White
Write-Host '  git commit -m "refactor: reorganise repo for FE coupling phase' -ForegroundColor White
Write-Host ""
Write-Host '  - Active scripts consolidated in scripts/ with plotting/ subfolder' -ForegroundColor Gray
Write-Host '  - Published paper code archived to archive/paper_v1/' -ForegroundColor Gray
Write-Host '  - Old generators (v13, v14, v16) archived to archive/generators/' -ForegroundColor Gray
Write-Host '  - Build artefacts cleaned (.venv, __pycache__)' -ForegroundColor Gray
Write-Host '  - New fe_coupling/ workspace for mechanical awareness work"' -ForegroundColor Gray
Write-Host ""
Write-Host '  git push' -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Green
