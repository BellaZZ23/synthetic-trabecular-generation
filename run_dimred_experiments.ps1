param(
    [string]$PythonExe = "python",
    [string]$PipelineScript = ".\dim_reduction_pipeline.py",
    [string]$DatasetDir = ".\dataset",
    [string]$BaseOutDir = ".\output\experiments",
    [int[]]$Components = @(2,4,8,12,16,24,32),
    [string[]]$FeatureTypes = @("texture","pixels"),
    [string[]]$SliceModes = @("mid","multi","mip"),
    [int[]]$Seeds = @(11,22,33,44,55),
    [int]$NSlices = 5,
    [int]$ImageSize = 64,
    [double]$TestSplit = 0.2
)

$ErrorActionPreference = "Stop"

function Run-Experiment {
    param(
        [string]$FeatureType,
        [string]$SliceMode,
        [int]$NComponents,
        [int]$Seed,
        [string]$RpMethod
    )

    $outDir = Join-Path $BaseOutDir ("feat_{0}\slice_{1}\k_{2}\seed_{3}\rp_{4}" -f $FeatureType, $SliceMode, $NComponents, $Seed, $RpMethod)
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    $args = @(
        $PipelineScript,
        "--dataset-dir", $DatasetDir,
        "--outdir", $outDir,
        "--n-components", $NComponents,
        "--slice-mode", $SliceMode,
        "--n-slices", $NSlices,
        "--image-size", $ImageSize,
        "--test-split", $TestSplit,
        "--seed", $Seed,
        "--feature-type", $FeatureType,
        "--rp-method", $RpMethod
    )

    Write-Host "Running: feature=$FeatureType slice=$SliceMode k=$NComponents seed=$Seed rp=$RpMethod"
    & $PythonExe @args

    if ($LASTEXITCODE -ne 0) {
        throw "Experiment failed: feature=$FeatureType slice=$SliceMode k=$NComponents seed=$Seed rp=$RpMethod"
    }
}

foreach ($feature in $FeatureTypes) {
    foreach ($sliceMode in $SliceModes) {
        foreach ($k in $Components) {
            foreach ($seed in $Seeds) {
                Run-Experiment -FeatureType $feature -SliceMode $sliceMode -NComponents $k -Seed $seed -RpMethod "both"
            }
        }
    }
}

Write-Host ""
Write-Host "All experiments completed."
Write-Host "Results saved under: $BaseOutDir"