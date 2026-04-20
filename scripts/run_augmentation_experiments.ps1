$ErrorActionPreference = "Stop"

function Get-PythonCommand {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return "python"
    }

    if (Get-Command py -ErrorAction SilentlyContinue) {
        return "py -3"
    }

    throw "Python interpreter was not found. Configure python or py launcher first."
}

$pythonCmd = Get-PythonCommand
$projectRoot = Split-Path -Parent $PSScriptRoot

$experiments = @(
    @{ Config = "configs/experiments/train_exp1_hflip_light.yaml"; RunName = "exp1_hflip_light" },
    @{ Config = "configs/experiments/train_exp2_rotation15.yaml"; RunName = "exp2_rotation15" },
    @{ Config = "configs/experiments/train_exp3_color_jitter.yaml"; RunName = "exp3_color_jitter" },
    @{ Config = "configs/experiments/train_exp4_flip_rotation.yaml"; RunName = "exp4_flip_rotation" },
    @{ Config = "configs/experiments/train_exp5_flip_color.yaml"; RunName = "exp5_flip_color" },
    @{ Config = "configs/experiments/train_exp6_strong_aug.yaml"; RunName = "exp6_strong_aug" },
    @{ Config = "configs/experiments/train_exp7_convnext_tiny_randaug_ls.yaml"; RunName = "exp7_convnext_tiny_randaug_ls" },
    @{ Config = "configs/experiments/train_exp8_maxvit_t_focal_weighted.yaml"; RunName = "exp8_maxvit_t_focal_weighted" },
    @{ Config = "configs/experiments/train_exp9_convnext_base_384_balanced.yaml"; RunName = "exp9_convnext_base_384_balanced" },
    @{ Config = "configs/experiments/train_exp10_convnext_tiny_night_blur.yaml"; RunName = "exp10_convnext_tiny_night_blur" },
    @{ Config = "configs/experiments/train_exp11_resnet34_focal_sampler.yaml"; RunName = "exp11_resnet34_focal_sampler" },
    @{ Config = "configs/experiments/train_exp12_convnext_tiny_head_only.yaml"; RunName = "exp12_convnext_tiny_head_only" },
    @{ Config = "configs/experiments/train_exp13_maxvit_t_384_focal.yaml"; RunName = "exp13_maxvit_t_384_focal" },
    @{ Config = "configs/experiments/train_exp14_convnext_tiny_randaug_sampler.yaml"; RunName = "exp14_convnext_tiny_randaug_sampler" }
)

foreach ($experiment in $experiments) {
    Write-Host "Running $($experiment.RunName) with $($experiment.Config)"
    Invoke-Expression "$pythonCmd `"$projectRoot\main.py`" --config `"$projectRoot\$($experiment.Config)`""

    $summary = Get-ChildItem "$projectRoot\mlruns\local_artifacts\experiments" -Directory |
        Where-Object { $_.Name -like "*_$($experiment.RunName)" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if ($null -eq $summary) {
        Write-Warning "Could not find output directory for $($experiment.RunName)"
        continue
    }

    $summaryPath = Join-Path $summary.FullName "summary.json"
    if (Test-Path $summaryPath) {
        Invoke-Expression "$pythonCmd `"$projectRoot\scripts\update_history.py`" --summary `"$summaryPath`" --history `"$projectRoot\HISTORY.md`""
    } else {
        Write-Warning "summary.json not found for $($experiment.RunName)"
    }
}
