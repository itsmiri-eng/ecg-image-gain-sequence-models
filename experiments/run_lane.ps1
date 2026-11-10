# Run one lane of training jobs one after another (used from 5 Nov 2026 with a CPU cap).
# Items: <config>:<seed> for src.run_queue (e.g. hybrid_vit:45, fg_cnn_pool:46) or wide:<seed> for the
# capacity-matched head (src.run_ptbxl_wide_head). Both skip runs that already exist.
# Started with:  cmd /c start "" /low /affinity FF powershell -ExecutionPolicy Bypass -File experiments\run_lane.ps1 <items>
# (/affinity FF = the first 8 of 16 logical CPUs, so training never takes more than half the processor;
#  child processes inherit the priority and the CPU limit.)
$python = "$PSScriptRoot\..\.venv\Scripts\python.exe"
Set-Location "$PSScriptRoot\.."
foreach ($item in $args) {
    $name, $seed = $item.Split(":")
    if ($name -eq "wide") {
        & $python -X utf8 -u -m src.run_ptbxl_wide_head --device cuda --seeds $seed
    } else {
        & $python -X utf8 -u -m src.run_queue --device cuda "${name}:${seed}"
    }
}
