# Run one lane of the 11 Nov 2026 amendment runs one after another (docs/protocol_amendment_2026-11-11.md).
# Items: <config>:<seed>, where experiments/configs/ptbxl_v2_<config>.json exists
# (e.g. resnet18:42, fg_resnet18:43, long40_cnn_pool_wide:44). src.run_ptbxl_extra skips runs that already exist.
# Started hidden, at Idle priority, on 8 logical CPUs (6-13):
#   cmd /c start "" /min /low /affinity 3FC0 powershell -WindowStyle Hidden -ExecutionPolicy Bypass -File experiments\run_lane_extra.ps1 <items>
$python = "$PSScriptRoot\..\.venv\Scripts\python.exe"
Set-Location "$PSScriptRoot\.."
foreach ($item in $args) {
    $name, $seed = $item.Split(":")
    & $python -X utf8 -u -m src.run_ptbxl_extra --config "experiments/configs/ptbxl_v2_$name.json" --seeds $seed --device cuda
}
