# Freeze or unfreeze every running project Python process (training lanes and downloads) without stopping it.
# Usage (from the project folder):
#   powershell -ExecutionPolicy Bypass -File experiments\pause_resume.ps1 -Action Pause
#   powershell -ExecutionPolicy Bypass -File experiments\pause_resume.ps1 -Action Resume
# Frozen processes use no CPU or GPU compute; they keep their memory (including some GPU memory) until resumed.
param([ValidateSet("Pause", "Resume")][string]$Action = "Pause")

Add-Type -Namespace Native -Name Proc -MemberDefinition @'
[DllImport("ntdll.dll")] public static extern int NtSuspendProcess(System.IntPtr handle);
[DllImport("ntdll.dll")] public static extern int NtResumeProcess(System.IntPtr handle);
'@

$project = (Resolve-Path "$PSScriptRoot\..").Path
$targets = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match "src\.(run_queue|run_ptbxl_|download_external)" }
foreach ($t in $targets) {
    $p = Get-Process -Id $t.ProcessId -ErrorAction SilentlyContinue
    if (-not $p) { continue }
    $code = if ($Action -eq "Pause") { [Native.Proc]::NtSuspendProcess($p.Handle) } else { [Native.Proc]::NtResumeProcess($p.Handle) }
    "{0} {1} -> {2}" -f $Action, $t.ProcessId, $(if ($code -eq 0) { "ok" } else { "error $code" })
}
Add-Content -Path "$project\experiments\logs\pause_resume.log" -Value ("{0} {1} {2} processes" -f (Get-Date -Format o), $Action, @($targets).Count)
