param([switch]$SearchOnly)
# Run in Windows PowerShell as the desktop user, after the WSL install.
# Defaults match this private household deployment; edit host script parameters
# and task arguments together if adapting it to another account or network.
$ErrorActionPreference = 'Stop'
$installDir = Join-Path $env:LOCALAPPDATA 'EchoMuseVoice'
New-Item -ItemType Directory -Force -Path $installDir | Out-Null
foreach ($name in @('Start-EchoMuse-LLM.ps1','Stop-EchoMuse-LLM.ps1')) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $name) -Destination (Join-Path $installDir $name)
}
$installedScript = Join-Path $installDir 'Start-EchoMuse-LLM.ps1'
$taskUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction -Execute "$env:WINDIR/System32/WindowsPowerShell/v1.0/powershell.exe" -Argument ('-NoProfile -WindowStyle Hidden -File "' + $installedScript + '"')
if ($SearchOnly) { $action.Arguments += ' -SearchOnly' }
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $taskUser
$principal = New-ScheduledTaskPrincipal -UserId $taskUser -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
$taskDescription = if ($SearchOnly) { 'Keep WSL web search active and forward only Pi loopback 12435; MiniMax is called directly by Home Assistant.' } else { 'Connect Home Assistant to WSL Ollama and search over private SSH loopback tunnels.' }
Register-ScheduledTask -TaskName 'EchoMuse LLM' -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description $taskDescription -Force | Out-Null
Start-ScheduledTask -TaskName 'EchoMuse LLM'
Get-ScheduledTask -TaskName 'EchoMuse LLM' | Select-Object TaskName,State
