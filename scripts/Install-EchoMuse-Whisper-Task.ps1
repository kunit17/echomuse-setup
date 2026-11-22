param([switch]$IncludeOrpheus)
# Run in Windows PowerShell as the desktop user, after the WSL install.
# Defaults match this private household deployment; edit host script parameters
# and task arguments together if adapting it to another account or network.
$ErrorActionPreference = 'Stop'
$installDir = Join-Path $env:LOCALAPPDATA 'EchoMuseVoice'
New-Item -ItemType Directory -Force -Path $installDir | Out-Null
foreach ($name in @('Start-EchoMuse-Whisper.ps1','Stop-EchoMuse-Whisper.ps1')) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $name) -Destination (Join-Path $installDir $name)
}
$installedScript = Join-Path $installDir 'Start-EchoMuse-Whisper.ps1'
$taskUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction -Execute "$env:WINDIR/System32/WindowsPowerShell/v1.0/powershell.exe" -Argument ('-NoProfile -WindowStyle Hidden -File "' + $installedScript + '"')
if ($IncludeOrpheus) { $action.Arguments += ' -IncludeOrpheus' }
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $taskUser
$principal = New-ScheduledTaskPrincipal -UserId $taskUser -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName 'EchoMuse Whisper' -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description 'Keep WSL Whisper active and forward only Pi loopback port 10300 over SSH.' -Force | Out-Null
Start-ScheduledTask -TaskName 'EchoMuse Whisper'
Get-ScheduledTask -TaskName 'EchoMuse Whisper' | Select-Object TaskName,State
