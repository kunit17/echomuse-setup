param([switch]$DisableStartup)
$ErrorActionPreference = 'Stop'
Stop-ScheduledTask -TaskName 'EchoMuse Whisper'
if ($DisableStartup) { Disable-ScheduledTask -TaskName 'EchoMuse Whisper' | Out-Null }
$stateFile = Join-Path $env:LOCALAPPDATA 'EchoMuseVoice/whisper-workers.json'
if (Test-Path -LiteralPath $stateFile) {
    $records = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
    foreach ($record in @($records.tunnel, $records.keepAlive)) {
        if ($null -eq $record) { continue }
        $worker = Get-Process -Id $record.Id -ErrorAction SilentlyContinue
        if ($null -eq $worker) { continue }
        # PIDs may have been reused by protected processes after a reboot.
        try { $workerStarted = $worker.StartTime } catch { continue }
        if ($null -eq $workerStarted) { continue }
        if ($workerStarted.ToUniversalTime().Ticks -eq ([datetime]$record.StartTime).ToUniversalTime().Ticks) {
            & "$env:WINDIR\System32\taskkill.exe" /PID $worker.Id /T /F | Out-Null
        }
    }
}
& "$env:WINDIR\System32\wsl.exe" -d Ubuntu --exec /bin/systemctl --user stop wyoming-whisper.service
if ($LASTEXITCODE -ne 0) { throw 'Could not stop the Whisper service.' }
Write-Output 'Whisper and its private tunnel stopped.'
