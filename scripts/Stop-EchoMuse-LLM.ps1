param([switch]$DisableStartup)
$ErrorActionPreference = 'Stop'
Stop-ScheduledTask -TaskName 'EchoMuse LLM'
if ($DisableStartup) { Disable-ScheduledTask -TaskName 'EchoMuse LLM' | Out-Null }
$stateFile = Join-Path $env:LOCALAPPDATA 'EchoMuseVoice/llm-workers.json'
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
Write-Output 'Private Ollama and search tunnels stopped; the existing Ollama service is unchanged.'
