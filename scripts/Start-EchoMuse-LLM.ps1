param(
    [string]$PiHost = '192.0.2.10',
    [string]$PiUser = 'piuser',
    [string]$IdentityFile = "$env:USERPROFILE\.ssh\echomuse_ed25519",
    [string]$Distribution = 'Ubuntu',
    [switch]$SearchOnly
)
$ErrorActionPreference = 'Stop'
$mutex = New-Object System.Threading.Mutex($false, 'Local\EchoMuseLLMHost')
$ownsMutex = $false
try { $ownsMutex = $mutex.WaitOne(0) }
catch [System.Threading.AbandonedMutexException] { $ownsMutex = $true }
if (-not $ownsMutex) { exit 0 }
$logDir = Join-Path $env:LOCALAPPDATA 'EchoMuseVoice'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir 'llm-tunnel.log'
$stateFile = Join-Path $logDir 'llm-workers.json'
$keepAlive = $null
$tunnel = $null
function Stop-OwnedWorker($record) {
    if ($null -eq $record) { return }
    $worker = Get-Process -Id $record.Id -ErrorAction SilentlyContinue
    if ($null -eq $worker) { return }
    # A PID can belong to a protected process after a reboot; never stop it.
    try { $workerStarted = $worker.StartTime } catch { return }
    if ($null -eq $workerStarted) { return }
    if ($workerStarted.ToUniversalTime().Ticks -eq ([datetime]$record.StartTime).ToUniversalTime().Ticks) {
        & "$env:WINDIR\System32\taskkill.exe" /PID $worker.Id /T /F | Out-Null
    }
}
function Get-WorkerRecord($worker) {
    if ($null -eq $worker -or $worker.HasExited) { return $null }
    return @{ Id = $worker.Id; StartTime = $worker.StartTime.ToUniversalTime().ToString('o') }
}
function Save-Workers {
    @{ keepAlive = (Get-WorkerRecord $keepAlive); tunnel = (Get-WorkerRecord $tunnel) } | ConvertTo-Json | Set-Content -LiteralPath $stateFile
}
try {
    if (Test-Path -LiteralPath $stateFile) {
        $previous = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
        Stop-OwnedWorker $previous.tunnel
        Stop-OwnedWorker $previous.keepAlive
    }
    while ($true) {
        try {
            if ((Test-Path -LiteralPath $logFile) -and (Get-Item -LiteralPath $logFile).Length -gt 1MB) {
                Move-Item -LiteralPath $logFile -Destination ($logFile + '.previous') -Force
            }
            if (-not $SearchOnly) {
                & "$env:WINDIR\System32\wsl.exe" -d $Distribution --exec /bin/systemctl --user start ollama.service
                if ($LASTEXITCODE -ne 0) { throw 'Ollama service could not start.' }
            }
            & "$env:WINDIR\System32\wsl.exe" -d $Distribution --exec /bin/systemctl --user start echomuse-search.service
            if ($LASTEXITCODE -ne 0) { throw 'Web search service could not start.' }
            if ($null -eq $keepAlive -or $keepAlive.HasExited) {
                $keepAlive = Start-Process -FilePath "$env:WINDIR\System32\wsl.exe" -ArgumentList @('-d', $Distribution, '--exec', '/bin/sleep', 'infinity') -WindowStyle Hidden -PassThru
                Save-Workers
            }
            $connections = if ($SearchOnly) { 'web search' } else { 'Ollama and web search' }
            Add-Content -LiteralPath $logFile -Value ((Get-Date -Format o) + ' Connecting private ' + $connections + ' tunnels.')
            $sshArgs = @('-N', '-T', '-i', ('"' + $IdentityFile + '"'), '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ExitOnForwardFailure=yes', '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3')
            if (-not $SearchOnly) { $sshArgs += @('-R', '127.0.0.1:12434:127.0.0.1:11434') }
            $sshArgs += @('-R', '127.0.0.1:12435:127.0.0.1:11435')
            $sshArgs += "$PiUser@$PiHost"
            $tunnel = Start-Process -FilePath "$env:WINDIR\System32\OpenSSH\ssh.exe" -ArgumentList $sshArgs -RedirectStandardError (Join-Path $logDir 'llm-ssh-error.log') -RedirectStandardOutput (Join-Path $logDir 'llm-ssh-output.log') -WindowStyle Hidden -PassThru
            Save-Workers
            $tunnel.WaitForExit()
        } catch {
            Add-Content -LiteralPath $logFile -Value ((Get-Date -Format o) + ' ' + $_.Exception.Message)
        }
        Start-Sleep -Seconds 5
    }
} finally {
    Stop-OwnedWorker (Get-WorkerRecord $tunnel)
    Stop-OwnedWorker (Get-WorkerRecord $keepAlive)
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
