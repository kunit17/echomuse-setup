$ErrorActionPreference='Stop'
$completed=@{}
while ($true) {
    try {
        $status = Invoke-RestMethod -Uri 'http://127.0.0.1:8814/status' -TimeoutSec 5
        if ($status.status -eq 'complete_listening_pending') { break }
        if ($status.status -eq 'browser_probe' -and !$completed.ContainsKey($status.configuration)) {
            $label=$status.configuration
            try { & (Join-Path $PSScriptRoot 'browser_probe.ps1') -Label $label }
            catch { @{status='failed';error=$_.Exception.Message;label=$label} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot ('browser-'+$label+'.json')) -Encoding utf8 }
            $completed[$label]=$true
        }
    } catch { Write-Output $_.Exception.Message }
    Start-Sleep -Seconds 1
}
