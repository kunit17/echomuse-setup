# ============================================================
#  3-chrome-webusb-flag.ps1
#
#  EchoMuse's provisioning wizard uses WebUSB, which Chrome only
#  permits on a secure origin. The controller runs on the Pi, so the
#  wizard page is served from http://192.0.2.10:8768 - plain HTTP,
#  which Chrome treats as insecure.
#
#  This adds that ONE origin to Chrome's
#  chrome://flags/#unsafely-treat-insecure-origin-as-secure allowlist,
#  which is the workaround EchoMuse's own docs prescribe
#  (controller/DOCS.md, docs/faq.md).
#
#  Run in a NORMAL PowerShell (admin not required). Safe to re-run.
#  Chrome MUST be closed - it rewrites Local State on exit and would
#  overwrite the change. This script refuses to run if Chrome is up.
# ============================================================
$ErrorActionPreference = 'Stop'

$ORIGIN = 'http://192.0.2.10:8768'
$FLAG   = 'unsafely-treat-insecure-origin-as-secure'

$chromeData = "$env:LOCALAPPDATA\Google\Chrome\User Data"
$localState = Join-Path $chromeData 'Local State'

if (-not (Test-Path $localState)) {
  Write-Host "Chrome's Local State not found at: $localState"
  Write-Host "Start Chrome once, close it, then run this again."
  exit 1
}

# Chrome holds a lock file while running; if it exists, it is open.
$lock = Join-Path $chromeData 'SingletonLock'
if (Test-Path $lock) {
  Write-Host "Chrome is RUNNING. Close all Chrome windows (and the background process), then run this again."
  Write-Host "Otherwise Chrome overwrites Local State on exit and the flag is lost."
  exit 1
}

# Back up before writing.
$backup = "$localState.hermes-backup"
Copy-Item $localState $backup -Force
Write-Host "Backup: $backup"

$json = Get-Content $localState -Raw -Encoding UTF8 | ConvertFrom-Json

if (-not $json.browser) { $json | Add-Member -MemberType NoteProperty -Name browser -Value (New-Object PSObject) }

$current = @()
if ($json.browser.PSObject.Properties.Name -contains 'enabled_labs_experiments' -and $json.browser.enabled_labs_experiments) {
  $current = @($json.browser.enabled_labs_experiments)
}

if ($current | Where-Object { $_ -like "$FLAG*" }) {
  Write-Host "Flag already present: $current"
} else {
  $updated = $current + "$FLAG=$ORIGIN"
  # -Force: the key exists (as $null) in a stock Local State, so a plain
  # Add-Member would fail with "member already exists".
  $json.browser | Add-Member -MemberType NoteProperty -Name enabled_labs_experiments -Value $updated -Force
  # WriteAllText with a BOM-less UTF8: Set-Content -Encoding UTF8 writes a BOM
  # on Windows PowerShell 5.1, which Chrome's Local State parser rejects.
  $out = $json | ConvertTo-Json -Depth 100 -Compress
  [System.IO.File]::WriteAllText($localState, $out, (New-Object System.Text.UTF8Encoding($false)))
  Write-Host ""
  Write-Host "Added:  $FLAG=$ORIGIN"
  Write-Host "Origin: $ORIGIN"
}

Write-Host ""
Write-Host "Now start Chrome and go to:"
Write-Host "  chrome://flags/#unsafely-treat-insecure-origin-as-secure"
Write-Host "Confirm it says 'Enabled' for $ORIGIN, then click 'Relaunch'."
Write-Host ""
Write-Host "Then open the dashboard:  $ORIGIN"
exit 0
