# ============================================================
#  2-unlock.ps1  -  unlock + root the Echo Dot (2nd gen, biscuit)
#
#  Run in an ADMINISTRATOR PowerShell, with the Dot plugged into
#  this computer over a DATA-capable micro-USB cable.
#
#  This is the irreversible part: it runs a bootrom exploit and
#  rewrites the bootloaders. The Dot can end up soft-bricked if it
#  goes wrong. Keep the Dot on a surface you can reach the buttons
#  on, and do not unplug it mid-run.
#
#  firebreak talks you through every button press. Follow its
#  instructions on screen - it prints the gesture it wants.
# ============================================================
$ErrorActionPreference = 'Continue'
$log = "$env:USERPROFILE\Desktop\echomuse\unlock.log"

function Say($m) {
  $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m
  Write-Host $line
  Add-Content -Path $log -Value $line
}

Say "=== EchoMuse unlock starting ==="
Say "Dot must be plugged in over USB now."

# Refresh PATH for adb/fastboot/uv installed by setup-windows.ps1
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

foreach ($t in @("adb","fastboot","uv")) {
  $c = Get-Command $t -ErrorAction SilentlyContinue
  if ($c) { Say "$t -> $($c.Source)" } else { Say "WARNING: $t not on PATH. Run setup-windows.ps1 first, in a NEW admin PowerShell." }
}

# adb server left running by another tool blocks the wizard later; clear it now.
Say "Resetting adb server..."
adb kill-server 2>&1 | ForEach-Object { Say $_ }

Say "Running: uvx --managed-python firebreak@latest unlock"
Say "(this takes ~10 minutes; it polls the Dot and asks for button holds)"

# --managed-python is required on Windows: firebreak cannot use a
# Microsoft-Store Python because adb/fastboot cannot see its files.
$env:UV_PYTHON_DOWNLOADS = "never"
uvx --managed-python firebreak@latest unlock 2>&1 | Tee-Object -FilePath $log -Append

Say ""
Say "=== firebreak finished (exit code $LASTEXITCODE) ==="
Say "Read the log above. Next step is the controller + the browser wizard."
exit $LASTEXITCODE
