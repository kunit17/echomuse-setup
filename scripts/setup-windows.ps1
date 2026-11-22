# EchoMuse — one-time Windows toolchain setup for the Echo Dot 2 unlock.
# Source of truth: https://github.com/bboe/firebreak (README, "Install > Windows").
# Run in an ADMINISTRATOR PowerShell. Safe to re-run.
#
# Everything here comes verbatim from firebreak's own Windows instructions.
# Nothing else is invented.

$ErrorActionPreference = 'Stop'
$log = "$env:USERPROFILE\Desktop\echomuse\setup-windows.log"
function Say($m) {
  $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m
  Write-Host $line
  Add-Content -Path $log -Value $line
}

Say "=== EchoMuse Windows setup starting ==="

# --- adb / fastboot (Android platform-tools) ---
if (-not (Get-Command adb -ErrorAction SilentlyContinue)) {
  Say "Installing Android platform-tools via winget (adb, fastboot)..."
  winget install --id Google.PlatformTools --accept-package-agreements --accept-source-agreements 2>&1 | Out-String | ForEach-Object { Say $_ }
} else { Say "adb already present" }

# --- uv (runs firebreak from PyPI) ---
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Say "Installing uv via winget..."
  winget install --id astral-sh.uv --accept-package-agreements --accept-source-agreements 2>&1 | Out-String | ForEach-Object { Say $_ }
} else { Say "uv already present" }

# Refresh PATH so this session sees the newly installed tools
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# --- Drivers: fastboot (Android) and MTK CDC-ACM (the Echo's serial) ---
$tmp = Join-Path $env:TEMP "fb-drv"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

if (-not (Test-Path (Join-Path $tmp "fastboot.inf"))) {
  Say "Downloading fastboot driver CAB..."
  curl.exe -fo (Join-Path $tmp "fastboot.cab") "https://catalog.s.download.windowsupdate.com/d/msdownload/update/driver/drvs/2016/10/20940218_0ae8695c01fb84d1fdea5c9a04baf336ec5b1ba7.cab"
  Say "Downloading MTK CDC-ACM driver CAB..."
  curl.exe -fo (Join-Path $tmp "mtk.cab") "https://catalog.s.download.windowsupdate.com/d/msdownload/update/driver/drvs/2016/07/20896845_fdc6bb5aa9a9bac99adf85d931d6c21d1130a96e.cab"
  Say "Extracting driver CABs..."
  # expand.exe is the Windows CAB extractor. Expand-Archive only reads .zip
  # and would fail on these .cab files.
  expand.exe -F:* (Join-Path $tmp "fastboot.cab") (Join-Path $tmp "fastboot") | Out-Null
  expand.exe -F:* (Join-Path $tmp "mtk.cab") (Join-Path $tmp "mtk") | Out-Null
  Say "Installing fastboot driver (requires admin)..."
  pnputil /add-driver (Join-Path $tmp "fastboot\tplink_winusb.inf") /install 2>&1 | ForEach-Object { Say $_ }
  Say "Installing MTK CDC-ACM driver (requires admin)..."
  pnputil /add-driver (Join-Path $tmp "mtk\cdc-acm.inf") /install 2>&1 | ForEach-Object { Say $_ }
} else { Say "driver CABs already downloaded" }

# --- libusb (adb/fastboot USB access) ---
$libusbDir = "$env:LOCALAPPDATA\Programs\libusb"
if (-not (Test-Path (Join-Path $libusbDir "VS2022\MS64\dll\libusb-1.0.dll"))) {
  Say "Downloading libusb 1.0.30..."
  curl.exe -fo (Join-Path $tmp "libusb.7z") "https://github.com/libusb/libusb/releases/download/v1.0.30/libusb-1.0.30.7z"
  Say "Extracting libusb..."
  New-Item -ItemType Directory -Force -Path $libusbDir | Out-Null
  & tar.exe -xf (Join-Path $tmp "libusb.7z") -C $libusbDir "VS2022/MS64/dll/libusb-1.0.dll"
  Say "Adding libusb to the user PATH..."
  [Environment]::SetEnvironmentVariable("Path", [Environment]::GetEnvironmentVariable("Path","User") + ";$libusbDir\VS2022\MS64\dll", "User")
} else { Say "libusb already installed" }

# --- UsbDk (lets Windows talk to the Echo's bootrom-mode USB device) ---
if (-not (Test-Path "$env:ProgramFiles\UsbDk_1.0.22_x64\UsbDk.inf")) {
  Say "Downloading UsbDk..."
  curl.exe -fLo (Join-Path $tmp "UsbDk.msi") "https://github.com/daynix/UsbDk/releases/download/v1.00-22/UsbDk_1.0.22_x64.msi"
  Say "Installing UsbDk (requires admin)..."
  Say "NOTE: UsbDk is a kernel-mode USB filter driver. The installer UI will appear - read it before accepting."
  # No /qn: this is a kernel driver install, so the UI is shown rather than
  # installing silently.
  $p = Start-Process msiexec.exe -ArgumentList "/i","$tmp\UsbDk.msi" -Wait -PassThru
  Say "UsbDk installer exit code: $($p.ExitCode)"
} else { Say "UsbDk already installed" }

Say ""
Say "=== setup finished ==="
Say "Open a NEW admin PowerShell before running the unlock (PATH + drivers)."
Say "Verify with:  adb version ; fastboot --version ; uv --version"
Say ""
Say "Next step: run  2-unlock.ps1  (the Dot must be plugged in over USB)."
exit 0
