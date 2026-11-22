# ============================================================
#  check-echo-usb.ps1
#
#  Confirms whether Windows sees the Echo Dot over USB.
#
#  Copy to the Windows machine that holds the USB cable and run it
#  there, in a normal PowerShell, AFTER setup-windows.ps1 has been run
#  in an ADMIN PowerShell (that installs adb, fastboot and UsbDk).
#
#  Read-only: it enumerates and reports. It writes nothing, flashes
#  nothing, and changes no driver or firewall state.
#
#  Why this has to run on Windows and not from WSL: WSL2 exposes no
#  USB bus, and Windows' setupapi.dev.log records driver installs
#  rather than USB enumerations, so neither side can see the device
#  from there.
# ============================================================
$ErrorActionPreference = 'Continue'

function Say($m) { Write-Host $m }

Say "=== Echo Dot USB check ==="
Say ""

# Refresh PATH so this session sees adb/fastboot just installed.
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

Say "--- tool versions ---"
foreach ($t in @("adb","fastboot","uv")) {
  $c = Get-Command $t -ErrorAction SilentlyContinue
  if ($c) {
    $v = (& $t --version 2>&1 | Select-Object -First 1)
    Say ("  {0,-9} {1}" -f $t, $v)
  } else {
    Say ("  {0,-9} NOT INSTALLED - run setup-windows.ps1 in an ADMIN PowerShell first" -f $t)
  }
}

# The four USB IDs an Echo Dot 2 presents across its modes.
$ids = @('VID_1949','VID_0E8D','VID_0BB4','VID_18D1')

Say ""
Say "--- connected USB devices (pnputil) ---"
$echoFound = $false
if (Get-Command pnputil -ErrorAction SilentlyContinue) {
  $out = & pnputil /enum-devices /connected /class USB 2>&1
  $lines = @($out)
  # pnputil prints a two-line record per device: "Instance ID:" then "Device Description:".
  for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match 'Instance ID:\s*(USB\\.+)$') {
      $inst = $Matches[1]
      $desc = ''
      if ($i + 1 -lt $lines.Count -and $lines[$i+1] -match 'Device Description:\s*(.+)') { $desc = $Matches[1].Trim() }
      if ($inst -match '($($ids -join '|'))') {
        Say "  >>> ECHO-RELATED DEVICE FOUND"
        Say ("      Instance ID : {0}" -f $inst)
        if ($desc) { Say ("      Description : {0}" -f $desc) }
        $echoFound = $true
      }
    }
  }
  if (-not $echoFound) {
    Say "  No Amazon/MediaTek/Android USB device in the connected list."
    Say "  EXPECTED if the Dot is powered but no driver is bound yet - that is what"
    Say "  setup-windows.ps1 fixes. Check the cable carries data, not just power."
  }
} else { Say "  pnputil unavailable" }

Say ""
Say "--- adb devices (Dot running FireOS with USB debugging exposed) ---"
$adbOut = @(& adb devices -l 2>&1)
$adbOut | ForEach-Object { Say ("  " + $_) }
if ($adbOut | Where-Object { $_ -match '^\S+\s+device' }) {
  Say "  >>> ADB SEES THE DOT."
} else {
  Say "  No device in 'device' state. NORMAL for a stock Dot - its adbd stays off"
  Say "  until unlocked, and the unlock makes fastboot appear on its own."
}

Say ""
Say "--- fastboot devices (Dot in fastboot / bootloader) ---"
@(& fastboot devices 2>&1) | ForEach-Object { Say ("  " + $_) }

Say ""
Say "=== reference: the USB IDs an Echo Dot 2 presents ==="
Say "  1949:2007  Echo in adb/serial mode       (needs the MTK CDC-ACM driver)"
Say "  0e8d:0003  Echo in bootrom mode          (needs UsbDk)"
Say "  0e8d:201c  Echo in recovery/serial mode  (needs UsbDk)"
Say "  0bb4:0c01  Echo in fastboot mode        (needs the fastboot driver)"
Say ""
Say "The unlock needs the Dot in fastboot; firebreak walks the owner through"
Say "the button gesture that gets it there."
exit 0
