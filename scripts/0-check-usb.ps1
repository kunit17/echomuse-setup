# ============================================================
#  0-check-usb.ps1  -  confirm Windows sees the Echo Dot over USB
#
#  Run in a NORMAL PowerShell AFTER setup-windows.ps1 has been run
#  in an ADMIN PowerShell (it installs adb, fastboot and UsbDk).
#
#  Read-only. Nothing is written, nothing is flashed.
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
  } else { Say ("  {0,-9} NOT INSTALLED - run setup-windows.ps1 in an ADMIN PowerShell first" -f $t) }
}

Say ""
Say "--- devices visible to Windows (all USB, any driver) ---"
$pnputil = Get-Command pnputil -ErrorAction SilentlyContinue
if ($pnputil) {
  $out = & pnputil /enum-devices /connected /class USB 2>&1
  $echo = $out | Select-String -Pattern "VID_1949|VID_0E8D|VID_0BB4|VID_18D1|Amazon|MediaTek|Fastboot|ADB" -Context 1,1
  if ($echo) {
    Say "  >>> ECHO-RELATED DEVICE FOUND:"
    $echo | ForEach-Object { Say ("     " + ($_ -replace '\s+',' ').Trim()) }
  } else {
    Say "  No Amazon/MediaTek/Android USB device in the connected list."
    Say "  (That is EXPECTED if the Dot is powered but the drivers are missing.)"
  }
} else { Say "  pnputil unavailable" }

Say ""
Say "--- adb devices (Dot running FireOS, USB debugging exposed) ---"
$adbOut = & adb devices -l 2>&1
$adbOut | ForEach-Object { Say ("  " + $_) }
$adbSeen = $adbOut | Where-Object { $_ -match "^\S+\s+device" }
if ($adbSeen) {
  Say "  >>> ADB SEES THE DOT."
} else {
  Say "  No device in 'device' state. Expected on a STOCK Dot (its adbd is off"
  Say "  until unlocked). This is NOT a problem - firebreak enters fastboot"
  Say "  itself with a button gesture."
}

Say ""
Say "--- fastboot devices (Dot in fastboot / bootloader) ---"
$fbOut = & fastboot devices 2>&1
$fbOut | ForEach-Object { Say ("  " + $_) }

Say ""
Say "=== notes ==="
Say "A stock Echo Dot 2 shows up on USB as one of:"
Say "  1949:2007  Echo in adb/serial mode      (needs MTK CDC-ACM driver)"
Say "  0e8d:0003  Echo in bootrom mode         (needs UsbDk)"
Say "  0e8d:201c  Echo in recovery/serial mode (needs UsbDk)"
Say "  0bb4:0c01  Echo in fastboot mode       (needs fastboot driver)"
Say ""
Say "If pnputil lists the device but adb does not see it, that is normal for a"
Say "stock Dot - the unlock makes fastboot appear."
exit 0
