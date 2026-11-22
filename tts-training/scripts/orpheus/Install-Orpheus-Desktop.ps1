$ErrorActionPreference='Stop'
$desktop=[Environment]::GetFolderPath('Desktop')
$install=Join-Path $env:LOCALAPPDATA 'OrpheusVoice'
New-Item -ItemType Directory -Path $install -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'Orpheus-Voice.ps1') -Destination (Join-Path $install 'Orpheus-Voice.ps1') -Force
$shell=New-Object -ComObject WScript.Shell
$scriptFile=Join-Path $install 'Orpheus-Voice.ps1'
foreach($entry in @(@{Name='Orpheus Voice';Option='';Icon=168},@{Name='Stop Orpheus Voice';Option=' -StopOnly';Icon=27})) {
    $path=Join-Path $desktop ($entry.Name+'.lnk')
    $shortcut=$shell.CreateShortcut($path)
    if((Test-Path -LiteralPath $path) -and $shortcut.Arguments -notlike '*Orpheus-Voice.ps1*'){throw "An unrelated shortcut already exists: $path"}
    $shortcut.TargetPath="$env:WINDIR\System32\WindowsPowerShell\v1.0\powershell.exe"
    $shortcut.Arguments='-NoProfile -ExecutionPolicy Bypass -STA -WindowStyle Hidden -File "'+$scriptFile+'"'+$entry.Option
    $shortcut.WorkingDirectory=$install
    $shortcut.Description=if($entry.Option){'Turn off Orpheus and release its GPU memory'}else{'Start Orpheus and open the voice page; close the control window to unload the GPU'}
    $shortcut.IconLocation="$env:WINDIR\System32\shell32.dll,"+$entry.Icon
    $shortcut.Save()
    Write-Output $path
}
