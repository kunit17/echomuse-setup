param([Parameter(Mandatory=$true)][string]$Label,[string]$TextFile,[string]$BaseUrl='http://localhost:8813')
$ErrorActionPreference='Stop'
$edgePath = Join-Path ${env:ProgramFiles(x86)} 'Microsoft/Edge/Application/msedge.exe'
if (!(Test-Path -LiteralPath $edgePath)) { $edgePath = Join-Path $env:ProgramFiles 'Microsoft/Edge/Application/msedge.exe' }
if (!(Test-Path -LiteralPath $edgePath)) { throw 'Microsoft Edge was not found for Windows playback measurement.' }
$profilePath = Join-Path $PSScriptRoot ('browser-profile-' + $Label)
$browserTask = Start-Process -FilePath $edgePath -ArgumentList @('--headless=new','--disable-gpu','--mute-audio','--autoplay-policy=no-user-gesture-required','--remote-debugging-port=9337',('--user-data-dir="'+$profilePath+'"'),'--no-first-run','about:blank') -WindowStyle Hidden -PassThru
try {
    $textArgument=if($TextFile){$TextFile}else{'-'}
    & (Get-Command node).Source (Join-Path $PSScriptRoot 'browser_probe.js') $Label $textArgument $BaseUrl
    if ($LASTEXITCODE -ne 0) { throw 'Browser playback probe failed; see log.' }
} finally {
    if (!$browserTask.HasExited) { Stop-Process -Id $browserTask.Id -ErrorAction SilentlyContinue }
}
