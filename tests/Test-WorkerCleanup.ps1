# Regression: stale/reused worker PIDs must never crash startup or kill others.
param([string]$RepoRoot = (Split-Path $PSScriptRoot -Parent))
$ErrorActionPreference = 'Stop'
$repo = $RepoRoot
function Get-Process {
    [CmdletBinding()]
    param([int]$Id)
    return $script:MockWorker
}
foreach ($name in @('Start-EchoMuse-Whisper.ps1','Start-EchoMuse-LLM.ps1')) {
    $parseErrors = $null
    $tree = [System.Management.Automation.Language.Parser]::ParseFile((Join-Path $repo "scripts/$name"),[ref]$null,[ref]$parseErrors)
    if ($parseErrors.Count) { throw ($parseErrors | Out-String) }
    $definition = $tree.Find({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Stop-OwnedWorker' }, $true)
    if ($null -eq $definition) { throw 'Cleanup function missing.' }
    . ([ScriptBlock]::Create($definition.Extent.Text))
    $record = [pscustomobject]@{Id=123;StartTime='2000-01-01T00:00:00Z'}
    $script:MockWorker = $null
    Stop-OwnedWorker $record
    $script:MockWorker = [pscustomobject]@{StartTime=$null}
    Stop-OwnedWorker $record
    $script:MockWorker = [pscustomobject]@{}
    $script:MockWorker | Add-Member -MemberType ScriptProperty -Name StartTime -Value {throw 'Protected process'}
    Stop-OwnedWorker $record
    $script:MockWorker = [pscustomobject]@{StartTime=[datetime]'2020-01-01T00:00:00Z'}
    Stop-OwnedWorker $record
    Write-Output "PASS $($name): missing, null, protected and reused PID skipped."
}
