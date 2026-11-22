param([switch]$StopOnly,[switch]$NoBrowser)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$root=$PSScriptRoot
if($StopOnly){
    Get-ChildItem -LiteralPath (Join-Path $root '.desktop-state') -Directory -ErrorAction SilentlyContinue | ForEach-Object { [IO.File]::WriteAllText((Join-Path $_.FullName 'stop'),'stop') }
    $p=Start-Process wsl.exe -ArgumentList @('-d','Ubuntu','--','/home/voiceuser/.local/bin/orpheus','stop') -WindowStyle Hidden -PassThru
    $p.WaitForExit();exit $p.ExitCode
}
$created=$false;$mutex=[Threading.Mutex]::new($true,'Local\OrpheusVoiceDesktopControl',[ref]$created)
if(!$created){[Windows.Forms.MessageBox]::Show('Orpheus Voice is already open. Use its Start, Stop or Open Page controls.','Orpheus Voice')|Out-Null;exit}
$script:hostProcess=$null;$script:session=$null;$script:opened=$false
$form=[Windows.Forms.Form]::new();$form.Text='Orpheus Voice';$form.ClientSize=[Drawing.Size]::new(440,180);$form.StartPosition='CenterScreen';$form.FormBorderStyle='FixedDialog';$form.MaximizeBox=$false
$status=[Windows.Forms.Label]::new();$status.Text='Voice is off';$status.SetBounds(20,18,400,28);$status.Font=[Drawing.Font]::new('Segoe UI',12);$form.Controls.Add($status)
$hint=[Windows.Forms.Label]::new();$hint.Text='Closing this window unloads the voice from your GPU.';$hint.SetBounds(20,55,400,36);$form.Controls.Add($hint)
$start=[Windows.Forms.Button]::new();$start.Text='Start';$start.SetBounds(20,110,115,36);$form.Controls.Add($start)
$open=[Windows.Forms.Button]::new();$open.Text='Open Page';$open.SetBounds(160,110,115,36);$open.Enabled=$false;$form.Controls.Add($open)
$stopButton=[Windows.Forms.Button]::new();$stopButton.Text='Stop';$stopButton.SetBounds(300,110,115,36);$stopButton.Enabled=$false;$form.Controls.Add($stopButton)
function Start-Voice {
    if($script:hostProcess -and !$script:hostProcess.HasExited){return}
    $script:session=Join-Path (Join-Path $root '.desktop-state') ([guid]::NewGuid().ToString('N'))
    [IO.Directory]::CreateDirectory($script:session)|Out-Null
    [IO.File]::WriteAllText((Join-Path $script:session 'heartbeat'),'alive')
    $linuxSession='/mnt/c/'+$script:session.Substring(3).Replace('\','/')
    $script:opened=$false;$status.Text='Loading voice...';$start.Enabled=$false;$open.Enabled=$false;$stopButton.Enabled=$true
    $script:hostProcess=Start-Process wsl.exe -ArgumentList @('-d','Ubuntu','--','python3','/home/voiceuser/projects/orpheus-lora/scripts/desktop_host.py','--session-dir',$linuxSession) -WindowStyle Hidden -RedirectStandardOutput (Join-Path $script:session 'host.out.log') -RedirectStandardError (Join-Path $script:session 'host.err.log') -PassThru
}
function Stop-Voice {
    if($script:session){[IO.File]::WriteAllText((Join-Path $script:session 'stop'),'stop')}
    $status.Text='Stopping voice...';$stopButton.Enabled=$false;$open.Enabled=$false
}
$start.Add_Click({Start-Voice});$stopButton.Add_Click({Stop-Voice});$open.Add_Click({Start-Process 'http://localhost:8812/'})
$timer=[Windows.Forms.Timer]::new();$timer.Interval=500
$timer.Add_Tick({
    if(!$script:session){return}
    [IO.File]::WriteAllText((Join-Path $script:session 'heartbeat'),'alive')
    $file=Join-Path $script:session 'state.json'
    if(Test-Path -LiteralPath $file){
        try{$state=Get-Content -Raw -LiteralPath $file|ConvertFrom-Json}catch{return}
        switch($state.status){
            'ready' {if(!(Test-Path -LiteralPath (Join-Path $script:session 'stop'))){$status.Text='Voice is on';$open.Enabled=$true;if(!$script:opened){$script:opened=$true;if(!$NoBrowser){Start-Process 'http://localhost:8812/'}}}}
            'off' {$status.Text='Voice is off - GPU released';$start.Enabled=$true;$open.Enabled=$false;$stopButton.Enabled=$false}
            'error' {$status.Text='Could not start voice';$hint.Text=$state.error;$start.Enabled=$true;$stopButton.Enabled=$false}
        }
    }
    if($script:hostProcess -and $script:hostProcess.HasExited -and !(Test-Path -LiteralPath $file)){$status.Text='Could not start voice';$hint.Text='See logs in '+$script:session;$start.Enabled=$true;$stopButton.Enabled=$false}
})
$form.Add_Shown({$timer.Start();Start-Voice})
$form.Add_FormClosing({
    $timer.Stop();Stop-Voice
    if($script:hostProcess -and !$script:hostProcess.HasExited){$script:hostProcess.WaitForExit(20000)|Out-Null}
    if($script:hostProcess -and !$script:hostProcess.HasExited){$p=Start-Process wsl.exe -ArgumentList @('-d','Ubuntu','--','/home/voiceuser/.local/bin/orpheus','stop') -WindowStyle Hidden -PassThru;$p.WaitForExit(15000)|Out-Null}
})
try{[Windows.Forms.Application]::Run($form)}finally{$mutex.ReleaseMutex();$mutex.Dispose()}
