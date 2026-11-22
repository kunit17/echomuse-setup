param([Parameter(Mandatory=$true)][string]$OutputDirectory)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$phrases = Get-Content -LiteralPath (Join-Path $PSScriptRoot '../voice/benchmark-phrases.json') -Raw | ConvertFrom-Json
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $synth.SelectVoice('Microsoft Zira Desktop')
    $format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
    foreach ($phrase in $phrases) {
        $wavPath = Join-Path $OutputDirectory $phrase.file
        if (Test-Path -LiteralPath $wavPath) { throw "Fixture already exists: $wavPath" }
        $synth.SetOutputToWaveFile($wavPath,$format)
        $synth.Speak($phrase.text)
        $synth.SetOutputToNull()
    }
    $phrases | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputDirectory 'manifest.json') -Encoding UTF8
} finally { $synth.Dispose() }
