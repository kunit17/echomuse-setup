param([Parameter(Mandatory=$true)][string]$OutputDirectory,
      [string]$ManifestPath = (Join-Path $PSScriptRoot '../samantha/evaluation-phrases.json'))
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$rows = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000,[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,[System.Speech.AudioFormat.AudioChannel]::Mono)
    foreach ($row in $rows) {
        $path = Join-Path $OutputDirectory $row.file
        if (Test-Path -LiteralPath $path) { throw "Evaluation file already exists: $path" }
        $synth.SelectVoice($row.voice)
        $synth.Rate = $row.rate
        $synth.SetOutputToWaveFile($path,$format)
        $synth.Speak($row.text)
        $synth.SetOutputToNull()
    }
    $rows | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputDirectory 'manifest.json') -Encoding UTF8
} finally { $synth.Dispose() }
