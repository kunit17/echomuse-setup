#!/usr/bin/env bash
set -euo pipefail
cd /home/voiceuser/projects/gpt-sovits
unit=she-corpus-asr-controller.service
if systemctl --user is-active --quiet "$unit"; then
  echo 'The ASR controller is already running.' >&2
  exit 1
fi
systemctl --user reset-failed "$unit" 2>/dev/null || true
# --wait keeps the Windows wsl.exe session alive for the entire corpus job.
exec systemd-run --user --wait --unit="$unit" \
  -p WorkingDirectory="$PWD" -p MemoryAccounting=true \
  -p MemoryHigh=infinity -p MemoryMax=infinity -p MemorySwapMax=infinity \
  -p RuntimeMaxSec=infinity -p KillMode=control-group \
  -p StandardOutput=append:"$PWD/logs/asr-controller.log" \
  -p StandardError=append:"$PWD/logs/asr-controller.log" \
  /bin/bash scripts/run_continuous_asr.sh
