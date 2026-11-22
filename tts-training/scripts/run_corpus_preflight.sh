#!/usr/bin/env bash
set -euo pipefail
cd /home/voiceuser/projects/gpt-sovits
compute=${1:-float16}
device=${2:-cuda}
if [[ "$compute" != float16 || "$device" != cuda ]]; then
  echo 'Only validated CUDA float16 is supported; no automatic fallback.' >&2
  exit 2
fi
exec bash scripts/run_asr_controller.sh preflight
