#!/usr/bin/env bash
set -euo pipefail
cd /home/voiceuser/projects/gpt-sovits
source scripts/env.sh
export PATH="/home/voiceuser/.local/bin:$PATH"
export PYTHONPATH=
exec python scripts/asr_controller.py "$@"
