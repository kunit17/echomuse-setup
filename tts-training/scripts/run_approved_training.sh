#!/usr/bin/env bash
set -euo pipefail
cd /home/voiceuser/projects/gpt-sovits
source scripts/env.sh
export PATH="/home/voiceuser/.local/bin:$PATH"
export PYTHONPATH="$PWD/scripts:$PYTHONPATH"
exec python scripts/approved_pipeline.py
