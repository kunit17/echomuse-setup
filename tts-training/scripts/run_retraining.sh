#!/usr/bin/env bash
set -euo pipefail
cd /home/voiceuser/projects/gpt-sovits
source scripts/env.sh
export PATH="/home/voiceuser/.local/bin:$PATH"
exec python scripts/run_retraining.py
