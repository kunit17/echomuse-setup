#!/usr/bin/env bash
set -euo pipefail
cd /home/voiceuser/projects/gpt-sovits
source scripts/env.sh
python scripts/run_pipeline.py
