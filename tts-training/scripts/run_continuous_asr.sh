#!/usr/bin/env bash
set -euo pipefail
cd /home/voiceuser/projects/gpt-sovits
exec bash scripts/run_asr_controller.sh corpus
