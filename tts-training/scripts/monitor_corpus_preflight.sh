#!/usr/bin/env bash
set -euo pipefail
cd /home/voiceuser/projects/gpt-sovits
echo 'Current ASR status and per-run resource records; this command never starts models.'
while [[ -f reports/asr-status.json ]]; do
  jq '{run_id,status,active_stage,updated,profile,failure_kind,error}' reports/asr-status.json
  run=$(jq -r '.run_id' reports/asr-status.json)
  stage=$(jq -r '.active_stage // empty' reports/asr-status.json)
  if [[ -n "$stage" && -f "reports/asr-runs/$run/$stage.json" ]]; then
    jq '{status,elapsed_seconds,heartbeat,memory,gpu_before,failure_kind}' "reports/asr-runs/$run/$stage.json"
  fi
  [[ $(jq -r '.status' reports/asr-status.json) == running || $(jq -r '.status' reports/asr-status.json) == starting ]] || break
  sleep 10
done
