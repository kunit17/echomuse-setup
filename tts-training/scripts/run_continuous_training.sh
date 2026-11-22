#!/usr/bin/env bash
set -euo pipefail
cd /home/voiceuser/projects/gpt-sovits
while [[ $(systemctl --user show she-corpus-asr-controller.service -p SubState --value) == running ]]; do sleep 3; done
[[ $(systemctl --user show she-corpus-asr-controller.service -p Result --value) == success ]]
exec 9>.cache/training-launch.lock
flock -n 9
[[ $(systemctl --user show she-gptsovits-tts.service -p ActiveState --value) == inactive ]]
jq -e '.status=="two_pass_corpus_built_pending_review" and .clips>0' reports/corpus-verification.json >/dev/null
run_stage() {
  local unit=$1 script=$2 arguments=$3 profile=${4:-model}
  systemctl --user stop "$unit" 2>/dev/null || true
  systemctl --user reset-failed "$unit" 2>/dev/null || true
  systemd-run --user --unit="$unit" -p MemoryAccounting=true -p MemoryMax=infinity -p MemoryHigh=infinity \
    -p MemorySwapMax=infinity -p OOMPolicy=stop -p WorkingDirectory="$PWD" -p RemainAfterExit=yes \
    -p StandardOutput=append:"$PWD/logs/$unit.log" -p StandardError=append:"$PWD/logs/$unit.log" \
    /bin/bash -c 'source scripts/env.sh; export PATH=/home/voiceuser/.local/bin:$PATH; export PYTHONHASHSEED=20261112; export PYTHONPATH="$PWD/scripts:$PYTHONPATH"; if [[ "$3" == asr ]]; then export PYTHONPATH=; fi; exec python "$1" $2' _ "$script" "$arguments" "$profile"
  while [[ $(systemctl --user show "$unit" -p SubState --value) == running ]]; do sleep 3; done
  systemctl --user show "$unit" -p Result -p ExecMainStatus -p MemoryPeak -p MemoryMax -p MemorySwapMax > "reports/$unit-resources.txt"
  [[ $(systemctl --user show "$unit" -p Result --value) == success ]]
}
for partition in train validation; do
  stages='text hubert speaker semantic verify'
  if [[ ${1:-} == resume ]]; then stages=verify; fi
  for stage in $stages; do
    run_stage "she-continuous-$partition-$stage" scripts/continuous_features.py "$partition $stage"
  done
done
if [[ ${1:-} != resume ]]; then run_stage she-continuous-voices scripts/continuous_voice_diagnostics.py ''; fi
run_stage she-continuous-s1 scripts/continuous_train.py s1
run_stage she-continuous-s2 scripts/continuous_train.py s2
run_stage she-continuous-synthesis scripts/continuous_synthesis.py ''
# ASR explicitly isolates PYTHONPATH from upstream's shadowing packages.
run_stage she-continuous-evaluate scripts/continuous_evaluate.py '' asr
run_stage she-continuous-report scripts/continuous_report.py '' asr
