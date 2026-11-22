#!/usr/bin/env bash
set -euo pipefail
cd /home/voiceuser/projects/gpt-sovits
exec 9>.cache/training-launch.lock
flock -n 9
unit=she-corpus-inventory.service
systemd-run --user --unit="$unit" \
  -p MemoryAccounting=true -p MemoryHigh=infinity -p MemoryMax=infinity \
  -p MemorySwapMax=infinity -p OOMPolicy=stop \
  -p RemainAfterExit=yes -p WorkingDirectory="$PWD" \
  -p StandardOutput=append:"$PWD/logs/corpus-inventory.log" \
  -p StandardError=append:"$PWD/logs/corpus-inventory.log" \
  /bin/bash -c 'source scripts/env.sh; export PATH="/home/voiceuser/.local/bin:$PATH"; exec python scripts/corpus_inventory.py'
trap 'systemctl --user stop "$unit"' EXIT
while [[ $(systemctl --user show "$unit" -p SubState --value) == running ]]; do sleep 1; done
systemctl --user show "$unit" -p ActiveState -p SubState -p ExecMainStatus \
  -p MemoryPeak -p MemorySwapPeak -p MemoryHigh -p MemoryMax -p MemorySwapMax \
  > reports/corpus-inventory-systemd.txt
test "$(systemctl --user show "$unit" -p ExecMainStatus --value)" = 0
jq -e '.duration_seconds == 9772 and .mono_peak <= 1 and (.summary.relative_2pct.count > 0)' reports/corpus-pause-inventory.json >/dev/null
sync reports/corpus-inventory-systemd.txt reports/corpus-pause-inventory.json
