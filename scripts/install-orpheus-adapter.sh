#!/usr/bin/env bash
# Lightweight bridge only. Requires the separately approved local Orpheus engine.
set -euo pipefail
record_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
voice_python="${VOICE_PYTHON:-$HOME/.local/bin/python3.11}"
selection="$HOME/projects/orpheus-lora/selected-model.json"
if [[ ! -f "$selection" ]]; then echo 'Approved Orpheus model selection is required.' >&2; exit 1; fi
"$voice_python" -m venv "$HOME/.local/share/echomuse-voice/tts-venv"
"$HOME/.local/share/echomuse-voice/tts-venv/bin/python" -m pip install -r "$record_dir/voice/orpheus-adapter-requirements.lock.txt"
"$HOME/.local/share/echomuse-voice/tts-venv/bin/python" -m pip check
"$HOME/.local/share/echomuse-voice/tts-venv/bin/python" -m unittest discover -s "$record_dir/tests" -p test_orpheus_adapter.py -v
mkdir -p "$HOME/.config/systemd/user"
cp "$record_dir/voice/wyoming-orpheus.service" "$HOME/.config/systemd/user/wyoming-orpheus.service"
systemctl --user daemon-reload
systemctl --user enable --now wyoming-orpheus.service
"$HOME/.local/share/echomuse-voice/tts-venv/bin/python" "$record_dir/scripts/probe-orpheus.py"
echo 'Adapter ready. Start the approved model explicitly with orpheus start.'
