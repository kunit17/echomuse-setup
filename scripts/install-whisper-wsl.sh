#!/usr/bin/env bash
# Run as the WSL user, not root. Never changes Orpheus or any shared environment.
set -euo pipefail
record_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
service_dir="$HOME/projects/wyoming-faster-whisper"
voice_python="${VOICE_PYTHON:-$HOME/.local/bin/python3.11}"
pinned_revision=1c407106530e09fc8f2477e4ae4cf86480fd6256
if [[ ! -x "$voice_python" ]]; then echo 'Set VOICE_PYTHON to an installed Python 3.11 interpreter.' >&2; exit 1; fi
if [[ ! -d "$service_dir" ]]; then
  git clone https://github.com/OHF-Voice/wyoming-faster-whisper.git "$service_dir"
  git -C "$service_dir" checkout --detach "$pinned_revision"
fi
if [[ "$(git -C "$service_dir" rev-parse HEAD)" != "$pinned_revision" ]]; then
  echo 'Existing service checkout is a different revision; review it manually.' >&2; exit 1
fi
"$voice_python" -m venv "$service_dir/.venv"
"$service_dir/.venv/bin/python" -m pip install -r "$record_dir/voice/whisper-requirements.lock.txt"
"$service_dir/.venv/bin/python" -m pip install --no-deps "$service_dir"
"$service_dir/.venv/bin/python" -m pip check
"$service_dir/.venv/bin/python" "$record_dir/scripts/download-whisper-models.py"
mkdir -p "$HOME/.config/systemd/user" "$HOME/.config/echomuse"
if [[ ! -f "$HOME/.config/echomuse/whisper.env" ]]; then
  cp "$record_dir/voice/whisper.env.example" "$HOME/.config/echomuse/whisper.env"
fi
cp "$record_dir/voice/wyoming-whisper.service" "$HOME/.config/systemd/user/wyoming-whisper.service"
systemctl --user daemon-reload
systemctl --user enable --now wyoming-whisper.service
"$service_dir/.venv/bin/python" "$record_dir/scripts/probe-whisper.py"
