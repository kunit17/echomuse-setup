#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-$HOME/projects/wyoming-faster-whisper/.venv/bin/python}"
venv="$HOME/.local/share/echomuse-llm-venv"
"$python_bin" -m venv "$venv"
"$venv/bin/pip" install -r "$repo/llm/requirements.lock"
mkdir -p "$HOME/.config/systemd/user"
# The unit uses the repository's existing documented checkout location.
cp "$repo/llm/echomuse-search.service" "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
systemctl --user enable --now echomuse-search.service
systemctl --user is-active echomuse-search.service
