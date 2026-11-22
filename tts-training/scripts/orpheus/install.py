"""Install command wrappers and source files; never starts preparation/training."""
from pathlib import Path
import shutil
ROOT=Path('/home/voiceuser/projects/orpheus-lora')
SOURCE=Path(__file__).resolve().parent
(ROOT/'scripts').mkdir(parents=True,exist_ok=True)
for path in SOURCE.glob('*.py'):
    destination=ROOT/'scripts'/path.name
    if path.resolve()!=destination.resolve():shutil.copy2(path,destination)
if (SOURCE/'README.md').resolve()!=(ROOT/'README.md').resolve():shutil.copy2(SOURCE/'README.md',ROOT/'README.md')
entry=Path('/home/voiceuser/.local/bin/orpheus');entry.parent.mkdir(exist_ok=True)
entry.write_text('''#!/usr/bin/env bash
set -euo pipefail
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
export HF_HOME=/home/voiceuser/projects/orpheus-lora/.cache/huggingface
exec /home/voiceuser/projects/orpheus-lora/.venv/bin/python /home/voiceuser/projects/orpheus-lora/scripts/cli.py "$@"
''');entry.chmod(0o755)
agents=ROOT/'AGENTS.md'
if not agents.exists():agents.write_text('''Read /home/voiceuser/projects/gpt-sovits/TTS_WORKFLOW.md and README.md.
This project only encodes approved immutable corpora into Orpheus features.
Do not segment or rewrite source WAVs here. Keep every SNAC frame, no row
filtering, and no sequence truncation. Use the shared reservation-aware
pipeline for GPU work. Do not edit shared hardlinked dependency files.
CSM remains the selected deployment until the owner chooses otherwise.
''')
print('Installed orpheus prepare|train|sample|status; no new job started.')
