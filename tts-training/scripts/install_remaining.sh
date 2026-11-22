#!/usr/bin/env bash
set -euo pipefail
cd /home/voiceuser/projects/gpt-sovits
unset PYTHONPATH
.venv/bin/python -m pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install -r configs/requirements-task.txt
source scripts/env.sh
python - <<'PY'
import nltk
for item in ['averaged_perceptron_tagger','averaged_perceptron_tagger_eng','cmudict','punkt','punkt_tab']:nltk.download(item,download_dir=__import__('os').environ['NLTK_DATA'])
PY
python scripts/verify_environment.py
python scripts/make_configs.py
