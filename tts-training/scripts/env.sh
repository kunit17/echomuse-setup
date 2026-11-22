#!/usr/bin/env bash
TASK_ROOT=/home/voiceuser/projects/gpt-sovits
export PATH="$TASK_ROOT/.venv/bin:/home/voiceuser/.hermes/cache/voices/espeak-ng/root/usr/bin:$PATH"
export LD_LIBRARY_PATH="$TASK_ROOT/.venv/lib/python3.11/site-packages/nvidia/cublas/lib:$TASK_ROOT/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/lib/wsl/lib:/home/voiceuser/.hermes/cache/voices/espeak-ng/root/usr/lib/x86_64-linux-gnu:/home/voiceuser/.hermes/cache/voices/espeak-ng/root/usr/lib/x86_64-linux-gnu/pulseaudio:${LD_LIBRARY_PATH:-}"
export ESPEAK_DATA_PATH=/home/voiceuser/.hermes/cache/voices/espeak-ng/root/usr/lib/x86_64-linux-gnu/espeak-ng-data
export HF_HOME="$TASK_ROOT/.cache/huggingface"
export NLTK_DATA="$TASK_ROOT/.cache/nltk_data"
export PYTHONPATH="$TASK_ROOT/upstream:$TASK_ROOT/upstream/GPT_SoVITS"
export version=v2ProPlus hz=25hz is_half=True CUDA_VISIBLE_DEVICES=0 _CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
