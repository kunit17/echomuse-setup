"""Inference must not load while an authorized GPU training/preparation lease exists."""
import fcntl
from pathlib import Path
with Path('/home/voiceuser/projects/gpt-sovits/.cache/training-launch.lock').open('a') as lease:
    try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit(1)
