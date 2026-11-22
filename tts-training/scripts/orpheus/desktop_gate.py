"""Prevent background jobs from loading a voice the owner has turned off."""
import json,time
from pathlib import Path
path=Path('/home/voiceuser/projects/orpheus-lora/activation-request.json')
try:
    state=json.loads(path.read_text());allowed=state.get('desired_on',False)
    if state.get('mode')=='desktop':allowed=allowed and time.time()-Path(state['heartbeat']).stat().st_mtime<20
except (OSError,ValueError,KeyError):allowed=False
raise SystemExit(0 if allowed else 1)
