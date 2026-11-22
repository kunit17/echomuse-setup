"""Independent benchmark dependencies; never upgrades a serving environment."""
import json,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/home/voiceuser/projects/orpheus-lora')
B=ROOT/'stream-benchmark';B.mkdir(exist_ok=True)
def state(**kw):(B/'setup-status.json').write_text(json.dumps(dict(time=time.time(),**kw),indent=2))
if (B/'setup-status.json').exists() and json.loads((B/'setup-status.json').read_text())['status']=='ready' and (B/'requirements.lock.txt').exists():
    print('Pinned benchmark environment already exists; no packages changed.');raise SystemExit(0)
try:
    state(status='installing')
    env=B/'.venv';python=env/'bin/python'
    if not python.exists():subprocess.run([str(ROOT/'.venv/bin/python'),'-m','venv',str(env)],check=True)
    subprocess.run([str(python),'-m','pip','install','--no-cache-dir','vllm','vllm-bnb-plugin==0.0.3','snac==1.2.1','bitsandbytes','soundfile'],check=True)
    subprocess.run([str(python),'-m','pip','freeze'],stdout=(B/'requirements.lock.txt').open('w'),check=True)
    state(status='ready')
except BaseException:
    state(status='failed',error=traceback.format_exc());raise
