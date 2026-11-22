"""Complete the first setup already in flight with the current vLLM BNB plugin."""
import json,time,subprocess
from pathlib import Path
B=Path('/home/voiceuser/projects/orpheus-lora/stream-benchmark')
while json.loads((B/'setup-status.json').read_text())['status']=='installing':time.sleep(5)
assert json.loads((B/'setup-status.json').read_text())['status']=='ready'
python=str(B/'.venv/bin/python')
subprocess.run([python,'-m','pip','install','--no-cache-dir','--no-deps','vllm-bnb-plugin==0.0.3'],check=True)
subprocess.run([python,'-m','pip','freeze'],stdout=(B/'requirements.lock.txt').open('w'),check=True)
(B/'bnb-plugin-ready.json').write_text(json.dumps(dict(version='0.0.3',time=time.time())))
