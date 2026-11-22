"""Create an isolated interpreter with immutable hardlinked dependency files."""
import hashlib,json,os,shutil,subprocess,time,urllib.request
from pathlib import Path
ROOT=Path('/home/voiceuser/projects/orpheus-lora')
SOURCE=Path('/home/voiceuser/projects/csm-lora/.venv')
ROOT.mkdir(parents=True,exist_ok=True)
def get(url):
    with urllib.request.urlopen(url,timeout=90) as response:return response.read()
def atomic(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)
atomic(ROOT/'setup-status.json',dict(status='creating_environment',started=time.time()))
python=ROOT/'.venv/bin/python'
if not python.exists():
    subprocess.run([str(SOURCE/'bin/python'),'-m','venv','--without-pip',str(ROOT/'.venv')],check=True)
    source=SOURCE/'lib/python3.11/site-packages'
    destination=ROOT/'.venv/lib/python3.11/site-packages'
    # Fresh directories and metadata; installed dependency payloads share storage.
    # Never upgrade shared packages in place. Adding SNAC creates only new files.
    shutil.copytree(source,destination,copy_function=os.link,dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
snac_version=json.loads(get('https://pypi.org/pypi/snac/json'))['info']['version']
subprocess.run([str(python),'-m','pip','install','--no-cache-dir','--no-deps','snac=='+snac_version],check=True)
revisions={}
for name in ['unsloth/orpheus-3b-0.1-ft','unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit','hubertsiuzdak/snac_24khz']:
    info=json.loads(get('https://huggingface.co/api/models/'+name))
    revisions[name]=info['sha']
notebook=get('https://raw.githubusercontent.com/unslothai/notebooks/main/nb/Orpheus_(3B)-TTS.ipynb')
(ROOT/'unsloth-orpheus.ipynb').write_bytes(notebook)
record=dict(status='ready',finished=time.time(),snac_version=snac_version,model_revisions=revisions,
    dependency_source=str(SOURCE),dependency_storage='hardlinked package files; isolated environment; no shared-package upgrades',
    notebook_sha256=hashlib.sha256(notebook).hexdigest())
atomic(ROOT/'setup.json',record)
subprocess.run([str(python),'-m','pip','freeze'],stdout=(ROOT/'requirements.lock.txt').open('w'),check=True)
atomic(ROOT/'setup-status.json',record)
print(json.dumps(record,indent=2),flush=True)
