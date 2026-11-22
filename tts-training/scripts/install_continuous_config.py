"""Owner-invoked checkpoint/reference installation; never activate a service."""
import csv
import json
import os
from pathlib import Path
import shutil
import time

import yaml

ROOT=Path(__file__).resolve().parents[1]
selections={stage:json.loads((ROOT/f'reports/continuous-{stage}-selection.json').read_text()) for stage in ('s1','s2')}
assert selections['s1']['final']['epoch']>=20 and selections['s2']['final']['epoch']>=25
assert all(Path(v['selected']['checkpoint']).is_file() and Path(v['selected']['checkpoint']).stat().st_size>0 for v in selections.values())
synthesis=json.loads((ROOT/'reports/continuous-synthesis.json').read_text())
assert synthesis['gpt_checkpoint']==selections['s1']['selected']['checkpoint']
assert synthesis['sovits_checkpoint']==selections['s2']['selected']['checkpoint']
rows=list(csv.DictReader((ROOT/'data/shecorpus/train/manifest.csv').open()))
reference=next(r for r in rows if r['filename']==synthesis['reference_filename'])
path=ROOT/'configs/tts-infer.yaml';reference_path=ROOT/'configs/tts-reference.json'
backup=ROOT.parent/'gpt-sovits-backups'/('owner-config-install-'+str(time.time_ns()))
backup.mkdir(parents=True)
for old in (path,reference_path):
    if old.exists():shutil.copy2(old,backup/old.name)
config=yaml.safe_load(path.read_text())
config['custom'].update(t2s_weights_path=synthesis['gpt_checkpoint'],vits_weights_path=synthesis['sovits_checkpoint'])

def write(destination,text):
    temporary=destination.with_name(destination.name+'.tmp')
    with temporary.open('w') as f:f.write(text);f.flush();os.fsync(f.fileno())
    os.replace(temporary,destination)
    fd=os.open(destination.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

write(reference_path,json.dumps(dict(audio_path=str(ROOT/reference['filename']),text=reference['text']),indent=2)+'\n')
write(path,yaml.safe_dump(config))
print('Installed selected checkpoints and training-only reference. Service activation and Hermes provider selection remain owner-controlled.')
