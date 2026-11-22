"""Derive an isolated V2 trainer from the tested V1 sources, not serving code."""
import hashlib,shutil,json
from pathlib import Path
ROOT=Path('/home/voiceuser/projects/orpheus-lora');source=ROOT/'scripts';target=source/'v2';target.mkdir(exist_ok=True)
workspace=Path(__file__).resolve().parent
provenance={}
for name in ['common.py','prepare.py','train.py','qa.py','qa_resolve.py','preview_qa.py','monitor.py','test_features.py']:
    text=(source/name).read_text();provenance[name]=hashlib.sha256(text.encode()).hexdigest()
    if name=='common.py':
        text=text.replace('boundary-v1-20261115','boundary-v2-20261116').replace('orpheus-v1-example01','orpheus-v2-example02').replace("data/v1-snac","data/v2-snac")
    if name in ['prepare.py','train.py']:
        text=text.replace('1083','1856').replace('121','192').replace('1204','2048')
    if name=='train.py':
        text=text.replace("wall_seconds=time.time()-started,", "wall_seconds=time.time()-started,adapter_sha256=sha(checkpoint/'adapter_model.safetensors'),revisions=setup()['model_revisions'],")
    if name=='monitor.py':text=text.replace('8769','8770').replace('v1 corpus','v2 corpus').replace('1204','2048')
    if name=='qa_resolve.py':text=text.replace('two Whisper suffix disagreements independently resolved','Whisper ending disagreements independently checked')
    (target/name).write_text(text);(workspace/name).write_text(text)
for name in ['pipeline.py','verify.py','record.py','README.md']:
    if (workspace/name).exists():shutil.copy2(workspace/name,target/name)
(workspace/'source-provenance.json').write_text(json.dumps(provenance,indent=2))
print('Installed isolated V2 sources; production and V1 sources unchanged.')
