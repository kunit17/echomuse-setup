"""Scoped V2 handoff; preserve unrelated work and the selected V1 deployment."""
from common import *
import difflib,shutil,subprocess
source=Path('/mnt/c/Users/DesktopUser/Projects/echomuse-provisioning/tts-training/orpheus-v2-training')
target=REPO/'scripts/orpheus/v2';target.mkdir(exist_ok=True)
for path in source.iterdir():
    if path.is_file() and (path.suffix in ['.py','.md'] or path.name=='source-provenance.json'):shutil.copy2(path,target/path.name)
shutil.copy2(source/'training_gate.py',REPO/'scripts/orpheus/training_gate.py')
record=dict(run=str(RUN),corpus=str(CORPUS),dataset_feature_path=str(DATA),epochs=3,
    selected_production_model=json.loads((ROOT/'selected-model.json').read_text()),deployment_changed=False,
    stopped_gpu_services=['orpheus-tts','orpheus-tts-16bit','cesame-tts','ollama','wyoming-whisper','wyoming-orpheus'],
    services_automatically_restored=False)
for name in ['status.json','authorization.json','corpus-verification.json','plan.json','smoke-16bit-passed.json','smoke-resume-16bit-passed.json','training-result.json','preview-qa.json']:
    if (RUN/name).exists():record[name]=read_status() if name=='status.json' else json.loads((RUN/name).read_text())
for name in ['dataset.json','qa-asr.json']:
    if (DATA/name).exists():
        value=json.loads((DATA/name).read_text())
        record[name]={k:v for k,v in value.items() if k not in ['splits','results']}
atomic(REPO/'reports/orpheus-v2-training-20261121.json',record)
guide=REPO/'TTS_WORKFLOW.md';section='\n## Orpheus V2 training — November 21\n\nOwner authorized fresh Orpheus training on the approved V2 corpus:1856 training clips (5.676h),192 validation clips (.604h), fingerprint example02f5e3af078842b6be580e2c811fb57d4d884590d94eda53649d677005. Three epochs with the tested V1 LoRA settings;16-bit first,4-bit fallback only on confirmed OOM. Read `scripts/orpheus/v2/README.md`. Run: `orpheus-lora/runs/orpheus-v2-example02`; progress: http://localhost:8770/.\n\nAll GPU services were stopped, including Ollama and Wyoming speech adapters, and must remain stopped at completion. The gateway/search services remain untouched. The selected V1 streaming model and the November20 inference-memory settings are preserved. V2 is not automatically deployed; inspect final metrics and listen before selecting an inference export. Machine handoff: `reports/orpheus-v2-training-20261121.json`.\n'
text=guide.read_text();marker='## Orpheus V2 training — November 21'
if marker not in text:guide.write_text(text.rstrip()+'\n'+section)
shutil.copy2(guide,source.parent/'TTS_WORKFLOW.md')
paths=['scripts/orpheus/v2','scripts/orpheus/training_gate.py','reports/orpheus-v2-training-20261121.json']
subprocess.run(['git','-C',str(REPO),'diff','--check','--',*paths],check=True)
subprocess.run(['git','-C',str(REPO),'add','--',*paths],check=True)
# Stage only this new guide section; leave the pre-existing November20 edit uncommitted.
head=subprocess.run(['git','-C',str(REPO),'show','HEAD:TTS_WORKFLOW.md'],capture_output=True,text=True,check=True).stdout
if marker not in head:
    patch=''.join(difflib.unified_diff(head.splitlines(True),(head.rstrip()+'\n'+section).splitlines(True),fromfile='a/TTS_WORKFLOW.md',tofile='b/TTS_WORKFLOW.md'))
    subprocess.run(['git','-C',str(REPO),'apply','--cached','-'],input=patch,text=True,check=True)
subprocess.run(['git','-C',str(REPO),'commit','-m','Start isolated Orpheus V2 training on the approved larger corpus'],check=True)
subprocess.run(['git','-C',str(REPO),'push','origin','master'],check=True)
