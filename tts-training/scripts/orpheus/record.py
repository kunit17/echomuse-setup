"""Persist implementation and a factual handoff without committing audio/weights."""
from common import *
import argparse,shutil,subprocess
import importlib.metadata
parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args()
target=REPO/'scripts/orpheus';target.mkdir(exist_ok=True)
for path in (ROOT/'scripts').glob('*.py'):shutil.copy2(path,target/path.name)
shutil.copy2(ROOT/'README.md',target/'README.md')
record=dict(project=str(ROOT),run=str(RUN),corpus=str(CORPUS),fingerprint=json.loads((DATA/'dataset.json').read_text())['fingerprint'],
    encoded_clips=1204,max_tokens=2499,epochs=3,setup=setup(),
    current_status=json.loads((RUN/'status.json').read_text()),
    codec_qa=dict(examples=11,all_endings_preserved=json.loads((DATA/'qa-asr.json').read_text())['all_final_words_preserved'],manual_listening_complete=False))
record['packages']={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
for name in ['smoke-16bit-passed.json','smoke-resume-16bit-passed.json','plan.json','training-result.json','preview-qa.json']:
    if (RUN/name).exists():record[name]=json.loads((RUN/name).read_text())
atomic(REPO/'reports/orpheus-v1-training.json',record)
guide=REPO/'TTS_WORKFLOW.md';text=guide.read_text();marker='## Orpheus experiment';suffix=''
if marker in text:
    start=text.index(marker);following=text.find('\n## ',start+len(marker))
    if following>=0:suffix=text[following:]
    text=text[:start].rstrip()+'\n'
state=record['current_status'];summary='completed at epoch 3' if args.final else 'running in 16-bit LoRA after memory/preview/resume preflight'
text+='\n'+marker+'\n\nOwner approved v1 corpus, 16-bit first and three epochs. Orpheus is '+summary+'.\n'
text+='Project: `/home/voiceuser/projects/orpheus-lora`; run `orpheus-v1-example01`.\n'
text+='Commands: `orpheus prepare`, `orpheus train`, `orpheus sample --text-file PATH --out PATH.mp3`, `orpheus status`.\n'
selection=json.loads((ROOT/'selected-model.json').read_text()) if (ROOT/'selected-model.json').exists() else None
text+='Progress/previews: http://localhost:8769/. The immutable v1 source manifests\nand recordings are unchanged. SNAC keeps all frames; context4096 holds the\nmaximum2499-token example. No speaker prefix or reference clip is needed for\nthis single-voice Orpheus run. '+('Orpheus epoch 3 is the owner-selected main model.\n' if selection else 'CSM stays the selected deployment.\n')
if selection:text+='Activation: `orpheus start`, `orpheus stop`, `orpheus status`. Samples reuse the warm service on loopback port 8812. Selection: `selected-tts-model.json`. Stop Orpheus before any GPU corpus/training job and restore it afterward.\n'
text+='Implementation: `scripts/orpheus/README.md`; exact revisions, preparation,\npreflight and final metrics: `reports/orpheus-v1-training.json`. Model weights,\ntoken caches, and MP3 previews remain local. Final checkpoint/adapter and\nper-epoch previews remain after temporary preflight states are removed.\n'
text+='Dependencies use a fresh interpreter with hardlinked pinned package payloads\nto avoid duplicating large CUDA libraries; do not edit or upgrade shared files\nin place. See the Orpheus README before changing its environment.\n'
guide.write_text(text.rstrip()+'\n'+suffix)
workspace=Path('/mnt/c/Users/DesktopUser/Projects/echomuse-provisioning/tts-training')
if workspace.is_dir():shutil.copy2(guide,workspace/'TTS_WORKFLOW.md')
paths=['TTS_WORKFLOW.md','scripts/orpheus','reports/orpheus-v1-training.json']
subprocess.run(['git','-C',str(REPO),'diff','--check','--',*paths],check=True)
subprocess.run(['git','-C',str(REPO),'add','--',*paths],check=True)
subprocess.run(['git','-C',str(REPO),'commit','-m','Record completed Orpheus run and preview checks' if args.final else 'Implement isolated Orpheus v1 LoRA pipeline and start approved three-epoch run'],check=True)
subprocess.run(['git','-C',str(REPO),'push','origin','master'],check=True)
