"""Save benchmark implementation and factual results; exclude audio and weights."""
from common import *
import shutil,subprocess
B=ROOT/'stream-benchmark';workspace=Path('/mnt/c/Users/DesktopUser/Projects/echomuse-provisioning/tts-training');source=workspace/'orpheus-training';target=REPO/'scripts/orpheus'
readme=ROOT/'README.md';readme_text=readme.read_text()
if '## Streaming benchmark' not in readme_text:
    readme_text+='\n## Streaming benchmark\n\nRead STREAMING.md for the isolated vLLM experiment and measured results. Listening: http://localhost:8814/. The selected production runtime is unchanged. For an explicit temporary live review, run `python3 scripts/stream_session.py`; Ctrl+C restores the previous production state.\n'
    readme.write_text(readme_text)
shutil.copy2(readme,target/'README.md');shutil.copy2(readme,source/'README.md')
shutil.copy2(source/'STREAMING.md',ROOT/'STREAMING.md')
for pattern in ['stream_*.py','test_stream.py','browser_probe.js','browser_probe.ps1','browser_watch.ps1','worklet.js','STREAMING.md','stream_review.html']:
    for path in source.glob(pattern):shutil.copy2(path,target/path.name)
if (B/'requirements.lock.txt').exists():shutil.copy2(B/'requirements.lock.txt',target/'requirements-stream.lock.txt')
record=dict(production_selection=json.loads((ROOT/'selected-model.json').read_text()),manual_listening_complete=False)
if (B/'buffer-policy-check.json').exists():
    buffered=json.loads((B/'buffer-policy-check.json').read_text())
    atomic(REPO/'reports/orpheus-streaming-buffer-20261118.json',buffered)
    report=json.loads((B/'report.json').read_text());report['current_live_policy_validation']=buffered;atomic(B/'report.json',report)
for name in ['status.json','setup-status.json','codec-check.json','baseline-startup.json','environment.json','screening.json','report.json','failures.json']:
    if (B/name).exists():record[name]=json.loads((B/name).read_text())
atomic(REPO/'reports/orpheus-streaming-benchmark-20261118.json',record)
guide=REPO/'TTS_WORKFLOW.md';text=guide.read_text();marker='## Streaming benchmark';suffix=''
if marker in text:
    start=text.index(marker);following=text.find('\n## ',start+len(marker))
    if following>=0:suffix=text[following:]
    text=text[:start].rstrip()+'\n'
text+='\n'+marker+'\n\nOrpheus epoch 3 remains selected on port 8812. The isolated vLLM streaming\nexperiment is benchmark-only; it never deploys a replacement. Read\n`scripts/orpheus/STREAMING.md`. Results/listening: http://localhost:8814/.\nExperimental PCM API uses port 8813 only while a candidate is running.\nState: `orpheus-lora/stream-benchmark/status.json`; compact report:\n`reports/orpheus-streaming-benchmark-20261118.json`. Preserve final weights,\ncorpora and source recordings. The benchmark owns the GPU reservation and\nrestores the selected service after workers exit; after a host crash inspect\nworker state before `orpheus start`. Human listening approval is still pending.\n'
if (B/'buffer-policy-check.json').exists():text+='\nCurrent live experimental policy: 1-second playback buffer and sentence-by-sentence text generation, preserving every complete audio ending. The original benchmark numbers retain their 500ms whole-passage policy. New two-clip Windows/ASR validation: `reports/orpheus-streaming-buffer-20261118.json`; both played with zero underruns and zero ASR word edits. Main production selection remains unchanged.\n'
if record['production_selection'].get('runtime')=='vllm-streaming':
    text=text[:text.index(marker)]+marker+'\n\nOwner completed listening review and approved 4-bit streaming as the default on November 18, 2026. Epoch 3 uses vLLM CUDA graphs, GPU SNAC, sentence-by-sentence generation and a 1-second client playback buffer. Live page: http://localhost:8812/. Existing `/synthesize` WAV requests and `orpheus sample` remain supported. Read `scripts/orpheus/ACTIVATION.md`; selection is in `selected-tts-model.json`.\n\nThe stream-benchmark/.venv and stream-benchmark/merged-epoch3 directories under orpheus-lora are now active production dependencies; do not delete them as temporary benchmark files. Original 16-bit fallback service and full epoch-3 checkpoint remain. Historical benchmark reports retain their original settings and are not deployment instructions. The selected service must stop before GPU training/preparation and be restored afterward. Owner approval scope and rollout checks are in `reports/orpheus-streaming-production-20261118.json`.\n'
guide.write_text(text.rstrip()+'\n'+suffix);shutil.copy2(guide,workspace/'TTS_WORKFLOW.md')
paths=['scripts/orpheus','TTS_WORKFLOW.md','reports/orpheus-streaming-benchmark-20261118.json']
if (B/'buffer-policy-check.json').exists():paths.append('reports/orpheus-streaming-buffer-20261118.json')
subprocess.run(['git','-C',str(REPO),'diff','--check','--',*paths],check=True)
subprocess.run(['git','-C',str(REPO),'add','--',*paths],check=True)
subprocess.run(['git','-C',str(REPO),'commit','-m','Add isolated epoch-3 streaming benchmark and record measured results'],check=True)
subprocess.run(['git','-C',str(REPO),'push','origin','master'],check=True)
