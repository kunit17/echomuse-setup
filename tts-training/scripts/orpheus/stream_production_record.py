"""Record owner approval, deployment checks and protected serving dependencies."""
from common import *
import shutil,subprocess,urllib.request
workspace=Path('/mnt/c/Users/DesktopUser/Projects/echomuse-provisioning/tts-training/orpheus-training');B=ROOT/'stream-benchmark'
browser=json.loads((workspace/'browser-production-streaming-epoch3.json').read_text());assert browser['status']=='complete' and browser['underruns']==0 and browser['bufferSeconds']==1
assert all(browser['apiChecks'].values())
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with http.open('http://127.0.0.1:8812/health',timeout=5) as response:health=json.load(response)
assert health['ready'] and health['precision']=='4bit' and health['runtime']=='vllm-streaming'
record=dict(owner_approval=json.loads((B/'owner-streaming-approval.json').read_text()),
    selection=json.loads((ROOT/'selected-model.json').read_text()),health=health,browser=browser,
    mp3_smoke=json.loads((workspace/'approved-streaming-tony.json').read_text()),
    fallback_retained=(Path.home()/'.config/systemd/user/orpheus-tts-16bit.service').exists(),
    production_dependencies=[str(B/'.venv'),str(B/'merged-epoch3')])
atomic(REPO/'reports/orpheus-streaming-production-20261118.json',record)
shutil.copy2(workspace/'STREAMING_ACTIVATION.md',ROOT/'ACTIVATION.md');shutil.copy2(ROOT/'ACTIVATION.md',workspace/'ACTIVATION.md')
shutil.copy2(ROOT/'ACTIVATION.md',REPO/'scripts/orpheus/ACTIVATION.md')
shutil.copy2(workspace/'STREAMING_ACTIVATION.md',REPO/'scripts/orpheus/STREAMING_ACTIVATION.md')
streaming=workspace/'STREAMING.md';text=streaming.read_text()
banner='## Current production selection\n\nOwner listening review is approved. The 4-bit vLLM runtime is now the default\non port8812 with a1-second buffer and sentence generation. Read ACTIVATION.md.\nThe stream-benchmark/.venv and merged-epoch3 folders are active serving\ndependencies; do not delete them as temporary files. The following benchmark\nsections retain their historical settings and do not override this selection.\n\n'
if '## Current production selection' not in text:text=text[:text.index('\n')+1]+'\n'+banner+text[text.index('\n')+1:]
streaming.write_text(text)
readme=ROOT/'README.md';text=readme.read_text()
if '## Approved streaming default' not in text:readme.write_text('# Orpheus voice\n\n## Approved streaming default\n\nEpoch3 now uses vLLM4-bit, CUDA graphs and GPU SNAC. Live playback: http://localhost:8812/. One-second audio buffer and sentence generation. `orpheus start|stop|status|sample` remain the public commands; read ACTIVATION.md for the protected production dependencies and retained16-bit fallback.\n\n'+text)
agents=ROOT/'AGENTS.md';text=agents.read_text()
if '4-bit streaming is approved' not in text:agents.write_text(text+'\n4-bit streaming is approved and selected. Read ACTIVATION.md. The stream-benchmark/.venv and merged-epoch3 directories are production dependencies, never disposable benchmark cache.\n')
shutil.copy2(workspace/'stream_review.html',B/'review.html')
subprocess.run(['/usr/bin/python3',str(workspace/'stream_record.py')],check=True)
paths=['selected-tts-model.json','reports/orpheus-streaming-production-20261118.json','scripts/orpheus/ACTIVATION.md','scripts/orpheus/STREAMING_ACTIVATION.md']
subprocess.run(['git','-C',str(REPO),'diff','--check','--',*paths],check=True)
subprocess.run(['git','-C',str(REPO),'add','--',*paths],check=True)
subprocess.run(['git','-C',str(REPO),'commit','-m','Promote owner-approved Orpheus 4-bit streaming default and retain 16-bit fallback'],check=True)
subprocess.run(['git','-C',str(REPO),'push','origin','master'],check=True)
