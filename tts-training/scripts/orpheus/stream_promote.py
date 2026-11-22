"""Deploy the owner-approved epoch-3 streaming runtime, retaining 16-bit fallback."""
from common import *
import shutil,subprocess
if json.loads((ROOT/'selected-model.json').read_text()).get('activation_mode')=='desktop-on-demand':
    raise SystemExit('Desktop on-demand policy is active. Do not re-enable automatic startup with this historical promotion script.')
source=Path('/mnt/c/Users/DesktopUser/Projects/echomuse-provisioning/tts-training/orpheus-training')
for name in ['stream_server.py','cli.py','stream_promote.py']:shutil.copy2(source/name,ROOT/'scripts'/name)
B=ROOT/'stream-benchmark';selection=json.loads((ROOT/'selected-model.json').read_text())
assert selection['epoch']==3 and sha(Path(selection['adapter_path'])/'adapter_model.safetensors')==selection['adapter_sha256']
qa=json.loads((B/'buffer-policy-check.json').read_text())
assert all(row['underruns']==0 and row['complete_audio'] for row in qa['samples'])
units=Path.home()/'.config/systemd/user';old=units/'orpheus-tts.service';fallback=units/'orpheus-tts-16bit.service'
if not (ROOT/'selected-model.16bit.json').exists():atomic(ROOT/'selected-model.16bit.json',selection)
if not fallback.exists():
    text=old.read_text();assert 'scripts/serve.py' in text
    fallback.write_text(text.replace('Conflicts=cesame-tts.service','Conflicts=cesame-tts.service orpheus-tts.service').replace('Description=Orpheus epoch-3 warm TTS','Description=Orpheus epoch-3 16-bit fallback TTS'))
approval=dict(approved_at=time.time(),approved_by='owner',scope='Owner completed listening review of updated Tony and longer passage samples and explicitly approved making 4-bit streaming the default.',
    adapter_sha256=selection['adapter_sha256'],precision='4bit',playback_buffer_seconds=1,text_policy='sentences-v1')
atomic(B/'owner-streaming-approval.json',approval)
selection.update(runtime='vllm-streaming',precision='4bit',playback_buffer_seconds=1,text_policy='sentences-v1',
    inference_model_path=str(B/'merged-epoch3'),python_environment=str(B/'.venv'),
    endpoint='http://127.0.0.1:8812',service='orpheus-tts.service',stream_endpoint='http://127.0.0.1:8812/stream',
    fallback_service='orpheus-tts-16bit.service',owner_approval=approval)
subprocess.run(['systemctl','--user','stop','orpheus-tts.service','orpheus-tts-16bit.service'],check=True)
old.write_text('''[Unit]
Description=Orpheus epoch-3 approved 4-bit streaming TTS
Conflicts=cesame-tts.service orpheus-tts-16bit.service
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/voiceuser/projects/orpheus-lora
Environment=PYTHONUNBUFFERED=1
Environment=HF_HUB_OFFLINE=1
ExecStart=/home/voiceuser/projects/orpheus-lora/stream-benchmark/.venv/bin/python /home/voiceuser/projects/orpheus-lora/scripts/stream_server.py --precision 4bit --codec cuda --port 8812
Restart=on-failure
RestartSec=10
TimeoutStartSec=300

[Install]
WantedBy=default.target
''')
atomic(ROOT/'selected-model.json',selection);atomic(REPO/'selected-tts-model.json',selection)
subprocess.run(['systemctl','--user','daemon-reload'],check=True)
subprocess.run(['systemctl','--user','enable','--now','orpheus-tts.service'],check=True)
print(json.dumps(selection,indent=2))
