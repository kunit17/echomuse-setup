#!/usr/bin/env python3
"""Runs under the existing asr_worker shared reservation, restores voice state."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--image',required=True);args=p.parse_args()
root=Path(__file__).resolve().parent
voice_root=Path('/home/voiceuser/projects/orpheus-lora')
request=voice_root/'activation-request.json'
user=['runuser','-u','voiceuser','--','env','XDG_RUNTIME_DIR=/run/user/1000']
voice=user+['/home/voiceuser/.local/bin/orpheus']
container='echomuse-samantha-training'
prior=json.loads(request.read_text())
was_active=subprocess.run(user+['systemctl','--user','is-active','--quiet','orpheus-tts.service']).returncode==0
state={'started':time.time(),'prior_voice_active':was_active,'prior_desired_on':prior.get('desired_on'), 'container':container,'status':'starting'}
state_path=root/'logs/training-supervisor.json'
stop_state=None

def write():state_path.write_text(json.dumps(state,indent=2)+'\n')
def cancel(signum,frame):raise KeyboardInterrupt()
signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
write()
try:
    # The global training lock is already held by asr_worker, not bypassed.
    if not os.environ.get('ASR_RUN_ID'):raise RuntimeError('Training reservation wrapper required')
    if was_active:
        subprocess.run(voice+['stop'],check=True)
        stop_state=json.loads(request.read_text())
        state['voice_temporarily_stopped']=True;write()
    command=['docker','run','--rm','--name',container,'--gpus','all','--shm-size','8g',
             '--user','1000:1000','-e','HOME=/data/home','-e','HF_HOME=/data/hf-cache',
             '-e','HF_XET_CHUNK_CACHE_SIZE_BYTES=0','-e','HF_HUB_DISABLE_TELEMETRY=1',
             '-v',str(root/'data')+':/data','--entrypoint','python',args.image,'/data/build_canonical.py']
    state['status']='training';write()
    subprocess.run(command,check=True)
    if not (root/'data/models/samantha.onnx').is_file():raise RuntimeError('No exported Samantha model')
    state['status']='trained'
except BaseException as error:
    state.update(status='failed',error=type(error).__name__+': '+str(error));raise
finally:
    subprocess.run(['docker','stop','--time','10',container],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
    current=json.loads(request.read_text())
    # A newer explicit owner Stop/Start takes precedence over the old state.
    if was_active and stop_state is not None and current==stop_state:
        state['voice_restore_requested']=True;write()
        restored=subprocess.run(voice+['start'],check=False)
        state['voice_restore_exit_code']=restored.returncode
    else:state['voice_restore_skipped']='Voice was off or owner activation changed during training'
    state['finished']=time.time();write()
