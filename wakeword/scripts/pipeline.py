#!/usr/bin/env python3
"""Samantha preparation, guarded downloads, reserved training and evaluation."""
import json,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent
state_path=root/'pipeline-status.json'
state={'status':'running','stage':'waiting-for-image','started':time.time()}
def save():
 state['updated']=time.time();state_path.write_text(json.dumps(state,indent=2)+'\n')
def run(command,stage):
 state['stage']=stage;save();print('STAGE '+stage,flush=True)
 subprocess.run(command,check=True)
def docker_args(image,name,gpu=False):
 values=['docker','run','--rm','--name',name,'--user','1000:1000','--shm-size','8g',
         '-e','HOME=/data/home','-e','HF_HOME=/data/hf-cache','-e','HF_XET_CHUNK_CACHE_SIZE_BYTES=0',
         '-e','HF_HUB_DISABLE_TELEMETRY=1','-v',str(root/'data')+':/data']
 if gpu:values+=['--gpus','all']
 return values+[image]
def guard(name,command,container=None):
 values=['python3',str(root/'guarded_job.py'),'--name',name,'--minimum-free-gib','10']
 if container:values+=['--container',container]
 return values+['--']+command
save()
try:
 while True:
  try:pull=json.loads((root/'logs/pull-trainer.json').read_text())
  except (FileNotFoundError,json.JSONDecodeError):time.sleep(2);continue
  if pull['status']=='running':time.sleep(3);continue
  if pull['status']!='success':raise RuntimeError('Image download stopped; see disk guard report')
  break
 inspect=json.loads(subprocess.check_output(['docker','image','inspect','ghcr.io/wilbowes/echomuse-forge:latest'],text=True))[0]
 image=next(d for d in inspect['RepoDigests'] if d.startswith('ghcr.io/wilbowes/echomuse-forge@sha256:'))
 (root/'published-image.json').write_text(json.dumps({'image':image,'labels':inspect['Config'].get('Labels'),'id':inspect['Id']},indent=2)+'\n')
 state['image']=image;save()
 config=root/'data/wakewords/samantha/config.yml'
 if not config.exists():
  run(docker_args(image,'echomuse-samantha-config')+['new','Samantha'],'create-samantha-config')
  # Add deliberate near-misses; retain default 30k/2k positives and 50k steps.
  text=config.read_text().replace('custom_negative_phrases: []',
      'custom_negative_phrases: ["Amanda", "Savannah", "Santa", "some other", "Sam and the", "Scarlett"]')
  config.write_text(text)
 run(guard('assets',docker_args(image,'echomuse-samantha-assets')+['assets'],
           'echomuse-samantha-assets'),'download-training-assets')
 counts={folder:len(list((root/'data/assets'/folder).glob('*.wav'))) for folder in ['mit_rirs','audioset_16k','fma_16k']}
 state['asset_counts']=counts;save()
 if counts['mit_rirs']<200 or counts['audioset_16k']<2000 or counts['fma_16k']<1000:
  raise RuntimeError('Asset download incomplete; refusing partial background corpus')
 reserved=['python3','/home/voiceuser/projects/gpt-sovits/scripts/asr_worker.py',
           '--root','/home/voiceuser/projects/gpt-sovits','--run-id','wake-samantha-20261120',
           '--stage','build','--profile','asr','--','python3',str(root/'train_reserved.py'),'--image',image]
 run(guard('training',reserved,'echomuse-samantha-training'),'train-samantha-with-gpu-reservation')
 # Evaluation only uses the CPU provider, after voice restoration.
 command=docker_args(image,'echomuse-samantha-evaluation')
 command=command[:-1]+['--entrypoint','python',image,'/data/evaluate_samantha.py']
 run(guard('evaluation',command,'echomuse-samantha-evaluation'),'evaluate-held-out-windows-voices')
 final=docker_args(image,'echomuse-samantha-final-evaluation')
 final=final[:-1]+['--entrypoint','python',image,'/data/evaluate_samantha.py','--suite','final-evaluation']
 run(guard('final-evaluation',final,'echomuse-samantha-final-evaluation'),'evaluate-fresh-final-voices')
 state.update(status='ready-to-install',stage='evaluation-passed',finished=time.time());save()
except BaseException as error:
 state.update(status='stopped',error=type(error).__name__+': '+str(error),finished=time.time());save();raise
