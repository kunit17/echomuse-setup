"""Explicit systemd lifecycle for ASR workers; importing this never loads models."""
import argparse
import fcntl
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import uuid
from asr_runtime import archive, atomic_json, read_json
from asr_worker import process_identity


def service_properties(unit):
    names = ['LoadState','ActiveState','SubState','Result','ExecMainStatus','InvocationID',
             'MemoryCurrent','MemoryPeak','MemorySwapPeak','MemoryHigh','MemoryMax','MemorySwapMax']
    result = subprocess.run(['systemctl','--user','show',unit,*['--property='+n for n in names]],
                            text=True,capture_output=True,timeout=15)
    return dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)


def verify_completion(properties, invocation, worker, run_id):
    if properties.get('LoadState') != 'loaded' or not invocation or properties.get('InvocationID') != invocation:
        raise RuntimeError('Worker identity changed or disappeared')
    if properties.get('Result') != 'success' or properties.get('ExecMainStatus') != '0':
        raise RuntimeError('Worker failed: '+properties.get('Result','unknown'))
    if not worker or worker.get('run_id') != run_id or not worker.get('complete') or worker.get('status') != 'success':
        raise RuntimeError('No matching durable worker completion record')


class Controller:
    def __init__(self, root, mode, memory_max=None):
        self.root=Path(root);self.mode=mode;self.memory_max=memory_max
        self.memory_high=max(1,memory_max*7//8) if memory_max else None
        self.run_id=time.strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:10]
        self.directory=self.root/'reports/asr-runs'/self.run_id
        self.directory.mkdir(parents=True,exist_ok=True)
        self.active=None
        self.state=dict(schema=2,run_id=self.run_id,mode=mode,status='starting',started=time.time(),
                        controller_pid=os.getpid(),controller_start_time=process_identity(os.getpid()),
                        profile=dict(device='cuda',compute_type='float16',memory_max_mib=memory_max,
                                     memory_high_mib=self.memory_high,memory_swap_max_mib=None))

    def save(self, **values):
        self.state.update(values,updated=time.time())
        atomic_json(self.directory/'controller.json',self.state)
        atomic_json(self.root/'reports/asr-status.json',self.state)

    def cancel(self):
        if self.active:
            subprocess.run(['systemctl','--user','stop',self.active],timeout=30,
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            self.active=None

    def run_stage(self, stage, script, arguments=(), profile='asr', runtime=0):
        unit='she-corpus-'+stage+'.service'
        if service_properties(unit).get('ActiveState') in ['active','activating','deactivating']:
            raise RuntimeError('A worker already owns '+unit)
        for action in ['stop','reset-failed']:
            subprocess.run(['systemctl','--user',action,unit],capture_output=True,timeout=20)
        self.save(active_stage=stage,status='running')
        python=self.root/'.venv/bin/python'
        command=[str(python),'scripts/asr_worker.py','--root',str(self.root),'--run-id',self.run_id,
                 '--stage',stage,'--profile',profile,'--runtime',str(runtime),
                 '--controller-pid',str(os.getpid()),'--controller-start-time',process_identity(os.getpid()),
                 '--',str(python),str(self.root/'scripts'/script),*arguments]
        shell=('source scripts/env.sh; export PATH=/home/voiceuser/.local/bin:$PATH; '
               'export PYTHONPATH=; exec '+shlex.join(command))
        parent='she-corpus-asr-controller.service' if self.mode=='corpus' else 'she-corpus-preflight-controller.service'
        props=dict(MemoryAccounting='true',MemoryHigh=f'{self.memory_high}M' if self.memory_max else 'infinity',
                   MemoryMax=f'{self.memory_max}M' if self.memory_max else 'infinity',MemorySwapMax='infinity',OOMPolicy='stop',
                   KillMode='control-group',TimeoutStopSec='15',RemainAfterExit='yes',
                   RuntimeMaxSec=str(runtime+30) if runtime else 'infinity',WorkingDirectory=str(self.root),PartOf=parent,
                   StandardOutput=f'append:{self.root}/logs/{unit}.log',
                   StandardError=f'append:{self.root}/logs/{unit}.log')
        parent_state=service_properties(parent)
        if parent_state.get('LoadState')=='loaded' and parent_state.get('ActiveState')=='active':
            props.update(BindsTo=parent,After=parent)
        args=['systemd-run','--user','--unit='+unit]
        for key,value in props.items():args.extend(['--property',key+'='+value])
        self.active=unit
        try:
            subprocess.run([*args,'/bin/bash','--noprofile','--norc','-c',shell],check=True,timeout=30)
            first=service_properties(unit);invocation=first.get('InvocationID')
            if first.get('LoadState')!='loaded' or not invocation:raise RuntimeError('Worker did not start')
            deadline=time.monotonic()+runtime+45
            while True:
                status=service_properties(unit)
                if status.get('LoadState')!='loaded':raise RuntimeError('Worker disappeared')
                if status.get('ActiveState') not in ['activating','deactivating'] and status.get('SubState')!='running':break
                if runtime and time.monotonic()>deadline:raise RuntimeError('Controller runtime deadline exceeded')
                time.sleep(1)
            worker=read_json(self.directory/(stage+'.json'))
            if not worker or worker.get('status')!='success':
                self.state['worker_failure_kind']=('cgroup_oom' if status.get('Result')=='oom-kill'
                                                   else (worker or {}).get('failure_kind',status.get('Result','unknown')))
            atomic_json(self.directory/(stage+'-resources.json'),dict(run_id=self.run_id,invocation_id=invocation,service=status,worker=worker))
            atomic_json(self.root/'reports'/(unit.removesuffix('.service')+'-resources.json'),dict(run_id=self.run_id,service=status))
            verify_completion(status,invocation,worker,self.run_id)
        finally:self.cancel()

    def run(self):
        lock_path=self.root/'.cache/asr-controller.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
        with lock_path.open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                self.state.update(status='failed',failure_kind='controller_busy',finished=time.time())
                atomic_json(self.directory/'controller.json',self.state)
                raise RuntimeError('Another ASR controller is active') from None
            self.save()
            try:
                subprocess.run(['systemctl','--user','stop','she-gptsovits-tts.service'],check=True,timeout=30)
                if service_properties('she-gptsovits-tts.service').get('ActiveState')!='inactive':
                    raise RuntimeError('Warm TTS must remain inactive')
                if self.mode=='preflight':
                    path=self.root/'reports/corpus-asr-preflight.json'
                    archive(path,self.root/'reports/asr-history')
                    atomic_json(path,dict(status='starting',run_id=self.run_id,device='cuda',compute_type='float16',model_ready=False))
                    self.run_stage('asr-preflight','corpus_asr_preflight.py')
                    result=read_json(path,{})
                    if result.get('run_id')!=self.run_id or result.get('status')!='passed':
                        raise RuntimeError('Preflight result is absent or stale')
                else:
                    self.run_stage('ground-map','corpus_ground_truth.py',['map'],'cpu')
                    self.run_stage('ground-asr','corpus_ground_truth.py',['asr'],)
                    result=read_json(self.root/'reports/corpus-ground-truth.json',{})
                    if result.get('status')!='complete' or result.get('scored_clips')!=result.get('mapped_clips'):
                        raise RuntimeError('Ground-truth validation is incomplete')
                    self.run_stage('two-pass','build_continuous_corpus.py')
                    result=read_json(self.root/'reports/corpus-verification.json',{})
                    if not result.get('corpus_built') or not result.get('clips') or not result.get('run_fingerprint'):
                        raise RuntimeError('Corpus completion is absent or lacks provenance')
                self.save(status='complete',active_stage=None,finished=time.time())
            except BaseException as error:
                self.save(status='failed',failure_kind=type(error).__name__,error=str(error),finished=time.time())
                if self.mode=='preflight':
                    path=self.root/'reports/corpus-asr-preflight.json'
                    value=read_json(path,{})
                    if value.get('run_id')==self.run_id:
                        value.update(status='failed',failure_kind=self.state.get('worker_failure_kind',str(error)))
                        atomic_json(path,value)
                raise
            finally:self.cancel()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['corpus','preflight'])
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--memory-max-mib',type=int,default=None,help='Optional explicit cap; default unlimited')
    args=parser.parse_args()
    if args.mode=='corpus':
        parser.error('Retired corpus mode. Read /home/voiceuser/projects/gpt-sovits/TTS_WORKFLOW.md; use scripts/corpus_repair/run.py')
    controller=Controller(args.root,args.mode,args.memory_max_mib)
    def cancel(signum,frame):raise InterruptedError('Controller cancelled')
    for signum in [signal.SIGINT,signal.SIGTERM]:signal.signal(signum,cancel)
    controller.run()


if __name__=='__main__':main()
