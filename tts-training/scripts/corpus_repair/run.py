"""Supervise sequential source-repair stages; never launch model training."""
import json,os,signal,subprocess,time,fcntl,runpy
from pathlib import Path
ROOT=Path('/home/voiceuser/projects/gpt-sovits')
SCRIPT=ROOT/'scripts/corpus_repair/build.py'
builder=runpy.run_path(str(SCRIPT))
from asr_worker import process_identity
WORK=builder['WORK']
WORK.mkdir(parents=True,exist_ok=True)
PYTHON=ROOT/'.venv/bin/python'
child=None

def cancel(signum,frame):
    if child is not None:
        child.terminate()
        try:child.wait(timeout=20)
        except subprocess.TimeoutExpired:child.kill();child.wait()
    raise InterruptedError('Owner or host stopped corpus repair')

if __name__=='__main__':
    builder['validate_recipe']()
    if (builder['OUT']/'corpus.json').exists():
        report=json.loads((builder['OUT']/'corpus.json').read_text())
        for split in report['splits'].values():
            assert builder['file_digest'](Path(split['path']))['sha256']==split['sha256']
        print('Completed dataset already exists; no preparation, service or training state was changed.')
        raise SystemExit(0)
    launch_lock=(ROOT/'.cache/training-launch.lock').open('a')
    fcntl.flock(launch_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for s in [signal.SIGTERM,signal.SIGINT]:signal.signal(s,cancel)
    selection=ROOT/'selected-tts-model.json'
    service=json.loads(selection.read_text()).get('service','cesame-tts.service') if selection.exists() else 'cesame-tts.service'
    assert service in ('cesame-tts.service','orpheus-tts.service')
    restore=subprocess.run(['systemctl','--user','is-active','--quiet',service]).returncode==0
    # Preserve initial service state across a restart of this supervisor.
    state_path=WORK/'supervisor.json'
    if state_path.exists():
        previous=json.loads(state_path.read_text())
        restore=restore or (previous.get('service','cesame-tts.service')==service and previous.get('restore_service',previous.get('restore_csm_service',False)))
    state=dict(started=time.time(),service=service,restore_service=restore,status='running',training_started=False)
    (Path('/home/voiceuser/projects/csm-lora')/'PAUSED_BY_OWNER').write_text('Corpus repair in progress; no model training requested.\n')
    try:
        subprocess.run(['systemctl','--user','stop',service],check=True)
        stages=[('scan','cpu',[]),('asr-pilot','asr',['--pilot']),('align-pilot','asr',['--pilot']),
                ('pilot-report','cpu',[]),('asr-full','asr',[]),('align-full','asr',[]),('publish','cpu',[])]
        for stage,profile,extra in stages:
            phase='asr' if stage.startswith('asr-') else 'align' if stage.startswith('align-') else stage
            state.update(stage=stage,updated=time.time());state_path.write_text(json.dumps(state,indent=2))
            command=[str(PYTHON),str(ROOT/'scripts/asr_worker.py'),'--root',str(ROOT),
                '--run-id','corpus-repair-'+builder['VERSION'],'--stage',stage,'--profile',profile,
                '--controller-pid',str(os.getpid()),'--controller-start-time',process_identity(os.getpid()),
                '--inherited-lock-fd',str(launch_lock.fileno()),
                '--',str(PYTHON),str(SCRIPT),phase,*extra]
            with (WORK/(stage+'.log')).open('a') as log:
                print('START',stage,flush=True)
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                                       pass_fds=(launch_lock.fileno(),))
                result=child.wait();child=None
                if result:raise RuntimeError(stage+' failed; see '+str(WORK/(stage+'.log')))
        state.update(status='complete',finished=time.time())
    except BaseException as error:
        state.update(status='failed',error=str(error),finished=time.time());raise
    finally:
        state_path.write_text(json.dumps(state,indent=2))
        if restore:
            # GPU workers have exited before restoring the previously active service.
            subprocess.run(['systemctl','--user','start','--no-block',service],check=False)
