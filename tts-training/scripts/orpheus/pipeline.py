"""Own the reservation across Orpheus preflight and three-epoch training."""
from common import *
import fcntl,signal,subprocess,sys,traceback
sys.path.insert(0,str(REPO/'scripts'))
from asr_worker import process_identity
child=None
service_stopped=False
def cancel(signum,frame):
    if child is not None:
        child.terminate()
        try:child.wait(timeout=20)
        except subprocess.TimeoutExpired:child.kill();child.wait()
    raise InterruptedError('Orpheus pipeline stopped')
def main():
    global child,service_stopped
    if (RUN/'training-result.json').exists():
        print('Three-epoch run already finished; no GPU or service state changed.',flush=True)
        return
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,cancel)
    assert json.loads((DATA/'dataset.json').read_text())['clips'] > 1
    assert (ROOT/'model.json').exists()
    assert json.loads((DATA/'qa-asr.json').read_text())['all_final_words_preserved']
    with (REPO/'.cache/training-launch.lock').open('a') as reservation:
        fcntl.flock(reservation,fcntl.LOCK_EX|fcntl.LOCK_NB)
        subprocess.run(['systemctl','--user','stop','cesame-tts.service'],check=True)
        service_stopped=True
        RUN.mkdir(parents=True,exist_ok=True)
        def stage(name,script,args):
            global child
            status(stage=name,error=None)
            cmd=[str(ROOT/'.venv/bin/python'),str(REPO/'scripts/asr_worker.py'),'--root',str(REPO),
                '--run-id','orpheus-v1-example01','--stage',name,'--profile','asr',
                '--controller-pid',str(os.getpid()),'--controller-start-time',process_identity(os.getpid()),
                '--inherited-lock-fd',str(reservation.fileno()),'--',str(ROOT/'.venv/bin/python'),str(ROOT/'scripts'/script),*args]
            with (RUN/(name+'.log')).open('a') as log:
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,pass_fds=(reservation.fileno(),),cwd=ROOT)
                result=child.wait();child=None
            return result
        precision='16bit'
        for candidate in ['16bit','4bit']:
            precision=candidate
            if candidate=='4bit':
                status(status='downloading_4bit_fallback')
                subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/download.py'),'--quantized'],check=True,cwd=ROOT)
            result=stage('smoke-'+candidate,'train.py',['--mode','smoke','--precision',candidate])
            if result==0:
                if stage('resume-'+candidate,'train.py',['--mode','smoke-resume','--precision',candidate]):raise RuntimeError('Checkpoint resume preflight failed')
                break
            failure=RUN/f'failure-smoke-{candidate}.json'
            info=json.loads(failure.read_text()) if failure.exists() else {}
            if candidate=='4bit' or not info.get('memory_failure'):
                raise RuntimeError('Preflight failed; see '+str(RUN/('smoke-'+candidate+'.log')))
        atomic(RUN/'plan.json',dict(model=MODEL,precision=precision,epochs=3,rank=64,alpha=64,
            learning_rate=2e-4,microbatch=1,gradient_accumulation=4,context=4096,seed=3407,
            corpus_fingerprint=json.loads((DATA/'dataset.json').read_text())['fingerprint'],
            validation='every epoch',preview='three sentences after every epoch',resource_caps=None,
            revisions=setup()['model_revisions'],trainer_sha256=sha(ROOT/'scripts/train.py')))
        if stage('train','train.py',['--mode','train','--precision',precision]):raise RuntimeError('Training failed; checkpoint retained')
        # Only disposable preflight model states; keep logs, passes and preview audio.
        import shutil
        for directory in RUN.glob('smoke-*'):
            if directory.is_dir() and directory.parent==RUN:shutil.rmtree(directory)
    print('Training completed; inference remains off until explicitly selected and started.')
if __name__=='__main__':
    try:main()
    except Exception:
        status(status='failed',error=traceback.format_exc())
        # Leave inference off after failure; preserve checkpoints for recovery.
        raise
