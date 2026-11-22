"""Train fresh Orpheus V2; serialize GPU stages and leave services stopped."""
from common import *
import fcntl,signal,subprocess,sys,traceback
sys.path.insert(0,str(REPO/'scripts'))
from asr_worker import process_identity
SERVICES=['orpheus-tts.service','orpheus-tts-16bit.service','cesame-tts.service','ollama.service','wyoming-whisper.service','wyoming-orpheus.service']
child=None
def cancel(*args):
    if child is not None:
        child.terminate()
        try:child.wait(timeout=25)
        except subprocess.TimeoutExpired:child.kill();child.wait()
    raise InterruptedError('V2 training controller stopped')
def main():
    global child
    for signum in [signal.SIGTERM,signal.SIGINT]:signal.signal(signum,cancel)
    if (RUN/'training-result.json').exists():print('V2 already trained; refusing accidental restart.');return
    with (REPO/'.cache/training-launch.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        subprocess.run(['/home/voiceuser/.local/bin/orpheus','stop'],check=True)
        subprocess.run(['systemctl','--user','stop',*SERVICES],check=True)
        RUN.mkdir(parents=True,exist_ok=True)
        atomic(RUN/'authorization.json',dict(owner_request='Stop all services on GPU and train Orpheus on the approved V2 corpus.',
            epochs=3,basis='Same Orpheus training configuration as V1; fresh LoRA from pinned original base.',
            services_stopped=SERVICES,services_restore_on_completion=False,production_selection_unchanged=True,time=time.time()))
        def stage(name,script,args=None,asr_env=False):
            global child
            status(status='preparing' if name in ['verify','encode','codec-qa','codec-resolve'] else 'preflight',stage=name,error=None)
            command=[str(ROOT/'.venv/bin/python'),str(Path(__file__).parent/script),*(args or [])]
            if asr_env:command=['/bin/bash','-c','source /home/voiceuser/projects/gpt-sovits/scripts/env.sh; exec python "$1"','v2-qa',str(Path(__file__).parent/script)]
            wrapped=[str(ROOT/'.venv/bin/python'),str(REPO/'scripts/asr_worker.py'),'--root',str(REPO),
                '--run-id',RUN.name,'--stage',name,'--profile','asr','--controller-pid',str(os.getpid()),
                '--controller-start-time',process_identity(os.getpid()),'--inherited-lock-fd',str(lease.fileno()),'--',*command]
            with (RUN/(name+'.log')).open('a') as log:
                child=subprocess.Popen(wrapped,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),),cwd=ROOT)
                result=child.wait();child=None
            return result
        if stage('verify','verify.py'):raise RuntimeError('Immutable V2 corpus audit failed')
        if stage('encode','prepare.py'):raise RuntimeError('V2 feature encoding failed')
        dataset=json.loads((DATA/'dataset.json').read_text())
        assert dataset['clips']==2048 and dataset['max_tokens']<=4096 and dataset['no_frames_removed']
        result=stage('codec-qa','qa.py',asr_env=True)
        if result:
            if not (DATA/'qa-asr.json').exists() or not json.loads((DATA/'qa-asr.json').read_text()).get('complete'):raise RuntimeError('Codec QA did not finish')
            if stage('codec-resolve','qa_resolve.py'):raise RuntimeError('Independent codec-ending QA failed')
        assert json.loads((DATA/'qa-asr.json').read_text())['all_final_words_preserved']
        precision='16bit'
        for precision in ['16bit','4bit']:
            if precision=='4bit':
                command=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/download.py'),'--quantized']
                subprocess.run(command,check=True,cwd=ROOT)
            result=stage('smoke-'+precision,'train.py',['--mode','smoke','--precision',precision])
            if result==0:
                if stage('resume-'+precision,'train.py',['--mode','smoke-resume','--precision',precision]):raise RuntimeError('V2 checkpoint resume test failed')
                break
            failure=RUN/f'failure-smoke-{precision}.json';info=json.loads(failure.read_text()) if failure.exists() else {}
            if precision=='4bit' or not info.get('memory_failure'):raise RuntimeError('V2 preflight failed; no non-memory fallback allowed')
        atomic(RUN/'plan.json',dict(model=MODEL,precision=precision,epochs=3,rank=64,alpha=64,learning_rate=2e-4,
            microbatch=1,gradient_accumulation=4,context=4096,seed=3407,expected_optimizer_steps=1392,
            train_clips=1856,validation_clips=192,corpus_fingerprint=dataset['fingerprint'],max_tokens=dataset['max_tokens'],
            validation='every epoch',preview='three identical comparison sentences after each epoch',resource_caps=None,
            revisions=setup()['model_revisions'],trainer_sha256=sha(Path(__file__).parent/'train.py'),
            production_selection=json.loads((ROOT/'selected-model.json').read_text()),no_automatic_service_restart=True))
        if stage('train','train.py',['--mode','train','--precision',precision]):raise RuntimeError('V2 training failed; checkpoints preserved')
        if stage('preview-qa','preview_qa.py',asr_env=True):raise RuntimeError('V2 preview ASR checks failed')
        status(status='complete_listening_pending',stage='complete',services_left_stopped=SERVICES)
        subprocess.run(['/usr/bin/python3',str(Path(__file__).parent/'record.py')],check=True)
if __name__=='__main__':
    try:main()
    except Exception:
        status(status='failed',error=traceback.format_exc());raise
