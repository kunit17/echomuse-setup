"""CPU-only process tests for real ASR-worker reservation and cleanup behavior."""
import fcntl,json,os,signal,subprocess,sys,tempfile,time,unittest,shutil
from pathlib import Path

DEFAULT_WORKER=Path(__file__).with_name('asr_worker.py')
if not DEFAULT_WORKER.exists():DEFAULT_WORKER=Path(__file__).resolve().parents[1]/'scripts/asr_worker.py'
WORKER=Path(os.environ.get('ASR_WORKER_UNDER_TEST',str(DEFAULT_WORKER))).resolve()
sys.path.insert(0,str(WORKER.parent))
from asr_worker import process_identity

def wait_for(predicate,timeout=10):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        value=predicate()
        if value:return value
        time.sleep(.03)
    raise AssertionError('Timed out waiting for test process/state')

class LockTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='asr-lock-test-')
        self.root=Path(self.temp.name);(self.root/'.cache').mkdir()
        self.lock=self.root/'.cache/training-launch.lock'
        self.processes=[];self.handles=[]
    def tearDown(self):
        for p in self.processes:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=8)
                except subprocess.TimeoutExpired:p.kill();p.wait()
            if p.stderr is not None:p.stderr.close()
        for f in self.handles:f.close()
        self.temp.cleanup()
    def reservation(self,locked=True):
        handle=self.lock.open('a');self.handles.append(handle)
        if locked:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return handle
    def free(self):
        with self.lock.open('a') as handle:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return False
            return True
    def state(self,stage):
        path=self.root/'reports/asr-runs/test'/f'{stage}.json'
        try:return json.loads(path.read_text())
        except (OSError,ValueError):return {}
    def worker(self,stage,handle=None,seconds=0):
        command=[sys.executable,str(WORKER),'--root',str(self.root),'--run-id','test','--stage',stage,'--profile','cpu']
        if handle is not None:
            command+=['--controller-pid',str(os.getpid()),'--controller-start-time',process_identity(os.getpid()),'--inherited-lock-fd',str(handle.fileno())]
        command+=['--',sys.executable,'-c',f'import time;time.sleep({seconds})']
        p=subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,
                           pass_fds=() if handle is None else (handle.fileno(),))
        self.processes.append(p);return p
    def running(self,stage):return wait_for(lambda:self.state(stage).get('child_pid'))
    def test_inherited_stages_keep_competitors_out_between_stages(self):
        handle=self.reservation()
        for stage in ['first','second']:
            p=self.worker(stage,handle)
            self.assertEqual(p.wait(timeout=10),0,p.stderr.read().decode())
            self.assertEqual(self.state(stage)['lock_mode'],'inherited')
            self.assertFalse(self.free())
            competitor=self.worker('competitor-'+stage)
            self.assertEqual(competitor.wait(timeout=10),1)
            self.assertEqual(self.state('competitor-'+stage)['failure_kind'],'resource_busy')
        handle.close();self.assertTrue(self.free())
    def test_standalone_lock_and_release(self):
        p=self.worker('standalone',seconds=30);self.running('standalone')
        self.assertFalse(self.free());p.terminate();self.assertEqual(p.wait(timeout=10),1)
        self.assertEqual(self.state('standalone')['lock_mode'],'standalone')
        self.assertTrue(self.free())
    def test_unlocked_inherited_descriptor_is_rejected(self):
        p=self.worker('unlocked',self.reservation(locked=False))
        self.assertEqual(p.wait(timeout=10),1)
        self.assertEqual(self.state('unlocked')['failure_kind'],'inherited_lock_not_exclusive')
        self.assertNotIn('child_pid',self.state('unlocked'))
    def test_wrong_file_descriptor_is_rejected(self):
        other=(self.root/'wrong-file').open('a');self.handles.append(other)
        fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=self.worker('wrong',other)
        self.assertEqual(p.wait(timeout=10),1)
        self.assertEqual(self.state('wrong')['failure_kind'],'invalid_inherited_lock_file')
        self.assertTrue(self.free())
    def test_cancellation_cleans_child_but_preserves_supervisor_lock(self):
        handle=self.reservation();p=self.worker('cancel',handle,seconds=30)
        child_pid=self.running('cancel');p.terminate()
        self.assertEqual(p.wait(timeout=10),1)
        self.assertIsNone(process_identity(child_pid))
        self.assertEqual(self.state('cancel')['failure_kind'],'cancelled')
        self.assertFalse(self.free());handle.close();self.assertTrue(self.free())
    def test_worker_keeps_inherited_lock_if_supervisor_closes_its_copy(self):
        handle=self.reservation();p=self.worker('retained',handle,seconds=30)
        self.running('retained');handle.close()
        self.assertFalse(self.free());p.terminate();p.wait(timeout=10)
        self.assertTrue(self.free())
    def test_supervisor_death_cleans_child_then_releases_lock(self):
        code='''import fcntl,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1]).parent))
from asr_worker import process_identity
root=Path(sys.argv[2])
with (root/'.cache/training-launch.lock').open('a') as handle:
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=subprocess.Popen([sys.executable,sys.argv[1],'--root',str(root),'--run-id','test','--stage','orphan','--profile','cpu','--controller-pid',str(os.getpid()),'--controller-start-time',process_identity(os.getpid()),'--inherited-lock-fd',str(handle.fileno()),'--',sys.executable,'-c','import time;time.sleep(30)'],pass_fds=(handle.fileno(),),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    (root/'worker-pid').write_text(str(p.pid))
    time.sleep(30)
'''
        controller=subprocess.Popen([sys.executable,'-c',code,str(WORKER),str(self.root)])
        self.processes.append(controller)
        child_pid=self.running('orphan');self.assertFalse(self.free())
        os.kill(controller.pid,signal.SIGKILL);controller.wait(timeout=5)
        wait_for(lambda:self.state('orphan').get('worker_stopped'))
        self.assertEqual(self.state('orphan')['failure_kind'],'controller_disappeared')
        self.assertIsNone(process_identity(child_pid));wait_for(self.free)
    def test_actual_supervisor_runs_all_stages_with_real_workers(self):
        # Exercise run.py's actual Popen/pass_fds wiring, not a duplicate command.
        scripts=self.root/'scripts';repair=scripts/'corpus_repair'
        repair.mkdir(parents=True)
        shutil.copy2(WORKER,scripts/'asr_worker.py')
        import asr_runtime
        shutil.copy2(asr_runtime.__file__,scripts/'asr_runtime.py')
        python=self.root/'.venv/bin/python';python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        pause=self.root/'csm-lora';pause.mkdir()
        builder='''import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
WORK=ROOT/'.cache/fixture';WORK.mkdir()
OUT=ROOT/'dataset';VERSION='fixture'
def validate_recipe():pass
if __name__=='__main__':
    with (ROOT/'phases').open('a') as stream:stream.write(sys.argv[1]+'\\n')
'''.replace('WORK.mkdir()','WORK.mkdir(exist_ok=True)')
        (repair/'build.py').write_text(builder)
        supervisor=WORKER.parent/'corpus_repair/run.py'
        if not supervisor.exists():supervisor=WORKER.with_name('run.py')
        source=supervisor.read_text().replace('/home/voiceuser/projects/gpt-sovits',str(self.root)).replace('/home/voiceuser/projects/csm-lora',str(pause))
        (repair/'run.py').write_text(source)
        binary=self.root/'bin';binary.mkdir()
        ctl=binary/'systemctl'
        ctl.write_text('#!/usr/bin/env python3\nimport sys\nraise SystemExit(1 if "is-active" in sys.argv else 0)\n')
        ctl.chmod(0o755)
        env=dict(os.environ,PATH=str(binary)+os.pathsep+os.environ['PATH'])
        p=subprocess.Popen([sys.executable,str(repair/'run.py')],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        self.processes.append(p)
        output,error=p.communicate(timeout=15)
        self.assertEqual(p.returncode,0,error.decode())
        self.assertEqual((self.root/'phases').read_text().splitlines(),['scan','asr','align','pilot-report','asr','align','publish'])
        self.assertEqual(json.loads((self.root/'.cache/fixture/supervisor.json').read_text())['status'],'complete')
        self.assertTrue(self.free())

if __name__=='__main__':unittest.main()
