"""Regression tests use fake ASR objects and CPU-only subprocesses; no CUDA imports."""
import csv
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import asr_runtime as runtime
import asr_controller as controller
from asr_worker import process_identity


class CacheTests(unittest.TestCase):
    def test_changed_input_and_bounds_invalidate_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.write_bytes(b'original')
            context=runtime.make_context({'source':source},{'model':'fixture','precision':'float16'})
            cache=runtime.Cache(root/'cache',context)
            key={'frames':[1,20],'filter_version':1}
            cache.put(key,{'accepted':True})
            self.assertEqual(cache.get(key),{'accepted':True})
            self.assertIsNone(cache.get({'frames':[1,21],'filter_version':1}))
            self.assertIsNone(cache.get({'frames':[1,20],'filter_version':2}))
            source.write_bytes(b'modified')
            changed=runtime.make_context({'source':source},{'model':'fixture','precision':'float16'})
            self.assertIsNone(runtime.Cache(root/'cache',changed).get(key))
            with self.assertRaises(RuntimeError):runtime.verify_inputs(context)

    def test_legacy_and_corrupt_cache_are_not_trusted(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);legacy=root/'clip-00000.json';legacy.write_text('{"accepted":true}')
            cache=runtime.Cache(root,{'fingerprint':'fixture'})
            key={'frames':[1,2]}
            self.assertIsNone(cache.get(key));self.assertTrue(legacy.exists())
            cache.put(key,{'ok':1});path=cache.path(key)
            damaged=json.loads(path.read_text());damaged['value']={'ok':2}
            path.write_text(json.dumps(damaged))
            self.assertIsNone(cache.get(key));cache.put(key,{'ok':3})
            self.assertTrue(list(path.parent.glob('*.invalid-*')))

    def test_missing_unit_cannot_be_a_success(self):
        with self.assertRaises(RuntimeError):
            controller.verify_completion({'LoadState':'not-found','Result':'success'},'x',
                                         {'run_id':'r','complete':True,'status':'success'},'r')
        with self.assertRaises(RuntimeError):
            controller.verify_completion({'LoadState':'loaded','Result':'success','ExecMainStatus':'0','InvocationID':'x'},
                                         'x',{'run_id':'old','complete':True,'status':'success'},'r')

    def test_agreement_is_not_wer_and_legacy_filters_are_honest(self):
        report=runtime.agreement_report([{'index':1}])
        self.assertNotIn('wer',json.dumps(report));self.assertFalse(report['independent_accuracy_measurement'])
        filters=runtime.summarize_filters([{'reason':'empty_or_identifying_reference'}],10,100)
        self.assertEqual(filters['rejected_clips_by_reason'],{'legacy_empty_or_identifying_reference_unclassified':1})

    def test_model_loader_requires_controller_before_import(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(RuntimeError):runtime.load_model('/no-model')
        self.assertNotIn('faster_whisper',sys.modules)
        self.assertNotIn('ctranslate2',sys.modules)

    def test_metadata_migration_preserves_history_and_is_idempotent(self):
        from migrate_asr_metadata import migrate
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);reports=root/'reports';reports.mkdir()
            runtime.atomic_json(reports/'corpus-asr-preflight.json',{'status':'failed','compute_type':'float32'})
            runtime.atomic_json(reports/'corpus-asr-cuda-float16-3200M.json',
                                {'status':'passed','device':'cuda','compute_type':'float16','audio_seconds':600})
            runtime.atomic_json(reports/'corpus-verification.json',dict(
                status='two_pass_corpus_built_pending_review',seconds=10,excluded_source_seconds=90,
                splits={'validation':{'clips':2}},validation_asr={'micro_wer':0},
                exclusions=[{'reason':'empty_or_identifying_reference'}]))
            migrate(root)
            current=json.loads((reports/'corpus-asr-preflight.json').read_text())
            self.assertEqual(current['status'],'not_run_since_hardening')
            self.assertEqual(current['historical_evidence'][0]['status'],'historical_passed')
            corpus=json.loads((reports/'corpus-verification.json').read_text())
            self.assertNotIn('validation_asr',corpus)
            self.assertEqual(corpus['asr_pass_agreement']['accepted_clips'],2)
            self.assertEqual(len(list((reports/'asr-history').glob('*.json'))),2)
            before=(reports/'corpus-verification.json').read_bytes();migrate(root)
            self.assertEqual(before,(reports/'corpus-verification.json').read_bytes())


class ScoringTests(unittest.TestCase):
    def test_interrupted_scoring_resumes_and_completed_run_loads_no_model(self):
        import numpy as np
        import corpus_ground_truth as ground
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old=root/'old';old.mkdir();(root/'reports').mkdir()
            source=root/'source';source.write_bytes(b'fixture-audio')
            model=root/'.cache/whisper-large-v3';model.mkdir(parents=True)
            (model/'model.bin').write_bytes(b'fake');(model/'config.json').write_text('{}')
            with (old/'manifest.csv').open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=['index','text']);writer.writeheader()
                writer.writerows([{'index':0,'text':'one two'},{'index':1,'text':'three four'}])
            mapping=runtime.make_context({'source':source,'reference':old/'manifest.csv'},{'fixture':1})
            rows=[dict(index=0,status='matched',start_seconds=0,end_seconds=2,duration_seconds=2),
                  dict(index=1,status='matched',start_seconds=3,end_seconds=5,duration_seconds=2)]
            report=root/'reports/corpus-ground-truth.json'
            runtime.atomic_json(report,dict(status='mapped',rows=rows,mapped_clips=2,covered_seconds=4,
                                mapping_fingerprint=mapping['fingerprint'],mapping_rows_hash=runtime.fingerprint(rows)))
            calls=[]
            class FakeModel:
                def __init__(self,fail=False):self.fail=fail;self.count=0
                def transcribe(self,audio,**kwargs):
                    self.count+=1;calls.append(float(audio[0]))
                    if self.fail and self.count==2:raise RuntimeError('planted interruption')
                    return iter([types.SimpleNamespace(text='one two' if audio[0]==0 else 'three four')]),None
            replacements=dict(ROOT=root,SOURCE=source,OLD=old,WORK=root/'.cache/truth',REPORT=report)
            with patch.multiple(ground,**replacements),patch.object(ground,'mapping_context',return_value=mapping), \
                 patch.object(ground,'audio',side_effect=lambda path,start,duration:np.array([start])), \
                 patch.object(ground,'load_model',return_value=FakeModel(True)):
                with self.assertRaisesRegex(RuntimeError,'planted'):ground.transcribe()
            partial=json.loads(report.read_text())
            self.assertEqual(partial['status'],'scoring');self.assertEqual(partial['scored_clips'],1)
            calls.clear()
            with patch.multiple(ground,**replacements),patch.object(ground,'mapping_context',return_value=mapping), \
                 patch.object(ground,'audio',side_effect=lambda path,start,duration:np.array([start])), \
                 patch.object(ground,'load_model',return_value=FakeModel()) as factory:
                ground.transcribe()
                self.assertEqual(calls,[3.0]);factory.assert_called_once()
            self.assertEqual(json.loads(report.read_text())['micro_wer'],0)
            with patch.multiple(ground,**replacements),patch.object(ground,'mapping_context',return_value=mapping), \
                 patch.object(ground,'load_model') as factory:
                ground.transcribe();factory.assert_not_called()


class CorpusTests(unittest.TestCase):
    def test_window_resume_filter_invalidation_and_no_false_wer(self):
        import numpy as np
        import wave
        import build_continuous_corpus as build
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);reports=root/'reports';reports.mkdir()
            source=root/'source';source.write_bytes(b'fixture')
            model=root/'.cache/whisper-large-v3';model.mkdir(parents=True)
            (model/'model.bin').write_bytes(b'fake');(model/'config.json').write_text('{}')
            runtime.atomic_json(reports/'corpus-split.json',{'train':[[0,8]],'validation':[[10,18]]})
            runtime.atomic_json(reports/'corpus-pause-inventory.json',dict(
                runs={'relative_2pct':[[0,.8],[6,6.8],[10,10.8],[16,16.8]]},thresholds={'relative_2pct':.01}))
            class TaggedAudio(np.ndarray):
                def __new__(cls,start,duration,rate=16000):
                    value=np.zeros(round(duration*rate),dtype=np.float32).view(cls)
                    value[rate:-rate]=.1;value.start=start
                    return value
            calls=[]
            class FakeModel:
                def __init__(self,fail=False):self.count=0;self.fail=fail
                def transcribe(self,audio,**kwargs):
                    self.count+=1;calls.append(audio.start)
                    if self.fail and self.count==2:raise RuntimeError('planted window interruption')
                    words=[' Jordan',' five',' six.'] if audio.start>=10 else [' Alex',' two',' three.']
                    pieces=[types.SimpleNamespace(word=w,start=1+i,end=1.2+i,probability=.9) for i,w in enumerate(words)]
                    return iter([types.SimpleNamespace(words=pieces)]),None
            def fake_run(command,**kwargs):
                if command[0]=='systemctl':return types.SimpleNamespace(returncode=1)
                destination=Path(command[-1]);duration=float(command[command.index('-t')+1])
                with wave.open(str(destination),'w') as stream:
                    stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(32000)
                    stream.writeframes(bytes(round(duration*32000)*2))
                return types.SimpleNamespace(returncode=0)
            def fake_probe(command,**kwargs):
                with wave.open(command[-1]) as stream:duration=stream.getnframes()/stream.getframerate()
                return json.dumps({'streams':[dict(sample_rate='32000',channels=1,codec_name='pcm_s16le',duration=str(duration))]}).encode()
            paths=dict(ROOT=root,SOURCE=source,WORK=root/'.cache/work',OUT=root/'data/shecorpus')
            (root/'data/shecorpus').mkdir(parents=True)
            def patches(factory,filter_version='v1'):
                from contextlib import ExitStack
                stack=ExitStack()
                stack.enter_context(patch.multiple(build,**paths))
                stack.enter_context(patch.object(build,'verify_frozen_inputs'))
                stack.enter_context(patch.object(build,'filter_fingerprint',return_value=filter_version))
                stack.enter_context(patch.object(build,'decode',side_effect=TaggedAudio))
                stack.enter_context(patch.object(build,'load_model',factory))
                stack.enter_context(patch.object(build.subprocess,'run',side_effect=fake_run))
                stack.enter_context(patch.object(build.subprocess,'check_output',side_effect=fake_probe))
                return stack
            with patches(lambda root:FakeModel(True)):
                with self.assertRaisesRegex(RuntimeError,'planted window'):build.main()
            calls.clear()
            with patches(lambda root:FakeModel()):build.main()
            self.assertEqual(len(calls),3)  # Cached first window is not transcribed again.
            result=json.loads((reports/'corpus-verification.json').read_text())
            self.assertEqual(result['clips'],2)
            self.assertFalse(result['proper_name_filter_enabled'])
            self.assertTrue(result['filtering']['privacy_policy_changed'])
            with (root/'data/shecorpus/train/manifest.csv').open() as stream:
                self.assertIn('Alex',next(csv.DictReader(stream))['text'])
            with (root/'data/shecorpus/validation/manifest.csv').open() as stream:
                self.assertIn('Jordan',next(csv.DictReader(stream))['text'])
            self.assertNotIn('validation_asr',result)
            self.assertFalse(result['asr_pass_agreement']['independent_accuracy_measurement'])
            with patches(lambda root:self.fail('completed caches should not load a model')):build.main()
            calls.clear()
            with patches(lambda root:FakeModel(),'v2'):build.main()
            self.assertEqual(len(calls),2)  # Filter change invalidates clips, not ASR windows.


class WorkerTests(unittest.TestCase):
    def run_worker(self,root,child,extra=()):
        script=root/'child.py';script.write_text(child)
        command=[sys.executable,str(SCRIPTS/'asr_worker.py'),'--root',str(root),'--run-id','fixture',
                 '--stage','cpu','--profile','cpu','--min-ram-mib','0','--runtime','20',
                 '--idle-timeout','10',*extra,'--',sys.executable,str(script)]
        process=subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
        self.addCleanup(process.stderr.close)
        return process

    def wait_for(self,path):
        deadline=time.monotonic()+8
        while not path.exists() and time.monotonic()<deadline:time.sleep(.05)
        self.assertTrue(path.exists(),str(path))

    def state(self,root):return json.loads((root/'reports/asr-runs/fixture/cpu.json').read_text())

    def test_default_worker_has_no_time_or_resource_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);script=root/'child.py'
            script.write_text('import time; time.sleep(.8)')
            result=subprocess.run([sys.executable,str(SCRIPTS/'asr_worker.py'),
                '--root',str(root),'--run-id','fixture','--stage','cpu','--profile','cpu',
                '--',sys.executable,str(script)],capture_output=True,text=True,timeout=8)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(self.state(root)['status'],'success')

    def test_successful_cpu_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);process=self.run_worker(root,'print("fixture complete")')
            self.assertEqual(process.wait(timeout=10),0,process.stderr.read())
            self.assertEqual(self.state(root)['status'],'success')

    def test_successful_child_cannot_leave_background_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pidfile=root/'grandchild'
            child=f'import subprocess,sys\nfrom pathlib import Path\np=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"])\nPath({str(pidfile)!r}).write_text(str(p.pid))\n'
            process=self.run_worker(root,child);self.assertEqual(process.wait(timeout=10),0)
            self.assertIsNone(process_identity(int(pidfile.read_text())))

    def test_cancel_releases_reservation_and_kills_child(self):
        import fcntl
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pidfile=root/'pid'
            child=f'import os,time,subprocess,sys,json\nfrom pathlib import Path\ngrandchild=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"])\nPath({str(pidfile)!r}).write_text(json.dumps([os.getpid(),grandchild.pid]))\ntime.sleep(30)\n'
            process=self.run_worker(root,child);self.wait_for(pidfile)
            pids=json.loads(pidfile.read_text());process.terminate();process.wait(timeout=10)
            self.assertEqual(self.state(root)['failure_kind'],'cancelled')
            for pid in pids:self.assertIsNone(process_identity(pid))
            with (root/'.cache/training-launch.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_second_worker_cannot_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pidfile=root/'pid'
            first=self.run_worker(root,f'import time\nfrom pathlib import Path\nPath({str(pidfile)!r}).write_text("ready")\ntime.sleep(30)')
            try:
                self.wait_for(pidfile)
                # Different stage preserves the first worker's status record.
                script=root/'second.py';script.write_text('raise SystemExit("must not run")')
                second=subprocess.run([sys.executable,str(SCRIPTS/'asr_worker.py'),'--root',str(root),
                    '--run-id','second','--stage','cpu','--profile','cpu','--min-ram-mib','0','--',sys.executable,str(script)],
                    capture_output=True,text=True,timeout=8)
                self.assertEqual(second.returncode,1)
                status=json.loads((root/'reports/asr-runs/second/cpu.json').read_text())
                self.assertEqual(status['failure_kind'],'resource_busy')
            finally:first.terminate();first.wait(timeout=10)

    def test_controller_death_stops_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pidfile=root/'pid'
            parent=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
            process=self.run_worker(root,f'import os,time\nfrom pathlib import Path\nPath({str(pidfile)!r}).write_text(str(os.getpid()))\ntime.sleep(30)',
                ['--controller-pid',str(parent.pid),'--controller-start-time',process_identity(parent.pid)])
            try:
                self.wait_for(pidfile);pid=int(pidfile.read_text());parent.kill();parent.wait(timeout=5)
                process.wait(timeout=10)
                self.assertEqual(self.state(root)['failure_kind'],'controller_disappeared')
                self.assertIsNone(process_identity(pid))
            finally:
                if parent.poll() is None:parent.kill();parent.wait()
                if process.poll() is None:process.terminate();process.wait(timeout=10)

    def test_no_progress_and_loader_deadlines(self):
        for loading in [False,True]:
            with self.subTest(loading=loading),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                child='import time\ntime.sleep(30)'
                options=['--idle-timeout','.6']
                if loading:
                    child='import os,json,time\nfrom pathlib import Path\nPath(os.environ["ASR_HEARTBEAT_PATH"]).write_text(json.dumps(dict(run_id=os.environ["ASR_RUN_ID"],phase="model_loading",monotonic=time.monotonic())))\ntime.sleep(30)'
                    options=['--load-timeout','.5','--idle-timeout','5']
                process=self.run_worker(root,child,options);process.wait(timeout=8)
                self.assertEqual(self.state(root)['failure_kind'],'loader_timeout' if loading else 'no_progress_timeout')


class SystemdFixtureTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('ASR_SYSTEMD_FIXTURE')=='1','explicit CPU-only systemd fixture')
    def test_real_systemd_lifecycle_without_any_model(self):
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ['scripts','.venv/bin','logs','reports']:(root/name).mkdir(parents=True)
            for name in ['asr_worker.py','asr_runtime.py']:shutil.copy2(SCRIPTS/name,root/'scripts'/name)
            (root/'scripts/env.sh').write_text('export PYTHONUNBUFFERED=1\n')
            (root/'.venv/bin/python').symlink_to(sys.executable)
            (root/'scripts/fixture.py').write_text('print("CPU fixture only")\n')
            job=controller.Controller(root,'preflight')
            job.run_stage('fixture-'+str(os.getpid()),'fixture.py',profile='cpu')
            record=json.loads((job.directory/('fixture-'+str(os.getpid())+'.json')).read_text())
            self.assertTrue(record['complete'])
            resources=json.loads((job.directory/('fixture-'+str(os.getpid())+'-resources.json')).read_text())
            for key in ['MemoryHigh','MemoryMax','MemorySwapMax']:
                self.assertEqual(resources['service'][key],'infinity')
            self.assertIsNone(job.state['profile']['memory_max_mib'])


if __name__=='__main__':unittest.main()
