"""Optional live listening session; Ctrl+C restores the prior production state."""
from common import *
import argparse,fcntl,signal,subprocess,sys,urllib.request
sys.path.insert(0,str(REPO/'scripts'))
from asr_worker import process_identity
def cancel(*args):raise KeyboardInterrupt()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    B=ROOT/'stream-benchmark';report=json.loads((B/'report.json').read_text())
    candidates=[r for r in report['configurations'] if r['name']!='baseline']
    chosen=min(candidates,key=lambda r:r.get('median_rtf',r['p95_rtf']))
    precision,mode,codec=chosen['name'].split('-');child=None;stopped=False
    signal.signal(signal.SIGTERM,cancel)
    with (REPO/'.cache/training-launch.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        was_active=subprocess.run(['systemctl','--user','is-active','--quiet','orpheus-tts.service']).returncode==0
        try:
            subprocess.run(['systemctl','--user','stop','orpheus-tts.service'],check=True);stopped=True
            command=[str(B/'.venv/bin/python'),str(ROOT/'scripts/stream_server.py'),'--precision',precision,'--codec',codec]
            if mode=='eager':command.append('--eager')
            wrapped=[str(ROOT/'.venv/bin/python'),str(REPO/'scripts/asr_worker.py'),'--root',str(REPO),
                '--run-id','orpheus-stream-review','--stage','live-review','--profile','asr','--controller-pid',str(os.getpid()),
                '--controller-start-time',process_identity(os.getpid()),'--inherited-lock-fd',str(lease.fileno()),'--',*command]
            with (B/'live-review.log').open('a') as log:
                child=subprocess.Popen(wrapped,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),))
                http=urllib.request.build_opener(urllib.request.ProxyHandler({}));deadline=time.monotonic()+600
                while time.monotonic()<deadline:
                    if child.poll() is not None:raise RuntimeError('Candidate exited; see live-review.log')
                    try:
                        with http.open('http://127.0.0.1:8813/health',timeout=3) as response:health=json.load(response)
                        if health.get('ready'):break
                    except OSError:pass
                    time.sleep(1)
                else:raise TimeoutError('Candidate startup timed out')
                print('Live experimental '+chosen['name']+': http://localhost:8813/ — Ctrl+C to restore production. No deployment change.',flush=True)
                child.wait()
        except KeyboardInterrupt:pass
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            if stopped and was_active:subprocess.run(['systemctl','--user','start','--no-block','orpheus-tts.service'],check=True)
if __name__=='__main__':main()
