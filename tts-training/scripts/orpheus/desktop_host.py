"""Keep WSL alive while the desktop control is open; unload on stop/crash."""
from common import *
import argparse,signal,subprocess,sys
parser=argparse.ArgumentParser();parser.add_argument('--session-dir',required=True,type=Path);args=parser.parse_args()
session=args.session_dir;session.mkdir(parents=True,exist_ok=True);heartbeat=session/'heartbeat';stop=session/'stop'
def state(**kw):atomic(session/'state.json',dict(updated=time.time(),**kw))
def cancel(*args):raise KeyboardInterrupt()
signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
child=None
try:
    assert heartbeat.exists() and time.time()-heartbeat.stat().st_mtime<20,'Desktop heartbeat missing'
    atomic(ROOT/'activation-request.json',dict(desired_on=True,mode='desktop',heartbeat=str(heartbeat),session=str(session)))
    state(status='starting')
    with (session/'start.log').open('w') as log:
        child=subprocess.Popen(['/home/voiceuser/.local/bin/orpheus','start'],stdout=log,stderr=subprocess.STDOUT)
        while child.poll() is None:
            if stop.exists() or time.time()-heartbeat.stat().st_mtime>20:raise KeyboardInterrupt()
            time.sleep(.5)
        if child.returncode:raise RuntimeError('Model startup failed; see start.log')
    state(status='ready',url='http://localhost:8812/')
    while not stop.exists() and time.time()-heartbeat.stat().st_mtime<20:time.sleep(.5)
except KeyboardInterrupt:pass
except Exception as exc:state(status='error',error=str(exc))
finally:
    current=json.loads((ROOT/'activation-request.json').read_text())
    if current.get('session')==str(session) or not current.get('desired_on'):
        atomic(ROOT/'activation-request.json',dict(desired_on=False,mode='desktop',session=str(session)))
        subprocess.run(['/home/voiceuser/.local/bin/orpheus','stop'],check=False)
    if child is not None and child.poll() is None:child.terminate();child.wait(timeout=10)
    previous=json.loads((session/'state.json').read_text()) if (session/'state.json').exists() else {}
    if previous.get('status')!='error':state(status='off')
