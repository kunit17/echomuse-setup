"""Archive changed-policy tests, release the live candidate and verify transcripts."""
from common import *
import fcntl,signal,subprocess,wave,sys,urllib.request
import stream_bench as bench
B=ROOT/'stream-benchmark';workspace=Path('/mnt/c/Users/DesktopUser/Projects/echomuse-provisioning/tts-training/orpheus-training')
samples=[]
for label in ['buffered-tony-v1','buffered-long-v1']:
    browser=json.loads((workspace/('browser-'+label+'.json')).read_text());assert browser['status']=='complete'
    trace=json.loads((B/'traces'/(browser['traceId']+'.json')).read_text());assert trace['status']=='complete'
    assert browser['bufferSeconds']==1 and browser['samples']==trace['sample_count']==trace['offline_sample_count']
    dest=B/'audio'/(label+'.wav');frames=[]
    for i,sentence in enumerate(trace['sentences']):
        assert sentence['sample_count']==sentence['offline_sample_count']
        with wave.open(str(B/'audio'/(browser['traceId']+f'-{i}.wav'))) as wav:
            assert wav.getframerate()==24000 and wav.getnchannels()==1 and wav.getsampwidth()==2
            frames.append(wav.readframes(wav.getnframes()))
    assert sum(len(f) for f in frames)//2==trace['sample_count']
    with wave.open(str(dest),'wb') as wav:wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(24000);wav.writeframes(b''.join(frames))
    bench.mp3(dest)
    samples.append(dict(label=label,text=trace['text'],wav=str(dest),file=dest.with_suffix('.mp3').name,
        first_playback_seconds=browser['firstPlayback'],underruns=browser['underruns'],underrun_seconds=browser['underrunSeconds'],
        sentence_count=len(trace['sentences']),audio_seconds=trace['audio_seconds'],sample_count=trace['sample_count'],complete_audio=True,api_checks=browser['apiChecks']))
atomic(B/'buffer-policy-check.json',dict(status='checking_transcripts',policy='sentences-v1',buffer_seconds=1,precision='4bit',samples=samples,manual_listening_complete=False))
controllers=[]
for path in Path('/proc').iterdir():
    if not path.name.isdigit():continue
    try:args=(path/'cmdline').read_bytes().split(b'\0')
    except OSError:continue
    if len(args)>1 and args[1]==str(ROOT/'scripts/stream_session.py').encode():controllers.append(int(path.name))
assert len(controllers)==1,controllers
for pid in controllers:
    identity=bench.process_identity(pid);os.kill(pid,signal.SIGTERM);deadline=time.monotonic()+60
    while bench.process_identity(pid)==identity and time.monotonic()<deadline:time.sleep(.2)
    assert bench.process_identity(pid)!=identity,'Live review controller did not exit'
signal.signal(signal.SIGTERM,bench.cancel)
with (REPO/'.cache/training-launch.lock').open('a') as lease:
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);bench.lease=lease
    try:
        subprocess.run(['systemctl','--user','stop','orpheus-tts.service'],check=True)
        bench.run_job('buffer-asr',['/bin/bash','-c','source /home/voiceuser/projects/gpt-sovits/scripts/env.sh; exec python "$1"','buffer-asr',str(ROOT/'scripts/stream_buffer_asr.py')])
    finally:
        bench.stop_child();subprocess.run(['systemctl','--user','start','--no-block','orpheus-tts.service'],check=True)
print(json.dumps(json.loads((B/'buffer-policy-check.json').read_text()),indent=2))
