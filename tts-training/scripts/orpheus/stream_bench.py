"""Reservation-owning sequential benchmark; restore production in finally."""
from common import *
from stream_core import *
import csv,fcntl,io,signal,subprocess,sys,traceback,urllib.request,urllib.error,wave,threading
sys.path.insert(0,str(REPO/'scripts'))
from asr_worker import process_identity
B=ROOT/'stream-benchmark';PYTHON=B/'.venv/bin/python';http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
PROMPTS= PROMPTS + [('tony',"How was your day today Tony? I've been thinking about you a lot lately. I had the strangest dream of you."),
 ('short','Good morning, Tony. How are you?'),
 ('long',"Good morning, my dear. I hope you have a wonderful week ahead of you. The weather is beautiful today, and I thought we could take a walk through the garden before heading down to the beach. Tell me about your plans for the weekend. I would love to hear what you have been thinking about, and perhaps we can arrange something special together.")]
child=None;lease=None;rows=[];telemetry_stop=threading.Event()
def state(**kw):atomic(B/'status.json',dict(updated=time.time(),pid=os.getpid(),**kw))
def cancel(*args):raise KeyboardInterrupt('Benchmark cancelled')
def get(path,port=8813):
    with http.open(f'http://127.0.0.1:{port}'+path,timeout=5) as response:return json.load(response)
def worker(stage,command):
    return [str(ROOT/'.venv/bin/python'),str(REPO/'scripts/asr_worker.py'),'--root',str(REPO),
        '--run-id','orpheus-stream-benchmark','--stage',stage,'--profile','asr','--controller-pid',str(os.getpid()),
        '--controller-start-time',process_identity(os.getpid()),'--inherited-lock-fd',str(lease.fileno()),'--',*command]
def stop_child():
    global child
    if child is not None:
        child.terminate()
        try:child.wait(timeout=30)
        except subprocess.TimeoutExpired:child.kill();child.wait()
        child=None
def run_job(name,command):
    global child
    with (B/(name+'.log')).open('a') as log:
        child=subprocess.Popen(worker(name,command),stdout=log,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),))
        code=child.wait();child=None
    if code:raise RuntimeError(name+' failed; see '+str(B/(name+'.log')))
def start_server(config):
    global child
    name=config['name'];state(status='starting',configuration=name)
    command=[str(PYTHON),str(ROOT/'scripts/stream_server.py'),'--precision',config['precision'],'--codec',config['codec']]
    if config['eager']:command.append('--eager')
    log=(B/(name+'.log')).open('a')
    child=subprocess.Popen(worker(name,command),stdout=log,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),));log.close()
    started=time.monotonic()
    while time.monotonic()-started<600:
        if child.poll() is not None:raise RuntimeError(name+' exited; see log')
        try:
            health=get('/health');config['startup_seconds']=time.monotonic()-started;return health
        except OSError:time.sleep(2)
    raise TimeoutError('Runtime startup timed out')
def mp3(wav):
    subprocess.run(['/home/voiceuser/.local/bin/ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(wav),'-codec:a','libmp3lame','-b:a','128k',str(wav.with_suffix('.mp3'))],check=True)
def api_checks(name):
    invalid=urllib.request.Request('http://127.0.0.1:8813/stream',data=b'{}',headers={'Content-Type':'application/json'})
    try:http.open(invalid,timeout=5);raise AssertionError('Invalid text accepted')
    except urllib.error.HTTPError as exc:assert exc.code==400
    request=urllib.request.Request('http://127.0.0.1:8813/stream',data=json.dumps(dict(text=PROMPTS[-1][1])).encode(),headers={'Content-Type':'application/json'})
    with http.open(request,timeout=60) as response:
        identifier=response.headers['X-Request-ID'];response.read1(4096)
    for _ in range(100):
        try:
            trace=get('/trace/'+identifier)
            assert trace['status']=='cancelled',trace
            atomic(B/(name+'-api-checks.json'),dict(invalid_text_rejected=True,disconnect_cancelled=True,trace_id=identifier));return
        except urllib.error.HTTPError:time.sleep(.1)
    raise RuntimeError('Cancelled request did not release the generation slot')
def request_sample(config,prompt,repeat,warmup=False):
    name,text=prompt;start=time.monotonic();events=[]
    request=urllib.request.Request(f"http://127.0.0.1:{8812 if config['name']=='baseline' else 8813}/"+('synthesize' if config['name']=='baseline' else 'stream'),data=json.dumps(dict(text=text)).encode(),headers={'Content-Type':'application/json'})
    with http.open(request,timeout=600) as response:
        identifier=response.headers.get('X-Request-ID');payload=[]
        while True:
            chunk=response.read1(65536)
            if not chunk:break
            payload.append(chunk);events.append([time.monotonic()-start,len(chunk)/48000])
    elapsed=time.monotonic()-start;data=b''.join(payload)
    if config['name']=='baseline':
        with wave.open(io.BytesIO(data)) as wav:duration=wav.getnframes()/wav.getframerate()
        events=[[elapsed,duration]];trace={}
    else:
        trace=None
        for _ in range(40):
            try:trace=get('/trace/'+identifier);break
            except OSError:time.sleep(.1)
        if not trace or trace['status']!='complete':raise RuntimeError('Stream did not complete: '+str(trace))
        if len(data)!=trace['sample_count']*2:raise RuntimeError('Client/server sample count differs')
        duration=len(data)/48000
    item=dict(configuration=config['name'],prompt=name,repeat=repeat,text=text,total_seconds=elapsed,audio_seconds=duration,
        rtf=elapsed/duration,first_token=trace.get('first_token'),events=events,trace_id=identifier,
        playback={str(buffer):simulate(events,buffer) for buffer in [.25,.5,1.]},warmup=warmup,
        stream_offline_mae=trace.get('stream_offline_mae'),stream_offline_peak=trace.get('stream_offline_peak'))
    if not warmup:
        out=B/'audio'/f"{config['name']}-{name}-{repeat}.wav";out.parent.mkdir(exist_ok=True)
        if config['name']=='baseline':out.write_bytes(data)
        else:
            with wave.open(str(out),'wb') as wav:wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(24000);wav.writeframes(data)
        mp3(out);item['file']=out.with_suffix('.mp3').name;item['wav']=str(out)
    with (B/'measurements.jsonl').open('a') as file:file.write(json.dumps(item)+'\n')
    return item
def telemetry():
    with (B/'gpu.csv').open('a') as log:
        while not telemetry_stop.is_set():
            result=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=timestamp,memory.used,power.draw,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
            log.write(result.stdout);log.flush();telemetry_stop.wait(1)
def summarize():
    configs=[]
    for name in sorted(set(row['configuration'] for row in rows)):
        items=[r for r in rows if r['configuration']==name]
        configs.append(dict(name=name,requests=len(items),p95_rtf=percentile([r['rtf'] for r in items]),
            p95_first_playback=percentile([r['playback']['0.5']['first_playback'] for r in items]),
            underruns=sum(r['playback']['0.5']['underruns'] for r in items),
            examples=[dict(prompt=r['prompt'],file=r['file']) for r in items if r['repeat']==0]))
    report=dict(status='benchmark_complete_listening_pending',configurations=configs,manual_listening_complete=False,
        acceptance=dict(p95_first_playback_seconds=2,p95_rtf=.8,buffer_seconds=.5,underruns=0),
        configurations_meeting_timing=[c['name'] for c in configs if c['name']!='baseline' and c['p95_rtf']<=.8 and c['p95_first_playback']<=2 and c['underruns']==0])
    atomic(B/'report.json',report)
    with (B/'measurements.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=['configuration','prompt','repeat','total_seconds','audio_seconds','rtf','first_token']);writer.writeheader()
        for row in rows:writer.writerow({key:row[key] for key in writer.fieldnames})
def main():
    global lease
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    if (B/'report.json').exists():raise RuntimeError('Completed benchmark exists; use a new version to rerun')
    if (B/'measurements.jsonl').exists():
        for line in (B/'measurements.jsonl').read_text().splitlines():
            item=json.loads(line)
            if not item['warmup']:rows.append(item)
    def done(config,prompt,repeat):return any(r['configuration']==config and r['prompt']==prompt and r['repeat']==repeat for r in rows)
    with (REPO/'.cache/training-launch.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            state(status='measuring_baseline')
            thread=threading.Thread(target=telemetry,daemon=True);thread.start()
            if sum(r['configuration']=='baseline' for r in rows)<30:
                subprocess.run(['systemctl','--user','stop','orpheus-tts.service'],check=True)
                cold=time.monotonic();subprocess.run(['/home/voiceuser/.local/bin/orpheus','start'],check=True)
                atomic(B/'baseline-startup.json',dict(seconds=time.monotonic()-cold,health=get('/health',8812),disk_cache='already populated'))
                baseline=dict(name='baseline');request_sample(baseline,PROMPTS[0],-1,True);request_sample(baseline,PROMPTS[0],-2,True)
                for repeat in range(5):
                    for prompt in PROMPTS:
                        if done('baseline',prompt[0],repeat):continue
                        state(status='measuring_baseline',prompt=prompt[0],repeat=repeat)
                        rows.append(request_sample(baseline,prompt,repeat))
            subprocess.run(['systemctl','--user','stop','orpheus-tts.service'],check=True)
            while json.loads((B/'setup-status.json').read_text())['status']=='installing':
                state(status='waiting_for_dependencies');time.sleep(10)
            if json.loads((B/'setup-status.json').read_text())['status']!='ready':raise RuntimeError('Isolated dependency installation failed')
            state(status='exporting_merged_model')
            run_job('export',[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/stream_export.py')])
            previous=json.loads((B/'screening.json').read_text()) if (B/'screening.json').exists() else {}
            failures=previous.get('failures',[]);screens=previous.get('successful',[])
            for precision in ['16bit','4bit']:
                successful=[c for c in screens if c['precision']==precision]
                for eager in [True,False]:
                    for codec in ['cpu','cuda']:
                        config=dict(precision=precision,eager=eager,codec=codec,name=f"{precision}-{'eager' if eager else 'graphs'}-{codec}")
                        if any(c['name']==config['name'] for c in successful):continue
                        try:
                            health=start_server(config)
                            first=request_sample(config,PROMPTS[2],-1,True);second=request_sample(config,PROMPTS[3],-2,True)
                            api_checks(config['name'])
                            config.update(screen_score=max(first['rtf'],second['rtf']),health=health)
                            successful.append(config);screens.append(config)
                        except Exception:failures.append(dict(configuration=config['name'],error=traceback.format_exc()))
                        finally:stop_child();atomic(B/'screening.json',dict(successful=screens,failures=failures))
                if not successful:continue
                chosen=min(successful,key=lambda x:x['screen_score']);start_server(chosen)
                try:
                    request_sample(chosen,PROMPTS[0],-1,True);request_sample(chosen,PROMPTS[0],-2,True)
                    for repeat in range(5):
                        for prompt in PROMPTS:
                            if done(chosen['name'],prompt[0],repeat):continue
                            state(status='measuring_stream',configuration=chosen['name'],prompt=prompt[0],repeat=repeat)
                            rows.append(request_sample(chosen,prompt,repeat))
                    # Browser probe uses this ready server without opening an audible window.
                    state(status='browser_probe',configuration=chosen['name'])
                    result_path=Path('/mnt/c/Users/DesktopUser/Projects/echomuse-provisioning/tts-training/orpheus-training')/('browser-'+chosen['name']+'.json')
                    requested=time.time();atomic(B/'browser-request.json',dict(configuration=chosen['name'],requested=requested))
                    deadline=time.monotonic()+180
                    while time.monotonic()<deadline:
                        if result_path.exists() and result_path.stat().st_mtime>=requested:
                            browser=json.loads(result_path.read_text(encoding='utf-8-sig'))
                            if browser.get('status')!='complete':raise RuntimeError('Windows playback failed: '+str(browser))
                            break
                        time.sleep(1)
                    else:raise TimeoutError('Windows playback bridge did not return a result')
                except Exception:failures.append(dict(configuration=chosen['name'],error=traceback.format_exc()))
                finally:stop_child()
            summarize();atomic(B/'failures.json',failures)
            state(status='checking_transcripts')
            run_job('asr',['/bin/bash','-c','source /home/voiceuser/projects/gpt-sovits/scripts/env.sh; exec python "$1"','stream-asr',str(ROOT/'scripts/stream_asr.py')])
            state(status='complete_listening_pending',measured_requests=len(rows))
        except BaseException:
            state(status='failed',error=traceback.format_exc());raise
        finally:
            stop_child();telemetry_stop.set()
            subprocess.run(['systemctl','--user','start','--no-block','orpheus-tts.service'],check=False)
if __name__=='__main__':main()
