"""Local Orpheus preparation, training, sample and status entry point."""
from common import *
import argparse,subprocess,sys,urllib.request
parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
for name in ['prepare','train','status','start','stop']:sub.add_parser(name)
p=sub.add_parser('sample');p.add_argument('--text-file',required=True);p.add_argument('--out',required=True)
args=parser.parse_args()
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def health():
    with http.open('http://127.0.0.1:8812/health',timeout=3) as response:value=json.load(response)
    if value.get('model')!='Orpheus' or value.get('epoch')!=3:raise RuntimeError('Port 8812 is not serving selected Orpheus epoch 3')
    selection=json.loads((ROOT/'selected-model.json').read_text())
    if value.get('adapter_sha256')!=selection['adapter_sha256']:raise RuntimeError('Loaded adapter does not match selection')
    if selection.get('runtime')=='vllm-streaming' and (value.get('runtime')!='vllm-streaming' or value.get('precision')!='4bit'):raise RuntimeError('Selected streaming runtime is not loaded')
    return value
def start():
    request=ROOT/'activation-request.json'
    current=json.loads(request.read_text()) if request.exists() else {}
    if not (current.get('desired_on') and current.get('mode')=='desktop' and Path(current.get('heartbeat','/nonexistent')).exists() and time.time()-Path(current['heartbeat']).stat().st_mtime<20):
        atomic(request,dict(desired_on=True,mode='manual',updated=time.time()))
    subprocess.run(['systemctl','--user','start','orpheus-tts.service'],check=True)
    deadline=time.monotonic()+300
    while time.monotonic()<deadline:
        try:return health()
        except (OSError,RuntimeError):time.sleep(2)
    raise RuntimeError('Orpheus readiness timed out; journalctl --user -u orpheus-tts.service')
if args.command=='start':print(json.dumps(start(),indent=2));raise SystemExit(0)
if args.command=='stop':
    atomic(ROOT/'activation-request.json',dict(desired_on=False,mode='manual',updated=time.time()))
    raise SystemExit(subprocess.call(['systemctl','--user','stop','orpheus-tts.service','orpheus-tts-16bit.service']))
if args.command=='status':
    value=read_status()
    try:value['service']=health()
    except (OSError,RuntimeError):value['service']={'ready':False}
    print(json.dumps(value,indent=2));raise SystemExit(0)
if args.command=='sample':
    text=Path(args.text_file).read_text().strip();out=Path(args.out).resolve();wav=out.with_suffix('.wav');mp3=out.with_suffix('.mp3');metadata=out.with_suffix('.json')
    if any(p.exists() for p in [wav,mp3,metadata]):raise FileExistsError('Choose a new output filename')
    try:selected=health()
    except (OSError,RuntimeError):raise SystemExit('Voice is off. Open Orpheus Voice on the desktop or run orpheus start first.')
    started=time.monotonic()
    request=urllib.request.Request('http://127.0.0.1:8812/synthesize',data=json.dumps(dict(text=text)).encode(),headers={'Content-Type':'application/json'})
    with http.open(request,timeout=1800) as response:audio=response.read()
    out.parent.mkdir(parents=True,exist_ok=True);wav.write_bytes(audio)
    subprocess.run(['/home/voiceuser/.local/bin/ffmpeg','-hide_banner','-loglevel','error','-n','-i',str(wav),'-codec:a','libmp3lame','-b:a','128k',str(mp3)],check=True)
    atomic(metadata,dict(text=text,wall_seconds=time.monotonic()-started,service=selected,postprocessing_policy='isolated-end-burst-v1'))
    print(mp3);raise SystemExit(0)
if args.command=='train':raise SystemExit(subprocess.call([sys.executable,str(ROOT/'scripts/pipeline.py')]))
command=[sys.executable,str(ROOT/'scripts/prepare.py')] if args.command=='prepare' else [sys.executable,str(ROOT/'scripts/sample.py'),'--text-file',args.text_file,'--out',args.out]
raise SystemExit(subprocess.call([sys.executable,str(REPO/'scripts/asr_worker.py'),'--root',str(REPO),
    '--run-id','orpheus-v1-example01','--stage',args.command+'-'+str(int(time.time())),
    '--profile','asr','--',*command]))
