import os
from pathlib import Path
ROOT=Path('/home/voiceuser/projects/orpheus-lora')
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import hashlib,json,time
REPO=Path('/home/voiceuser/projects/gpt-sovits')
CORPUS=REPO/'data/shecorpus/repaired/boundary-v1-20261115'
RUN=ROOT/'runs/orpheus-v1-example01'
DATA=ROOT/'data/v1-snac'
MODEL='unsloth/orpheus-3b-0.1-ft'
QUANT='unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit'
CODEC='hubertsiuzdak/snac_24khz'
TOKENS=dict(start_human=128259,end_human=128260,start_ai=128261,end_ai=128262,
            start_speech=128257,end_speech=128258,end_text=128009,pad=128263,audio=128266)
PROMPTS=[('greeting',"Hi Alex, it's great to finally meet you after all this time! We are going to get along swimmingly."),
 ('alternative',"I don't know what you think you're doing, but I am not into that. Let's try something else, shall we?"),
 ('endings','The flowers are blooming in the garden. We can see them from the house.')]
def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def setup():return json.loads((ROOT/'setup.json').read_text())
def status(**kw):
    path=RUN/'status.json'
    previous=json.loads(path.read_text()) if path.exists() else {}
    previous.update(kw,updated=time.time(),pid=os.getpid());atomic(path,previous)
def pack_codes(codes):
    n=len(codes[0]);assert len(codes)==3 and len(codes[1])==2*n and len(codes[2])==4*n
    result=[]
    for i in range(n):
        values=[codes[0][i],codes[1][2*i],codes[2][4*i],codes[2][4*i+1],codes[1][2*i+1],codes[2][4*i+2],codes[2][4*i+3]]
        assert all(0<=int(v)<4096 for v in values)
        result.extend(int(v)+TOKENS['audio']+j*4096 for j,v in enumerate(values))
    return result
def unpack_codes(tokens):
    if not tokens or len(tokens)%7:raise ValueError('Incomplete SNAC frame group')
    layers=[[],[],[]]
    for i in range(0,len(tokens),7):
        v=[int(tokens[i+j])-TOKENS['audio']-j*4096 for j in range(7)]
        if not all(0<=x<4096 for x in v):raise ValueError('Invalid audio token position/range')
        layers[0].append(v[0]);layers[1].extend([v[1],v[4]]);layers[2].extend([v[2],v[3],v[5],v[6]])
    return layers
def format_example(text_ids,codes):
    return [TOKENS['start_human']]+list(text_ids)+[TOKENS['end_text'],TOKENS['end_human'],TOKENS['start_ai'],TOKENS['start_speech']]+list(codes)+[TOKENS['end_speech'],TOKENS['end_ai']]
def collate(examples):
    import torch
    n=((max(len(e['input_ids']) for e in examples)+7)//8)*8
    result={}
    for key,pad in [('input_ids',TOKENS['pad']),('attention_mask',0),('labels',-100)]:
        result[key]=torch.tensor([e[key]+[pad]*(n-len(e[key])) for e in examples],dtype=torch.long)
    return result
def refresh_model_placement(model):
    """Refresh Unsloth's layer routing after staging frozen weights on CPU."""
    from unsloth_zoo.patching_utils import verify_and_set_device
    if any(p.device.type=='cpu' for p in model.parameters()):model.to('cuda')
    for module in model.modules():
        if '_per_layer_device' in module.__dict__ or '_per_layer_device_index' in module.__dict__:
            verify_and_set_device(module)
def read_status():
    path=RUN/'status.json';value=json.loads(path.read_text()) if path.exists() else {}
    if value.get('mode')=='train':
        metrics=RUN/'metrics.jsonl';latest=None
        if metrics.exists():
            for line in metrics.read_text().splitlines():
                try:row=json.loads(line)
                except ValueError:continue
                if row.get('mode')=='train' and 'eval_loss' in row:latest=row
        value['last_validation']=latest
    return value
