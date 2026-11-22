"""Epoch-3 PCM/WAV service; port8812 is approved production,8813 is review."""
from common import *
from stream_core import *
import argparse,asyncio,sys,uuid,logging,io,wave
import numpy as np
import torch,soundfile as sf
from fastapi import FastAPI,Request,HTTPException
from fastapi.responses import StreamingResponse,FileResponse,Response
from starlette.middleware.trustedhost import TrustedHostMiddleware
from vllm import AsyncLLMEngine,AsyncEngineArgs,SamplingParams
from transformers import AutoTokenizer
from snac import SNAC
from huggingface_hub import snapshot_download
sys.path.insert(0,str(REPO/'scripts'))
from audio_postprocess import clean_generated_tail
parser=argparse.ArgumentParser();parser.add_argument('--precision',choices=['16bit','4bit'],required=True)
parser.add_argument('--eager',action='store_true');parser.add_argument('--codec',choices=['cpu','cuda'],default='cpu')
parser.add_argument('--port',type=int,choices=[8812,8813],default=8813)
parser.add_argument('--kv-cache-mib',type=int,default=1024,help='GPU generation cache; default 1 GiB for one voice request')
args=parser.parse_args();B=ROOT/'stream-benchmark';started=time.monotonic()
if args.kv_cache_mib<1024:parser.error('--kv-cache-mib must be at least 1024 for the 4096-token context')
torch.set_num_threads(4)
codec=SNAC.from_pretrained(snapshot_download(CODEC,revision=setup()['model_revisions'][CODEC],local_files_only=True)).to(args.codec).eval()
tokenizer=AutoTokenizer.from_pretrained(B/'merged-epoch3',local_files_only=True)
selection=json.loads((ROOT/'selected-model.json').read_text())
assert sha(Path(selection['adapter_path'])/'adapter_model.safetensors')==selection['adapter_sha256']
assert json.loads((B/'merged-epoch3/provenance.json').read_text())['selection']['adapter_sha256']==selection['adapter_sha256']
options=dict(model=str(B/'merged-epoch3'),dtype='bfloat16',max_model_len=4096,max_num_seqs=1,
    enforce_eager=args.eager,disable_log_stats=True,gpu_memory_utilization=0.5,kv_cache_memory_bytes=args.kv_cache_mib*1024**2)
if args.precision=='4bit':options['quantization']='bitsandbytes'
engine=None
health=dict(ready=False,precision=args.precision,eager=args.eager,codec=args.codec,
    model='Orpheus',model_loaded=True,runtime='vllm-streaming',port=args.port,
    kv_cache_mib=args.kv_cache_mib,max_model_len=4096,max_num_seqs=1,
    playback_buffer_seconds=1.0,text_policy='sentences-v1',
    load_seconds=time.monotonic()-started,pid=os.getpid(),epoch=3,
    adapter_sha256=json.loads((ROOT/'selected-model.json').read_text())['adapter_sha256'])
app=FastAPI();gate=asyncio.Lock()
app.add_middleware(TrustedHostMiddleware,allowed_hosts=['localhost','127.0.0.1'])
@app.on_event('startup')
async def load_engine():
    global engine
    engine=AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**options))
    health.update(ready=True,load_seconds=time.monotonic()-started)
    print(json.dumps(health),flush=True)
def decode(tokens):
    validate_codes(tokens);layers=unpack_codes(tokens)
    with torch.inference_mode():
        return codec.decode([torch.tensor([v],device=args.codec) for v in layers])[0,0].float().cpu().numpy()
def pcm(audio):return (np.clip(audio,-1,1)*32767).astype('<i2').tobytes()
@app.get('/health')
def get_health():return health
@app.get('/')
def page():return FileResponse(B/'review.html')
@app.get('/worklet.js')
def worklet():return FileResponse(B/'worklet.js',media_type='application/javascript')
@app.get('/report')
def report():return json.loads((B/'report.json').read_text()) if (B/'report.json').exists() else {'status':'benchmarking'}
@app.get('/trace/{request_id}')
def trace(request_id:str):
    if not request_id.replace('-','').isalnum():raise HTTPException(400)
    path=B/'traces'/(request_id+'.json')
    if not path.exists():raise HTTPException(404)
    return json.loads(path.read_text())
@app.get('/audio/{name}')
def audio_file(name:str):
    if Path(name).name!=name or not name.endswith(('.mp3','.wav')):raise HTTPException(400)
    path=B/'audio'/name
    if not path.is_file():raise HTTPException(404)
    return FileResponse(path)
@app.post('/stream')
async def stream(request:Request):
    if request.headers.get('host') not in (f'localhost:{args.port}',f'127.0.0.1:{args.port}'):raise HTTPException(400,'Local requests only')
    if request.headers.get('origin') not in (None,f'http://localhost:{args.port}',f'http://127.0.0.1:{args.port}'):raise HTTPException(403)
    try:data=await request.json()
    except ValueError:raise HTTPException(400,'Expected JSON')
    text=data.get('text') if isinstance(data,dict) else None
    if not isinstance(text,str) or not text.strip() or len(text)>5000:raise HTTPException(400,'Use 1–5000 characters')
    if isinstance(data.get('speed',1),bool) or data.get('speed',1)!=1:raise HTTPException(400,'Only speed=1 supported')
    identifier=str(uuid.uuid4());clock=time.monotonic();events=[];utterances=split_sentences(text)
    stats=dict(id=identifier,text=text,health=health,status='running',events=events,sentences=[])
    async def sentence_body(sentence,index):
        tokens=[];pieces=[];emitted=0;first_token=None;generation_id=f'{identifier}-{index}'
        ids=[TOKENS['start_human']]+tokenizer.encode(sentence,add_special_tokens=True)+[TOKENS['end_text'],TOKENS['end_human'],TOKENS['start_ai'],TOKENS['start_speech']]
        sampling=SamplingParams(seed=1700,temperature=.6,top_p=.95,repetition_penalty=1.1,
            max_tokens=min(3000,4096-len(ids)),stop_token_ids=[TOKENS['end_speech']],ignore_eos=True,detokenize=False)
        previous=0;last_decoded=0;eos=False
        try:
            async for result in engine.generate({'prompt_token_ids':ids},sampling,generation_id):
                if await request.is_disconnected():raise asyncio.CancelledError()
                output=result.outputs[0];new=list(output.token_ids[previous:]);previous=len(output.token_ids)
                if new and first_token is None:first_token=time.monotonic()-clock
                for token in new:
                    if token==TOKENS['end_speech']:eos=True;break
                    if not 0<=token-TOKENS['audio']-(len(tokens)%7)*4096<4096:raise ValueError('Invalid audio token')
                    tokens.append(token)
                if output.finish_reason=='stop' and output.stop_reason==TOKENS['end_speech']:eos=True
                groups=len(tokens)//7
                if groups>=12 and groups-last_decoded>=4:
                    full=await asyncio.to_thread(decode,tokens[:groups*7]);last_decoded=groups
                    end=safe_end(len(full),groups)
                    if end>emitted:
                        part=full[emitted:end].copy();emitted=end;pieces.append(part)
                        events.append([time.monotonic()-clock,len(part)/RATE]);yield pcm(part)
            if not eos:raise RuntimeError('Missing EOS; output is incomplete')
            validate_codes(tokens);full=await asyncio.to_thread(decode,tokens)
            if not np.isfinite(full).all():raise ValueError('Nonfinite audio')
            cleaned,decision=clean_generated_tail(full,RATE)
            tail=cleaned[emitted:].copy();pieces.append(tail)
            if len(tail):events.append([time.monotonic()-clock,len(tail)/RATE]);yield pcm(tail)
            streamed=np.concatenate(pieces)
            if len(streamed)!=len(full) or len(full)!=len(tokens)//7*GROUP_SAMPLES:raise RuntimeError('Frame/sample coverage mismatch')
            audio_dir=B/'audio';audio_dir.mkdir(exist_ok=True)
            sf.write(audio_dir/(generation_id+'.wav'),streamed,RATE,subtype='PCM_16')
            sf.write(audio_dir/(generation_id+'-offline.wav'),cleaned,RATE,subtype='PCM_16')
            stats['sentences'].append(dict(text=sentence,first_token=first_token,audio_tokens=len(tokens),tokens=tokens,
                tail_cleanup=decision,sample_count=len(streamed),offline_sample_count=len(full),
                stream_offline_mae=float(np.mean(np.abs(streamed-cleaned))),stream_offline_peak=float(np.max(np.abs(streamed-cleaned)))))
        finally:await engine.abort(generation_id)
    async def body():
        async with gate:
            try:
                for index,sentence in enumerate(utterances):
                    async for chunk in sentence_body(sentence,index):yield chunk
                count=sum(row['sample_count'] for row in stats['sentences'])
                stats.update(status='complete',first_token=stats['sentences'][0]['first_token'],
                    total_seconds=time.monotonic()-clock,audio_seconds=count/RATE,
                    sample_count=count,offline_sample_count=sum(row['offline_sample_count'] for row in stats['sentences']),
                    audio_tokens=sum(row['audio_tokens'] for row in stats['sentences']))
            except BaseException as exc:
                stats.update(status='cancelled' if isinstance(exc,asyncio.CancelledError) else 'failed',error=str(exc))
                raise
            finally:atomic(B/'traces'/(identifier+'.json'),stats)
    return StreamingResponse(body(),media_type='audio/pcm',headers={'X-Request-ID':identifier,'X-Sample-Rate':'24000','X-Channels':'1','X-Sample-Format':'s16le','Cache-Control':'no-store'})
@app.post('/synthesize')
async def synthesize(request:Request):
    stream_response=await stream(request)
    chunks=[]
    async for chunk in stream_response.body_iterator:chunks.append(chunk)
    buffer=io.BytesIO()
    with wave.open(buffer,'wb') as wav:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(RATE);wav.writeframes(b''.join(chunks))
    identifier=stream_response.headers['X-Request-ID'];trace=json.loads((B/'traces'/(identifier+'.json')).read_text())
    if trace['status']!='complete':raise HTTPException(500,'Generation incomplete')
    return Response(buffer.getvalue(),media_type='audio/wav',headers={'X-Request-ID':identifier,'X-Audio-Chunks':str(len(trace['sentences'])),'X-Model':'orpheus-epoch3-4bit','Cache-Control':'no-store'})
if __name__=='__main__':
    import uvicorn
    print(json.dumps(health),flush=True)
    uvicorn.run(app,host='127.0.0.1',port=args.port,log_level='warning')
