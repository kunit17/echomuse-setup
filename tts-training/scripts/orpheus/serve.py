"""Loopback speech endpoint; one epoch-3 model stays loaded until stopped."""
from common import *
import io,re,sys,threading,wave,logging
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

def main():
    from unsloth import FastLanguageModel
    import torch,numpy as np
    from transformers import set_seed
    from peft import PeftModel
    from snac import SNAC
    from huggingface_hub import snapshot_download
    sys.path.insert(0,str(REPO/'scripts'))
    from audio_postprocess import clean_generated_tail
    selected=json.loads((ROOT/'selected-model.json').read_text())
    checkpoint=Path(selected['adapter_path'])
    assert sha(checkpoint/'adapter_model.safetensors')==selected['adapter_sha256']
    started=time.monotonic()
    base,tokenizer=FastLanguageModel.from_pretrained(MODEL,revision=setup()['model_revisions'][MODEL],
        max_seq_length=4096,dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        load_in_4bit=False,device_map='cpu')
    model=PeftModel.from_pretrained(base,str(checkpoint),is_trainable=False).to('cuda')
    FastLanguageModel.for_inference(model);refresh_model_placement(model)
    codec=SNAC.from_pretrained(snapshot_download(CODEC,revision=setup()['model_revisions'][CODEC],local_files_only=True)).cpu().eval()
    health=dict(ready=True,model_loaded=True,model='Orpheus',epoch=3,reference_id=None,
        adapter_sha256=selected['adapter_sha256'],adapter_path=str(checkpoint),
        postprocessing_policy='isolated-end-burst-v1',load_seconds=time.monotonic()-started,pid=os.getpid())
    lock=threading.Lock()
    def synthesize(text):
        # Sentence/space boundaries only; each chunk must generate its own EOS.
        parts=[];remaining=text.strip()
        while remaining:
            if len(remaining)<=400:parts.append(remaining);break
            matches=list(re.finditer(r'[.!?]\s+',remaining[:401]))
            cut=matches[-1].end() if matches else remaining.rfind(' ',0,401)
            if cut<=0:raise ValueError('A single unbroken word exceeds 400 characters')
            parts.append(remaining[:cut].strip());remaining=remaining[cut:].strip()
        chunks=[];decisions=[]
        with lock,torch.inference_mode():
            for text in parts:
                ids=[TOKENS['start_human']]+tokenizer.encode(text,add_special_tokens=True)+[TOKENS['end_text'],TOKENS['end_human'],TOKENS['start_ai'],TOKENS['start_speech']]
                inputs=torch.tensor([ids],device='cuda');set_seed(1700)
                output=model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),max_new_tokens=2000,
                    do_sample=True,temperature=.6,top_p=.95,repetition_penalty=1.1,
                    eos_token_id=TOKENS['end_speech'],pad_token_id=TOKENS['pad'],use_cache=True)
                tokens=output[0,len(ids):].tolist()
                if TOKENS['end_speech'] not in tokens:raise RuntimeError('No end-of-speech; refusing truncated audio')
                layers=unpack_codes(tokens[:tokens.index(TOKENS['end_speech'])])
                audio=codec.decode([torch.tensor([v]) for v in layers])[0,0].float().numpy()
                if not len(audio) or not np.isfinite(audio).all():raise RuntimeError('Invalid decoded audio')
                audio,decision=clean_generated_tail(audio,24000);decisions.append(decision);chunks.append(audio)
            pcm=(np.clip(np.concatenate(chunks),-1,1)*32767).astype('<i2').tobytes()
        buffer=io.BytesIO()
        with wave.open(buffer,'wb') as wav:
            wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(24000);wav.writeframes(pcm)
        return buffer.getvalue(),len(parts),decisions
    class Handler(BaseHTTPRequestHandler):
        def reply(self,code,body,kind='application/json',headers=None):
            self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            for key,value in (headers or {}).items():self.send_header(key,str(value))
            self.end_headers();self.wfile.write(body)
        def do_GET(self):
            if self.path!='/health':return self.reply(404,b'{}')
            self.reply(200,json.dumps(dict(health,allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved())).encode())
        def do_POST(self):
            if self.path!='/synthesize':return self.reply(404,b'{}')
            try:
                if self.headers.get('Origin') or self.headers.get('Host') not in ('127.0.0.1:8812','localhost:8812'):raise ValueError('Local requests only')
                if self.headers.get_content_type()!='application/json':raise ValueError('Expected JSON')
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=65536:raise ValueError('Invalid request length')
                data=json.loads(self.rfile.read(size));text=data.get('text')
                if not isinstance(text,str) or not text.strip() or len(text)>5000:raise ValueError('Use 1–5000 characters')
                if isinstance(data.get('speed',1),bool) or data.get('speed',1)!=1:raise ValueError('Only speed=1 supported')
            except (ValueError,TypeError,AttributeError) as exc:return self.reply(400,json.dumps({'error':str(exc)}).encode())
            started=time.monotonic()
            try:
                audio,n,decisions=synthesize(text);elapsed=time.monotonic()-started
                atomic(ROOT/'last-generation.json',dict(text=text,seconds=elapsed,chunks=n,tail_cleanup=decisions,**health))
                self.reply(200,audio,'audio/wav',{'X-Audio-Chunks':n,'X-Generation-Seconds':elapsed,'X-Model':'orpheus-epoch3'})
            except Exception:
                logging.exception('Synthesis failed');self.reply(500,b'{"error":"Synthesis failed; see journal"}')
    server=ThreadingHTTPServer(('127.0.0.1',8812),Handler);server.daemon_threads=True
    print(json.dumps(health),flush=True);server.serve_forever()
if __name__=='__main__':main()
