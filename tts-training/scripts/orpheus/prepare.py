"""Read immutable approved clips; create lossless-in-time SNAC token features."""
from common import *
import csv,html,math,sys
def main():
    import numpy as np,soundfile as sf,torch
    from scipy.signal import resample_poly
    from snac import SNAC
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    report=json.loads((CORPUS/'corpus.json').read_text())
    approval=json.loads((CORPUS/'owner-review-approved.json').read_text())
    assert approval['fingerprint']==report['fingerprint']
    plan=json.loads((CORPUS/'source-plan.json').read_text())
    revision=setup()['model_revisions']
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=revision[MODEL])
    assert len(tokenizer)>TOKENS['audio']+7*4096-1
    status(status='encoding',prepared=0,error=None)
    codec_dir=snapshot_download(CODEC,revision=revision[CODEC])
    codec=SNAC.from_pretrained(codec_dir).to('cuda').eval()
    splits={};count=0;qa=[]
    for split in ['train','validation']:
        manifest=CORPUS/split/'manifest.csv'
        assert sha(manifest)==report['splits'][split]['sha256']
        with manifest.open() as stream:rows=list(csv.DictReader(stream))
        assert rows, "Corpus split is empty"
        longest={r['id'] for r in sorted(rows,key=lambda r:float(r['duration_seconds']),reverse=True)[:2]}
        chosen=longest|{r['id'] for r in rows[:2]+rows[-2:]}
        output=[]
        for row in rows:
            path=REPO/row['filename'];assert sha(path)==row['audio_sha256'] and not row['flags']
            a=int(row['source_start_frame'])/int(row['source_sample_rate']);b=int(row['source_end_frame'])/int(row['source_sample_rate'])
            assert any(a>=lo and b<=hi for lo,hi in plan['partitions'][row['source']][split])
            assert not any(a<hi and b>lo for lo,hi in plan['exclusions'][row['source']])
            key=hashlib.sha256(json.dumps(dict(audio=row['audio_sha256'],text=row['text'],codec=revision[CODEC],tokenizer=revision[MODEL],resampling='scipy-poly-24k',frames='all'),sort_keys=True).encode()).hexdigest()
            cache=DATA/'tokens'/(key+'.json')
            if cache.exists():item=json.loads(cache.read_text())
            else:
                audio,sr=sf.read(path,dtype='float32');assert audio.ndim==1 and np.isfinite(audio).all()
                divisor=math.gcd(sr,24000)
                if sr!=24000:audio=resample_poly(audio,24000//divisor,sr//divisor)
                waveform=torch.from_numpy(audio).view(1,1,-1).to('cuda')
                with torch.inference_mode():encoded=codec.encode(waveform)
                codes=pack_codes([layer[0].cpu().tolist() for layer in encoded])
                ids=format_example(tokenizer.encode(row['text'],add_special_tokens=True),codes)
                assert len(ids)<=4096, f'Clip exceeds context without truncation: {row["id"]} {len(ids)}'
                item=dict(id=row['id'],input_ids=ids,text=row['text'],audio_sha256=row['audio_sha256'],
                    split=split,duration_seconds=float(row['duration_seconds']),source=row['source'],
                    audio_token_count=len(codes),resampled_samples=len(audio),cache_key=key)
                atomic(cache,item)
            assert item['audio_sha256']==row['audio_sha256'] and item['text']==row['text']
            assert item['input_ids'][-2:]==[TOKENS['end_speech'],TOKENS['end_ai']]
            assert item['audio_token_count']%7==0 and len(item['input_ids'])<=4096
            output.append(str(cache));count+=1
            if row['id'] in chosen:
                qa_dir=DATA/'qa';qa_dir.mkdir(parents=True,exist_ok=True)
                dest=qa_dir/(row['id']+'.wav')
                if not dest.exists():
                    ids=item['input_ids'];start=ids.index(TOKENS['start_speech'])+1
                    layers=unpack_codes(ids[start:-2])
                    with torch.inference_mode():decoded=codec.decode([torch.tensor([v],device='cuda') for v in layers])[0,0].float().cpu().numpy()
                    assert len(decoded)>=item['resampled_samples'] and len(decoded)-item['resampled_samples']<2048
                    # Remove only known codec right-padding, never source samples.
                    decoded=decoded[:item['resampled_samples']]
                    assert np.isfinite(decoded).all()
                    sf.write(dest,decoded,24000,subtype='PCM_16')
                qa.append(dict(id=row['id'],text=row['text'],original=str(path),roundtrip=str(dest),split=split))
            if count%10==0:status(status='encoding',prepared=count,total_clips=1204)
        splits[split]=output
    all_items=[json.loads(Path(p).read_text()) for group in splits.values() for p in group]
    assert not ({r['audio_sha256'] for r in all_items if r['split']=='train'} & {r['audio_sha256'] for r in all_items if r['split']=='validation'})
    encoded=dict(fingerprint=report['fingerprint'],splits=splits,clips=count,max_tokens=max(len(r['input_ids']) for r in all_items),
        codec=CODEC,revisions=revision,no_frames_removed=True,source_manifests={s:sha(CORPUS/s/'manifest.csv') for s in splits})
    atomic(DATA/'dataset.json',encoded);atomic(DATA/'qa.json',qa)
    status(status='prepared',prepared=count,max_tokens=encoded['max_tokens'])
    print(json.dumps({k:v for k,v in encoded.items() if k!='splits'}),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        status(status='failed',error=traceback.format_exc());raise
