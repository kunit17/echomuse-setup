"""Preserve every manifest transcript and resample every clip without slicing."""
import concurrent.futures,csv,hashlib,json,pathlib,subprocess,wave
ROOT=pathlib.Path(__file__).resolve().parents[1]
SRC=pathlib.Path('/home/voiceuser/.hermes/cache/voices/she-xtts')
rows=list(csv.DictReader((SRC/'manifest.csv').open()))
assert len(rows)==1178 and len({r['filename'] for r in rows})==1178
out=ROOT/'data/wavs32k';out.mkdir(parents=True,exist_ok=True)
def process(r):
 src=SRC/'clips'/r['filename']; dst=out/src.name
 digest=hashlib.sha256(src.read_bytes()).hexdigest()
 if not dst.exists():
  subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(src),'-af','aresample=32000:resampler=soxr:precision=28','-ac','1','-c:a','pcm_s16le',str(dst)],check=True)
 with wave.open(str(dst)) as w:
  assert (w.getframerate(),w.getnchannels(),w.getsampwidth())==(32000,1,2)
  duration=w.getnframes()/32000
 assert not any(c in r['text'] for c in '|\n\r'),src
 return {'filename':src.name,'source_sha256':digest,'duration':duration,'original_index':r['index']}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as p: clips=list(p.map(process,rows))
(ROOT/'data/she.list').write_text(''.join(f"{out/r['filename']}|she|en|{r['text']}\n" for r in rows))
report={'clips':len(clips),'seconds':sum(c['duration'] for c in clips),'hours':sum(c['duration'] for c in clips)/3600,'speaker':'she','exclusions':[],'manifest_sha256':hashlib.sha256((SRC/'manifest.csv').read_bytes()).hexdigest(),'method':'ffmpeg soxr precision=28, 32000 Hz mono PCM s16le','source_clips':clips}
(ROOT/'reports/retrain-dataset.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='source_clips'},indent=2))
subprocess.run(['ffprobe','-v','error','-show_streams','-of','json',str(out/rows[0]['filename'])],check=True,stdout=(ROOT/'reports/retrain-resampled-ffprobe.json').open('w'))
print('Transcript spot checks:')
for r in rows[::300]:print(r['filename'],r['text'])
