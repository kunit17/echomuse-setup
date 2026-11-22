"""Measure each sample and compare independent faster-whisper ASR to its prompt."""
import difflib,json,os,pathlib,re,subprocess
from faster_whisper import WhisperModel
R=pathlib.Path(__file__).resolve().parents[1]
model=WhisperModel(str(R/'.cache/whisper-large-v3'),device='cuda',compute_type='float16',download_root=str(R/'.cache/whisper'))
def words(s):return re.findall(r"[a-z0-9]+",s.lower().replace('’',"'"))
def wer(a,b):
 prev=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  cur=[i]
  for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
  prev=cur
 return prev[-1]/max(1,len(a))
results=[]
for item in json.loads((R/'samples/manifest.json').read_text()):
 p=R/item['path'];probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(p)]))
 vol=subprocess.run(['ffmpeg','-nostdin','-i',str(p),'-af','volumedetect','-f','null','-'],capture_output=True,text=True,check=True).stderr
 mean=float(re.search(r'mean_volume: ([\-\d.]+) dB',vol).group(1));peak=float(re.search(r'max_volume: ([\-\d.]+) dB',vol).group(1))
 segments,info=model.transcribe(str(p),language='en',beam_size=5,condition_on_previous_text=False,vad_filter=False)
 transcript=' '.join(s.text.strip() for s in segments)
 error=wer(words(item['text']),words(transcript))
 result={**item,'duration_seconds':float(probe['format']['duration']),'codec':probe['streams'][0]['codec_name'],'channels':probe['streams'][0]['channels'],'mean_volume_db':mean,'max_volume_db':peak,'transcript':transcript,'word_error_rate':error,'words_match':words(item['text'])==words(transcript),'diff':list(difflib.ndiff(words(item['text']),words(transcript))),'ffprobe':probe}
 assert mean>-50 and result['duration_seconds']>1 and p.stat().st_size>1000
 assert result['codec']=='pcm_s16le' and result['channels']==1 and int(probe['streams'][0]['sample_rate'])==32000
 results.append(result); print(json.dumps({k:v for k,v in result.items() if k not in ['ffprobe','diff']}),flush=True)
 (R/'reports/sample-verification.json').write_text(json.dumps(results,indent=2)+'\n')
assert len(results)>=5
assert all(x['words_match'] for x in results),'ASR word mismatch: inspect reports/sample-verification.json; do not claim verification passed'
