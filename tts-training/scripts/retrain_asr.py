"""Independent float32 large-v3 verification; PYTHONPATH must be empty."""
import json,os,pathlib,re,subprocess
assert not os.environ.get('PYTHONPATH')
from faster_whisper import WhisperModel
R=pathlib.Path(__file__).resolve().parents[1]
assert subprocess.run(['systemctl','--user','is-active','--quiet','she-gptsovits-tts.service']).returncode!=0
model=WhisperModel(str(R/'.cache/whisper-large-v3'),device='cuda',compute_type='float32')
def words(t):return re.findall(r"[a-z0-9]+",t.lower().replace('’',"'"))
def distance(a,b):
 prev=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  cur=[i]
  for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
  prev=cur
 return prev[-1]
results=[]
for item in json.loads((R/'reports/retrain-samples.json').read_text()):
 p=item['path'];probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams','-of','json',p]));s=probe['streams'][0]
 assert (s['sample_rate'],s['channels'],s['codec_name'])==('32000',1,'pcm_s16le');assert float(probe['format']['duration'])>0.5
 vol=subprocess.run(['ffmpeg','-nostdin','-i',p,'-af','volumedetect','-f','null','-'],capture_output=True,text=True,check=True).stderr
 mean=float(re.search(r'mean_volume: ([-\d.]+)',vol).group(1));assert mean>-50
 segments,_=model.transcribe(p,language='en',beam_size=5,condition_on_previous_text=False,vad_filter=False);text=' '.join(s.text.strip() for s in segments)
 a,b=words(item['text']),words(text);n=distance(a,b)
 result={**item,'transcript':text,'word_errors':n,'reference_words':len(a),'wer':n/len(a),'words_match':a==b,'mean_volume_db':mean,'ffprobe':probe,'asr_model':'large-v3','compute_type':'float32'}
 if 'word' in item:result['target_word_rendered']=item['word'] in b
 results.append(result);(R/'reports/retrain-asr.json').write_text(json.dumps(results,indent=2)+'\n');print(item['name'],result['wer'],text,flush=True)
