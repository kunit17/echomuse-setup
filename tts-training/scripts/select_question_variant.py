"""Generate a fixed candidate batch; retain all ASR outcomes, select exact words."""
import csv,json,os,pathlib,re,shutil
import soundfile as sf
R=pathlib.Path(__file__).resolve().parents[1];os.chdir(R/'upstream')
manifest=json.loads((R/'samples/manifest.json').read_text());item=next(s for s in manifest if s['path']=='samples/03-question.wav')
rows={r['filename']:r for r in csv.DictReader(open('/home/voiceuser/.hermes/cache/voices/she-xtts/manifest.csv'))}
from TTS_infer_pack.TTS import TTS,TTS_Config
config=json.loads((R/'configs/inference.json').read_text());assert 'checkpoints/gpt' in config['t2s_weights_path'] and 'checkpoints/sovits' in config['vits_weights_path']
cfg=TTS_Config({'custom':config});cfg.configs_path=str(R/'configs/tts-infer.yaml');tts=TTS(cfg)
out=R/'samples/attempts/question-candidates';out.mkdir(parents=True,exist_ok=True)
candidates=[]
for i in range(10):
 ref=rows['clip_00033.wav' if i%2==0 else 'clip_00406.wav'];seed=3000+i
 inputs={'text':item['text'],'text_lang':'en','ref_audio_path':str(R/'data/wavs32k'/ref['filename']),'prompt_text':ref['text'],'prompt_lang':'en','top_k':15,'top_p':1.0,'temperature':0.8,'text_split_method':'cut5','batch_size':1,'seed':seed,'parallel_infer':False,'use_cuda_graph':False,'use_flash_attention':False,'speed_factor':0.95,'repetition_penalty':1.35}
 chunks=list(tts.run(inputs));assert len(chunks)==1
 sr,audio=chunks[0];path=out/f'candidate-{seed}.wav';sf.write(path,audio,sr,subtype='PCM_16')
 candidates.append({**item,'path':str(path.relative_to(R)),'seed':seed,'reference_filename':ref['filename'],'top_k':15,'temperature':0.8,'speed_factor':0.95,'duration_seconds':len(audio)/sr})
(R/'reports/question-candidates.json').write_text(json.dumps(candidates,indent=2)+'\n')
del tts
import gc,torch
gc.collect();torch.cuda.empty_cache()
from faster_whisper import WhisperModel
model=WhisperModel(str(R/'.cache/whisper-large-v3'),device='cuda',compute_type='float16')
def words(s):return re.findall(r'[a-z0-9]+',s.lower().replace('’',"'"))
for c in candidates:
 seg,_=model.transcribe(str(R/c['path']),language='en',beam_size=5,condition_on_previous_text=False,vad_filter=False)
 c['transcript']=' '.join(s.text.strip() for s in seg);c['words_match']=words(c['text'])==words(c['transcript'])
 print(c['seed'],c['words_match'],c['transcript'],flush=True)
 (R/'reports/question-candidates.json').write_text(json.dumps(candidates,indent=2)+'\n')
passing=[c for c in candidates if c['words_match']]
assert passing,'No candidate passed; see question-candidates.json'
selected=passing[0]
backup=out/'prior-final-question.wav';shutil.copy2(R/item['path'],backup)
shutil.copy2(R/selected['path'],R/item['path'])
replacement={k:v for k,v in selected.items() if k not in ['transcript','words_match','duration_seconds']};replacement['path']=item['path']
manifest=[replacement if s['path']==item['path'] else s for s in manifest]
(R/'samples/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('SELECTED',selected['seed'],'PASSING',len(passing),'OF',len(candidates),flush=True)
