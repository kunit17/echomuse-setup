from common import *
import sys
sys.path.insert(0,str(REPO/'scripts'))
from faster_whisper import WhisperModel
from corpus_ground_truth import tokens,edits
model=WhisperModel(str(REPO/'.cache/whisper-large-v3'),device='cuda',compute_type='float16',local_files_only=True)
results=[]
for path in sorted((RUN/'samples').glob('*/results.json')):
    if path.parent.name.startswith('smoke'):continue
    report=json.loads(path.read_text())
    for item in report['samples']:
        audio=path.parent/(item['name']+'.wav')
        segments,_=model.transcribe(str(audio),language='en',beam_size=5,temperature=0,condition_on_previous_text=False,vad_filter=False)
        transcript=' '.join(s.text.strip() for s in segments)
        expected,heard=tokens(item['text']),tokens(transcript)
        item['asr']=dict(transcript=transcript,word_edits=edits(expected,heard),final_word_match=bool(heard and heard[-1]==expected[-1]))
        results.append(dict(folder=path.parent.name,name=item['name'],**item['asr']))
    atomic(path,report)
atomic(RUN/'preview-qa.json',dict(complete=True,results=results,manual_listening_complete=False))
print('ASR review complete',len(results),flush=True)
