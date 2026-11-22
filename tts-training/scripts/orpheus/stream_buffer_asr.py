from common import *
import sys
sys.path.insert(0,str(REPO/'scripts'))
from faster_whisper import WhisperModel
from corpus_ground_truth import tokens,edits
B=ROOT/'stream-benchmark';path=B/'buffer-policy-check.json';report=json.loads(path.read_text())
model=WhisperModel(str(REPO/'.cache/whisper-large-v3'),device='cuda',compute_type='float16',local_files_only=True)
for item in report['samples']:
    segments,_=model.transcribe(item['wav'],language='en',beam_size=5,temperature=0,condition_on_previous_text=False,vad_filter=False)
    text=' '.join(s.text.strip() for s in segments);expected,heard=tokens(item['text']),tokens(text)
    item['asr']=dict(transcript=text,word_edits=edits(expected,heard),first_word_match=bool(heard and heard[0]==expected[0]),final_word_match=bool(heard and heard[-1]==expected[-1]))
report['status']='complete_listening_pending';atomic(path,report)
print('Buffered sentence-streaming ASR checks complete',flush=True)
