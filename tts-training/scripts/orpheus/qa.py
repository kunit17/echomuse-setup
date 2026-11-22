"""Check SNAC round-trip transcript endings against the unchanged originals."""
from common import *
import sys
sys.path.insert(0,str(REPO/'scripts'))
from faster_whisper import WhisperModel
from corpus_ground_truth import tokens,edits
model=WhisperModel(str(REPO/'.cache/whisper-large-v3'),device='cuda',compute_type='float16',local_files_only=True)
results=[]
def transcribe(path):
    segments,_=model.transcribe(path,language='en',beam_size=5,temperature=0,condition_on_previous_text=False,vad_filter=False)
    return ' '.join(s.text.strip() for s in segments)
for item in json.loads((DATA/'qa.json').read_text()):
    actual=transcribe(item['roundtrip']);expected=tokens(item['text']);heard=tokens(actual)
    matches=bool(heard and heard[-1]==expected[-1]);baseline=None
    if not matches:
        baseline=transcribe(item['original']);original=tokens(baseline)
        matches=bool(heard and original and heard[-1]==original[-1])
    results.append(dict(item,asr=actual,word_edits=edits(expected,heard),final_word_preserved=matches,original_asr=baseline))
    atomic(DATA/'qa-asr.json',dict(results=results,complete=False))
atomic(DATA/'qa-asr.json',dict(results=results,complete=True,all_final_words_preserved=all(r['final_word_preserved'] for r in results),manual_listening_complete=False))
assert all(r['final_word_preserved'] for r in results),'Codec ending QA failed; inspect qa-asr.json'
print('PASS: codec round-trip final-word checks',len(results),flush=True)
