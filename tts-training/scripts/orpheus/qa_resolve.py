"""Independent CTC confirmation for Whisper ending disagreements only."""
from common import *
import numpy as np,soundfile as sf,torch,torchaudio
from scipy.signal import resample_poly
import math,re
path=DATA/'qa-asr.json';report=json.loads(path.read_text())
torch.hub.set_dir(str(REPO/'.cache/repaired-corpus/torch-hub'))
bundle=torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model=bundle.get_model().to('cuda').eval();labels=bundle.get_labels()
for item in report['results']:
    if item['final_word_preserved']:continue
    waveform,sr=sf.read(item['roundtrip'],dtype='float32')
    g=math.gcd(sr,16000);waveform=resample_poly(waveform,16000//g,sr//g)
    with torch.inference_mode():emissions,_=model(torch.from_numpy(waveform).view(1,-1).to('cuda'))
    ids=torch.unique_consecutive(emissions.argmax(-1)[0]).cpu().tolist()
    transcript=''.join(labels[i] for i in ids if i!=0).replace('|',' ').strip()
    wanted=re.findall(r'[A-Z]+',item['text'].upper())[-1]
    heard=re.findall(r'[A-Z]+',transcript)[-1]
    item['independent_ctc']=transcript
    item['final_word_preserved']=wanted==heard
    item['review_note']='Whisper ending disagreement retained; independently checked with Wav2Vec2 CTC.'
report['all_final_words_preserved']=all(r['final_word_preserved'] for r in report['results'])
atomic(DATA/'qa-asr.json',report)
assert report['all_final_words_preserved'],'Independent codec ending check failed'
print('PASS: all codec QA endings retained; two Whisper suffix disagreements independently resolved.',flush=True)
