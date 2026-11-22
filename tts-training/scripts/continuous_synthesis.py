"""Generate frozen probes once from the selected new-corpus checkpoints."""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import importlib

import numpy as np
import soundfile as sf
import torch
from resource_safe import bounded_hubert_loading

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT/'upstream')
from TTS_infer_pack.TTS import TTS,TTS_Config
bounded_hubert_loading()
tts_module=importlib.import_module('TTS_infer_pack.TTS')
def bounded_sovits_load(path):
    # Versioned exports change only the first ZIP-signature bytes. Restore
    # those bytes in a bounded disk copy instead of constructing a BytesIO.
    destination=ROOT/'.cache/continuous-evaluation-sovits.pth'
    with open(path,'rb') as source,destination.open('wb') as output:
        shutil.copyfileobj(source,output,length=1024*1024)
    with destination.open('r+b') as stream:
        head=stream.read(2)
        if head in (b'05',b'06'):stream.seek(0);stream.write(b'PK')
    return torch.load(destination,map_location='cuda',weights_only=False)
tts_module.load_sovits_new=bounded_sovits_load
# English text features are zeros in the upstream preprocessor. Avoid loading
# an unused Chinese BERT model; both target and reference languages stay en.
def english_bert(self,base_path):
    self.bert_model=None;self.bert_tokenizer=None
TTS.init_bert_weights=english_bert
G=json.loads((ROOT/'reports/continuous-s1-selection.json').read_text())['selected']
S=json.loads((ROOT/'reports/continuous-s2-selection.json').read_text())['selected']
for row in (G,S):assert Path(row['checkpoint']).stat().st_size>0
rows=list(csv.DictReader((ROOT/'data/shecorpus/train/manifest.csv').open()))
ref=next(r for r in rows if 5<=float(r['duration_seconds'])<=9)
config=dict(device='cuda',is_half=True,version='v2ProPlus',t2s_weights_path=G['checkpoint'],vits_weights_path=S['checkpoint'],
            cnhuhbert_base_path=str(ROOT/'upstream/GPT_SoVITS/pretrained_models/chinese-hubert-base'),
            bert_base_path=str(ROOT/'upstream/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large'))
cfg=TTS_Config({'custom':config});cfg.configs_path=str(ROOT/'.cache/continuous-evaluation-infer.yaml')
with torch.device('cuda'):tts=TTS(cfg)
tts.configs.max_sec=60
plan=json.loads((ROOT/'configs/corpus-evaluation-prompts.json').read_text())
probes=[dict(name=f'novel-{i:02d}',group='novel',text=text) for i,text in enumerate(plan['novel_text'])]
probes += [dict(name=f'termination-{i}',group='termination',text=text) for i,text in enumerate(plan['termination_probes'])]
probes += [dict(name='punctuation-unsplit',group='punctuation',text=plan['punctuation_probe']),dict(name='five-unsplit',group='five-unsplit',text=plan['five_sentence_probe'])]
OUT=ROOT/'samples/continuous';OUT.mkdir(parents=True,exist_ok=True)
def file_hash(path):
    value=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()
report=dict(gpt_checkpoint=G['checkpoint'],sovits_checkpoint=S['checkpoint'],gpt_epoch=G['epoch'],sovits_epoch=S['epoch'],reference_filename=ref['filename'],
    inference_max_seconds=60,english_bert_features='upstream zero features; unused model not loaded',
    checkpoint_sha256={k:file_hash(v['checkpoint']) for k,v in [('gpt',G),('sovits',S)]},samples=[])
def run(text,seed):
    inputs=dict(text=text,text_lang='en',ref_audio_path=str(ROOT/ref['filename']),prompt_text=ref['text'],prompt_lang='en',
        top_k=15,top_p=1.,temperature=.8,speed_factor=1.,text_split_method='cut0',batch_size=1,seed=seed,
        parallel_infer=False,use_cuda_graph=False,use_flash_attention=False,repetition_penalty=1.35,fragment_interval=0.)
    chunks=list(tts.run(inputs));assert chunks
    parts=[]
    for sr,audio in chunks:
        assert sr==32000 and len(audio)>0
        parts.append(audio.astype(np.float32)/32768 if audio.dtype==np.int16 else audio.astype(np.float32))
    return np.concatenate(parts),len(chunks)
for i,probe in enumerate(probes):
    path=OUT/(probe['name']+'.wav')
    if path.exists():raise RuntimeError('Refusing an unrecorded synthesis retry: '+probe['name'])
    audio,n=run(probe['text'],20261112+i)
    sf.write(path,audio,32000,subtype='PCM_16')
    report['samples'].append(dict(**probe,path=str(path),duration_seconds=len(audio)/32000,yielded_chunks=n,fragment_interval=0.,seed=20261112+i))
    (ROOT/'reports/continuous-synthesis.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Generated',probe['name'],len(audio)/32000,flush=True)
# Local application segmentation, independently reported from learned pauses.
sentences=re.split(r'(?<=[.!?])\s+',plan['five_sentence_probe'])
assert len(sentences)==5
parts=[];boundaries=[]
for i,text in enumerate(sentences):
    audio,n=run(text,20261212+i)
    if parts:
        start=sum(len(p) for p in parts)/32000
        parts.append(np.zeros(9600,dtype=np.float32));boundaries.append([start,start+.3])
    parts.append(audio)
path=OUT/'five-application-segmented.wav';sf.write(path,np.concatenate(parts),32000,subtype='PCM_16')
report['application_segmentation']=dict(path=str(path),sentence_count=5,inserted_gap_seconds=.3,gap_intervals=boundaries,
    proves_learned_pauses=False,live_provider_invoked=False)
(ROOT/'reports/continuous-synthesis.json').write_text(json.dumps(report,indent=2)+'\n')
