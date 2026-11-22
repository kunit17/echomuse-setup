"""Controlled greeting comparison: fixed GPT, reference, text and paired seeds."""
import argparse
import csv
import fcntl
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

ROOT=Path('/home/voiceuser/projects/gpt-sovits')
OUT=Path(os.environ.get('GREETING_TEST_OUTPUT',str(ROOT/'samples/greeting-tests')));OUT.mkdir(parents=True,exist_ok=True)
TEXT="Hi Alex, it's great to finally meet you after all this time! We are going to get along swimmingly."
sys.path.insert(0,str(ROOT/'scripts'))
from asr_runtime import atomic_json,file_digest

def synthesize():
    import numpy as np
    import soundfile as sf
    import torch
    from resource_safe import bounded_hubert_loading
    from training_run import verify_inputs
    assert torch.cuda.is_available()
    status=json.loads((ROOT/'reports/training-status.json').read_text())
    assert status['status']=='complete'
    reports=ROOT/'reports/training-runs'/status['run_id']
    run=json.loads((reports/'run.json').read_text());verify_inputs(run)
    gpt=json.loads((reports/'s1-selection.json').read_text())['selected']
    selection=json.loads((reports/'s2-selection.json').read_text())
    selected,latest=selection['selected'],selection['final']
    assert selected['epoch']==10 and latest['epoch']==15
    with (ROOT/'data/shecorpus/train/manifest.csv').open(encoding='utf8') as stream:
        reference=next(row for row in csv.DictReader(stream) if 5<=float(row['duration_seconds'])<=9)
    os.chdir(ROOT/'upstream')
    from TTS_infer_pack.TTS import TTS,TTS_Config
    bounded_hubert_loading()
    tts_module=importlib.import_module('TTS_infer_pack.TTS')
    def load_sovits(path):
        destination=ROOT/'.cache/greeting-comparison-sovits.pth'
        with open(path,'rb') as source,destination.open('wb') as output:shutil.copyfileobj(source,output,1024*1024)
        with destination.open('r+b') as stream:
            if stream.read(2) in (b'05',b'06'):stream.seek(0);stream.write(b'PK')
        return torch.load(destination,map_location='cuda',weights_only=False)
    tts_module.load_sovits_new=load_sovits
    def english_bert(self,base_path):self.bert_model=None;self.bert_tokenizer=None
    TTS.init_bert_weights=english_bert
    report=dict(schema=1,status='generating',text=TEXT,created=time.time(),run_id=run['run_id'],
        corpus_fingerprint=run['corpus_fingerprint'],gpt_epoch=gpt['epoch'],gpt_checkpoint=gpt['checkpoint'],
        gpt_sha256=file_digest(gpt['checkpoint'])['sha256'],reference_filename=reference['filename'],
        reference_sha256=file_digest(ROOT/reference['filename'])['sha256'],samples=[],
        inference=dict(device='cuda',precision='float16',top_k=15,top_p=1,temperature=.8,
            speed_factor=1,text_split_method='cut0',repetition_penalty=1.35,fragment_interval=0),
        postprocessing='Native generated audio retained; no trimming, padding or inserted pauses',live_provider_changed=False)
    atomic_json(OUT/'results.json',report)
    for stage in [selected,latest]:
        config=dict(device='cuda',is_half=True,version='v2ProPlus',t2s_weights_path=gpt['checkpoint'],vits_weights_path=stage['checkpoint'],
            cnhuhbert_base_path=str(ROOT/'upstream/GPT_SoVITS/pretrained_models/chinese-hubert-base'),
            bert_base_path=str(ROOT/'upstream/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large'))
        cfg=TTS_Config({'custom':config});cfg.configs_path=str(ROOT/'.cache/greeting-comparison-infer.yaml')
        with torch.device('cuda'):tts=TTS(cfg)
        tts.configs.max_sec=60
        assert str(tts.configs.vits_weights_path)==stage['checkpoint']
        checkpoint_hash=file_digest(stage['checkpoint'])['sha256']
        for seed in [1700,1701]:
            path=OUT/f'sovits-e{stage["epoch"]}-seed{seed}.wav'
            if path.exists():raise RuntimeError('Refusing to overwrite an existing test: '+path.name)
            inputs=dict(text=TEXT,text_lang='en',ref_audio_path=str(ROOT/reference['filename']),prompt_text=reference['text'],prompt_lang='en',
                top_k=15,top_p=1.,temperature=.8,speed_factor=1.,text_split_method='cut0',batch_size=1,seed=seed,
                parallel_infer=False,use_cuda_graph=False,use_flash_attention=False,repetition_penalty=1.35,fragment_interval=0.)
            chunks=list(tts.run(inputs));assert chunks
            parts=[]
            for sr,audio in chunks:
                assert sr==32000 and audio.ndim==1 and len(audio)
                parts.append(audio.astype(np.float32)/32768 if audio.dtype==np.int16 else audio.astype(np.float32))
            signal=np.concatenate(parts);assert np.isfinite(signal).all()
            sf.write(path,signal,32000,subtype='PCM_16')
            rms=np.sqrt(np.mean(signal[:len(signal)//320*320].reshape(-1,320).astype(np.float64)**2,axis=1))
            tail=next((i for i,v in enumerate(rms[::-1]) if v>=.01),len(rms))*.01
            item=dict(filename=path.name,sovits_epoch=stage['epoch'],seed=seed,seconds=len(signal)/32000,
                sovits_checkpoint=stage['checkpoint'],sovits_sha256=checkpoint_hash,audio_sha256=file_digest(path)['sha256'],
                yielded_chunks=len(chunks),quiet_tail_seconds=tail,peak=float(np.max(abs(signal))))
            report['samples'].append(item);atomic_json(OUT/'results.json',report)
            print('GENERATED',path.name,'seconds',item['seconds'],flush=True)
        del tts;gc.collect();torch.cuda.empty_cache()
    report['status']='generated_pending_asr';atomic_json(OUT/'results.json',report)

def score():
    from faster_whisper import WhisperModel
    from corpus_ground_truth import tokens,edits
    report=json.loads((OUT/'results.json').read_text());assert len(report['samples'])==4
    model=WhisperModel(str(ROOT/'.cache/whisper-large-v3'),device='cuda',compute_type='float16',local_files_only=True)
    expected=tokens(TEXT)
    for item in report['samples']:
        segments,_=model.transcribe(str(OUT/item['filename']),language='en',task='transcribe',beam_size=5,
            temperature=0,condition_on_previous_text=False,vad_filter=False,word_timestamps=True)
        words=[word for segment in segments for word in (segment.words or [])]
        transcript=''.join(w.word for w in words).strip();got=tokens(transcript)
        distance=edits(expected,got)
        item['asr']=dict(model='large-v3',precision='float16',transcript=transcript,word_edits=distance,
            reference_words=len(expected),wer=distance/len(expected),final_word_match=bool(got and got[-1]==expected[-1]),
            independent_human_verification=False)
        atomic_json(OUT/'results.json',report)
        print('CHECKED',item['filename'],'word edits',distance,'final word',item['asr']['final_word_match'],flush=True)
    report.update(status='complete_listening_pending',manual_listening_complete=False)
    atomic_json(OUT/'results.json',report)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['synth','asr']);args=parser.parse_args()
    with (ROOT/'.cache/training-launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if __import__('subprocess').run(['systemctl','--user','is-active','--quiet','she-gptsovits-tts.service']).returncode==0:
            raise RuntimeError('TTS service is active; model comparison not started')
        synthesize() if args.stage=='synth' else score()
