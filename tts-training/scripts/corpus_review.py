"""Read-only corpus audit, fresh speaker screening and a portable listening queue."""
import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
import wave

import numpy as np
from asr_runtime import atomic_json, atomic_text, file_digest, fingerprint


def normalized(text):
    return ' '.join(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower().replace('’', "'")))


def repetition_flags(text):
    words=normalized(text).split();flags=[]
    for width in range(1,9):
        minimum=6 if width==1 else 3
        if any(words[start:start+width]*minimum==words[start:start+width*minimum]
               for start in range(max(0,len(words)-width*minimum+1))):
            flags.append('repeated_word_or_phrase');break
    sentences=[normalized(s) for s in re.split(r'[.!?]+',text)]
    if any(count>1 and len(s.split())>=4 for s,count in Counter(sentences).items()):
        flags.append('repeated_sentence')
    return flags


def quiet_metrics(signal,rate,threshold):
    width=rate//100;frames=signal[:len(signal)//width*width].reshape(-1,width)
    rms=np.sqrt(np.mean(frames.astype(np.float64)**2,axis=1))
    quiet=rms<threshold
    lead=next((i for i,v in enumerate(quiet) if not v),len(quiet))
    tail=next((i for i,v in enumerate(quiet[::-1]) if not v),len(quiet))
    longest=current=0
    for value in quiet[lead:len(quiet)-tail if tail else len(quiet)]:
        current=current+1 if value else 0;longest=max(longest,current)
    return dict(quiet_lead_seconds=lead*.01,quiet_tail_seconds=tail*.01,
                longest_interior_quiet_seconds=longest*.01),rms


def structural_checks(rows,spans):
    """Validate source containment and detect leaks; never change assignments."""
    errors=[];groups={key:defaultdict(list) for key in ['audio','text','filename']}
    for row in rows:
        lo,hi=row['source_start_frame'],row['source_end_frame'];partition=row['split']
        if not any(a<=lo<hi<=b for a,b in spans[partition]):
            errors.append(dict(kind='outside_frozen_split',clips=[row['id']]))
        for field,key in [('audio','audio_sha256'),('filename','filename')]:
            if row.get(key):groups[field][row[key]].append(row)
        text=normalized(row['text'])
        if text:groups['text'][text].append(row)
    duplicates=[]
    for kind,values in groups.items():
        for group in values.values():
            if len(group)<2:continue
            issue=dict(kind=kind+'_duplicate',clips=[r['id'] for r in group],
                       cross_split=len({r['split'] for r in group})>1)
            duplicates.append(issue)
            if issue['cross_split'] or kind=='filename':errors.append(issue)
    ordered=sorted(rows,key=lambda r:r['source_start_frame'])
    for index,row in enumerate(ordered):
        for other in ordered[index+1:]:
            if other['source_start_frame']>=row['source_end_frame']:break
            if other['split']!=row['split']:
                errors.append(dict(kind='cross_split_source_overlap',clips=[row['id'],other['id']]))
    return errors,duplicates


def choose_queue(rows,seed=20261113):
    chosen=[]
    for split,limit in [('train',40),('validation',10)]:
        pool=[r for r in rows if r['split']==split];limit=min(limit,len(pool));seen=set();selected=[]
        def add(items,reason,number):
            count=0
            for row in items:
                if len(selected)>=limit or count>=number:break
                if row['id'] in seen:continue
                seen.add(row['id']);copy=dict(row,selection_reason=reason)
                selected.append(copy);count+=1
        add(sorted([r for r in pool if r['hard_errors']],key=lambda r:-len(r['hard_errors'])),'integrity flag',limit)
        blocks=defaultdict(list)
        for row in pool:blocks[row.get('source_span',0)].append(row)
        for block in blocks.values():
            ordered=sorted(block,key=lambda r:r['source_start_frame'])
            add([ordered[len(ordered)//2]],'source-span coverage',1)
        flag_types=sorted({f for r in pool for f in r['flags']},key=lambda f:(not f.startswith('repeated'),f))
        for flag in flag_types:
            add(sorted([r for r in pool if flag in r['flags']],key=lambda r:-len(r['flags'])),'flag: '+flag.replace('_',' '),1)
        add(sorted([r for r in pool if r['flags']],key=lambda r:-len(r['flags'])),'automated review flag',5 if split=='train' else 1)
        add(sorted(pool,key=lambda r:-r['speaker']['distance_to_training_center']),'unusual speaker embedding',6 if split=='train' else 2)
        add(sorted(pool,key=lambda r:-r.get('pitch_deviation',0)),'pitch variation',4 if split=='train' else 1)
        add(sorted(pool,key=lambda r:r['duration_seconds']),'short clip',3 if split=='train' else 1)
        add(sorted(pool,key=lambda r:-r['duration_seconds']),'long clip',3 if split=='train' else 1)
        chronological=sorted(pool,key=lambda r:r['source_start_frame'])
        positions=np.linspace(0,max(0,len(pool)-1),4 if split=='train' else 2,dtype=int)
        add([chronological[i] for i in positions] if pool else [],'source-span coverage',len(positions))
        shuffled=pool.copy();random.Random(seed+(0 if split=='train' else 1)).shuffle(shuffled)
        add(shuffled,'seeded random coverage',limit)
        chosen.extend(selected)
    return chosen


def screen_speakers(root,rows,out,status):
    import torch
    import torchaudio
    import librosa
    sys.path.insert(0,str(root/'upstream/GPT_SoVITS/eres2net'))
    from ERes2NetV2 import ERes2NetV2
    import kaldi
    checkpoint=root/'upstream/GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt'
    model_hash=file_digest(checkpoint)['sha256']
    provenance=dict(model_sha256=model_hash,method='ERes2NetV2.forward 192-dimensional speaker embedding',
                    preprocessing='32k PCM / 32768; resample16k; 80-bin Kaldi fbank; dither0',
                    precision='float16',device='cuda')
    model=resample=None
    cache=root/'.cache/corpus-review'/fingerprint(provenance);cache.mkdir(parents=True,exist_ok=True)
    matrix=[];cache_hits=0
    for index,row in enumerate(rows):
        path=cache/(row['audio_sha256']+'.json')
        saved=json.loads(path.read_text()) if path.exists() else None
        if saved and saved.get('audio_sha256')==row['audio_sha256'] and len(saved.get('embedding',[]))==192:
            cache_hits+=1
            embedding=np.asarray(saved['embedding'],dtype=np.float32);pitch=saved['pitch_median_hz']
        else:
            if model is None:
                if not torch.cuda.is_available():raise RuntimeError('CUDA required; no silent precision/device fallback')
                model=ERes2NetV2(baseWidth=24,scale=4,expansion=4)
                model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
                model=model.eval().half().cuda()
                resample=torchaudio.transforms.Resample(32000,16000).cuda()
            with wave.open(str(root/row['filename'])) as stream:
                audio=np.frombuffer(stream.readframes(stream.getnframes()),dtype='<i2').astype(np.float32)/32768
            with torch.inference_mode():
                wav=resample(torch.from_numpy(audio).unsqueeze(0).cuda())
                features=kaldi.fbank(wav,num_mel_bins=80,sample_frequency=16000,dither=0)
                embedding=model(features.unsqueeze(0).half()).float().cpu().numpy()[0]
            assert embedding.shape==(192,) and np.isfinite(embedding).all()
            # Descriptive pitch only: quiet frames excluded; YIN can make octave errors.
            small=librosa.resample(audio,orig_sr=32000,target_sr=8000)
            f0=librosa.yin(small,fmin=60,fmax=500,sr=8000,frame_length=512,hop_length=160)
            rms=librosa.feature.rms(y=small,frame_length=512,hop_length=160)[0]
            voiced=f0[:len(rms)][rms[:len(f0)]>.02]
            pitch=float(np.median(voiced)) if len(voiced) else None
            atomic_json(path,dict(audio_sha256=row['audio_sha256'],embedding=embedding.tolist(),pitch_median_hz=pitch))
        norm=np.linalg.norm(embedding)
        if not np.isfinite(embedding).all() or not np.isfinite(norm) or norm<1e-8:raise RuntimeError('Invalid speaker embedding')
        matrix.append(embedding/norm);row['pitch_median_hz']=pitch
        if index%25==0:
            status('speaker_screening',completed_clips=index+1,total_clips=len(rows))
            print('Speaker screening',index+1,'/',len(rows),flush=True)
    del model,resample;torch.cuda.empty_cache()
    matrix=np.asarray(matrix);mask=np.asarray([r['split']=='train' for r in rows]);train=matrix[mask]
    center=train.mean(axis=0);center/=np.linalg.norm(center)
    distance=1-matrix@center
    median=float(np.median(distance[mask]));mad=float(np.median(abs(distance[mask]-median)))
    threshold=median+3*1.4826*max(mad,1e-6)
    pitch_train=np.asarray([r['pitch_median_hz'] for r in rows if r['split']=='train' and r['pitch_median_hz'] is not None])
    pmed=float(np.median(pitch_train));pmad=float(np.median(abs(pitch_train-pmed)))
    for i,row in enumerate(rows):
        row['speaker']=dict(distance_to_training_center=float(distance[i]),embedding_outlier=bool(distance[i]>threshold))
        pitch=row['pitch_median_hz'];deviation=abs(pitch-pmed) if pitch else 0
        row['pitch_deviation']=deviation
        if distance[i]>threshold:row['flags'].append('speaker_embedding_outlier')
        if pitch is None:row['flags'].append('pitch_unmeasurable')
        elif deviation>3*1.4826*max(pmad,1):row['flags'].append('pitch_outlier')
    return dict(status='complete_listening_pending',provenance=provenance,
                cached_embeddings_used=cache_hits,embeddings_computed_this_invocation=len(rows)-cache_hits,
                thresholds_fitted_on='training_only',distance_median=median,distance_mad=mad,
                distance_review_threshold=threshold,pitch_median_hz=pmed,pitch_mad_hz=pmad,
                flagged_embedding_outliers=sum(r['speaker']['embedding_outlier'] for r in rows),
                pitch_warning='RMS-gated YIN is descriptive and can make octave errors.',
                interpretation='Variation flags are review priorities, not speaker identities or automatic exclusions.')


def run(root,out):
    started=time.time();out.mkdir(parents=True,exist_ok=True)
    def status(phase,**values):atomic_json(out/'status.json',dict(phase=phase,updated=time.time(),**values))
    report_path=root/'reports/corpus-verification.json';verification=json.loads(report_path.read_text())
    assert verification['corpus_built']
    protected=[root/'reports/corpus-split.json',root/'reports/corpus-pause-inventory.json',report_path]
    protected += [root/f'data/shecorpus/{split}/{name}' for split in ['train','validation'] for name in ['manifest.csv','she.list']]
    before={str(p.relative_to(root)):file_digest(p)['sha256'] for p in protected}
    split=json.loads(protected[0].read_text());inventory=json.loads(protected[1].read_text())
    threshold=inventory['thresholds']['relative_2pct'];rows=[];errors=[]
    status('audio_checks')
    for partition in ['train','validation']:
        with (root/f'data/shecorpus/{partition}/manifest.csv').open() as stream:
            manifest=list(csv.DictReader(stream))
        expected=[]
        for r in manifest:
            row=dict(r,id=partition+'-'+str(r['index']),split=partition,flags=[],hard_errors=[])
            for field in ['source_start_frame','source_end_frame']:row[field]=int(r[field])
            row['source_span']=next((i for i,(a,b) in enumerate(split['source_frames'][partition])
                                     if a<=row['source_start_frame']<row['source_end_frame']<=b),-1)
            row['duration_seconds']=float(r['duration_seconds'])
            path=(root/r['filename']).resolve()
            if not path.is_relative_to((root/'data/shecorpus').resolve()):raise RuntimeError('Audio outside corpus root')
            expected.append(f"{root/r['filename']}|she|en|{r['text']}")
            try:
                with wave.open(str(path)) as stream:
                    params=stream.getparams()
                    if (params.nchannels,params.sampwidth,params.framerate,params.comptype)!=(1,2,32000,'NONE'):
                        raise ValueError('Expected 32kHz mono PCM16')
                    raw=stream.readframes(params.nframes)
                    if len(raw)!=params.nframes*2 or not raw:raise ValueError('Incomplete/empty WAV')
                samples=np.frombuffer(raw,dtype='<i2');audio=samples.astype(np.float32)/32768
                actual=len(audio)/32000
                if abs(actual-row['duration_seconds'])>1/32000+1e-6:row['hard_errors'].append('duration_mismatch')
                if (int(r['sample_rate']),int(r['channels']))!=(32000,1):row['hard_errors'].append('manifest_format_mismatch')
                source_duration=(row['source_end_frame']-row['source_start_frame'])/48000
                if abs(actual-source_duration)>1/32000+1/48000:row['hard_errors'].append('source_duration_mismatch')
                row['duration_seconds']=actual;row['audio_sha256']=hashlib.sha256(raw).hexdigest()
                row['file_sha256']=file_digest(path)['sha256']
                quiet,_=quiet_metrics(audio,32000,threshold);row.update(quiet)
                row['peak_amplitude']=float(np.max(abs(audio)))
                row['clipped_samples']=int(np.sum((samples==32767)|(samples==-32768)))
                row['dc_offset']=float(np.mean(audio))
                if quiet['quiet_lead_seconds']<.1:row['flags'].append('short_leading_quiet')
                if quiet['quiet_tail_seconds']<.15:row['flags'].append('short_trailing_quiet')
                if quiet['longest_interior_quiet_seconds']>1.5:row['flags'].append('long_interior_quiet')
                if row['clipped_samples']:row['flags'].append('full_scale_samples')
                if abs(row['dc_offset'])>.01:row['flags'].append('dc_offset')
                row['words_per_second']=len(normalized(row['text']).split())/actual
                if row['words_per_second']<.5 or row['words_per_second']>5:row['flags'].append('unusual_word_rate')
                row['flags'].extend(repetition_flags(row['text']))
                if not normalized(row['text']):row['hard_errors'].append('empty_text')
                if re.search(r'<\||\[|\]',row['text']):row['flags'].append('annotation_artifact')
                if r['split']!=partition:row['hard_errors'].append('manifest_split_mismatch')
                if not 3<=actual<=15:row['flags'].append('duration_outside_design')
                for issue in row['hard_errors']:errors.append(dict(kind=issue,clips=[row['id']]))
            except (OSError,EOFError,wave.Error,ValueError) as error:
                row['hard_errors'].append('unreadable_or_invalid_audio')
                errors.append(dict(kind='unreadable_or_invalid_audio',clips=[row['id']],detail=str(error)))
            rows.append(row)
        actual=(root/f'data/shecorpus/{partition}/she.list').read_text().splitlines()
        if actual!=expected:errors.append(dict(kind='list_manifest_mismatch',split=partition))
    structural,duplicates=structural_checks(rows,split['source_frames']);errors.extend(structural)
    for issue in structural:
        for row in rows:
            if row['id'] in issue['clips']:row['hard_errors'].append(issue['kind'])
    for issue in duplicates:
        for row in rows:
            if row['id'] in issue['clips']:row['flags'].append(issue['kind'])
    if len(rows)!=verification['clips']:errors.append(dict(kind='verification_count_mismatch'))
    atomic_json(out/'audio-audit.json',dict(errors=errors,duplicates=duplicates,clips=len(rows),rows=rows))
    if any('unreadable_or_invalid_audio' in r['hard_errors'] for r in rows):
        raise RuntimeError('Invalid WAVs; inspect audio-audit.json')
    speakers=screen_speakers(root,rows,out,status)
    queue=choose_queue(rows);audio_dir=out/'audio';context_dir=out/'context'
    audio_dir.mkdir(exist_ok=True);context_dir.mkdir(exist_ok=True)
    status('packaging',selected_clips=len(queue))
    ffmpeg=shutil.which('ffmpeg') or '/home/voiceuser/.hermes/tools/ffmpeg-9.0.1-linux-x64/bin/ffmpeg'
    for row in queue:
        filename=row['id']+'.wav';shutil.copy2(root/row['filename'],audio_dir/filename)
        row['audio_url']='audio/'+filename
        start=max(0,row['source_start_frame']/48000-1);end=min(9772,row['source_end_frame']/48000+1)
        subprocess.run([ffmpeg,'-nostdin','-v','error','-y','-ss',str(start),'-i',
                        '/home/voiceuser/projects/she-training/SheCorpus.flac','-t',str(end-start),
                        '-af','pan=mono|c0=0.5*c0+0.5*c1','-ar','32000','-c:a','pcm_s16le',str(context_dir/filename)],check=True)
        row['context_url']='context/'+filename;row['context_clip_start_seconds']=row['source_start_frame']/48000-start
    after={str(p.relative_to(root)):file_digest(p)['sha256'] for p in protected}
    if before!=after:raise RuntimeError('Corpus inputs changed during review; do not use this package')
    summary=dict(schema=1,status='automated_checks_complete_listening_pending',corpus_fingerprint=verification['run_fingerprint'],
        created=time.time(),elapsed_seconds=time.time()-started,clips=len(rows),seconds=sum(r['duration_seconds'] for r in rows),
        split_counts=dict(Counter(r['split'] for r in rows)),hard_errors=errors,duplicates=duplicates,
        flagged_clip_count=sum(bool(r['flags']) for r in rows),flag_counts=dict(Counter(f for r in rows for f in r['flags'])),
        speaker_screening=speakers,selected_for_listening=len(queue),selection_seed=20261113,
        selection_counts=dict(Counter(r['split'] for r in queue)),protected_sha256=before,corpus_unchanged=True,
        manual_review_complete=False,automatically_excluded_clips=0,
        limitations=['Same-model transcript agreement does not prove accuracy.',
                     'Boundary silence does not prove first/last words are complete.',
                     'Identical text/PCM checks do not detect all near duplicates.',
                     'Voice outlier flags require human interpretation.'])
    atomic_json(out/'summary.json',summary);atomic_json(out/'all-clips.json',rows)
    payload=dict(schema=1,corpus_fingerprint=verification['run_fingerprint'],summary=summary,clips=queue)
    atomic_json(out/'queue.json',payload)
    template=Path(__file__).with_name('corpus_review.html').read_text()
    encoded=json.dumps(payload,ensure_ascii=True).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    atomic_text(out/'index.html',template.replace('__REVIEW_PAYLOAD__',encoded))
    atomic_text(out/'README.txt','Open index.html in a browser. Start with the 50-clip queue.\n'
        'Listen, compare the transcript, and mark Keep / Reject / Correct. Use source context for boundaries.\n'
        'Decisions autosave in this browser when supported. Export JSON before closing or moving the folder.\n'
        'Import restores a matching export. Nothing edits the training corpus automatically.\n'
        'The full automated report is summary.json; all-clips.json contains every clip measurement.\n')
    status('complete',selected_clips=len(queue),hard_errors=len(errors))
    print(json.dumps({k:summary[k] for k in ['status','clips','hard_errors','flag_counts','selected_for_listening','elapsed_seconds']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    try:
        with (args.root/'.cache/training-launch.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if subprocess.run(['systemctl','--user','is-active','--quiet','she-gptsovits-tts.service']).returncode==0:
                raise RuntimeError('TTS is active; review did not load a model')
            run(args.root,args.out)
    except BaseException as error:
        atomic_json(args.out/'status.json',dict(phase='failed',error=str(error),updated=time.time()));raise
