"""Map trusted evaluation spans and score fp16 CUDA without emitting source text."""
import argparse
import csv
import difflib
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

import numpy as np
from asr_runtime import (Cache, DECODE_SETTINGS, archive, asr_context, atomic_json,
                         fingerprint, heartbeat, load_model, make_context, read_json,
                         verify_inputs)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/voiceuser/projects/she-training/SheCorpus.flac')
OLD = Path('/home/voiceuser/.hermes/cache/voices/she-xtts')
WORK = ROOT / '.cache/corpus-ground-truth'
WORK.mkdir(parents=True, exist_ok=True)
REPORT = ROOT / 'reports/corpus-ground-truth.json'
MATCHER = ROOT.parent / 'she-training/verify_same_recording6.py'

def save(path, value):
    atomic_json(path, value)


def mapping_context():
    files={'source':SOURCE,'manifest':OLD/'manifest.csv',
           'audit':ROOT/'reports/corpus-alignment-audit.json','matcher':MATCHER}
    with (OLD/'manifest.csv').open() as stream:
        references=list(csv.DictReader(stream))
    for row in references:
        if int(row['index'])<595 or int(row['index'])==620:
            files['reference/'+row['filename']]=OLD/'clips'/row['filename']
    return make_context(files, dict(mapping_version=2,source_rate=8000,positive_min=.999,
                                    negative_max=.5,match_min=.7,tail_difference_max=.06))

def tokens(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower().replace('’', "'"))

def edits(a, b):
    last = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(row[-1]+1, last[j]+1, last[j-1]+(x != y)))
        last = row
    return last[-1]

def audio(path, start, duration, rate=16000):
    command = ['ffmpeg','-nostdin','-v','error','-ss',str(start),'-i',str(path),
               '-t',str(duration),'-ac','1','-ar',str(rate),'-f','f32le','-']
    return np.frombuffer(subprocess.check_output(command), dtype='<f4')

def union_seconds(rows):
    spans = sorted((r['start_seconds'],r['end_seconds']) for r in rows)
    end = -1.; total = 0.
    for a,b in spans:
        total += max(0.,b-max(a,end)); end = max(end,b)
    return total

def mapping():
    heartbeat('mapping_provenance')
    context=mapping_context()
    previous=read_json(REPORT,{})
    if (previous.get('mapping_fingerprint')==context['fingerprint'] and
        previous.get('mapping_rows_hash')==fingerprint(previous.get('rows')) and
        previous.get('status') in ['mapped','scoring','complete']):
        heartbeat('mapping_cached',mapped_clips=previous['mapped_clips'])
        return
    cache=Cache(WORK/'mapping',context)
    cached=cache.get({'stage':'mapping'}) or {}
    cached_rows={row['index']:row for row in cached.get('rows',[])}
    spec = importlib.util.spec_from_file_location('matcher', str(MATCHER))
    matcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(matcher)
    assert matcher.self_test()
    # Reuse the verified contiguous disk decode, never whole-recording RAM decode.
    cache.directory.mkdir(parents=True,exist_ok=True)
    matcher.SHE=str(SOURCE)
    matcher.RAW=str(cache.directory/'source-8k.raw')
    raw=Path(matcher.RAW)
    if raw.exists() and raw.stat().st_size!=9772*8000*2:
        raw.rename(raw.with_name(raw.name+'.incomplete-'+str(time.time_ns())))
    heartbeat('mapping_decode')
    assert matcher.decode_whole() == 9772*8000
    data = np.memmap(matcher.RAW, dtype='<i2', mode='r')
    _, positive = matcher.scan(np.asarray(data[3000*8000:3012*8000],dtype=np.float32),data,'positive control')
    _, negative = matcher.scan(matcher.dec_clip(str(OLD/'clips/clip_00620.wav')),data,'negative control')
    assert positive > .999 and negative < .5
    audit = json.loads((ROOT/'reports/corpus-alignment-audit.json').read_text())['per_clip']
    with (OLD/'manifest.csv').open() as stream:
        available={int(r['index']):r for r in csv.DictReader(stream)}
    rows = []; cumulative = 0.; bias = 0.
    for r in sorted(audit,key=lambda x:x['index']):
        index = int(r['index']); duration = float(r['duration_seconds'])
        if index >= 595: break
        predicted = cumulative+bias; cumulative += duration
        if index in cached_rows:
            result=cached_rows[index];rows.append(result)
            if result['status']=='matched':bias=result['start_seconds']-(cumulative-duration)
            heartbeat('mapping_cached_clip',completed_clips=len(rows))
            continue
        result = {'index':index,'filename':r['filename'],'duration_seconds':duration}
        if index not in available or r['repeated_sentences'] or r['wer'] > .25:
            result['status']='excluded_audit_flag_or_removed';rows.append(result);continue
        if predicted > 9802:
            result['status']='outside_recording';rows.append(result);continue
        ref = matcher.dec_clip(str(OLD/'clips'/r['filename']))
        left = max(0,int((predicted-35)*8000)); right = min(len(data),int((predicted+duration+35)*8000))
        lag,score = matcher.best_match(ref,np.asarray(data[left:right],dtype=np.float32))
        offset = (left+lag)/8000
        if score < .7 or lag < 0:
            offset,score = matcher.scan(ref,data,r['filename'])
        result.update(start_seconds=offset,end_seconds=offset+duration,match_score=score)
        if score >= .7 and offset >= 0 and offset+duration <= 9772:
            # Verify end alignment too, not just the first twelve seconds.
            tail_start = max(0,duration-8)
            tail = audio(OLD/'clips'/r['filename'],tail_start,duration-tail_start,8000)
            pos = int((offset+tail_start)*8000)
            lo=max(0,pos-800);hi=min(len(data),pos+len(tail)+800)
            lag2,score2=matcher.best_match(tail,np.asarray(data[lo:hi],dtype=np.float32))
            tail_offset=(lo+lag2)/8000-tail_start
            result.update(tail_match_score=score2,tail_offset_difference_seconds=tail_offset-offset)
            result['status']='matched' if score2 >= .7 and abs(tail_offset-offset) <= .06 else 'tail_mismatch'
            if result['status']=='matched': bias=offset-(cumulative-duration)
        else: result['status']='outside_or_unmatched'
        rows.append(result)
        if index%20==0: print('Mapped through index',index,flush=True)
        cache.put({'stage':'mapping'},dict(positive_control=positive,negative_control=negative,rows=rows))
        heartbeat('mapping',completed_clips=len(rows))
    mapped=[r for r in rows if r['status']=='matched']
    missing=sorted(set(range(595))-{r['index'] for r in rows})
    rows.extend(dict(index=i,status='excluded_missing_audit_row') for i in missing)
    rows.sort(key=lambda r:r['index'])
    result=dict(status='mapped',decoder='large-v3 CUDA float16',positive_control=positive,negative_control=negative,
                candidate_index_range=[0,594],mapped_clips=len(mapped),covered_seconds=union_seconds(mapped),
                corpus_seconds=9772,coverage_fraction=union_seconds(mapped)/9772,rows=rows)
    result.update(missing_audit_indices=missing,excluded_missing_audit_rows=len(missing),filter_counts=dict(
        manifest_rows=len(available),audit_rows=len(audit),audit_rows_removed_from_manifest=sum(r['index'] not in available for r in audit),
        audit_repetition_flagged_global=sum(bool(r['repeated_sentences']) for r in audit),
        subset_removed_from_manifest=sum(r['index']<595 and r['index'] not in available for r in audit),
        subset_repetition_flagged=sum(r['index']<595 and bool(r['repeated_sentences']) for r in audit),
        subset_legacy_disagreement_over_025=sum(r['index']<595 and r['wer']>.25 for r in audit)))
    verify_inputs(context)
    result.update(mapping_fingerprint=context['fingerprint'],mapping_rows_hash=fingerprint(rows))
    archive(REPORT,ROOT/'reports/asr-history')
    save(REPORT,result)
    heartbeat('mapping_complete',mapped_clips=len(mapped))

def transcribe():
    result=json.loads(REPORT.read_text())
    mapping_provenance=mapping_context()
    if (result.get('mapping_fingerprint')!=mapping_provenance['fingerprint'] or
        result.get('mapping_rows_hash')!=fingerprint(result.get('rows'))):
        raise RuntimeError('Mapping provenance is missing or stale; run the map stage first')
    with (OLD/'manifest.csv').open() as stream:
        manifest={int(r['index']):r for r in csv.DictReader(stream)}
    context=asr_context(ROOT,SOURCE,normalizer_version=2,source_mix='ffmpeg-ac1')
    cache=Cache(WORK/'scores',context)
    model=None
    scored=[]
    original_scores={r['index']:r for r in result.get('scores',[])} if result.get('scoring_fingerprint')==context['fingerprint'] else {}
    archive(REPORT,ROOT/'reports/asr-history')
    for name in ['micro_wer','median_wer','mean_wer','median_legacy_sequence_disagreement']:
        result.pop(name,None)
    result.update(status='scoring',scoring_fingerprint=context['fingerprint'],scored_clips=0,scores=[])
    save(REPORT,result)
    for row in result['rows']:
        if row['status']!='matched':continue
        want=manifest[row['index']]['text']
        key=dict(index=row['index'],source_start_frame=round(row['start_seconds']*48000),
                 source_end_frame=round(row['end_seconds']*48000),
                 reference_hash=fingerprint(tokens(want)),mapping_fingerprint=result['mapping_fingerprint'])
        score=cache.get(key) or original_scores.get(row['index'])
        if score and score.get('score_key')==fingerprint(key):
            scored.append(score)
            result.update(scored_clips=len(scored),scores=scored)
            save(REPORT,result)
            heartbeat('ground_truth_cached',completed_clips=len(scored))
            continue
        if model is None:
            verify_inputs(context);model=load_model(ROOT)
        segments=[]
        for start in np.arange(0,row['duration_seconds'],30):
            signal=audio(SOURCE,row['start_seconds']+start,min(30,row['duration_seconds']-start))
            pieces,_=model.transcribe(signal,**DECODE_SETTINGS)
            segments.extend(s.text for s in pieces)
            heartbeat('ground_truth_window',completed_clips=len(scored))
        got=' '.join(segments)
        a,b=tokens(want),tokens(got); n=edits(a,b)
        # Match the historical tokenizer exactly for its separately named score.
        def old_norm(t):
            return [w.lower().strip(".,!?\"';:") for w in t.replace('—',' ').replace('-',' ').split() if w.strip(".,!?\"';:")]
        if not a:raise RuntimeError('Ground-truth reference is empty')
        score=dict(index=row['index'],reference_words=len(a),word_edits=n,wer=n/len(a),score_key=fingerprint(key),
                   legacy_sequence_disagreement=1-difflib.SequenceMatcher(None,old_norm(want),old_norm(got)).ratio())
        cache.put(key,score);scored.append(score)
        result.update(status='scoring',scored_clips=len(scored),scores=scored)
        save(REPORT,result)
        heartbeat('ground_truth',completed_clips=len(scored),total_clips=result['mapped_clips'])
        if len(scored)%20==0:print('Scored clips',len(scored),flush=True)
    verify_inputs(context);verify_inputs(mapping_provenance)
    if len(scored)!=result['mapped_clips'] or not scored:raise RuntimeError('Incomplete ground-truth scores')
    result.update(status='complete',micro_wer=sum(r['word_edits'] for r in scored)/sum(r['reference_words'] for r in scored),
                  median_wer=statistics.median(r['wer'] for r in scored),
                  mean_wer=statistics.mean(r['wer'] for r in scored),
                  median_legacy_sequence_disagreement=statistics.median(r['legacy_sequence_disagreement'] for r in scored))
    save(REPORT,result)
    heartbeat('complete',completed_clips=len(scored))
    print('Ground-truth validation complete',result['scored_clips'],result['covered_seconds'],result['micro_wer'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['map','asr']);args=parser.parse_args()
    if args.stage=='map':mapping()
    else:transcribe()
