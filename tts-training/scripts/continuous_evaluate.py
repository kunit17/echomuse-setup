"""Measure generated WER and punctuation quiet runs, including unsplit output."""
import json
from pathlib import Path
import re
import statistics

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from corpus_ground_truth import ROOT, edits, tokens, save
from corpus_audio_metrics import frame_rms, longest_quiet as longest, quiet_between, quiet_at

report=json.loads((ROOT/'reports/continuous-synthesis.json').read_text())
model=WhisperModel(str(ROOT/'.cache/whisper-large-v3'),device='cuda',compute_type='float16',cpu_threads=0,num_workers=1,local_files_only=True)

def align(a,b):
    table=[list(range(len(b)+1))]
    for i,x in enumerate(a,1):
        row=[i]
        for j,y in enumerate(b,1):row.append(min(row[-1]+1,table[-1][j]+1,table[-1][j-1]+(x!=y)))
        table.append(row)
    mapping={};i,j=len(a),len(b)
    while i or j:
        if i and j and table[i][j]==table[i-1][j-1]+(a[i-1]!=b[j-1]):
            if a[i-1]==b[j-1]:mapping[i-1]=j-1
            i-=1;j-=1
        elif i and table[i][j]==table[i-1][j]+1:i-=1
        else:j-=1
    return mapping

results=[]
for item in report['samples']:
    segments,_=model.transcribe(item['path'],language='en',task='transcribe',beam_size=5,temperature=0,
        condition_on_previous_text=False,vad_filter=False,word_timestamps=True)
    words=[w for s in segments for w in (s.words or [])]
    got=''.join(w.word for w in words);a=tokens(item['text']);b=[];timings=[]
    for w in words:
        for token in tokens(w.word):b.append(token);timings.append((w.start,w.end))
    errors=edits(a,b);mapping=align(a,b)
    signal,sr=sf.read(item['path'],dtype='float32');assert sr==32000 and signal.ndim==1 and len(signal)>0
    rms=frame_rms(signal,sr)
    thresholds={'minus_40_dbfs':.01,'relative_2pct':float(np.max(np.abs(signal)))*.02}
    row=dict(name=item['name'],group=item['group'],text=item['text'],transcript=got,wer=errors/max(1,len(a)),word_edits=errors,reference_words=len(a),
        final_word_match=bool(b and a[-1]==b[-1]),duration_seconds=len(signal)/sr,
        quiet_tail_seconds={key:longest(rms[next((i for i in range(len(rms)-1,-1,-1) if rms[i]>=threshold),-1)+1:],threshold) for key,threshold in thresholds.items()})
    if item['group'] in ('punctuation','five-unsplit'):
        punctuation=[]
        for match in re.finditer(r'[,.;]',item['text']):
            before=tokens(item['text'][:match.start()]);i=len(before)-1
            if i+1>=len(a):continue
            mark={',':'comma','.':'period',';':'semicolon'}[match.group()]
            if i not in mapping or i+1 not in mapping:
                punctuation.append(dict(mark=mark,status='unaligned_due_to_word_edits'));continue
            left=timings[mapping[i]][1];right=timings[mapping[i+1]][0]
            punctuation.append(dict(mark=mark,status='asr_timestamp_alignment',preceding_word_end=left,following_word_start=right,
                interword_seconds=right-left,longest_quiet_seconds={key:quiet_between(rms,left,right,threshold) for key,threshold in thresholds.items()}))
        row['punctuation']=punctuation
    results.append(row);save(ROOT/'reports/continuous-evaluation.json',dict(status='running',samples=results))
novel=[r for r in results if r['group']=='novel'];termination=[r for r in results if r['group']=='termination']
punctuation=next(r for r in results if r['group']=='punctuation')
measured={r['mark']:r.get('longest_quiet_seconds') for r in punctuation['punctuation']}
limits={'comma':.08,'period':.15,'semicolon':.1}
pauses_pass=all(measured.get(key) is not None and all(value>=limits[key] for value in measured[key].values()) for key in limits)
application=report['application_segmentation']
signal,sr=sf.read(application['path'],dtype='float32');assert sr==32000
rms=frame_rms(signal,sr)
thresholds={'minus_40_dbfs':.01,'relative_2pct':float(np.max(np.abs(signal)))*.02}
gaps=[]
for a,b in application['gap_intervals']:
    measured_gap={}
    for key,threshold in thresholds.items():
        measured_gap[key]=quiet_at(rms,int((a+b)*50),threshold)
    gaps.append(dict(inserted_start_seconds=a,inserted_end_seconds=b,measured_quiet_seconds=measured_gap))
application['measured_gaps']=gaps
final=dict(status='complete',gpt_checkpoint=report['gpt_checkpoint'],sovits_checkpoint=report['sovits_checkpoint'],samples=results,
    novel_micro_wer=sum(r['word_edits'] for r in novel)/sum(r['reference_words'] for r in novel),
    novel_exact_match_count=sum(r['word_edits']==0 for r in novel),termination_zero_edits=all(r['word_edits']==0 for r in termination),
    punctuation_quiet_seconds=measured,punctuation_thresholds_met=pauses_pass,
    baseline_punctuation_quiet_seconds=dict(comma=0.,period=0.,semicolon=0.),
    application_segmentation=application,
    listening_review_complete=False,asr_alignment_limitation='Word timestamps are model estimates; ambiguous boundaries require listening.',
    defects_fixed='unverified_pending_listening' if pauses_pass and all(r['word_edits']==0 for r in termination) else False,
    live_provider_changed=False)
save(ROOT/'reports/continuous-evaluation.json',final)
print('Measured punctuation quiet seconds',json.dumps(measured),flush=True)
