"""Source-first corpus rebuild with acoustic pause cuts and independent CTC QA."""
import argparse,collections,csv,hashlib,html,io,json,math,os,re,sys,time,unicodedata
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT=Path('/home/voiceuser/projects/gpt-sovits')
sys.path.insert(0,str(ROOT/'scripts'))
from asr_runtime import atomic_json,atomic_text,file_digest,fingerprint,heartbeat
RECIPE_PATH=Path(os.environ.get('CORPUS_REPAIR_RECIPE',str(Path(__file__).with_name('recipe.json')))).resolve()
RECIPE=json.loads(RECIPE_PATH.read_text())
VERSION=os.environ.get('CORPUS_REPAIR_VERSION','boundary-v1-20261115')
if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}',VERSION):
    raise ValueError('Invalid CORPUS_REPAIR_VERSION')
WORK=ROOT/'.cache/repaired-corpus'/VERSION
WORK.mkdir(parents=True,exist_ok=True)
OUT=ROOT/'data/shecorpus/repaired'/VERSION
STATUS=ROOT/'reports/corpus-repair-status.json'
SOURCES={name:Path(item['path']) for name,item in RECIPE['sources'].items()}
POLICY=dict(quiet_seconds=.30,edge_seconds=.12,vad_padding_seconds=.08,
    vad_threshold=.35,quiet_rms_floor=.0005,quiet_rms_ceiling=.004,
    quiet_speech_ratio=.025,preferred_seconds=8,max_seconds=35,
    alignment_first_last_score=.35,alignment_word_score=.2,
    independent_max_word_error_rate=.15,minimum_aligned_start_margin=.10,
    minimum_aligned_end_margin=.12,
    partition_guard_seconds=5,previous_owner_exclusions_preserved=True)

def validate_recipe(check_sources=True):
    assert RECIPE['schema']==1 and RECIPE['policy']==POLICY, 'Recipe policy differs from implemented policy'
    for source,path in SOURCES.items():
        if check_sources:
            assert file_digest(path)['sha256']==RECIPE['sources'][source]['sha256'], 'Source changed; create a reviewed recipe for '+source
        groups=RECIPE['partitions'][source]
        spans=[]
        for split in ['train','validation']:
            for lo,hi in groups[split]:
                assert 0<=lo<hi
                spans.append((lo,hi))
        spans.sort()
        assert all(a[1]<=b[0] for a,b in zip(spans,spans[1:])), 'Source partitions overlap'
        assert all(0<=lo<hi for lo,hi in RECIPE['exclusions'][source])
    if (WORK/'plan.json').exists():
        old=json.loads((WORK/'plan.json').read_text())
        assert old['partitions']==RECIPE['partitions'] and old['exclusions']==RECIPE['exclusions'], 'Use a new version for a changed recipe'
        assert all(old['sources'][s]['sha256']==RECIPE['sources'][s]['sha256'] for s in SOURCES), 'Cached source mismatch'
        assert all(POLICY[k]==v for k,v in old['policy'].items()), 'Cached policy mismatch'
        if 'recipe_sha256' in old:
            assert old['recipe_sha256']==file_digest(RECIPE_PATH)['sha256'], 'Use a new version for a changed recipe'
    return True

def progress(stage,**kw):
    value=json.loads(STATUS.read_text()) if STATUS.exists() else {}
    value.update(status=stage,updated=time.time(),pid=os.getpid(),training_started=False,**kw)
    atomic_json(STATUS,value);heartbeat(stage,**kw)

def rows(path):
    with open(path,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def clean(t):return ' '.join(unicodedata.normalize('NFC',t).replace('’',"'").split())
def words(t):return re.findall(r"[a-z']+",clean(t).lower())

def read(source,a,b,rate=16000):
    path=SOURCES[source];info=sf.info(path)
    lo=max(0,round(a*info.samplerate));hi=min(info.frames,round(b*info.samplerate))
    with sf.SoundFile(path) as stream:
        stream.seek(lo);x=stream.read(hi-lo,dtype='float32',always_2d=True).mean(axis=1)
    divisor=math.gcd(info.samplerate,rate)
    if info.samplerate!=rate:x=resample_poly(x,rate//divisor,info.samplerate//divisor)
    assert np.isfinite(x).all()
    return x

def rms_frames(x,rate=16000):
    width=rate//100;n=len(x)//width
    return np.sqrt(np.mean(x[:n*width].reshape(n,width).astype(np.float64)**2,axis=1))

def runs(mask):
    changes=np.diff(np.r_[False,mask,False].astype(np.int8))
    return [(a/100,b/100) for a,b in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1))]

def merge(intervals):
    result=[]
    for a,b in sorted(intervals):
        if result and a<=result[-1][1]:result[-1][1]=max(result[-1][1],b)
        else:result.append([a,b])
    return result

def subtract(spans,excluded):
    for a,b in excluded:
        after=[]
        for lo,hi in spans:
            if b<=lo or a>=hi:after.append([lo,hi]);continue
            if lo<a:after.append([lo,a])
            if hi>b:after.append([b,hi])
        spans=after
    return spans

def exclusions():
    return RECIPE['exclusions']

def partitions():
    return RECIPE['partitions']

def scan():
    validate_recipe()
    from faster_whisper.vad import get_speech_timestamps,VadOptions
    context=dict(policy=POLICY,sources={s:file_digest(p) for s,p in SOURCES.items()},
                 partitions=partitions(),exclusions=exclusions(),recipe_sha256=file_digest(RECIPE_PATH)['sha256'])
    atomic_json(WORK/'plan.json',context)
    atomic_json(WORK/'recipe.json',RECIPE)
    for source,path in SOURCES.items():
        cache=WORK/(source+'-acoustics.json')
        if cache.exists():continue
        duration=sf.info(path).duration;rms=[];speech=[]
        for core in range(0,math.ceil(duration),60):
            end=min(duration,core+60);left=max(0,core-1);right=min(duration,end+1)
            x=read(source,left,right)
            found=get_speech_timestamps(x,VadOptions(threshold=.35,min_speech_duration_ms=0,
                min_silence_duration_ms=150,speech_pad_ms=0),sampling_rate=16000)
            for v in found:
                a=max(core,left+v['start']/16000-.08);b=min(end,left+v['end']/16000+.08)
                if b>a:speech.append([a,b])
            y=read(source,core,end);rms.extend(rms_frames(y).tolist())
            progress('acoustic_scan',source=source,through_seconds=end,source_seconds=duration)
        rms=np.asarray(rms);threshold=max(.0005,min(.004,float(np.quantile(rms,.95))*.025))
        quiet=rms<threshold
        for a,b in merge(speech):quiet[max(0,math.floor(a*100)):min(len(quiet),math.ceil(b*100))]=False
        gaps=[(a,b) for a,b in runs(quiet) if b-a>=.3-1e-6]
        assert gaps,source+' has no verified acoustic pauses'
        atomic_json(cache,dict(source=source,duration=duration,quiet_threshold=threshold,
            gaps=gaps,speech=merge(speech),eligible_pause_count=len(gaps)))
        print('SCANNED',source,'safe pauses',len(gaps),'threshold',threshold,flush=True)
    candidates=[];quarantine=[]
    for source in SOURCES:
        acoustics=json.loads((WORK/(source+'-acoustics.json')).read_text())
        for split,intervals in context['partitions'][source].items():
            allowed=subtract(intervals,context['exclusions'][source])
            for lo,hi in allowed:
                gaps=[(a,b) for a,b in acoustics['gaps'] if a>=lo and b<=hi]
                cursor=0
                while cursor<len(gaps)-1:
                    first=gaps[cursor];start=first[1]-min(.25,(first[1]-first[0])/2)
                    selected=None;short=None
                    for i in range(cursor+1,len(gaps)):
                        a,b=gaps[i];end=a+min(.25,(b-a)/2);seconds=end-start
                        if seconds>=3:short=i
                        if seconds>=8:
                            selected=i;break
                    if selected is None:selected=short
                    if selected is None:break
                    last=gaps[selected];end=last[0]+min(.25,(last[1]-last[0])/2)
                    item=dict(source=source,split=split,start_frame=round(start*32000),
                        end_frame=round(end*32000),duration_seconds=round(end-start,6),
                        left_quiet_interval=first,right_quiet_interval=last,
                        quiet_threshold=acoustics['quiet_threshold'])
                    item['id']=source+'-'+fingerprint(item)[:16]
                    if end-start<=35:candidates.append(item)
                    else:quarantine.append(dict(item,reason='no_safe_pause_within_35_seconds'))
                    cursor=selected
    assert candidates
    atomic_json(WORK/'candidates.json',candidates);atomic_json(WORK/'segmentation-quarantine.json',quarantine)
    # Pilot covers the reported region and examples from each source/split.
    pilot=[]
    for source in SOURCES:
        available=[r for r in candidates if r['source']==source]
        pilot.extend(available[:3]);pilot.extend(available[len(available)//2:len(available)//2+2])
    near=[r for r in candidates if r['source']=='additional' and r['start_frame']/32000<1238.44 and r['end_frame']/32000>1227.59]
    pilot.extend(near)
    pilot=list({r['id']:r for r in pilot}.values())
    atomic_json(WORK/'pilot.json',pilot)
    progress('segmented',candidate_clips=len(candidates),candidate_hours=sum(r['duration_seconds'] for r in candidates)/3600,
             pilot_clips=len(pilot),oversize_quarantined=len(quarantine))
    print('SEGMENTED',len(candidates),'candidates',len(pilot),'pilot clips',flush=True)

def edge_check(x,threshold):
    rms=rms_frames(x,32000);n=12
    quiet=rms<threshold
    leading=next((i for i,v in enumerate(quiet) if not v),len(quiet))*.01
    trailing=next((i for i,v in enumerate(quiet[::-1]) if not v),len(quiet))*.01
    return dict(leading_quiet=bool(len(rms)>=n and np.all(rms[:n]<threshold)),
        trailing_quiet=bool(len(rms)>=n and np.all(rms[-n:]<threshold)),
        leading_quiet_seconds=leading,trailing_quiet_seconds=trailing,
        leading_rms_max=float(np.max(rms[:n])),trailing_rms_max=float(np.max(rms[-n:])),
        threshold=threshold)

def selected(pilot):return json.loads((WORK/('pilot.json' if pilot else 'candidates.json')).read_text())

def transcribe(pilot):
    from asr_runtime import load_model,DECODE_SETTINGS
    model=load_model(ROOT);items=selected(pilot)
    for i,item in enumerate(items):
        destination=WORK/'transcripts'/(item['id']+'.json')
        if not destination.exists():
            x=read(item['source'],item['start_frame']/32000,item['end_frame']/32000)
            segments,_=model.transcribe(x,**DECODE_SETTINGS);segments=list(segments)
            text=clean(' '.join(s.text.strip() for s in segments))
            aligned=[dict(word=w.word,start=float(w.start),end=float(w.end),probability=float(w.probability)) for s in segments for w in (s.words or [])]
            atomic_json(destination,dict(text=text,whisper_words=aligned,
                normalized=words(text),source_item=item,model='large-v3',compute_type='float16'))
        progress('pilot_transcribing' if pilot else 'transcribing',done=i+1,total=len(items))
        if i%25==0:print('TRANSCRIBED',i+1,'/',len(items),flush=True)

def normalize_ctc(text):
    vendor=WORK.parent/'vendor'
    sys.path.insert(0,str(vendor))
    from num2words import num2words
    def number(match):
        try:return num2words(match.group())
        except Exception:raise ValueError('Unnormalizable number '+match.group())
    text=re.sub(r'\d+(?:\.\d+)?',number,clean(text).lower())
    text=unicodedata.normalize('NFKD',text).encode('ascii','ignore').decode()
    text=re.sub('[^a-z\' ]',' ',text)
    return ' '.join(text.split()).upper()

def align(pilot):
    import torch,torchaudio
    torch.hub.set_dir(str(WORK.parent/'torch-hub'))
    bundle=torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model=bundle.get_model().to('cuda').eval();labels=bundle.get_labels();dictionary={c:i for i,c in enumerate(labels)}
    items=selected(pilot);passed=0;held=0
    for i,item in enumerate(items):
        destination=WORK/'alignment'/(item['id']+'.json')
        if destination.exists():
            previous=json.loads(destination.read_text())
            passed+=int(previous['automatic_pass']);held+=int(not previous['automatic_pass'])
            continue
        transcript=json.loads((WORK/'transcripts'/(item['id']+'.json')).read_text())
        x=read(item['source'],item['start_frame']/32000,item['end_frame']/32000,32000)
        edges=edge_check(x,item['quiet_threshold']);flags=[]
        if not edges['leading_quiet']:flags.append('leading_edge_not_quiet')
        if not edges['trailing_quiet']:flags.append('trailing_edge_not_quiet')
        if not transcript['text']:flags.append('empty_transcript')
        if re.search(r'<\||\[|\]',transcript['text']):flags.append('annotation_artifact')
        if np.max(np.abs(x))>=.999:flags.append('near_clipping')
        normalized='';aligned_words=[];greedy='';scores=[];margins={};distance=None;wer=None
        try:
            normalized=normalize_ctc(transcript['text']);assert normalized
            pieces=normalized.split();target='|'.join(pieces)
            targets=torch.tensor([[dictionary[c] for c in target]],dtype=torch.int32,device='cuda')
            audio=torch.from_numpy(resample_poly(x,1,2)).unsqueeze(0).to('cuda')
            with torch.inference_mode():
                emission,_=model(audio);emission=torch.log_softmax(emission,dim=-1)
                alignment,log_scores=torchaudio.functional.forced_align(emission,targets,blank=0)
                spans=torchaudio.functional.merge_tokens(alignment[0].cpu(),log_scores[0].exp().cpu(),blank=0)
            assert len(spans)==len(target)
            ratio=(len(x)/32000)/emission.shape[1]
            position=0
            for piece in pieces:
                chars=spans[position:position+len(piece)]
                score=float(sum(float(c.score) for c in chars)/len(chars));scores.append(score)
                aligned_words.append(dict(word=piece,start=chars[0].start*ratio,end=chars[-1].end*ratio,score=score))
                position+=len(piece)+1
            margins=dict(first_word_seconds=aligned_words[0]['start'],last_word_seconds=len(x)/32000-aligned_words[-1]['end'])
            if min(scores[0],scores[-1])<.35:flags.append('weak_first_or_last_word_alignment')
            if sum(s<.2 for s in scores)>max(1,len(scores)*.1):flags.append('weak_word_alignment')
            if margins['first_word_seconds']<.10 or margins['last_word_seconds']<.12:flags.append('aligned_speech_too_close_to_edge')
            ids=emission.argmax(-1)[0].cpu().tolist();collapsed=[v for j,v in enumerate(ids) if v and (j==0 or v!=ids[j-1])]
            greedy=''.join(labels[v] for v in collapsed).replace('|',' ').strip()
            independent=words(greedy);expected=words(normalized)
            from corpus_ground_truth import edits
            distance=edits(expected,independent);wer=distance/max(1,len(expected))
            if not independent or independent[0]!=expected[0] or independent[-1]!=expected[-1]:flags.append('independent_edge_word_disagreement')
            if wer>.15:flags.append('independent_transcript_disagreement')
        except Exception as error:
            flags.append('alignment_failed');margins['error']=str(error)
        evidence=dict(item=item,transcript=transcript['text'],normalized=normalized,edges=edges,
            words=aligned_words,independent_ctc_transcript=greedy,margins=margins,
            independent_word_edits=distance,independent_word_error_rate=wer,
            flags=sorted(set(flags)),automatic_pass=not flags,
            alignment_model='WAV2VEC2_ASR_BASE_960H',manual_listening_complete=False)
        atomic_json(destination,evidence)
        passed+=int(not flags);held+=int(bool(flags))
        progress('pilot_aligning' if pilot else 'aligning',done=i+1,total=len(items),
                 automatic_pass_clips=passed,quarantined_so_far=held)
        if i%25==0:print('ALIGNED',i+1,'/',len(items),flush=True)

def pilot_report():
    items=selected(True);folder=OUT/'pilot';folder.mkdir(parents=True,exist_ok=True);cards=[];passed=0
    for item in items:
        evidence=json.loads((WORK/'alignment'/(item['id']+'.json')).read_text())
        passed+=evidence['automatic_pass'];name=item['id'];a=item['start_frame']/32000;b=item['end_frame']/32000
        sf.write(folder/(name+'.wav'),read(item['source'],a,b,32000),32000,subtype='PCM_16')
        sf.write(folder/(name+'-context.wav'),read(item['source'],max(0,a-1.5),min(sf.info(SOURCES[item['source']]).duration,b+1.5),32000),32000,subtype='PCM_16')
        text=html.escape(evidence['transcript']);flags=html.escape(', '.join(evidence['flags']) or 'Automatic acoustic/alignment checks passed')
        cards.append(f'<article><h2>{name}</h2><p>{text}</p><p>{flags}</p><p>New cut</p><audio controls src="{name}.wav"></audio><p>Original source with 1.5 seconds of context on each side</p><audio controls src="{name}-context.wav"></audio></article>')
    page='<!doctype html><html><meta charset="utf-8"><title>Boundary repair pilot</title><style>body{font:17px/1.6 system-ui;max-width:1000px;margin:auto;padding:24px;background:#f2f4ef}article{background:white;padding:24px;margin:20px 0;border-radius:12px}audio{width:100%}</style><h1>Boundary repair pilot</h1><p>New cuts versus source context. Automatic checks are not a human listening review. Includes the region of reported clip 111.</p>'+''.join(cards)+'</html>'
    atomic_text(folder/'index.html',page)
    atomic_json(WORK/'pilot-result.json',dict(clips=len(items),passed=passed,review=str(folder/'index.html')))
    assert passed>=max(2,len(items)//3),'Pilot acceptance too low; inspect before full processing'
    progress('pilot_ready',pilot_total=len(items),pilot_automatic_pass=passed,pilot_review=str(folder/'index.html'))
    print('PILOT',passed,'/',len(items),'passed',flush=True)

def review_page(accepted):
    folder=OUT/'review';(folder/'context').mkdir(parents=True,exist_ok=True)
    chosen=[]
    for source in SOURCES:
        group=sorted([r for r in accepted if r['source']==source and r['split']=='train'],key=lambda r:r['source_start_frame'])
        chosen.extend(group[:10]);chosen.extend(group[-10:])
    validation=[r for r in accepted if r['split']=='validation']
    if validation:
        positions=np.linspace(0,len(validation)-1,min(10,len(validation)),dtype=int)
        chosen.extend(validation[i] for i in positions)
    # Include accepted clips closest to the minimum permitted acoustic margins.
    chosen.extend(sorted(accepted,key=lambda r:min(r['quiet_lead_seconds'],r['quiet_tail_seconds']))[:10])
    chosen=list({r['id']:r for r in chosen}.values());cards=[]
    for r in chosen:
        source_rate=r['source_sample_rate'];a=r['source_start_frame']/source_rate;b=r['source_end_frame']/source_rate
        context=folder/'context'/(r['id']+'.wav')
        sf.write(context,read(r['source'],max(0,a-1.5),min(sf.info(SOURCES[r['source']]).duration,b+1.5),32000),32000,subtype='PCM_16')
        text=html.escape(r['text']);label=html.escape(r['id']+' · '+r['split'])
        cards.append(f'<article><h2>{label}</h2><p>{text}</p><p>Training/validation cut</p><audio controls preload="metadata" src="../{r["split"]}/audio/{r["id"]}.wav"></audio><details><summary>Compare original context and boundary measurements</summary><p>Original recording with 1.5 seconds of context on each side. This context file is not training data.</p><audio controls preload="none" src="context/{r["id"]}.wav"></audio><p>Measured quiet at start: {r["quiet_lead_seconds"]*1000:.0f} ms; end: {r["quiet_tail_seconds"]*1000:.0f} ms.</p></details></article>')
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Repaired corpus listening review</title><style>body{font:17px/1.6 system-ui;max-width:1000px;margin:auto;padding:24px;background:#f2f4ef}article{background:white;padding:24px;margin:20px 0;border-radius:12px}audio{width:100%}summary{cursor:pointer}</style><h1>Repaired corpus listening review</h1><p>Accepted clips only: first/last training examples from both sources, validation examples, and clips closest to the minimum quiet-edge margin. All passed automatic checks; human listening remains incomplete. Model training has not started.</p>'+''.join(cards)+'</html>'
    atomic_text(folder/'index.html',page)
    atomic_json(folder/'selection.json',dict(ids=[r['id'] for r in chosen],manual_listening_complete=False))
    return str(folder/'index.html')

def publish():
    plan=json.loads((WORK/'plan.json').read_text());accepted=[];rejected=[];seen_audio=set();seen_text=set()
    if (OUT/'corpus.json').exists():
        previous=json.loads((OUT/'corpus.json').read_text())
        for value in previous['splits'].values():
            assert file_digest(value['path'])['sha256']==value['sha256']
        progress('complete_training_paused',clips=previous['clips'],hours=previous['hours'],
                 splits=previous['splits'],pilot=previous['pilot'])
        return
    for item in sorted(selected(False),key=lambda x:(x['split']!='validation',x['source'],x['start_frame'])):
        evidence=json.loads((WORK/'alignment'/(item['id']+'.json')).read_text())
        if evidence['flags']:rejected.append(evidence);continue
        x=read(item['source'],item['start_frame']/32000,item['end_frame']/32000,32000)
        assert all(edge_check(x,item['quiet_threshold'])[k] for k in ['leading_quiet','trailing_quiet'])
        audio_sha=hashlib.sha256((np.clip(x,-1,1)*32767).astype('<i2').tobytes()).hexdigest();text_sha=fingerprint(words(evidence['normalized']))
        if audio_sha in seen_audio or text_sha in seen_text:
            evidence['flags']=['duplicate_audio_or_transcript'];rejected.append(evidence);continue
        dest=OUT/item['split']/'audio'/(item['id']+'.wav');dest.parent.mkdir(parents=True,exist_ok=True)
        temp=dest.with_suffix('.part.wav');sf.write(temp,x,32000,subtype='PCM_16');temp.replace(dest)
        info=sf.info(dest);assert info.samplerate==32000 and info.channels==1 and info.frames==len(x)
        decoded,sample_rate=sf.read(dest,dtype='float32')
        assert sample_rate==32000 and np.allclose(decoded,x,rtol=0,atol=1/32768+1e-7)
        exported_edges=edge_check(decoded,item['quiet_threshold'])
        if not exported_edges['leading_quiet'] or not exported_edges['trailing_quiet']:
            evidence['flags']=['exported_pcm_edge_not_quiet'];rejected.append(evidence);dest.unlink();continue
        seen_audio.add(audio_sha);seen_text.add(text_sha)
        source_rate=sf.info(SOURCES[item['source']]).samplerate
        measured_edges=exported_edges
        evidence_path=OUT/'alignment'/(item['id']+'.json')
        atomic_json(evidence_path,evidence)
        row=dict(index=len(accepted),id=item['id'],filename=str(dest.relative_to(ROOT)),text=evidence['transcript'],
            duration_seconds=info.duration,sample_rate=32000,channels=1,split=item['split'],source=item['source'],
            source_filename=str(SOURCES[item['source']]),source_start_frame=round(item['start_frame']*source_rate/32000),source_end_frame=round(item['end_frame']*source_rate/32000),source_sample_rate=source_rate,
            quiet_lead_seconds=measured_edges['leading_quiet_seconds'],quiet_tail_seconds=measured_edges['trailing_quiet_seconds'],flags='',audio_sha256=file_digest(dest)['sha256'],
            alignment_evidence=str(evidence_path))
        accepted.append(row)
    assert accepted and all(any(r['split']==s for r in accepted) for s in ['train','validation'])
    # Prove source frame disjointness and exclusion compliance after export.
    for source in SOURCES:
        ordered=sorted([r for r in accepted if r['source']==source],key=lambda r:r['source_start_frame'])
        assert all(a['source_end_frame']<=b['source_start_frame'] for a,b in zip(ordered,ordered[1:]))
        for r in ordered:
            a,b=r['source_start_frame']/r['source_sample_rate'],r['source_end_frame']/r['source_sample_rate']
            assert not any(a<hi and b>lo for lo,hi in plan['exclusions'][source])
            assert any(a>=lo and b<=hi for lo,hi in plan['partitions'][source][r['split']])
    manifests={}
    for split in ['train','validation']:
        subset=[r for r in accepted if r['split']==split];buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=list(subset[0]));writer.writeheader();writer.writerows(subset)
        path=OUT/split/'manifest.csv';atomic_text(path,buf.getvalue())
        atomic_text(OUT/split/'she.list',''.join(f"{ROOT/r['filename']}|she|en|{r['text'].replace('|',' ')}\n" for r in subset))
        assert len(rows(path))==len(subset)
        manifests[split]=dict(path=str(path),sha256=file_digest(path)['sha256'],clips=len(subset),seconds=sum(r['duration_seconds'] for r in subset))
    for s,p in SOURCES.items():assert file_digest(p)['sha256']==plan['sources'][s]['sha256']
    atomic_json(OUT/'quarantine.json',rejected)
    segmentation_quarantine=json.loads((WORK/'segmentation-quarantine.json').read_text())
    atomic_json(OUT/'segmentation-quarantine.json',segmentation_quarantine)
    coverage={}
    candidates=selected(False)
    for source in SOURCES:
        candidate_seconds=sum(r['duration_seconds'] for r in candidates if r['source']==source)
        accepted_seconds=sum(r['duration_seconds'] for r in accepted if r['source']==source)
        approved_spans=[span for values in plan['partitions'][source].values() for span in values]
        approved_available=sum(b-a for a,b in subtract(approved_spans,plan['exclusions'][source]))
        coverage[source]=dict(source_seconds=sf.info(SOURCES[source]).duration,
            seconds_after_partition_guards_and_owner_exclusions=approved_available,
            candidate_seconds=candidate_seconds,accepted_seconds=accepted_seconds,
            candidate_quarantine_seconds=candidate_seconds-accepted_seconds,
            silence_or_no_usable_boundary_seconds=max(0,approved_available-candidate_seconds))
    model_files={'transcription':ROOT/'.cache/whisper-large-v3/model.bin',
                 'alignment':WORK.parent/'torch-hub/checkpoints/wav2vec2_fairseq_base_ls960_asr_ls960.pth'}
    model_hashes={name:file_digest(path)['sha256'] for name,path in model_files.items()}
    from asr_runtime import DECODE_SETTINGS
    import importlib.metadata
    report=dict(status='prepared_training_paused',schema=1,policy=POLICY,source_fingerprints=plan['sources'],
        splits=manifests,clips=len(accepted),hours=sum(r['duration_seconds'] for r in accepted)/3600,
        quarantined_clips=len(rejected),quarantine_reason_counts=dict(collections.Counter(f for r in rejected for f in r['flags'])),
        candidate_clips=len(selected(False)),manual_listening_complete=False,pilot=str(OUT/'pilot/index.html'),
        training_started=False,previous_corpus_preserved=True,created=time.time(),
        coverage=coverage,model_sha256=model_hashes,code_sha256=file_digest(Path(__file__))['sha256'])
    report.update(asr_decode_settings=DECODE_SETTINGS,
        software_versions={name:importlib.metadata.version(name) for name in ['torch','torchaudio','faster-whisper','ctranslate2']},
        source_audio_formats={s:dict(sample_rate=sf.info(p).samplerate,channels=sf.info(p).channels,duration_seconds=sf.info(p).duration) for s,p in SOURCES.items()},
        oversize_quarantined_clips=len(segmentation_quarantine),
        all_exported_clips_pass_actual_edge_checks=True,source_intervals_disjoint=True,
        owner_exclusions_preserved=True,model_training_requires_new_request=True)
    report['listening_review']=review_page(accepted)
    report['fingerprint']=fingerprint(dict(sources={s:v['sha256'] for s,v in plan['sources'].items()},
        policy=POLICY,manifests=manifests,models=model_hashes,code=report['code_sha256']))
    atomic_json(OUT/'source-plan.json',plan)
    atomic_json(OUT/'recipe.json',RECIPE)
    atomic_json(OUT/'corpus.json',report);atomic_json(ROOT/'reports/corpus-repair-result.json',report)
    # Publishing a candidate never selects it for training or deployment.
    atomic_json(ROOT/'data/shecorpus/repaired-corpus.json',report)
    atomic_text(Path('/home/voiceuser/projects/csm-lora/PAUSED_BY_OWNER'),'Dataset rebuilt; model training requires a new owner request.\n')
    progress('complete_training_paused',**{k:report[k] for k in ['clips','hours','quarantined_clips','splits','pilot']})
    for path in list(SOURCES.values())+list(model_files.values()):
        with path.open('rb') as stream:
            os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    print('REPAIRED DATASETS COMPLETE',json.dumps(manifests),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['check','scan','asr','align','pilot-report','publish']);parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    try:
        if args.phase=='check':
            validate_recipe();print('Recipe, source hashes, partitions, exclusions and cache identity verified; no preparation or training started.')
        elif args.phase=='scan':scan()
        elif args.phase=='asr':transcribe(args.pilot)
        elif args.phase=='align':align(args.pilot)
        elif args.phase=='pilot-report':pilot_report()
        elif args.phase=='publish':publish()
    except BaseException:
        import traceback
        progress('failed',error=traceback.format_exc());raise
