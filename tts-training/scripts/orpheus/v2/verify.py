"""Read-only audit of the approved immutable V2 clips and frozen source spans."""
from common import *
import csv,re,math
import numpy as np,soundfile as sf
report=json.loads((CORPUS/'corpus.json').read_text());approval=json.loads((CORPUS/'owner-review-approved.json').read_text());plan=json.loads((CORPUS/'source-plan.json').read_text())
assert approval['fingerprint']==report['fingerprint']=='example02f5e3af078842b6be580e2c811fb57d4d884590d94eda53649d677005'
seen={};texts={};counts={};longest={}
for split,count in [('train',1856),('validation',192)]:
    manifest=CORPUS/split/'manifest.csv';assert sha(manifest)==report['splits'][split]['sha256']
    with manifest.open() as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==count
    counts[split]=len(rows);longest[split]=max(float(r['duration_seconds']) for r in rows)
    for row in rows:
        path=REPO/row['filename'];assert path.resolve().is_relative_to((CORPUS/split/'audio').resolve())
        assert not row['flags'] and sha(path)==row['audio_sha256']
        assert Path(row['alignment_evidence']).is_file()
        evidence=json.loads(Path(row['alignment_evidence']).read_text());assert not evidence['flags']
        audio,sr=sf.read(path,dtype='float32');assert audio.ndim==1 and len(audio)>0 and np.isfinite(audio).all()
        assert abs(len(audio)/sr-float(row['duration_seconds']))<=1/sr+.00001
        # 300ms is the eligible source pause; cuts retain half that pause.
        # Recheck the publisher's actual 120ms exported-PCM edge requirement.
        assert float(row['quiet_lead_seconds'])>=report['policy']['edge_seconds'] and float(row['quiet_tail_seconds'])>=report['policy']['edge_seconds']
        threshold=evidence['item']['quiet_threshold'];window=sr//100
        rms=np.sqrt(np.mean(audio[:len(audio)//window*window].reshape(-1,window).astype('float64')**2,axis=1))
        assert np.all(rms[:12]<threshold) and np.all(rms[-12:]<threshold),row['id']+' has an active PCM edge'
        assert evidence['margins']['first_word_seconds']>=report['policy']['minimum_aligned_start_margin']
        assert evidence['margins']['last_word_seconds']>=report['policy']['minimum_aligned_end_margin']
        for key in ['left_quiet_interval','right_quiet_interval']:
            lo,hi=evidence['item'][key];assert hi-lo>=report['policy']['quiet_seconds']-1e-6
        a=int(row['source_start_frame'])/int(row['source_sample_rate']);b=int(row['source_end_frame'])/int(row['source_sample_rate'])
        assert any(a>=lo and b<=hi for lo,hi in plan['partitions'][row['source']][split])
        assert not any(a<hi and b>lo for lo,hi in plan['exclusions'][row['source']])
        digest=row['audio_sha256'];assert digest not in seen;seen[digest]=split
        normalized=' '.join(re.sub(r'[^a-z0-9 ]',' ',row['text'].lower()).split());assert normalized
        assert normalized not in texts or texts[normalized]==split,'Transcript crosses train/validation split'
        texts[normalized]=split
    assert len(list((CORPUS/split/'audio').glob('*.wav')))==count
for source in report['source_fingerprints'].values():assert sha(Path(source['path']))==source['sha256']
atomic(RUN/'corpus-verification.json',dict(passed=True,fingerprint=report['fingerprint'],counts=counts,longest_seconds=longest,
    owner_approval=approval,manifest_hashes={s:sha(CORPUS/s/'manifest.csv') for s in counts},source_intervals_and_exclusions_checked=True,no_audio_modified=True))
print('PASS immutable V2 corpus',counts,longest,flush=True)
