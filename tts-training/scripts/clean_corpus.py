"""Transactional filename-based removal; rollback on failure, retain audio quarantine."""
import collections,csv,json,pathlib,statistics,os,shutil,wave
R=pathlib.Path(__file__).resolve().parents[1]; D=pathlib.Path('/home/voiceuser/.hermes/cache/voices/she-xtts')
drop=json.loads((R/'reports/drop-list.json').read_text()); assert len(drop)==len(set(drop))==39
rows=list(csv.DictReader((D/'manifest.csv').open())); assert len(rows)==1217
assert set(drop)<={r['filename'] for r in rows}
backup=D/'manifest.csv.bak-pre-drop-39'; assert not backup.exists();shutil.copy2(D/'manifest.csv',backup)
summary_backup=(D/'summary.json').read_bytes(); quarantine=D/'.drop-39-backup';quarantine.mkdir()
keep=[r for r in rows if r['filename'] not in drop]
try:
 for name in drop:os.replace(D/'clips'/name,quarantine/name)
 with (D/'manifest.csv.tmp').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(keep)
 os.replace(D/'manifest.csv.tmp',D/'manifest.csv')
 durations=[]
 for r in keep:
  with wave.open(str(D/'clips'/r['filename'])) as w:durations.append(w.getnframes()/w.getframerate())
 names={r['filename'] for r in keep};disk={p.name for p in (D/'clips').glob('*.wav')};assert names==disk and len(names)==1178
 s=json.loads(summary_backup);s.update(total_clip_count=len(keep),total_duration_seconds=sum(durations),mean_duration_seconds=statistics.mean(durations),median_duration_seconds=statistics.median(durations),min_duration_seconds=min(durations),max_duration_seconds=max(durations),sources=dict(collections.Counter(r['source'] for r in keep)),ffprobe_verified_count=0)
 for key in list(s):
  if key.startswith('random_sample'):del s[key]
 s['duration_histogram_seconds']={f'{a}-{a+5}':sum(a<=v<a+5 for v in durations) for a in range(0,30,5)}
 s['removed_clips'].update({r['index']:r['filename']+' (owner drop-list: mismatched or inflated transcript)' for r in rows if r['filename'] in drop})
 s['consistency']={'manifest_rows_resolving':len(names),'wavs_with_manifest_rows':len(disk),'missing':[],'orphaned':[]}
 (D/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
 report={'dropped':drop,'before_clips':len(rows),'after_clips':len(keep),'before_hours':sum(float(r['duration_seconds']) for r in rows)/3600,'after_hours':sum(durations)/3600,**s['consistency'],'backup':str(backup),'audio_recovery':str(quarantine),'parquet_recovery':'Original data/ parquet shards retain every clip.'}
 (R/'reports/retrain-cleanup.json').write_text(json.dumps(report,indent=2)+'\n')
except BaseException:
 shutil.copy2(backup,D/'manifest.csv');(D/'summary.json').write_bytes(summary_backup)
 for p in quarantine.glob('*.wav'):os.replace(p,D/'clips'/p.name)
 raise
print(json.dumps(report,indent=2))
