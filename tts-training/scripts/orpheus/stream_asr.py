from common import *
import sys,statistics,csv
sys.path.insert(0,str(REPO/'scripts'))
from faster_whisper import WhisperModel
from corpus_ground_truth import tokens,edits
B=ROOT/'stream-benchmark'
model=WhisperModel(str(REPO/'.cache/whisper-large-v3'),device='cuda',compute_type='float16',local_files_only=True)
results=[]
measured=[]
for line in (B/'measurements.jsonl').read_text().splitlines():
    row=json.loads(line)
    if row['warmup'] or not row.get('wav'):continue
    measured.append(row)
    segments,_=model.transcribe(row['wav'],language='en',beam_size=5,temperature=0,condition_on_previous_text=False,vad_filter=False)
    transcript=' '.join(s.text.strip() for s in segments);expected,heard=tokens(row['text']),tokens(transcript)
    results.append(dict(configuration=row['configuration'],prompt=row['prompt'],repeat=row['repeat'],
        transcript=transcript,word_edits=edits(expected,heard),first_word_match=bool(heard and expected[0]==heard[0]),
        final_word_match=bool(heard and expected[-1]==heard[-1])))
    atomic(B/'asr.json',dict(complete=False,results=results))
atomic(B/'asr.json',dict(complete=True,results=results,manual_listening_complete=False))
report=json.loads((B/'report.json').read_text());report['asr_complete']=True
workspace=Path('/mnt/c/Users/DesktopUser/Projects/echomuse-provisioning/tts-training/orpheus-training')
report['screening']=json.loads((B/'screening.json').read_text()) if (B/'screening.json').exists() else {}
report['failures']=json.loads((B/'failures.json').read_text()) if (B/'failures.json').exists() else []
report['browser_results']={path.stem.removeprefix('browser-'):json.loads(path.read_text(encoding='utf-8-sig')) for path in workspace.glob('browser-*.json')}
for config in report['configurations']:
    items=[r for r in results if r['configuration']==config['name']]
    config['asr_flagged']=sum(r['word_edits']>0 or not r['first_word_match'] or not r['final_word_match'] for r in items)
    timed=[r for r in measured if r['configuration']==config['name']]
    config['median_rtf']=statistics.median(r['rtf'] for r in timed)
    config['median_first_playback']=statistics.median(r['playback']['0.5']['first_playback'] for r in timed)
    browser=report['browser_results'].get(config['name'],{})
    config['passes_automated_deployment_gates']=bool(config['name']!='baseline' and config['requests']==30 and config['asr_flagged']==0
        and config['name'] in report['configurations_meeting_timing'] and browser.get('status')=='complete'
        and browser.get('firstPlayback',999)<=2 and browser.get('underruns',999)==0
        and browser.get('apiChecks',{}).get('invalidTextRejected') and browser.get('apiChecks',{}).get('disconnectCancelled'))
telemetry=list(csv.reader((B/'gpu.csv').read_text().splitlines()))
def maximum(column):
    values=[]
    for row in telemetry:
        try:values.append(float(row[column]))
        except (ValueError,IndexError):pass
    return max(values) if values else None
report['whole_benchmark_driver_telemetry']=dict(peak_used_mib_including_desktop=maximum(1),peak_power_watts=maximum(2),peak_temperature_c=maximum(3))
traces=[json.loads(p.read_text()) for p in (B/'traces').glob('*.json')]
complete=[t for t in traces if t['status']=='complete']
report['audio_coverage']=dict(complete_traces=len(complete),sample_count_failures=sum(t['sample_count']!=t['offline_sample_count'] for t in complete),
    cancelled_traces=sum(t['status']=='cancelled' for t in traces),failed_traces=sum(t['status']=='failed' for t in traces))
report['status']='complete_listening_pending';atomic(B/'report.json',report)
print('ASR complete',len(results),flush=True)
