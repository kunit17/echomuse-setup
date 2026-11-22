"""Read benchmark progress without loading a CUDA context."""
from common import *
from stream_core import percentile
import statistics
B=ROOT/'stream-benchmark';result={}
for name in ['status.json','setup-status.json','screening.json']:
    if (B/name).exists():result[name]=json.loads((B/name).read_text())
if 'screening.json' in result:
    screen=result['screening.json']
    result['screening.json']=dict(successful=[dict(name=c['name'],rtf=c['screen_score']) for c in screen['successful']],
        failures=[dict(configuration=c['configuration'],error=c['error'].splitlines()[-1]) for c in screen['failures']])
rows=[]
if (B/'measurements.jsonl').exists():
    for line in (B/'measurements.jsonl').read_text().splitlines():
        try:row=json.loads(line)
        except ValueError:continue
        if not row['warmup']:rows.append(row)
result['measurements']={name:dict(requests=sum(r['configuration']==name for r in rows),
    median_rtf=statistics.median(r['rtf'] for r in rows if r['configuration']==name),
    p95_rtf=percentile([r['rtf'] for r in rows if r['configuration']==name])) for name in set(r['configuration'] for r in rows)}
print(json.dumps(result,indent=2))
