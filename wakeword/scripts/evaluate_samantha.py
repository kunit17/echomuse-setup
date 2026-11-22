import argparse,json,wave
from pathlib import Path
import numpy as np
import onnxruntime as ort
from openwakeword.model import Model
parser=argparse.ArgumentParser();parser.add_argument('--suite',choices=['evaluation','final-evaluation'],default='evaluation');args=parser.parse_args()
root=Path('/data');suite=root/args.suite;model_path=root/'models/samantha.onnx'
session=ort.InferenceSession(str(model_path),providers=['CPUExecutionProvider'])
inp=session.get_inputs()[0];out=session.get_outputs()[0]
if len(inp.shape)!=3 or inp.shape[1:]!=[16,96] or out.shape[1:]!=[1]:
 raise RuntimeError('Classifier is incompatible with EchoMuse on-device window')
model=Model(wakeword_models=[str(model_path)],inference_framework='onnx')
rows=json.loads((suite/'manifest.json').read_text());results=[]
for row in rows:
 model.reset()
 with wave.open(str(suite/row['file']),'rb') as wav:
  audio=np.frombuffer(wav.readframes(wav.getnframes()),dtype=np.int16)
 scores=[]
 for start in range(0,len(audio)-1279,1280):
  prediction=model.predict(audio[start:start+1280])
  values=prediction.get('samantha')
  if values is None:raise RuntimeError('Model prediction key mismatch')
  scores.append(float(np.ravel(values)[0]))
 peak=max(scores,default=0)
 results.append(dict(**row,peak=peak,detected=peak>=.5))
positive=[r for r in results if r['label']=='positive'];negative=[r for r in results if r['label']=='negative']
recall=sum(r['detected'] for r in positive)/len(positive)
false_positives=sum(r['detected'] for r in negative)
report=dict(suite=args.suite,threshold=.5,input_shape=inp.shape,output_shape=out.shape,
            positive_detected=sum(r['detected'] for r in positive),positive_total=len(positive),
            negative_false_positives=false_positives,negative_total=len(negative),
            positive_recall=recall,passed=recall>=.8 and false_positives==0,
            limitation='Held-out synthetic Windows voices, not real Dot recordings',rows=results)
(suite/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
raise SystemExit(0 if report['passed'] else 2)
