"""Generate only from this experiment's trained exports; never allow fallback."""
import argparse,csv,hashlib,json,os,pathlib,re,sys
parser=argparse.ArgumentParser()
parser.add_argument("--only",nargs="*",default=[])
parser.add_argument("--seed-offset",type=int,default=0)
parser.add_argument("--temperature",type=float,default=0.8)
parser.add_argument("--speed",type=float,default=1.0)
parser.add_argument("--top-k",type=int,default=15)
parser.add_argument("--reference",default="")
parser.add_argument("--force",action="store_true")
args=parser.parse_args()
import numpy as np
import soundfile as sf
R=pathlib.Path(__file__).resolve().parents[1];os.chdir(R/'upstream')
from TTS_infer_pack.TTS import TTS,TTS_Config
from process_ckpt import load_sovits_new
import torch
G=R/'checkpoints/gpt/she-v2proplus-e3.ckpt'
candidates=list((R/'checkpoints/sovits').glob('she-v2proplus_e5_s*.pth'))
assert G.is_file() and len(candidates)==1,'Both final trained exports required'
S=candidates[0]
gpt_meta=torch.load(G,map_location='cpu',weights_only=False);sovits_meta=load_sovits_new(str(S))
assert gpt_meta['info']=='GPT-e3'
assert sovits_meta['info'].startswith('5epoch_')
provenance={'gpt':{'path':str(G.relative_to(R)),'info':gpt_meta['info'],'bytes':G.stat().st_size,'sha256':hashlib.sha256(G.read_bytes()).hexdigest()},'sovits':{'path':str(S.relative_to(R)),'info':sovits_meta['info'],'bytes':S.stat().st_size,'sha256':hashlib.sha256(S.read_bytes()).hexdigest()}}
(R/'reports/trained-checkpoints.json').write_text(json.dumps(provenance,indent=2)+'\n')
del gpt_meta,sovits_meta
config={'device':'cuda','is_half':True,'version':'v2ProPlus','t2s_weights_path':str(G),'vits_weights_path':str(S),'cnhuhbert_base_path':str(R/'upstream/GPT_SoVITS/pretrained_models/chinese-hubert-base'),'bert_base_path':str(R/'upstream/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large')}
(R/'configs/inference.json').write_text(json.dumps(config,indent=2)+'\n')
if not args.only and not args.force and (R/'reports/sample-verification.json').exists():
 previous=json.loads((R/'reports/sample-verification.json').read_text())
 if len(previous)==5 and all(s['words_match'] and s['gpt_checkpoint']==str(G.relative_to(R)) and s['sovits_checkpoint']==str(S.relative_to(R)) and (R/s['path']).is_file() and s.get('sha256')==hashlib.sha256((R/s['path']).read_bytes()).hexdigest() for s in previous):
  print('Preserving five existing verified samples; use --force to regenerate.')
  sys.exit(0)
tts_config=TTS_Config({'custom':config})
tts_config.configs_path=str(R/'configs/tts-infer.yaml')
tts=TTS(tts_config)
assert pathlib.Path(tts.configs.t2s_weights_path)==G
assert pathlib.Path(tts.configs.vits_weights_path)==S
rows=list(csv.DictReader(open('/home/voiceuser/.hermes/cache/voices/she-xtts/manifest.csv')))
# A complete original short clip meets upstream reference length of 3-10 seconds.
ref=next(r for r in rows if r['filename']==args.reference) if args.reference else next(r for r in rows if 5<=float(r['duration_seconds'])<=9)
plan=json.loads((R/'configs/sample-plan.json').read_text())
prompts=[(p['name'],p['text']) for p in plan]
results=json.loads((R/"samples/manifest.json").read_text()) if args.only else []
for i,(name,text) in enumerate(prompts):
 if args.only and name not in args.only:continue
 settings=plan[i] if not args.only else {'seed':1700+i+args.seed_offset,'top_k':args.top_k,'temperature':args.temperature,'speed_factor':args.speed,'reference_filename':ref['filename']}
 chosen_ref=next(r for r in rows if r['filename']==settings['reference_filename'])
 inputs={'text':text,'text_lang':'en','ref_audio_path':str(R/'data/wavs32k'/chosen_ref['filename']),'prompt_text':chosen_ref['text'],'prompt_lang':'en','top_k':settings['top_k'],'top_p':1.0,'temperature':settings['temperature'],'speed_factor':settings['speed_factor'],'text_split_method':'cut5','batch_size':1,'seed':settings['seed'],'parallel_infer':False,'use_cuda_graph':False,'use_flash_attention':False,'repetition_penalty':1.35}
 chunks=list(tts.run(inputs)); assert len(chunks)==1
 sr,audio=chunks[0];assert len(audio)>sr
 out=R/'samples'/f'{name}.wav';sf.write(out,audio,sr,subtype='PCM_16')
 results=[r for r in results if r['path']!=str(out.relative_to(R))]
 results.append({'temperature':settings['temperature'],'top_k':settings['top_k'],'speed_factor':settings['speed_factor'],'path':str(out.relative_to(R)),'text':text,'seed':settings['seed'],'sample_rate':sr,'reference_filename':chosen_ref['filename'],'gpt_checkpoint':str(G.relative_to(R)),'sovits_checkpoint':str(S.relative_to(R))})
 results.sort(key=lambda r:r['path'])
 (R/'samples/manifest.json').write_text(json.dumps(results,indent=2)+'\n')
 print(name,len(audio)/sr,flush=True)
