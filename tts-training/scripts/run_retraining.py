"""Isolated resumable retraining. No upstream edits; plateau decisions are recorded."""
import csv,json,os,pathlib,random,subprocess,sys,time,yaml
R=pathlib.Path(__file__).resolve().parents[1]; E=R/'experiments/she-retrain-v2proplus'; U=R/'upstream'; P=U/'GPT_SoVITS/pretrained_models'; E.mkdir(parents=True,exist_ok=True)
STATUS=R/'reports/retrain-status.json'
def status(stage,**kw):STATUS.write_text(json.dumps({'stage':stage,'time':time.time(),**kw},indent=2)+'\n')
def run(stage,args,env=None):
 status(stage);log=R/'logs'/('retrain-'+stage+'.log')
 with log.open('a') as f:
  result=subprocess.run([sys.executable,*args],cwd=U,env=env,stdout=f,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError(f'{stage}: exit {result.returncode}; exact error in {log}')
def features():
 env=os.environ.copy();env.update(inp_text=str(R/'data/she.list'),inp_wav_dir='',exp_name=E.name,i_part='0',all_parts='1',opt_dir=str(E),bert_pretrained_dir=str(P/'chinese-roberta-wwm-ext-large'),cnhubert_base_dir=str(P/'chinese-hubert-base'),sv_path=str(P/'sv/pretrained_eres2netv2w24s4ep4.ckpt'),pretrained_s2G=str(P/'v2Pro/s2Gv2ProPlus.pth'))
 # Extractor already passes version explicitly. Remove only the duplicate config key.
 cfg=json.loads((U/'GPT_SoVITS/configs/s2v2ProPlus.json').read_text());cfg['model'].pop('version',None)
 semantic_cfg=R/'configs/retrain-semantic.json';semantic_cfg.write_text(json.dumps(cfg,indent=2)+'\n');env['s2config_path']=str(semantic_cfg)
 semantic_part=E/'6-name2semantic-0.tsv'
 if semantic_part.exists() and len(semantic_part.read_text().splitlines())!=1178:semantic_part.unlink()
 for stage,script in [('text','1-get-text.py'),('hubert','2-get-hubert-wav32k.py'),('speaker','2-get-sv.py'),('semantic','3-get-semantic.py')]:run('features-'+stage,['GPT_SoVITS/prepare_datasets/'+script],env)
 rows=list(csv.DictReader(open('/home/voiceuser/.hermes/cache/voices/she-xtts/manifest.csv'))); names={r['filename'] for r in rows};assert len(names)==1178
 held=sorted(random.Random(20261112).sample(sorted(names),20));(R/'reports/retrain-heldout.json').write_text(json.dumps({'seed':20261112,'filenames':held,'training_clips':1158,'heldout_clips':20},indent=2)+'\n')
 for directory,pattern in [('4-cnhubert','*.pt'),('5-wav32k','*.wav'),('7-sv_cn','*.pt')]:
  found={p.name.removesuffix('.pt') for p in (E/directory).glob(pattern)};assert found==names,(directory,len(found))
 for src,dst,header in [('2-name2text-0.txt','2-name2text.txt',''),('6-name2semantic-0.tsv','6-name2semantic.tsv','item_name\tsemantic_audio\n')]:
  lines=(E/src).read_text().splitlines();assert {l.split('\t')[0] for l in lines}==names
  (E/('all-'+dst)).write_text(header+'\n'.join(lines)+'\n')
  (E/dst).write_text(header+'\n'.join(l for l in lines if l.split('\t')[0] not in held)+'\n')
 # Remove held-out feature files too, making leakage impossible for either loader.
 for n in held:
  for d,suffix in [('4-cnhubert','.pt'),('5-wav32k',''),('7-sv_cn','.pt')]:
   p=E/d/(n+suffix)
   if p.exists():p.unlink()
 (E/'features.done').touch()
def configure(stage,epochs):
 (R/'checkpoints/retrain/gpt').mkdir(parents=True,exist_ok=True)
 (R/'checkpoints/retrain/sovits').mkdir(parents=True,exist_ok=True)
 if stage=='s1':
  cfg=yaml.safe_load((R/'configs/baseline-s1.yaml').read_text());cfg['output_dir']=str(E/'logs_s1_v2ProPlus');cfg['train'].update(epochs=epochs,exp_name=E.name,half_weights_save_dir=str(R/'checkpoints/retrain/gpt'))
  cfg['train_phoneme_path']=str(E/'2-name2text.txt');cfg['train_semantic_path']=str(E/'6-name2semantic.tsv');(R/'configs/retrain-s1.yaml').write_text(yaml.safe_dump(cfg))
 else:
  cfg=json.loads((R/'configs/baseline-s2.json').read_text());cfg['train']['epochs']=epochs;cfg['data']['exp_dir']=str(E);cfg.update(s2_ckpt_dir=str(E),save_weight_dir=str(R/'checkpoints/retrain/sovits'),name=E.name);(R/'configs/retrain-s2.json').write_text(json.dumps(cfg,indent=2)+'\n');(E/'logs_s2_v2ProPlus').mkdir(exist_ok=True)
def epochs(stage):
 p=R/'reports/retrain-epochs.jsonl';return sorted({v['epoch']:v for v in map(json.loads,p.read_text().splitlines()) if v['stage']==stage}.values(),key=lambda v:v['epoch']) if p.exists() else []
def plateau(stage):
 a=epochs(stage)
 if len(a)<10:return False
 old,new=a[-10:-5],a[-5:]
 keys=['total_loss_epoch','top_3_acc_epoch'] if stage=='s1' else ['loss/g/total','loss/g/mel','loss/g/kl']
 checks={}
 for k in keys:
  x=sum(v['metrics'][k] for v in old)/5;y=sum(v['metrics'][k] for v in new)/5
  improvement=y-x if 'acc' in k else (x-y)/max(abs(x),1e-9)
  checks[k]={'previous_5_mean':x,'last_5_mean':y,'improvement':improvement,'threshold':0.002 if 'acc' in k else 0.005,'final_epoch_delta':a[-1]['metrics'][k]-a[-2]['metrics'][k]}
 done=all(v['improvement']<=v['threshold'] for v in checks.values())
 (R/f'reports/retrain-{stage}-convergence.json').write_text(json.dumps({'converged':done,'epochs':len(a),'global_step':a[-1]['global_step'],'checks':checks},indent=2)+'\n');return done
try:
 if not (E/'features.done').exists():features()
 configure('s1',20);configure('s2',25)
 os.environ['SHE_RETRAIN']='1'
 run('audit-training-data',[str(R/'scripts/audit_training_data.py')])
 for stage,floor in [('s1',20),('s2',25)]:
  while True:
   a=epochs(stage)
   if a and a[-1]['epoch']>=floor and plateau(stage):break
   target=max(floor,(a[-1]['epoch']+5) if a else floor);configure(stage,target)
   run('train-'+stage,[str(R/'scripts/run_training.py'),stage,'--retrain'])
 status('training-complete',s1=epochs('s1')[-1],s2=epochs('s2')[-1])
 run('finalize',[str(R/'scripts/finalize_retraining.py')])
 status('complete')
except BaseException as exc:
 status('failed',error=repr(exc));raise
