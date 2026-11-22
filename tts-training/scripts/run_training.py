"""Run upstream trainers with timing instrumentation, without changing their code."""
import argparse,importlib,json,pathlib,sys,time
R=pathlib.Path(__file__).resolve().parents[1]
stage=sys.argv.pop(1)
retrain='--retrain' in sys.argv
if retrain:sys.argv.remove('--retrain')
prefix='retrain-' if retrain else ''
def record(item):
 with (R/('reports/'+prefix+'epochs.jsonl')).open('a') as f:f.write(json.dumps(item)+'\n')
 print('EPOCH_METRICS',json.dumps(item),flush=True)
if stage=='s1':
 import pytorch_lightning as pl
 import s1_train as train
 OriginalTrainer=train.Trainer
 class EpochRecorder(pl.Callback):
  def on_train_epoch_start(self,trainer,module):self.started=time.time();self.start_step=trainer.global_step
  def on_train_epoch_end(self,trainer,module):
   record({'stage':'s1','epoch':trainer.current_epoch+1,'started':self.started,'seconds':time.time()-self.started,'global_step':trainer.global_step,'optimizer_steps':trainer.global_step-self.start_step,'metrics':{k:float(v.detach().cpu()) for k,v in trainer.callback_metrics.items() if v.numel()==1}})
 def trainer_factory(*args,**kwargs):
  kwargs['callbacks'].append(EpochRecorder());kwargs['enable_progress_bar']=False
  return OriginalTrainer(*args,**kwargs)
 train.Trainer=trainer_factory
 train.main(argparse.Namespace(config_file=str(R/('configs/'+prefix+'s1.yaml'))))
elif stage=='s2':
 (R/('experiments/'+('she-retrain-v2proplus' if retrain else 'she-v2proplus')+'/logs_s2_v2ProPlus')).mkdir(parents=True,exist_ok=True)
 sys.argv=[sys.argv[0],'--config',str(R/('configs/'+prefix+'s2.json'))]
 import s2_train as train
 original=train.train_and_evaluate
 original_summarize=train.utils.summarize
 logged={}
 def summarize(*args,**kwargs):
  for key,value in kwargs.get('scalars',{}).items():
   if key.startswith('loss/'):logged.setdefault(key,[]).append(float(value))
  return original_summarize(*args,**kwargs)
 train.utils.summarize=summarize
 def timed(rank,epoch,*args,**kwargs):
  logged.clear();started=time.time();start_step=train.global_step;result=original(rank,epoch,*args,**kwargs)
  if rank==0:record({'stage':'s2','epoch':epoch,'started':started,'seconds':time.time()-started,'global_step':train.global_step,'optimizer_steps':train.global_step-start_step,'metrics':{k:sum(v)/len(v) for k,v in logged.items()}})
  return result
 train.train_and_evaluate=timed
 train.main()
else:raise ValueError(stage)
