"""Train only the new corpus, validate each epoch, and retain the best and latest checkpoints."""
import argparse
import contextlib
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
import yaml
from resource_safe import loader, require_cuda, atomic_save, gpt_training_step
from training_run import run_context, POLICIES, should_stop, prune_checkpoints

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT/'upstream')
RUN=run_context(ROOT)
REPORTS=Path(RUN['reports'])
EXP=Path(RUN['features'])
TRAIN=EXP/'train';VALID=EXP/'validation'
CKPT=Path(RUN['checkpoints'])
CURVES=REPORTS/'epochs.jsonl'
require_cuda();torch.save=atomic_save(torch.save)
parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['s1','s2']);args=parser.parse_args()
for directory in (CKPT/'gpt',CKPT/'sovits',TRAIN/'logs_s2_v2ProPlus'):directory.mkdir(parents=True,exist_ok=True)

def history(stage):
    if not CURVES.exists():return []
    return sorted({r['epoch']:r for r in map(json.loads,CURVES.read_text().splitlines()) if r['stage']==stage}.values(),key=lambda r:r['epoch'])

def record(value):
    value.update(run_id=RUN['run_id'],corpus_fingerprint=RUN['corpus_fingerprint'],curation_revision=RUN.get('curation_revision'))
    assert all(math.isfinite(float(v)) for v in value.get('validation',{}).values())
    with CURVES.open('a') as f:f.write(json.dumps(value)+'\n');f.flush();os.fsync(f.fileno())
    print('CONTINUOUS_EPOCH',json.dumps(value),flush=True)

def stop(stage):
    return should_stop(history(stage),stage)

def select(stage):
    rows=history(stage);key='nll_per_token' if stage=='s1' else 'mel'
    # Epoch zero is reported as a control; select among the adapted checkpoints.
    best=min((r for r in rows if r['epoch']>0),key=lambda r:(r['validation'][key],r['validation'].get('kl',0),r['epoch']))
    path=REPORTS/f'{stage}-selection.json'
    with path.open('w') as f:json.dump(dict(selected=best,final=rows[-1],stopping_criteria_met=stop(stage),policy=POLICIES[stage],run_id=RUN['run_id']),f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

if stop(args.stage):
    select(args.stage)
    print('Required floor and plateau already verified for',args.stage,flush=True)
    sys.exit(0)

if args.stage=='s1':
    import pytorch_lightning as pl
    from lightning_fabric.plugins.io import CheckpointIO
    import s1_train as train
    from AR.data.dataset import Text2SemanticDataset
    import AR.data.data_module as data_module
    from AR.data.bucket_sampler import DistributedBucketSampler
    cfg=yaml.safe_load((ROOT/'configs/s1.yaml').read_text())
    cfg['output_dir']=str(EXP/'gpt-training')
    cfg['train'].update(epochs=POLICIES['s1']['max_epochs'],batch_size=1,exp_name=RUN['run_name'],half_weights_save_dir=str(CKPT/'gpt'),if_save_latest=False,save_every_n_epoch=1)
    cfg['data'].update(num_workers=0,max_sec=54)
    cfg['train_phoneme_path']=str(TRAIN/'2-name2text.txt');cfg['train_semantic_path']=str(TRAIN/'6-name2semantic.tsv')
    config=REPORTS/'s1-config.yaml';config.write_text(yaml.safe_dump(cfg))
    class CorpusDataset(Text2SemanticDataset):
        def __init__(self,*a,**kw):
            kw.update(max_sec=54,min_ps_ratio=0,max_ps_ratio=float('inf'))
            super().__init__(*a,**kw)
            unique=dict(zip(self.item_names,self.semantic_phoneme))
            self.item_names=list(unique);self.semantic_phoneme=list(unique.values())
    data_module.Text2SemanticDataset=CorpusDataset
    validation=CorpusDataset(str(VALID/'2-name2text.txt'),str(VALID/'6-name2semantic.tsv'))
    expected={Path(r['filename']).name for r in csv.DictReader((ROOT/'data/shecorpus/validation/manifest.csv').open(encoding='utf-8'))}
    assert set(validation.item_names)==expected,'Validation loader silently excluded clips'
    train.Text2SemanticLightningModule.training_step=gpt_training_step
    OriginalModule=train.Text2SemanticLightningModule
    class DeviceModule(OriginalModule):
        def __init__(self,config,output_dir,is_train=True):
            # Construct parameters on CUDA and stream checkpoint storages to
            # CUDA, avoiding simultaneous full host copies during loading.
            with torch.device('cuda'):
                super().__init__(config,output_dir,is_train=False)
            if is_train:
                state=torch.load(config['pretrained_s1'],map_location='cuda',weights_only=False)
                self.load_state_dict(state['weight']);del state
                self.automatic_optimization=False;self.save_hyperparameters()
                self.eval_dir=output_dir/'eval';self.eval_dir.mkdir(parents=True,exist_ok=True)
    train.Text2SemanticLightningModule=DeviceModule
    class StreamingCheckpointIO(CheckpointIO):
        def save_checkpoint(self,checkpoint,path,storage_options=None):torch.save(checkpoint,path)
        def load_checkpoint(self,path,map_location=None,weights_only=None):return torch.load(path,map_location='cuda',weights_only=False)
        def remove_checkpoint(self,path):Path(path).unlink(missing_ok=True)
    # Upstream's inference writer uses an in-memory BytesIO. Stream the same
    # weight dictionary to disk, retaining atomic fsync behavior.
    train.my_save=lambda value,path:torch.save(value,path)
    def training_loader(self):
        expected_train={Path(r['filename']).name for r in csv.DictReader((ROOT/'data/shecorpus/train/manifest.csv').open(encoding='utf-8'))}
        assert set(self._train_dataset.item_names)==expected_train and not expected_train & expected
        sampler=DistributedBucketSampler(self._train_dataset,batch_size=1,num_replicas=1,rank=0)
        return loader(self._train_dataset,batch_size=1,sampler=sampler,collate_fn=self._train_dataset.collate)
    train.Text2SemanticDataModule.train_dataloader=training_loader
    def evaluate(module):
        was_training=module.training;module.eval();loss_sum=0.;count=0;correct=0.;semantic=0
        with torch.inference_mode():
            for i in range(len(validation)):
                batch={k:v.cuda() for k,v in validation.collate([validation[i]]).items() if torch.is_tensor(v)}
                module.model.ar_accuracy_metric.reset()
                with torch.autocast('cuda',dtype=torch.float16):
                    loss,accuracy=module.model.forward_old(batch['phoneme_ids'],batch['phoneme_ids_len'],batch['semantic_ids'],batch['semantic_ids_len'],batch['bert_feature'])
                n=int(batch['semantic_ids_len'].item());loss_sum+=float(loss);count+=n+1;correct+=float(accuracy)*n;semantic+=n
        module.model.ar_accuracy_metric.reset()
        module.train(was_training)
        return dict(nll_per_token=loss_sum/count,top3_accuracy=correct/semantic)
    class Validation(pl.Callback):
        def on_train_start(self,trainer,module):
            if not history('s1'):record(dict(stage='s1',epoch=0,optimizer_steps=0,global_step=0,validation=evaluate(module)))
        def on_train_epoch_start(self,trainer,module):
            self.started=time.time();self.start=trainer.global_step;self.calls=0
            def counted(*unused):self.calls+=1
            self.handle=module.optimizers().optimizer.register_step_post_hook(counted)
        def on_train_epoch_end(self,trainer,module):
            self.handle.remove()
            trainer._continuous_record=dict(stage='s1',epoch=trainer.current_epoch+1,seconds=time.time()-self.started,
                global_step=trainer.global_step,optimizer_steps=self.calls,scheduled_optimizer_steps=trainer.global_step-self.start,
                train={k:float(v) for k,v in trainer.callback_metrics.items() if v.numel()==1},validation=evaluate(module))
    OriginalCheckpoint=train.my_model_ckpt
    class Checkpoint(OriginalCheckpoint):
        def on_train_epoch_end(self,trainer,module):
            super().on_train_epoch_end(trainer,module)
            value=trainer._continuous_record
            path=CKPT/'gpt'/f"{RUN['run_name']}-e{value['epoch']}.ckpt"
            expected_bytes=sum(v.numel()*2 for v in module.state_dict().values())
            assert path.is_file() and path.stat().st_size>=expected_bytes
            full=Path(self._last_checkpoint_saved)
            assert full.is_file() and full.stat().st_size>expected_bytes
            value['full_checkpoint']=str(full)
            value['checkpoint']=str(path);record(value);select('s1')
            prune_checkpoints(RUN,history('s1'),'s1')
            if stop('s1'):trainer.should_stop=True
    train.my_model_ckpt=Checkpoint
    original_newest=train.get_newest_ckpt
    train.get_newest_ckpt=lambda names:original_newest([n for n in names if n.endswith('.ckpt')])
    OriginalTrainer=train.Trainer
    def factory(*a,**kw):
        kw['callbacks'].append(Validation());kw.update(accelerator='gpu',devices=1,strategy='auto',enable_progress_bar=False,plugins=[StreamingCheckpointIO()])
        return OriginalTrainer(*a,**kw)
    train.Trainer=factory
    train.main(argparse.Namespace(config_file=str(config)))
else:
    cfg=json.loads((ROOT/'configs/s2.json').read_text())
    cfg['train'].update(epochs=POLICIES['s2']['max_epochs'],batch_size=2,log_interval=1,if_save_latest=False,save_every_epoch=1)
    cfg['data']['exp_dir']=str(TRAIN)
    cfg.update(s2_ckpt_dir=str(TRAIN),save_weight_dir=str(CKPT/'sovits'),name=RUN['run_name'])
    config=REPORTS/'s2-config.json';config.write_text(json.dumps(cfg,indent=2))
    sys.argv=[sys.argv[0],'--config',str(config)]
    import s2_train as train
    train.DataLoader=loader
    import process_ckpt
    def streaming_versioned_save(value,path,model_version):
        path=Path(path);temporary=path.with_name(path.name+'.header-saving')
        torch.save(value,temporary)
        with temporary.open('r+b') as stream:
            stream.write(process_ckpt.model_version2byte[model_version]);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
        fd=os.open(path.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    process_ckpt.my_save2=streaming_versioned_save
    original_generator=train.SynthesizerTrn
    original_discriminator=train.MultiPeriodDiscriminator
    def generator(*a,**kw):
        with torch.device('cuda'):return original_generator(*a,**kw)
    def discriminator(*a,**kw):
        with torch.device('cuda'):return original_discriminator(*a,**kw)
    train.SynthesizerTrn=generator;train.MultiPeriodDiscriminator=discriminator
    original_load=torch.load
    def device_load(*a,**kw):
        if a and isinstance(a[0],(str,Path)) and str(a[0]).endswith('.pth'):
            path=Path(a[0])
            if path.parent.name.startswith('logs_s2_'):
                # Resume files include optimizer state: page them from disk on
                # CPU, then let optimizer/model loaders place only needed tensors.
                kw.update(map_location='cpu',mmap=True)
            else:kw['map_location']='cuda'
        return original_load(*a,**kw)
    torch.load=device_load
    original_resume=train.utils.load_checkpoint
    def resume_checkpoint(*a,**kw):
        result=original_resume(*a,**kw)
        import gc
        gc.collect();torch.cuda.empty_cache()
        print('RESUME_STATE_LOADED',a[0],'allocated_mib',torch.cuda.memory_allocated()/1024**2,flush=True)
        return result
    train.utils.load_checkpoint=resume_checkpoint
    train.utils.plot_spectrogram_to_numpy=lambda *a,**kw:None
    from module.data_utils import TextAudioSpeakerLoader,TextAudioSpeakerCollate
    from utils import HParams
    validation_hps=HParams(**cfg['data']);validation_hps.exp_dir=str(VALID)
    validation=TextAudioSpeakerLoader(validation_hps,version='v2ProPlus')
    # The upstream loader repeats sets below 100 clips. Validation counts each
    # source clip exactly once, with stable order for deterministic seeds.
    unique={item[0]:(item,length) for item,length in zip(validation.audiopaths_sid_text,validation.lengths)}
    validation.audiopaths_sid_text=[unique[name][0] for name in sorted(unique)]
    validation.lengths=[unique[name][1] for name in sorted(unique)]
    collate=TextAudioSpeakerCollate(version='v2ProPlus')
    expected={Path(r['filename']).name for r in csv.DictReader((ROOT/'data/shecorpus/validation/manifest.csv').open(encoding='utf-8'))}
    assert len(validation)==len(expected),'Validation loader silently excluded clips'
    def evaluate(net,hps):
        was_training=net.training;net.eval();mels=[];kls=[]
        with torch.no_grad(),torch.random.fork_rng(devices=[0]):
            for i in range(len(validation)):
                torch.manual_seed(20261112+i);torch.cuda.manual_seed_all(20261112+i)
                ssl,ssl_lengths,spec,spec_lengths,y,y_lengths,text,text_lengths,sv=collate([validation[i]])
                ssl,spec,spec_lengths,text,text_lengths,sv=[v.cuda() for v in (ssl,spec,spec_lengths,text,text_lengths,sv)]
                with torch.autocast('cuda',dtype=torch.float16):
                    output=net(ssl,spec,spec_lengths,text,text_lengths,sv)
                    y_hat,kl_ssl,ids_slice,x_mask,z_mask,(z,z_p,m_p,logs_p,m_q,logs_q),stats=output
                    mel=train.spec_to_mel_torch(spec,hps.data.filter_length,hps.data.n_mel_channels,hps.data.sampling_rate,hps.data.mel_fmin,hps.data.mel_fmax)
                    target=train.commons.slice_segments(mel,ids_slice,hps.train.segment_size//hps.data.hop_length)
                    generated=train.mel_spectrogram_torch(y_hat.squeeze(1),hps.data.filter_length,hps.data.n_mel_channels,hps.data.sampling_rate,hps.data.hop_length,hps.data.win_length,hps.data.mel_fmin,hps.data.mel_fmax)
                mels.append(float(torch.nn.functional.l1_loss(target.float(),generated.float())))
                kls.append(float(train.kl_loss(z_p.float(),logs_q.float(),m_p.float(),logs_p.float(),z_mask.float())))
        net.train(was_training)
        return dict(mel=sum(mels)/len(mels),kl=sum(kls)/len(kls))
    logged={}
    def summarize(*a,**kw):
        for key,value in kw.get('scalars',{}).items():
            if key.startswith('loss/'):logged.setdefault(key,[]).append(float(value))
    train.utils.summarize=summarize
    original=train.train_and_evaluate
    class PlateauReached(Exception):pass
    def wrapped(rank,epoch,hps,nets,optims,schedulers,scaler,loaders,logger,writers):
        previous=max((r['epoch'] for r in history('s2')),default=0)
        assert epoch==previous+1,'Resume did not continue the recorded epoch; refusing updates'
        recorded_step=max((r['global_step'] for r in history('s2') if r['epoch']==previous),default=0)
        if train.global_step!=recorded_step:
            print('RESTORE_RECORDED_GLOBAL_STEP',train.global_step,'to',recorded_step,flush=True)
            train.global_step=recorded_step
        train_names={item[0] for item in loaders[0].dataset.audiopaths_sid_text}
        expected_train={Path(r['filename']).name for r in csv.DictReader((ROOT/'data/shecorpus/train/manifest.csv').open(encoding='utf-8'))}
        assert train_names==expected_train and not train_names & expected
        if not history('s2'):record(dict(stage='s2',epoch=0,global_step=0,optimizer_steps=0,validation=evaluate(nets[0],hps)))
        logged.clear();start=train.global_step;started=time.time()
        calls=[0,0];handles=[]
        for i,opt in enumerate(optims):
            handles.append(opt.register_step_post_hook(lambda opt,a,kw,i=i:calls.__setitem__(i,calls[i]+1)))
        try:original(rank,epoch,hps,nets,optims,schedulers,scaler,loaders,logger,writers)
        finally:
            for handle in handles:handle.remove()
        value=dict(stage='s2',epoch=epoch,seconds=time.time()-started,global_step=train.global_step,
            optimizer_steps=train.global_step-start,generator_optimizer_calls=calls[0],discriminator_optimizer_calls=calls[1],
            train={k:sum(v)/len(v) for k,v in logged.items()},validation=evaluate(nets[0],hps))
        path=CKPT/'sovits'/f"{RUN['run_name']}_e{epoch}_s{train.global_step}.pth"
        expected_bytes=sum(v.numel()*2 for k,v in nets[0].state_dict().items() if 'enc_q' not in k)
        assert path.is_file() and path.stat().st_size>=expected_bytes
        full=[TRAIN/'logs_s2_v2ProPlus'/f'{kind}_{train.global_step}.pth' for kind in ('G','D')]
        assert all(p.is_file() and p.stat().st_size>0 for p in full)
        value['full_checkpoints']=[str(p) for p in full]
        value['checkpoint']=str(path);record(value);select('s2')
        prune_checkpoints(RUN,history('s2'),'s2')
        if stop('s2'):raise PlateauReached()
    train.train_and_evaluate=wrapped
    try:train.main()
    except PlateauReached:print('Validation-aware plateau reached after required floor',flush=True)
