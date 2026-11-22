from common import *
import argparse,copy,ctypes,gc,random,sys,traceback

def main(mode,precision):
    from unsloth import FastLanguageModel
    import numpy as np,soundfile as sf,torch
    from torch.utils.data import Dataset,Subset
    from transformers import Trainer,TrainingArguments,TrainerCallback,set_seed
    dataset=json.loads((DATA/'dataset.json').read_text())
    assert dataset['clips'] > 1 and dataset['max_tokens']<=4096
    assert all(dataset['splits'].get(s) for s in ['train','validation'])
    assert all(sha(CORPUS/s/'manifest.csv')==v for s,v in dataset['source_manifests'].items())
    smoke=mode!='train';directory=RUN/('smoke-'+precision if smoke else 'checkpoints')
    model_name=QUANT if precision=='4bit' else MODEL
    status(status='loading_model',mode=mode,precision=precision,error=None)
    assert torch.cuda.is_available()
    torch.mm(torch.ones((8,8),device='cuda'),torch.ones((8,8),device='cuda'))
    model,tokenizer=FastLanguageModel.from_pretrained(model_name=model_name,
        revision=setup()['model_revisions'][model_name],max_seq_length=4096,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        load_in_4bit=precision=='4bit',device_map={'':'cuda:0'} if precision=='4bit' else 'cpu')
    if precision!='4bit':model.to('cuda')
    gc.collect();ctypes.CDLL(None).malloc_trim(0)
    for path in (ROOT/'.cache/huggingface').rglob('*.safetensors'):
        with path.open('rb') as stream:os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    model=FastLanguageModel.get_peft_model(model,r=64,lora_alpha=64,lora_dropout=0,bias='none',
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        use_gradient_checkpointing='unsloth',random_state=3407,use_rslora=False,loftq_config=None)
    tokenizer.pad_token_id=TOKENS['pad']
    model.config.use_cache=False
    class TokenDataset(Dataset):
        def __init__(self,paths):self.paths=paths
        def __len__(self):return len(self.paths)
        def __getitem__(self,index):
            item=json.loads(Path(self.paths[index]).read_text());ids=item['input_ids']
            assert len(ids)<=4096 and ids[-2:]==[TOKENS['end_speech'],TOKENS['end_ai']]
            return dict(input_ids=ids,attention_mask=[1]*len(ids),labels=ids.copy())
    sets={s:TokenDataset(p) for s,p in dataset['splits'].items()}
    if smoke:
        longest=max(range(len(sets['train'])),key=lambda i:len(sets['train'][i]['input_ids']))
        val_longest=max(range(len(sets['validation'])),key=lambda i:len(sets['validation'][i]['input_ids']))
        sets['train']=Subset(sets['train'],[longest]*4)
        sets['validation']=Subset(sets['validation'],[val_longest,0])
    decoder=None
    def previews(epoch,checkpoint):
        nonlocal decoder
        folder=RUN/'samples'/('smoke-'+precision if smoke else f'epoch-{epoch}')
        if (folder/'results.json').exists():return
        from snac import SNAC
        from huggingface_hub import snapshot_download
        sys.path.insert(0,str(REPO/'scripts'))
        from audio_postprocess import clean_generated_tail
        if decoder is None:
            path=snapshot_download(CODEC,revision=setup()['model_revisions'][CODEC],local_files_only=True)
            decoder=SNAC.from_pretrained(path).cpu().eval()
        python_rng=random.getstate();numpy_rng=np.random.get_state();cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
        FastLanguageModel.for_inference(model)
        refresh_model_placement(model)
        folder.mkdir(parents=True,exist_ok=True);results=[]
        try:
            for name,text in PROMPTS[:1] if smoke else PROMPTS:
                ids=[TOKENS['start_human']]+tokenizer.encode(text,add_special_tokens=True)+[TOKENS['end_text'],TOKENS['end_human'],TOKENS['start_ai'],TOKENS['start_speech']]
                inputs=torch.tensor([ids],device='cuda');set_seed(1700);started=time.time()
                with torch.inference_mode():
                    output=model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),
                        max_new_tokens=2000,do_sample=True,temperature=.6,top_p=.95,repetition_penalty=1.1,
                        eos_token_id=TOKENS['end_speech'],pad_token_id=TOKENS['pad'],use_cache=True)
                new=output[0,len(ids):].tolist()
                if TOKENS['end_speech'] not in new:raise RuntimeError('Preview hit token ceiling before end-of-speech')
                codes=new[:new.index(TOKENS['end_speech'])]
                layers=unpack_codes(codes)
                with torch.inference_mode():audio=decoder.decode([torch.tensor([v]) for v in layers])[0,0].float().numpy()
                assert np.isfinite(audio).all() and len(audio)>24000
                sf.write(folder/(name+'-raw.wav'),audio,24000,subtype='PCM_16')
                audio,decision=clean_generated_tail(audio,24000)
                sf.write(folder/(name+'.wav'),audio,24000,subtype='PCM_16')
                subprocess_module=__import__('subprocess')
                subprocess_module.run(['/home/voiceuser/.local/bin/ffmpeg','-hide_banner','-loglevel','error','-n','-i',str(folder/(name+'.wav')),'-codec:a','libmp3lame','-b:a','128k',str(folder/(name+'.mp3'))],check=True)
                results.append(dict(name=name,text=text,epoch=epoch,checkpoint=str(checkpoint),seed=1700,
                    audio_tokens=len(codes),audio_seconds=len(audio)/24000,wall_seconds=time.time()-started,
                    tail_cleanup=decision,manual_listening_complete=False))
            atomic(folder/'results.json',dict(model=MODEL,precision=precision,samples=results))
        finally:
            del decoder;decoder=None;gc.collect();ctypes.CDLL(None).malloc_trim(0)
            FastLanguageModel.for_training(model);model.train();model.config.use_cache=False
            random.setstate(python_rng);np.random.set_state(numpy_rng);torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
            torch.cuda.empty_cache()
    class Progress(TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kwargs):
            row=dict(logs or {},step=state.global_step,epoch=state.epoch,time=time.time(),mode=mode,precision=precision)
            RUN.mkdir(parents=True,exist_ok=True)
            with (RUN/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            status(status='training' if not smoke else 'preflight',step=state.global_step,epoch=state.epoch,
                total_steps=state.max_steps,max_epochs=3,metrics=row,peak_gpu_gib=torch.cuda.max_memory_reserved()/1024**3)
        def on_evaluate(self,args,state,control,metrics=None,**kwargs):
            assert np.isfinite(metrics['eval_loss'])
            status(last_validation=metrics)
        def on_save(self,args,state,control,**kwargs):
            if (not smoke) or (mode=='smoke' and state.global_step==3):
                try:previews(int(state.epoch) if not smoke else 0,Path(args.output_dir)/f'checkpoint-{state.global_step}')
                except Exception:
                    atomic(RUN/f'preview-error-{mode}-{state.global_step}.json',dict(error=traceback.format_exc()))
                    raise
    bf16=next(p for p in model.parameters() if p.requires_grad).dtype==torch.bfloat16
    args=TrainingArguments(output_dir=str(directory),per_device_train_batch_size=1,per_device_eval_batch_size=1,
        gradient_accumulation_steps=1 if smoke else 4,num_train_epochs=3,max_steps=4 if smoke else -1,
        learning_rate=2e-4,optim='adamw_8bit',warmup_steps=5,weight_decay=.001,lr_scheduler_type='linear',
        bf16=bf16,fp16=not bf16,logging_steps=1,seed=3407,data_seed=3407,report_to='none',
        eval_strategy='no' if smoke else 'epoch',save_strategy='steps' if smoke else 'epoch',save_steps=3,
        save_total_limit=2 if smoke else 1,load_best_model_at_end=False,prediction_loss_only=True,
        dataloader_num_workers=0,dataloader_pin_memory=False,remove_unused_columns=False,
        restore_callback_states_from_checkpoint=True)
    trainer=Trainer(model=model,args=args,train_dataset=sets['train'],eval_dataset=sets['validation'],
        data_collator=collate,callbacks=[Progress()])
    resume=str(directory/'checkpoint-3') if mode=='smoke-resume' else None
    if mode=='train':
        checkpoints=list(directory.glob('checkpoint-*'))
        if checkpoints:resume=str(max(checkpoints,key=lambda p:int(p.name.split('-')[-1])))
    status(status='preflight' if smoke else 'training',mode=mode,precision=precision,resume_from=resume,
        train_clips=len(dataset['splits']['train']),validation_clips=len(dataset['splits']['validation']),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    started=time.time();result=trainer.train(resume_from_checkpoint=resume)
    metrics=trainer.evaluate() if smoke else {}
    if smoke:
        assert trainer.state.global_step==4 and np.isfinite(metrics['eval_loss'])
        atomic(RUN/f'{mode}-{precision}-passed.json',dict(metrics=metrics,step=4,resume_from=resume,
            seconds=time.time()-started,peak_gpu_gib=torch.cuda.max_memory_reserved()/1024**3))
    else:
        final=directory/f'checkpoint-{trainer.state.global_step}';tokenizer.save_pretrained(final)
        model.save_pretrained(RUN/'adapter-final');tokenizer.save_pretrained(RUN/'adapter-final')
        history=[r for r in trainer.state.log_history if 'eval_loss' in r]
        best=min(history,key=lambda r:r['eval_loss'])
        atomic(RUN/'training-result.json',dict(final_epoch=trainer.state.epoch,final_checkpoint=str(final),
            best_validation=best,validation_history=history,metrics=result.metrics,precision=precision,
            corpus_fingerprint=dataset['fingerprint'],revisions=setup()['model_revisions']))
        status(status='complete',final_epoch=trainer.state.epoch,final_checkpoint=str(final),best_validation=best)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['smoke','smoke-resume','train'],required=True)
    parser.add_argument('--precision',choices=['16bit','4bit'],default='16bit');args=parser.parse_args()
    try:main(args.mode,args.precision)
    except Exception as error:
        memory=('out of memory' in str(error).lower() or error.__class__.__name__=='OutOfMemoryError')
        atomic(RUN/f'failure-{args.mode}-{args.precision}.json',dict(memory_failure=memory,error=traceback.format_exc()))
        status(status='failed',error=traceback.format_exc());raise
