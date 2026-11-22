"""Owner-authorized retention: final checkpoints and active deployments only."""
import argparse,csv,fcntl,hashlib,json,os,shutil,time
from pathlib import Path
CSM=Path('/home/voiceuser/projects/csm-lora')
REPO=Path('/home/voiceuser/projects/gpt-sovits')
HOME=Path('/home/voiceuser')
REPORT=REPO/'reports/training-storage-cleanup-20261118.json'
ARCHIVE=REPO/'reports/training-storage-cleanup-20261118-metadata'
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def bytes_in(path):
    if path.is_file():return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink())
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    selected=json.loads((CSM/'selected-model.json').read_text())
    deployment=json.loads(Path('/home/voiceuser/projects/csm-serve/deployment.json').read_text())
    assert selected['adapter_path']==deployment['adapter_path']
    protected=[Path(selected['adapter_path']).resolve(),Path(selected['reference']['path']).resolve(),
               REPO/'data/shecorpus',HOME/'projects/she-training',HOME/'projects/corpus-source',
               CSM/'.cache/huggingface',REPO/'.cache/whisper-large-v3',REPO/'.cache/repaired-corpus']
    config=json.loads((CSM/'corpus-config.json').read_text())
    manifests={config[s+'_manifest']:sha(Path(config[s+'_manifest'])) for s in ['train','validation']}
    model_sha=sha(Path(selected['adapter_path'])/'adapter_model.safetensors')
    assert model_sha==selected['adapter_sha256']
    targets=[];retained=[];processor_copies=[]
    def add(path,reason):
        if not path.exists():return
        assert not path.is_symlink(),str(path)
        absolute=path.resolve()
        permitted=[CSM/'runs',REPO/'checkpoints',REPO/'experiments',HOME/'.cache/pip']
        assert any(absolute.is_relative_to(p.resolve()) and absolute!=p.resolve() for p in permitted),str(path)
        assert not any(absolute==p or absolute.is_relative_to(p) or p.is_relative_to(absolute) for p in protected),str(path)
        assert path not in [p for p,_ in targets]
        targets.append((path,reason))
    for run in sorted((CSM/'runs').iterdir()):
        checkpoints=[]
        for checkpoint in (run/'checkpoints').glob('checkpoint-*'):
            state=json.loads((checkpoint/'trainer_state.json').read_text())
            checkpoints.append((state['epoch'],state['global_step'],checkpoint))
        if not checkpoints:continue
        latest=max(checkpoints)[2]
        retained.append(dict(run=run.name,path=str(latest),epoch=max(checkpoints)[0]))
        for _,_,checkpoint in checkpoints:
            if checkpoint!=latest:add(checkpoint,'older CSM full checkpoint')
        best=run/'adapter-best'
        if best.exists():
            # Keep processor files with the final adapter so it remains directly usable.
            processor_copies.append((best,latest))
            if best.resolve()!=Path(selected['adapter_path']).resolve():add(best,'non-final best adapter; owner requested final epochs only')
    old=REPO/'experiments/approved-a098bf96-20261113T104538'
    for name in ['G_3945.pth','D_3945.pth','G_6033.pth','D_6033.pth']:
        add(old/'train/logs_s2_v2ProPlus'/name,'older SoVITS optimizer/model state')
    add(old/'gpt-training/ckpt/epoch=1-step=132.ckpt','older GPT full training checkpoint')
    for split in ['train','validation']:
        for name in ['4-cnhubert','5-wav32k','7-sv_cn']:
            add(old/split/name,'obsolete extracted training features from retired corpus')
    exports=REPO/'checkpoints/approved-a098bf96-20261113T104538'
    add(exports/'gpt/she-approved-a098bf96-20261113T104538-e2.ckpt','older GPT inference checkpoint')
    for epoch,step in [(15,3945),(23,6033)]:
        add(exports/f'sovits/she-approved-a098bf96-20261113T104538_e{epoch}_s{step}.pth','older SoVITS inference checkpoint')
    for name in ['http-v2','http','wheels']:
        add(HOME/'.cache/pip'/name,'downloaded package cache; installed environments unchanged')
    records=[dict(path=str(path),reason=reason,logical_bytes=bytes_in(path)) for path,reason in targets]
    report=dict(created=time.time(),policy='keep final epoch checkpoints and active selected deployment; remove older best weights',
        retained_csm=retained,retained_sovits_epoch=25,retained_gpt_epoch=7,
        configured_legacy_sovits_fallback_preserved=True,targets=records,
        logical_bytes_to_remove=sum(r['logical_bytes'] for r in records),applied=False)
    print(json.dumps(report,indent=2),flush=True)
    if not args.apply:return
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:argv=(entry/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        names={Path(arg.decode(errors='ignore')).name for arg in argv if arg}
        if names & {'train.py','asr_worker.py'}:raise RuntimeError('Training/ASR job active; do not clean')
    with (REPO/'.cache/training-launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        free_before=shutil.disk_usage(CSM).free
        ARCHIVE.mkdir(exist_ok=True)
        for best,latest in processor_copies:
            for name in ['chat_template.jinja','preprocessor_config.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json']:
                if (best/name).exists() and not (latest/name).exists():shutil.copy2(best/name,latest/name)
        for path,reason in targets:
            # Preserve lightweight checkpoint provenance before removing weights.
            if path.is_dir() and 'package cache' not in reason:
                for metadata in path.rglob('*.json'):
                    if metadata.stat().st_size>2_000_000:continue
                    relative=metadata.relative_to(HOME)
                    destination=ARCHIVE/relative
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(metadata,destination)
            if path.is_dir():shutil.rmtree(path)
            else:path.unlink()
        for item in retained:
            run=CSM/'runs'/item['run']
            retention=dict(retained_checkpoint=item['path'],retained_epoch=item['epoch'],
                best_validation_weights_removed_by_owner=True,metadata_archive=str(ARCHIVE))
            (run/'retention.json').write_text(json.dumps(retention,indent=2))
            status=run/'status.json'
            if status.exists():
                state=json.loads(status.read_text())
                if state.get('adapter')==str(run/'adapter-best'):state['adapter']=item['path']
                state['retention']=retention;status.write_text(json.dumps(state,indent=2))
        assert sha(Path(selected['adapter_path'])/'adapter_model.safetensors')==model_sha
        assert all(sha(Path(path))==expected for path,expected in manifests.items())
        report.update(applied=True,finished=time.time(),free_before=free_before,free_after=shutil.disk_usage(CSM).free,
            retained_selected_model_sha256=model_sha,current_dataset_hashes_unchanged=True)
        report['filesystem_bytes_reclaimed']=report['free_after']-free_before
        REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(applied=True,filesystem_bytes_reclaimed=report['filesystem_bytes_reclaimed'],report=str(REPORT))),flush=True)
if __name__=='__main__':main()
