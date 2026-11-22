"""Run identity, immutable inputs, stopping and checkpoint retention (no model imports)."""
import hashlib
import json
import os
from pathlib import Path
from asr_runtime import atomic_json

POLICIES={'s1':dict(max_epochs=20,min_epochs=3,patience=5,min_relative_improvement=.005,metric='nll_per_token'),
          's2':dict(max_epochs=25,min_epochs=25,patience=5,min_relative_improvement=.005,metric='mel')}

def verify_inputs(run):
    for name,expected in run['input_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Corpus input changed: '+name)

def run_context(root):
    run_id=os.environ.get('SHE_TRAINING_RUN','')
    if not run_id or not all(c.isalnum() or c in '-_' for c in run_id):
        raise RuntimeError('Use the approved run launcher with a recorded run ID')
    run=json.loads((Path(root)/'reports/training-runs'/run_id/'run.json').read_text())
    assert run['run_id']==run_id and run['policies']==POLICIES
    verify_inputs(run)
    return run

def should_stop(rows,stage):
    policy=POLICIES[stage];rows=sorted([r for r in rows if r['stage']==stage and r['epoch']>0],key=lambda r:r['epoch'])
    if not rows:return False
    if rows[-1]['epoch']>=policy['max_epochs']:return True
    if rows[-1]['epoch']<policy['min_epochs']:return False
    best=None;best_epoch=0
    for row in rows:
        value=float(row['validation'][policy['metric']])
        if best is None or value<best*(1-policy['min_relative_improvement']):
            best=value;best_epoch=row['epoch']
    return rows[-1]['epoch']-best_epoch>=policy['patience']

def prune_checkpoints(run,rows,stage):
    adapted=[r for r in rows if r['stage']==stage and r['epoch']>0]
    if not adapted:return
    metric=POLICIES[stage]['metric']
    best=min(adapted,key=lambda r:(r['validation'][metric],r['validation'].get('kl',0),r['epoch']))
    latest=max(adapted,key=lambda r:r['epoch'])
    keep=set()
    for row in [best,latest]:
        keep.add(Path(row['checkpoint']).resolve())
        if row.get('full_checkpoint'):keep.add(Path(row['full_checkpoint']).resolve())
        keep.update(Path(p).resolve() for p in row.get('full_checkpoints',[]))
    checkpoint_root=Path(run['checkpoints']).resolve();features=Path(run['features']).resolve()
    folders=[checkpoint_root/('gpt' if stage=='s1' else 'sovits'),
             features/'gpt-training/ckpt' if stage=='s1' else features/'train/logs_s2_v2ProPlus']
    if stage=='s2':
        keep.update(Path(path).resolve() for path in run.get('pinned_checkpoints',[]))
    if any(not (p.is_relative_to(checkpoint_root) or p.is_relative_to(features)) for p in keep):
        raise RuntimeError('Retained checkpoint escaped this run')
    if not all(p.is_file() for p in keep):raise RuntimeError('Best/latest checkpoint is missing; refusing pruning')
    removed=[]
    for folder in folders:
        if folder.is_symlink():raise RuntimeError('Unexpected checkpoint-directory alias')
        for path in folder.iterdir():
            if path.suffix not in ['.ckpt','.pth'] or not path.is_file():continue
            resolved=path.resolve()
            if resolved in keep:continue
            if path.is_symlink() or not (resolved.is_relative_to(checkpoint_root) or resolved.is_relative_to(features)):
                raise RuntimeError('Checkpoint path escaped this run')
            path.unlink();removed.append(str(path))
    atomic_json(Path(run['reports'])/(stage+'-retention.json'),dict(best_epoch=best['epoch'],latest_epoch=latest['epoch'],
        kept=sorted(map(str,keep)),removed_this_epoch=removed,policy='best and latest only'))
