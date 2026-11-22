"""Select an approved dataset for a new CSM run; never starts training."""
import argparse,csv,hashlib,json,runpy,time,tempfile
from pathlib import Path
ROOT=Path('/home/voiceuser/projects/csm-lora')
REPO=Path('/home/voiceuser/projects/gpt-sovits')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,type=Path)
    parser.add_argument('--run-name',required=True)
    parser.add_argument('--check-only',action='store_true',help='Validate without selecting a dataset or changing training state')
    args=parser.parse_args()
    import re,fcntl
    assert re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}',args.run_name)
    dataset=args.dataset.resolve()
    assert dataset.is_relative_to(REPO/'data/shecorpus/repaired')
    assert not (ROOT/'runs'/args.run_name).exists(), 'Use a new run name for fresh training'
    report=json.loads((dataset/'corpus.json').read_text())
    approval=json.loads((dataset/'owner-review-approved.json').read_text())
    assert approval.get('fingerprint')==report['fingerprint'], 'Record owner approval for this exact fingerprint first'
    plan=dataset/'source-plan.json';assert plan.exists()
    config=dict(schema='repaired-v1',run_name=args.run_name,corpus_report=str(dataset/'corpus.json'),
        fingerprint=report['fingerprint'],source_plan=str(plan),excluded_filenames=[],
        training_paused=True,owner_listening_review_approved=True,approved_at=approval['time'])
    for split in ['train','validation']:
        manifest=dataset/split/'manifest.csv'
        assert sha(manifest)==report['splits'][split]['sha256']
        config[split+'_manifest']=str(manifest);config[split+'_clips']=report['splits'][split]['clips']
    with (REPO/'.cache/training-launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Run the real trainer's full integrity/exclusion/split checks before selection.
        module=runpy.run_path(str(ROOT/'train.py'))
        validation=module['prepare'].__globals__
        with tempfile.TemporaryDirectory(prefix='corpus-training-check-') as temporary:
            validation.update(CONFIG=config,COUNTS={s:config[s+'_clips'] for s in ['train','validation']},RUN=Path(temporary))
            module['prepare']()
        if args.check_only:
            print('PASS: full dataset integrity, source exclusions and split containment; no configuration or training state changed.')
            return
        existing=ROOT/'corpus-config.json'
        backup=ROOT/f'corpus-config.before-{args.run_name}.json'
        backup.write_bytes(existing.read_bytes())
        temporary=existing.with_suffix('.tmp');temporary.write_text(json.dumps(config,indent=2));temporary.replace(existing)
        (ROOT/'PAUSED_BY_OWNER').write_text('Dataset selected; remove this guard only with owner authorization to start training.\n')
    print('Dataset verified and selected. Training remains paused. See TTS_WORKFLOW.md.')
if __name__=='__main__':main()
