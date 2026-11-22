"""Metadata-only migration. Does not decode audio, import ASR, or start services."""
import argparse
from pathlib import Path
import time
from asr_runtime import agreement_report,archive,atomic_json,file_digest,read_json,summarize_filters


def migrate(root):
    root=Path(root);reports=root/'reports';history=reports/'asr-history'
    preflight=reports/'corpus-asr-preflight.json'
    value=read_json(preflight,{})
    if value and not value.get('schema_version') and not value.get('run_id'):
        archive(preflight,history)
        candidates=[]
        for path in reports.glob('corpus-asr-cuda-float16-*.json'):
            item=read_json(path,{})
            if item.get('status')=='passed' and item.get('device')=='cuda' and item.get('compute_type')=='float16':
                candidates.append((path,item))
        evidence=[]
        for path,item in candidates:
            evidence.append(dict(path=str(path.relative_to(root)),sha256=file_digest(path)['sha256'],
                                 status='historical_passed',compute_type='float16',device='cuda',
                                 transcription_seconds=item.get('transcription_seconds'),audio_seconds=item.get('audio_seconds')))
        atomic_json(preflight,dict(schema_version=2,status='not_run_since_hardening',
                                  historical_evidence=evidence,previous_result_archived=True,
                                  note='Historical failures and passes are not a new execution of the hardened job'))
    path=reports/'corpus-verification.json';value=read_json(path,{})
    if 'validation_asr' in value:
        archive(path,history)
        old=value.pop('validation_asr')
        count=value.get('splits',{}).get('validation',{}).get('clips',len(old.get('per_clip',[])))
        value['asr_pass_agreement']=agreement_report([None]*count)
        source=value.get('excluded_source_seconds',0)+value.get('seconds',0)
        value['filtering']=summarize_filters(value.get('exclusions',[]),value.get('seconds',0),source)
        value['report_schema_version']=2
        value['provenance_status']='historical_result_without_hardened_run_fingerprint'
        value['metadata_migration']=dict(time=time.time(),asr_rerun=False,privacy_policy_changed=False,
                                         change='Report two-pass agreement separately from reference-based WER')
        atomic_json(path,value)
    if not (reports/'asr-status.json').exists():
        atomic_json(reports/'asr-status.json',dict(schema=2,status='not_started',run_id=None,
                                                note='Hardening installed; no model job was started',gpu_model_loaded=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    migrate(parser.parse_args().root)
