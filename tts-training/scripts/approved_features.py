"""Extract new-recording features into separate training/validation roots."""
import argparse
import csv
import json
import os
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
from training_run import run_context
RUN=run_context(ROOT)
BASE=Path(RUN['features'])
PRETRAINED=ROOT/'upstream/GPT_SoVITS/pretrained_models'
args=argparse.ArgumentParser();args.add_argument('partition',choices=['train','validation']);args.add_argument('stage',choices=['text','hubert','speaker','semantic','verify']);args=args.parse_args()
EXP=BASE/args.partition;EXP.mkdir(parents=True,exist_ok=True)
rows=list(csv.DictReader((ROOT/f'data/shecorpus/{args.partition}/manifest.csv').open(encoding='utf-8')))
names={Path(r['filename']).name for r in rows};assert names
frozen=json.loads((ROOT/'reports/corpus-split.json').read_text())['source_frames']
assert len(names)==len(rows)
for row in rows:
    start,end=int(row['source_start_frame']),int(row['source_end_frame'])
    assert start<end and any(a<=start<end<=b for a,b in frozen[args.partition])
other_partition='validation' if args.partition=='train' else 'train'
other=list(csv.DictReader((ROOT/f'data/shecorpus/{other_partition}/manifest.csv').open(encoding='utf-8')))
assert not names & {Path(r['filename']).name for r in other}
for row in rows:
    start,end=int(row['source_start_frame']),int(row['source_end_frame'])
    assert not any(start<int(r['source_end_frame']) and end>int(r['source_start_frame']) for r in other)
os.environ.update(inp_text=str(ROOT/f'data/shecorpus/{args.partition}/she.list'),inp_wav_dir='',exp_name=RUN['run_name'],
    i_part='0',all_parts='1',opt_dir=str(EXP),bert_pretrained_dir=str(PRETRAINED/'chinese-roberta-wwm-ext-large'),
    cnhubert_base_dir=str(PRETRAINED/'chinese-hubert-base'),sv_path=str(PRETRAINED/'sv/pretrained_eres2netv2w24s4ep4.ckpt'),
    pretrained_s2G=str(PRETRAINED/'v2Pro/s2Gv2ProPlus.pth'))
os.chdir(ROOT/'upstream')
if args.stage=='text':
    # English uses zero BERT features upstream. Avoid loading an unused Chinese
    # model and preserve the established English cleaner without upstream edits.
    from text.cleaner import clean_text
    result=[]
    for row in rows:
        phones,word2ph,text=clean_text(row['text'],'en','v2ProPlus')
        assert phones
        result.append(f"{Path(row['filename']).name}\t{' '.join(phones)}\t{word2ph}\t{text}")
    with (EXP/'2-name2text.txt').open('w') as f:f.write('\n'.join(result)+'\n');f.flush();os.fsync(f.fileno())
elif args.stage=='verify':
    import torch
    import numpy as np
    from scipy.io import wavfile
    from corpus_audio_metrics import frame_rms,quiet_at
    embeddings=[]
    normalization=[]
    for row in rows:
        name=Path(row['filename']).name
        for directory in ('4-cnhubert','7-sv_cn'):
            value=torch.load(EXP/directory/(name+'.pt'),map_location='cpu',weights_only=False)
            assert value.numel()>0 and torch.isfinite(value).all()
            if directory=='7-sv_cn':embeddings.append(value.float().reshape(-1).numpy())
        sr,signal=wavfile.read(EXP/'5-wav32k'/name);assert sr==32000 and len(signal)>0
        source_sr,source_signal=wavfile.read(ROOT/row['filename'])
        assert source_sr==sr and signal.shape==source_signal.shape
        before=source_signal.astype(np.float32)/32768;after=signal.astype(np.float32)/32768
        measurement={}
        for label,values in [('corpus',before),('feature',after)]:
            rms=frame_rms(values,sr);peak=float(np.max(np.abs(values)))
            measurement[label]=dict(peak_amplitude=peak,quiet_tail_seconds={
                key:quiet_at(rms,len(rms)-1,threshold) for key,threshold in [('minus_40_dbfs',.01),('relative_2pct_of_clip_peak',peak*.02)]})
        normalization.append(dict(filename=name,**measurement))
    text_names={line.split('\t')[0] for line in (EXP/'2-name2text.txt').read_text().splitlines()}
    semantic=(EXP/'6-name2semantic.tsv').read_text().splitlines()
    assert semantic.pop(0)=='item_name\tsemantic_audio'
    assert text_names==names=={line.split('\t')[0] for line in semantic}
    assert all(len(line.split('\t')[1].split())>0 for line in semantic)
    np.save(EXP/'verified-speaker-embeddings.npy',np.stack(embeddings))
    from corpus_ground_truth import save
    save(Path(RUN['reports'])/f'{args.partition}-feature-audit.json',dict(verified_clips=len(names),
        normalization_method='upstream linear gain 0.5 * 0.95 / peak + 0.5; no padding',per_clip=normalization))
    print('Verified isolated feature rows',args.partition,len(names),flush=True)
else:
    script={'hubert':'2-get-hubert-wav32k.py','speaker':'2-get-sv.py','semantic':'3-get-semantic.py'}[args.stage]
    if args.stage=='hubert':
        from resource_safe import bounded_hubert_loading
        bounded_hubert_loading()
    if args.stage=='semantic':
        import torch
        import module.models as models
        original_generator=models.SynthesizerTrn
        def generator(*a,**kw):
            with torch.device('cuda'):return original_generator(*a,**kw)
        models.SynthesizerTrn=generator
        original_load=torch.load
        def load(*a,**kw):
            if a and isinstance(a[0],(str,Path)) and str(a[0]).endswith('.pth'):kw['map_location']='cuda'
            return original_load(*a,**kw)
        torch.load=load
        cfg=json.loads((ROOT/'upstream/GPT_SoVITS/configs/s2v2ProPlus.json').read_text());cfg['model'].pop('version',None)
        config=BASE/'semantic-config.json';config.write_text(json.dumps(cfg));os.environ['s2config_path']=str(config)
    runpy.run_path(str(ROOT/'upstream/GPT_SoVITS/prepare_datasets'/script),run_name='__main__')
    if args.stage=='semantic':
        original=EXP/'6-name2semantic-0.tsv'
        lines=original.read_text().splitlines();assert {line.split('\t')[0] for line in lines}==names
        with (EXP/'6-name2semantic.tsv').open('w') as f:f.write('item_name\tsemantic_audio\n'+'\n'.join(lines)+'\n');f.flush();os.fsync(f.fileno())

# Non-empty / exact-name assertions after every feature stage.
if args.stage=='text':
    lines=(EXP/'2-name2text.txt').read_text().splitlines()
    assert len(lines)==len(names) and {line.split('\t')[0] for line in lines}==names
elif args.stage in ['hubert','speaker']:
    directories=['4-cnhubert','5-wav32k'] if args.stage=='hubert' else ['7-sv_cn']
    for directory in directories:
        suffix='.pt' if directory!='5-wav32k' else ''
        actual={p.name for p in (EXP/directory).iterdir() if p.is_file()}
        assert actual=={name+suffix for name in names},f'Unexpected output names in {directory}'
        assert all((EXP/directory/(name+suffix)).stat().st_size>0 for name in names)
elif args.stage=='semantic':
    lines=(EXP/'6-name2semantic.tsv').read_text().splitlines()
    assert lines[0]=='item_name\tsemantic_audio' and len(lines)==len(names)+1
    assert {line.split('\t')[0] for line in lines[1:]}==names
    assert all(len(line.split('\t')[1].split())>0 for line in lines[1:])
print('FEATURE_STAGE_VERIFIED',args.partition,args.stage,len(names),flush=True)
