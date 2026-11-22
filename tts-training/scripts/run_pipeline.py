"""Resume-safe feature extraction and upstream two-stage training."""
import datetime,json,os,pathlib,subprocess,sys,time
R=pathlib.Path(__file__).resolve().parents[1]; U=R/'upstream'; E=R/'experiments/she-v2proplus'; P=U/'GPT_SoVITS/pretrained_models'
E.mkdir(parents=True,exist_ok=True)
os.chdir(U)
os.environ.update(inp_text=str(R/'data/she.list'),inp_wav_dir='',exp_name='she-v2proplus',i_part='0',all_parts='1',opt_dir=str(E),bert_pretrained_dir=str(P/'chinese-roberta-wwm-ext-large'),cnhubert_base_dir=str(P/'chinese-hubert-base'),sv_path=str(P/'sv/pretrained_eres2netv2w24s4ep4.ckpt'),pretrained_s2G=str(P/'v2Pro/s2Gv2ProPlus.pth'),s2config_path=str(U/'GPT_SoVITS/configs/s2v2ProPlus.json'))
def run(stage,args):
 start=time.time(); print(datetime.datetime.now().isoformat(),stage,flush=True)
 with (R/f'logs/{stage}.log').open('a') as log:
  result=subprocess.run([sys.executable,*args],stdout=log,stderr=subprocess.STDOUT)
 with (R/'reports/stages.jsonl').open('a') as f:f.write(json.dumps({'stage':stage,'started':start,'seconds':time.time()-start,'exit_code':result.returncode})+'\n')
 if result.returncode:raise RuntimeError(f'{stage} failed with exit code {result.returncode}; see logs/{stage}.log')
def count(path):return len(path.read_text().strip().splitlines())
run('features-text',['GPT_SoVITS/prepare_datasets/1-get-text.py'])
assert count(E/'2-name2text-0.txt')==1217,'text extraction lost clips'
(E/'2-name2text.txt').write_text((E/'2-name2text-0.txt').read_text())
run('features-hubert',['GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py'])
for d,pattern in [('4-cnhubert','*.pt'),('5-wav32k','*.wav')]:assert len(list((E/d).glob(pattern)))==1217,d
run('features-speaker',['GPT_SoVITS/prepare_datasets/2-get-sv.py'])
assert len(list((E/'7-sv_cn').glob('*.pt')))==1217
run('features-semantic',['GPT_SoVITS/prepare_datasets/3-get-semantic.py'])
assert count(E/'6-name2semantic-0.tsv')==1217
(E/'6-name2semantic.tsv').write_text('item_name\tsemantic_audio\n'+(E/'6-name2semantic-0.tsv').read_text().strip()+'\n')
run('audit-training-data',[str(R/'scripts/audit_training_data.py')])
run('train-s1',[str(R/'scripts/run_training.py'),'s1'])
run('train-s2',[str(R/'scripts/run_training.py'),'s2'])
(R/'reports/training-complete.json').write_text(json.dumps({'completed':datetime.datetime.now().isoformat()})+'\n')
run('generate-samples',[str(R/'scripts/generate_samples.py')])
run('verify-samples',[str(R/'scripts/verify_samples.py')])
run('verify-source',[str(R/'scripts/verify_source.py')])
run('export-metrics',[str(R/'scripts/export_metrics.py')])
run('build-report',[str(R/'scripts/build_report.py')])
print('PIPELINE COMPLETE',flush=True)
