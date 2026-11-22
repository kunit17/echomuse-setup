import json,pathlib,yaml
R=pathlib.Path(__file__).resolve().parents[1]; E=R/'experiments/she-v2proplus'; P=R/'upstream/GPT_SoVITS/pretrained_models'; E.mkdir(parents=True,exist_ok=True)
s1=yaml.safe_load((R/'upstream/GPT_SoVITS/configs/s1longer-v2.yaml').read_text())
s1['train'].update(epochs=3,batch_size=2,precision='16-mixed',save_every_n_epoch=1,if_save_every_weights=True,if_save_latest=False,if_dpo=False,half_weights_save_dir=str(R/'checkpoints/gpt'),exp_name='she-v2proplus')
s1['data']['num_workers']=2
s1.update(pretrained_s1=str(P/'gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt'),train_semantic_path=str(E/'6-name2semantic.tsv'),train_phoneme_path=str(E/'2-name2text.txt'),output_dir=str(E/'logs_s1_v2ProPlus'))
s2=json.loads((R/'upstream/GPT_SoVITS/configs/s2v2ProPlus.json').read_text())
s2['train'].update(epochs=5,batch_size=2,pretrained_s2G=str(P/'v2Pro/s2Gv2ProPlus.pth'),pretrained_s2D=str(P/'v2Pro/s2Dv2ProPlus.pth'),if_save_latest=False,if_save_every_weights=True,save_every_epoch=1,gpu_numbers='0',grad_ckpt=True,log_interval=25)
s2['model']['version']='v2ProPlus'; s2['data']['exp_dir']=str(E)
s2.update(s2_ckpt_dir=str(E),save_weight_dir=str(R/'checkpoints/sovits'),name='she-v2proplus',version='v2ProPlus')
(R/'configs/s1.yaml').write_text(yaml.safe_dump(s1));(R/'configs/s2.json').write_text(json.dumps(s2,indent=2)+'\n')
for p in ['checkpoints/gpt','checkpoints/sovits','experiments/she-v2proplus/logs_s2_v2ProPlus']:(R/p).mkdir(parents=True,exist_ok=True)
