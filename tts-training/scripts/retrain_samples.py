"""Predeclared, single-attempt generalization and vocabulary evaluation."""
import csv,json,os,pathlib,re,collections
import soundfile as sf
R=pathlib.Path(__file__).resolve().parents[1];os.chdir(R/'upstream')
from TTS_infer_pack.TTS import TTS,TTS_Config
metrics=[json.loads(x) for x in (R/'reports/retrain-epochs.jsonl').read_text().splitlines()]
s1=max(x['epoch'] for x in metrics if x['stage']=='s1');s2=max(x['epoch'] for x in metrics if x['stage']=='s2')
G=R/f'checkpoints/retrain/gpt/she-retrain-v2proplus-e{s1}.ckpt';candidates=list((R/'checkpoints/retrain/sovits').glob(f'she-retrain-v2proplus_e{s2}_s*.pth'));assert G.exists() and len(candidates)==1;S=candidates[0]
config={'device':'cuda','is_half':True,'version':'v2ProPlus','t2s_weights_path':str(G),'vits_weights_path':str(S),'cnhuhbert_base_path':str(R/'upstream/GPT_SoVITS/pretrained_models/chinese-hubert-base'),'bert_base_path':str(R/'upstream/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large')}
(R/'reports/retrain-checkpoints.json').write_text(json.dumps({'s1_epoch':s1,'s2_epoch':s2,'gpt':str(G),'sovits':str(S),'training_clips':1158,'heldout_excluded':True},indent=2)+'\n')
# Never write the live inference config during evaluation.
cfg=TTS_Config({'custom':config});cfg.configs_path=str(R/'configs/retrain-evaluation-infer.yaml');tts=TTS(cfg)
rows=list(csv.DictReader(open('/home/voiceuser/.hermes/cache/voices/she-xtts/manifest.csv')));held=json.loads((R/'reports/retrain-heldout.json').read_text())['filenames'];ref=next(r for r in rows if 5<=float(r['duration_seconds'])<=9 and r['filename'] not in held)
texts=[
'The little girl stood beside the garden gate and wondered who had left it open.',
'I have never seen a rabbit carry an umbrella, she said, looking down the path.',
'Please tell me which road will take us back to the river before nightfall.',
'There was a small wooden table beneath the tree, with three empty cups upon it.',
'She thought the room would look much larger if someone opened the curtains.',
'What a curious morning this has been! I hardly know what will happen next.',
'The old man shook his head and pointed toward a narrow door behind the stairs.',
'For a moment nobody spoke, and all she could hear was the rain against the window.',
'You may borrow my hat, he said, but you must promise to bring it back tomorrow.',
'They followed the winding path until the sound of the river grew louder.',
'I would rather sit here quietly than argue about a question nobody can answer.',
'A bright yellow bird landed on the wall and watched them with great interest.',
'She reached into her pocket and found a little silver key she had almost forgotten.',
'If the boat is ready, we ought to leave while the water is still calm.',
'The woman smiled as though she had been expecting this very question all along.',
'He looked from one face to the other, but neither of them offered an explanation.',
'Beyond the bridge lay a quiet village where every house had a blue front door.',
'Now listen carefully, she whispered, because I shall only tell you this once.',
'The clock struck nine just as they arrived at the end of the long passage.',
'She could not remember the dream, though she was certain it had been a pleasant one.'
]
plan=[{'name':f'heldout-{i:02d}','group':'heldout-style-new-text','style_clip':n,'text':t} for i,(n,t) in enumerate(zip(held,texts),1)]
# Fix evaluation prompts before synthesis. No reference audio or transcript comes from held-out clips.
words=['server','computer','window','garden','river','mother','table','phone','music','coffee','train','water','book','door','friend','morning','weather','message','kitchen','light']
vocab=collections.Counter(re.findall(r"[a-z]+",' '.join(r['text'] for r in rows).lower()))
for word in words:plan.append({'name':'vocabulary-'+word,'group':'vocabulary','word':word,'corpus_occurrences':vocab[word],'text':f'She said the word {word} clearly.'})
assert all(p['text'] not in {r['text'] for r in rows} for p in plan)
(R/'configs/retrain-sample-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
out=R/'samples/retrain';out.mkdir(parents=True,exist_ok=True);results=[]
for i,p in enumerate(plan):
 inputs={'text':p['text'],'text_lang':'en','ref_audio_path':str(R/'data/wavs32k'/ref['filename']),'prompt_text':ref['text'],'prompt_lang':'en','top_k':15,'top_p':1.0,'temperature':0.8,'speed_factor':1.0,'text_split_method':'cut5','batch_size':1,'seed':20261112+i,'parallel_infer':False,'use_cuda_graph':False,'use_flash_attention':False,'repetition_penalty':1.35}
 chunks=list(tts.run(inputs));assert len(chunks)==1;sr,audio=chunks[0];assert sr==32000 and len(audio)>0
 path=out/(p['name']+'.wav');sf.write(path,audio,sr,subtype='PCM_16');results.append({**p,'path':str(path),'reference_filename':ref['filename'],'gpt_checkpoint':str(G),'sovits_checkpoint':str(S),'seed':20261112+i})
 (R/'reports/retrain-samples.json').write_text(json.dumps(results,indent=2)+'\n');print(p['name'],len(audio)/sr,flush=True)
