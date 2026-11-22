import pathlib,torch,json
R=pathlib.Path(__file__).resolve().parents[1]
p=R/'upstream/GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt'
d=torch.load(p,map_location='cpu',weights_only=False)
print(d.keys());print(d.get('config'));print([(k,tuple(v.shape)) for k,v in d['weight'].items() if 'embedding' in k])
(R/'reports/pretrained-gpt-architecture.json').write_text(json.dumps({'config':d.get('config'),'embeddings':{k:list(v.shape) for k,v in d['weight'].items() if 'embedding' in k}},indent=2)+'\n')
