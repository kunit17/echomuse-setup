"""Download only the upstream assets needed for English V2ProPlus."""
import concurrent.futures, hashlib, json, pathlib, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[1]
tree = json.loads((ROOT/'reports/hf-tree.json').read_text())
names = ['gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt','v2Pro/s2Dv2ProPlus.pth','v2Pro/s2Gv2ProPlus.pth','s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt']
files = [f for f in tree if f['type']=='file' and (f['path'] in names or f['path'].startswith(('sv/','chinese-hubert-base/','chinese-roberta-wwm-ext-large/')))]
def get(f):
 p=ROOT/'upstream/GPT_SoVITS/pretrained_models'/f['path']; p.parent.mkdir(parents=True,exist_ok=True)
 url='https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/'+f['path']
 if not p.exists() or p.stat().st_size!=f['size']:
  subprocess.run(['curl','-L','--fail','--retry','5','-C','-','-o',str(p)+'.part',url],check=True)
  pathlib.Path(str(p)+'.part').rename(p)
 assert p.stat().st_size==f['size'],p
 result={'path':str(p.relative_to(ROOT)),'url':url,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 print(json.dumps(result),flush=True); return result
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool: results=list(pool.map(get,files))
(ROOT/'reports/assets.json').write_text(json.dumps(results,indent=2)+'\n')
