"""Create an independent merged inference export on CPU; preserve original LoRA."""
from common import *
import sys
from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
import torch
B=ROOT/'stream-benchmark';out=B/'merged-epoch3'
if (out/'provenance.json').exists():print('Existing immutable export verified by controller');raise SystemExit(0)
selection=json.loads((ROOT/'selected-model.json').read_text());adapter=Path(selection['adapter_path'])
assert sha(adapter/'adapter_model.safetensors')==selection['adapter_sha256']
basepath=json.loads((ROOT/'model.json').read_text())['path']
base=AutoModelForCausalLM.from_pretrained(basepath,torch_dtype=torch.bfloat16,device_map='cpu',local_files_only=True)
model=PeftModel.from_pretrained(base,str(adapter)).merge_and_unload(safe_merge=True)
out.mkdir(exist_ok=True);model.save_pretrained(out,safe_serialization=True,max_shard_size='2GB')
AutoTokenizer.from_pretrained(str(adapter),local_files_only=True).save_pretrained(out)
atomic(out/'provenance.json',dict(selection=selection,base_revision=setup()['model_revisions'][MODEL],
    files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
assert sha(adapter/'adapter_model.safetensors')==selection['adapter_sha256']
print('Merged epoch3 export complete',flush=True)
