"""Exercise real SNAC prefix coverage and final flush on frozen feature tokens."""
from common import *
from stream_core import *
import numpy as np,torch
from snac import SNAC
from huggingface_hub import snapshot_download
torch.set_num_threads(4)
codec=SNAC.from_pretrained(snapshot_download(CODEC,revision=setup()['model_revisions'][CODEC],local_files_only=True)).cpu().eval()
item=json.loads(next((DATA/'tokens').glob('*.json')).read_text());ids=item['input_ids']
codes=ids[ids.index(TOKENS['start_speech'])+1:ids.index(TOKENS['end_speech'])]
validate_codes(codes)
def decode(values):
    with torch.inference_mode():return codec.decode([torch.tensor([v]) for v in unpack_codes(values)])[0,0].numpy()
pieces=[];emitted=0
for groups in range(12,len(codes)//7+1,4):
    full=decode(codes[:groups*7]);end=safe_end(len(full),groups)
    assert len(full)==groups*2048
    if end>emitted:pieces.append(full[emitted:end].copy());emitted=end
offline=decode(codes);pieces.append(offline[emitted:]);stream=np.concatenate(pieces)
assert len(stream)==len(offline)==len(codes)//7*2048
assert np.array_equal(stream[-min(7200,len(stream)):],offline[-min(7200,len(offline)):])
atomic(ROOT/'stream-benchmark/codec-check.json',dict(passed=True,groups=len(codes)//7,samples=len(stream),
    missing_samples=0,terminal_samples_identical=True,mae=float(np.mean(np.abs(stream-offline))),peak=float(np.max(np.abs(stream-offline)))))
print('Real codec prefix/final-flush check passed',flush=True)
