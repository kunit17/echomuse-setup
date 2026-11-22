from common import *
from huggingface_hub import snapshot_download
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--quantized',action='store_true');args=parser.parse_args()
name=QUANT if args.quantized else MODEL
path=snapshot_download(name,revision=setup()['model_revisions'][name])
atomic(ROOT/('quantized-model.json' if args.quantized else 'model.json'),dict(model=name,path=path,revision=setup()['model_revisions'][name]))
print('Model cached',name,path,flush=True)
