"""Record explicit owner listening approval for one immutable dataset."""
import argparse,hashlib,json,time
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset',type=Path,required=True)
parser.add_argument('--fingerprint',required=True)
parser.add_argument('--note',required=True)
parser.add_argument('--owner-approved',action='store_true',required=True)
args=parser.parse_args()
report=json.loads((args.dataset/'corpus.json').read_text())
assert report['fingerprint']==args.fingerprint
for split in ['train','validation']:
    path=args.dataset/split/'manifest.csv'
    with path.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==report['splits'][split]['sha256']
record=dict(time=time.time(),fingerprint=args.fingerprint,scope='owner listening review approved dataset for training',note=args.note)
(args.dataset/'owner-review-approved.json').write_text(json.dumps(record,indent=2))
print('Owner approval recorded. No model training started.')
