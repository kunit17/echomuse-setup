import json,os,pathlib,yaml
from AR.data.dataset import Text2SemanticDataset
from module.data_utils import TextAudioSpeakerLoader
from utils import get_hparams_from_file
R=pathlib.Path(__file__).resolve().parents[1];retrain=os.environ.get('SHE_RETRAIN')=='1';prefix='retrain-' if retrain else '';E=R/('experiments/'+('she-retrain-v2proplus' if retrain else 'she-v2proplus'))
s1=Text2SemanticDataset(str(E/'2-name2text.txt'),str(E/'6-name2semantic.tsv'),max_sec=54)
s2=TextAudioSpeakerLoader(get_hparams_from_file(str(R/('configs/'+prefix+'s2.json'))).data,version='v2ProPlus')
expected={p.name for p in (R/'data/wavs32k').glob('*.wav')}
if retrain:expected-=set(json.loads((R/'reports/retrain-heldout.json').read_text())['filenames'])
assert set(s1.item_names)==expected,sorted(expected-set(s1.item_names))
assert {x[0] for x in s2.audiopaths_sid_text}==expected
(R/('reports/'+prefix+'training-data-audit.json')).write_text(json.dumps({'s1_clips':len(s1),'s2_clips':len(s2),'exclusions':[]},indent=2)+'\n')
