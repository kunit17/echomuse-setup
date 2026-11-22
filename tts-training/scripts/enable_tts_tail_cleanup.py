"""Enable the shared inference-only tail cleanup in the existing CSM service."""
from pathlib import Path
import ast,shutil
SERVE=Path('/home/voiceuser/projects/csm-serve')
MODULE='/home/voiceuser/projects/gpt-sovits/scripts'
path=SERVE/'serve_csm.py';text=path.read_text()
if 'from audio_postprocess import clean_generated_tail, POLICY_ID' not in text:
    backup=SERVE/'.runtime/serve_csm.before-tail-cleanup.py'
    backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,backup)
    imports="import sys\nsys.path.insert(0,"+repr(MODULE)+")\nfrom audio_postprocess import clean_generated_tail, POLICY_ID\n"
    text=text.replace("BASE = 'unsloth/csm-1b'",imports+"\nBASE = 'unsloth/csm-1b'",1)
    needle='                parts.append(audio)'
    assert text.count(needle)==1
    text=text.replace(needle,"                audio, tail_cleanup = clean_generated_tail(audio, SR)\n                logging.info('audio_tail_cleanup=%s', json.dumps(tail_cleanup))\n"+needle,1)
    needle="mode='with-context, attached LoRA', adapter_proof=proof, epoch=DEPLOYMENT['epoch'],"
    assert needle in text
    text=text.replace(needle,needle+' postprocessing_policy=POLICY_ID,',1)
    ast.parse(text);path.write_text(text)
print('CSM service uses shared guarded tail cleanup; restart it to activate the policy.')
