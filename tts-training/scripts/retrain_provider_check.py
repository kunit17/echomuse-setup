"""Run only with Hermes gateway Python 3.14; never sends Telegram messages."""
import os,sys,json,pathlib,subprocess
os.environ.pop('PYTHONHOME',None);os.environ.pop('PYTHONPATH',None)
sys.path.insert(0,'/home/voiceuser/.hermes/hermes-agent')
from pm.environments import activate_dependencies
from pm import paths
activate_dependencies(paths.repo_root())
from gateway.platforms.base import build_auto_tts_output_path,Platform
from tools.tts_tool import text_to_speech_tool
assert sys.version_info[:2]==(3,14),sys.version
p=build_auto_tts_output_path(Platform.TELEGRAM)
result=text_to_speech_tool(text='The new voice is now serving.',output_path=p)
if isinstance(result,str):result=json.loads(result)
print(p,result,flush=True)
R=pathlib.Path('/home/voiceuser/projects/gpt-sovits');(R/'reports/retrain-hermes-result.json').write_text(json.dumps({'python':sys.version,'output_path':str(p),'result':result},indent=2)+'\n')
assert result['success'] is True,result
assert result.get('provider')=='she-gptsovits',result
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(p)]));file=subprocess.check_output(['file',str(p)],text=True)
assert probe['streams'][0]['codec_name']=='opus' and probe['format']['format_name']=='ogg',(file,probe)
(R/'reports/retrain-hermes-audio.json').write_text(json.dumps({'file':file,'ffprobe':probe},indent=2)+'\n')
