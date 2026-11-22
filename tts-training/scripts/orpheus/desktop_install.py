"""Install on-demand activation policy and disable every TTS auto-start unit."""
from common import *
import shutil,subprocess
source=Path('/mnt/c/Users/DesktopUser/Projects/echomuse-provisioning/tts-training/orpheus-training')
for name in ['desktop_gate.py','desktop_host.py','desktop_install.py','cli.py']:shutil.copy2(source/name,ROOT/'scripts'/name)
atomic(ROOT/'activation-request.json',dict(desired_on=False,mode='desktop',updated=time.time()))
for selection_path in [ROOT/'selected-model.json',REPO/'selected-tts-model.json']:
    selection=json.loads(selection_path.read_text());selection.update(activation_mode='desktop-on-demand',autostart=False);atomic(selection_path,selection)
unit=Path.home()/'.config/systemd/user/orpheus-tts.service.d';unit.mkdir(exist_ok=True)
(unit/'on-demand.conf').write_text('[Service]\nExecCondition=/usr/bin/python3 /home/voiceuser/projects/orpheus-lora/scripts/desktop_gate.py\n')
subprocess.run(['systemctl','--user','disable','--now','orpheus-tts.service','orpheus-tts-16bit.service','cesame-tts.service'],check=True)
subprocess.run(['systemctl','--user','daemon-reload'],check=True)
print('Automatic TTS startup disabled; models stopped. Desktop or explicit CLI start required.',flush=True)
