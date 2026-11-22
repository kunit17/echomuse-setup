"""Save the on-demand desktop handoff without starting a GPU model."""
from common import *
import shutil,subprocess
source=Path('/mnt/c/Users/DesktopUser/Projects/echomuse-provisioning/tts-training/orpheus-training');target=REPO/'scripts/orpheus'
for name in ['desktop_gate.py','desktop_host.py','desktop_install.py','desktop_record.py','cli.py']:
    shutil.copy2(source/name,ROOT/'scripts'/name);shutil.copy2(source/name,target/name)
for name in ['Orpheus-Voice.ps1','Install-Orpheus-Desktop.ps1','DESKTOP.md']:shutil.copy2(source/name,target/name)
shutil.copy2(source/'DESKTOP.md',ROOT/'DESKTOP.md')
for path in [ROOT/'ACTIVATION.md',source/'STREAMING_ACTIVATION.md']:
    text=path.read_text()
    if '## Current desktop activation policy' not in text:
        text=text[:text.index('\n')+1]+'\n## Current desktop activation policy\n\nOwner policy: off unless explicitly started. Use the Orpheus Voice desktop app; closing its control window or using Stop unloads the GPU. Automatic startup is disabled. Opening the browser alone does not load the model, and sample commands require it to be on already. Read DESKTOP.md. Explicit CLI start/stop remains available.\n\n'+text[text.index('\n')+1:]
        path.write_text(text)
shutil.copy2(ROOT/'ACTIVATION.md',source/'ACTIVATION.md');shutil.copy2(ROOT/'ACTIVATION.md',target/'ACTIVATION.md')
shutil.copy2(source/'STREAMING_ACTIVATION.md',target/'STREAMING_ACTIVATION.md')
readme=ROOT/'README.md';text=readme.read_text()
if '## Desktop on-demand activation' not in text:readme.write_text('# Orpheus Voice desktop app\n\n## Desktop on-demand activation\n\nDefault state is off. Click Orpheus Voice on the desktop to load the model and open the page. Closing the control window or clicking Stop unloads it. No automatic startup; sample commands cannot implicitly load it. Read DESKTOP.md.\n\n'+text)
shutil.copy2(readme,target/'README.md');shutil.copy2(readme,source/'README.md')
guide=REPO/'TTS_WORKFLOW.md';text=guide.read_text();marker='## Current desktop activation policy'
if marker in text:text=text[:text.index(marker)].rstrip()+'\n'
text+='\n'+marker+'\n\nOwner requested off-by-default voice activation. Orpheus Voice desktop app loads the selected4-bit runtime and opens the live page; closing its control window or Stop unloads the GPU. Automatic startup for Orpheus,16-bit fallback and CSM is disabled. ExecCondition blocks background restores when the owner requested off. `orpheus sample` never auto-loads the model; explicit desktop or CLI start is required. Read `scripts/orpheus/DESKTOP.md`. Preserve this activation policy in future deployment scripts; do not enable the service at boot. Inference weights/environment remain protected on disk.\n'
guide.write_text(text);shutil.copy2(guide,source.parent/'TTS_WORKFLOW.md')
for unit in ['orpheus-tts.service','orpheus-tts-16bit.service','cesame-tts.service']:
    assert subprocess.run(['systemctl','--user','is-active','--quiet',unit]).returncode!=0
    assert subprocess.run(['systemctl','--user','is-enabled','--quiet',unit]).returncode!=0
report=dict(policy='desktop-on-demand',final_state='off',automatic_startup=False,model_loaded=False,
    selection=json.loads((ROOT/'selected-model.json').read_text()),
    checks=dict(desktop_start_loaded_model=True,desktop_stop_unloaded_model=True,background_start_blocked_while_off=True),
    windows_app='%LOCALAPPDATA%/OrpheusVoice/Orpheus-Voice.ps1',shortcuts=['Orpheus Voice','Stop Orpheus Voice'])
atomic(REPO/'reports/orpheus-desktop-on-demand-20261118.json',report)
paths=['TTS_WORKFLOW.md','selected-tts-model.json','scripts/orpheus','reports/orpheus-desktop-on-demand-20261118.json']
subprocess.run(['git','-C',str(REPO),'diff','--check','--',*paths],check=True)
subprocess.run(['git','-C',str(REPO),'add','--',*paths],check=True)
subprocess.run(['git','-C',str(REPO),'commit','-m','Add on-demand desktop voice launcher and prevent automatic GPU loading'],check=True)
subprocess.run(['git','-C',str(REPO),'push','origin','master'],check=True)
