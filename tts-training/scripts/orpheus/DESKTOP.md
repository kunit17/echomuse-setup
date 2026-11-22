# On-demand desktop voice

The owner's current policy is **off unless explicitly turned on**. Automatic
startup is disabled for Orpheus main,16-bit fallback and CSM. The service has an
ExecCondition checking activation-request.json; background restores must not
override an off request. No model is loaded just because WSL starts or the page
is opened. `orpheus sample` now requires an already-running voice.

Desktop shortcuts:

- **Orpheus Voice**: opens the control app, loads the selected4-bit model, waits
  until ready, then opens http://localhost:8812/ in the default browser.
- **Stop Orpheus Voice**: turns the voice off and releases its GPU memory.

Inside the control app use Start,Stop,Open Page. Closing its control window
stops the model. Closing only the browser tab does not stop it. The control
app keeps WSL alive while on. If the app is killed, a heartbeat watchdog stops
the model after approximately20seconds. Reopening the app starts a new session.
No automatic startup entry or scheduled task is installed.

Windows installation: `%LOCALAPPDATA%\OrpheusVoice\Orpheus-Voice.ps1`.
Shortcuts use the actual Windows desktop folder (currently OneDrive\Desktop).
The source installer is Install-Orpheus-Desktop.ps1. The app's small session
logs and heartbeat files are under its .desktop-state folder; they contain no
model weights. Do not move the app files without reinstalling its shortcuts.

Ubuntu helper: `orpheus-lora/scripts/desktop_host.py`; activation gate:
`desktop_gate.py`. The systemd override is
`~/.config/systemd/user/orpheus-tts.service.d/on-demand.conf`.
Explicit manual `orpheus start` and `orpheus stop` remain supported. Sample
commands do not implicitly load the GPU model. GPU preparation/training must
preserve the owner's requested off state when restoring a service.

The approved epoch3 weights,4-bit streaming runtime,1-second playback buffer,
sentence generation and16-bit fallback are retained. Active inference export
and environment remain protected even while the model is unloaded.
