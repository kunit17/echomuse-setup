# Whisper and private tunnels

1. Install WSL with systemd and a separate user environment. Review and run
   scripts/install-whisper-wsl.sh; package versions are in
   voice/whisper-requirements.lock.txt. Use scripts/download-whisper-models.py
   for model download; inspect --help rather than assuming a model is cached.
2. The included voice/whisper.env.example selects base.en with CPU INT8.
   Update model/cache paths for your machine. Install voice/wyoming-whisper.service
   as a user unit, daemon-reload and start it. Use scripts/probe-whisper.py.
3. Create your own desktop SSH key and authorize its public key on the Pi.
   Verify the Pi host-key fingerprint yourself; retain StrictHostKeyChecking.
4. Configure scripts/Start-EchoMuse-Whisper.ps1 parameters PiHost, PiUser,
   IdentityFile and Distribution. Its reverse forwarding exposes Whisper only
   on Pi loopback 10300; Orpheus adapter optionally uses loopback 10200.
5. Use Install-EchoMuse-Whisper-Task.ps1 for the Windows sign-in task after
   reviewing its parameters. Keepalive and process-identity cleanup prevent
   duplicated or stale workers. Start-EchoMuse-Whisper -IncludeOrpheus starts
   lightweight adapters, not the GPU model itself.
6. Add Wyoming in HA using 127.0.0.1:10300 from HA's host-network container.
   Select faster-whisper in the assistant and English. Test one real phrase.

Keep ports private; do not publish Wyoming or the GPU API to the Internet.
Model speed/accuracy depends on your CPU/GPU. benchmark-whisper.py and fixture
helpers are available; do not repeatedly benchmark an unchanged working system.
