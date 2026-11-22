# EchoMuse voice assistant: reproducible public setup

Turn a supported Echo Dot into a Home Assistant voice satellite, with a custom
wake word, local Whisper recognition, a self-trained Orpheus voice streamed by
vLLM, optional MiniMax conversation/web search, and Spotify through Music Assistant.

This is a sanitized source distribution with fresh history, not a backup of a
working home. It does not include credentials, recordings, trained weights or
private training/run records. Read [publication boundaries](PUBLICATION.md).
You need your own authorized voice corpus, accounts and compatible hardware.
The source was syntax/audit checked; a clean-machine hardware/GPU installation
has not been executed for this public edition. Stop and resolve a failed check
at each stage before continuing. Historical helpers require review before use.

## Architecture

Echo Dot → EchoMuse controller → Home Assistant Assist → Whisper STT →
MiniMax + exposed HA tools → Orpheus/vLLM → Wyoming adapter → Dot speaker.
Music Assistant separately streams Spotify audio to the same Dot. The LLM is
optional/cloud; speech recognition, wake detection and voice synthesis are local.

## Requirements and example values

- Supported Echo hardware; the main procedure targets Echo Dot 2nd generation.
- An always-on 64-bit Linux controller host with Docker and Home Assistant.
- Windows/Chrome for WebUSB provisioning; follow hardware-specific unlock docs.
- WSL/Linux desktop with a compatible NVIDIA GPU. The serving profile targets
  roughly 12 GB VRAM; training memory varies with model, precision and corpus.
- SSH connectivity from desktop to controller; no router port forwarding needed.
- Spotify Premium for Spotify music. Cloud LLM/search need their own account/network.

Example controller 192.0.2.10, desktop 192.0.2.20, Dot 192.0.2.40 are deliberately
non-routable documentation addresses. Replace them with your own addresses.
Example accounts are piuser (controller), voiceuser (WSL) and DesktopUser (Windows).
Use your own SSH key, never a key copied from another installation.

Clone this repo into ~/projects/echomuse-provisioning to retain the service-unit
layout, or update those units. Run tools/configure_examples.py in a LOCAL clone
to replace example usernames and IPs. Do not push the personalized copy.

## Follow these stages in order

1. [Identify supported hardware](docs/model-support.md). Check model and firmware
   against current upstream EchoMuse documentation before any destructive step.
2. [Host the controller](docs/controller-hosting.md), using controller/ compose
   and .env example. Set SERVER_IP yourself; create controller credentials via
   [first-run setup](docs/controller-first-run.md). Keep .env private.
3. [Unlock and provision](docs/provisioning.md). Preserve your OWN stock boot
   image. USB/browser constraints and recovery are part of this step.
4. [Add Whisper and private speech tunnels](docs/whisper-deployment.md). Confirm
   a voice command transcribes correctly before introducing an LLM/custom voice.
5. [Train and evaluate Samantha](docs/samantha-wakeword.md). The corrected
   canonical clip path is essential; no pre-trained owner model is included.
6. [Prepare your corpus, train Orpheus, export and serve with vLLM](docs/tts-training.md).
   This covers isolated environments, GPU reservation, full SNAC frames, EOS,
   checkpoint selection, 4-bit inference, 1 GiB KV cache and streaming setup.
7. [Connect the Orpheus Wyoming adapter](docs/orpheus-deployment.md), test full
   streaming audio, and keep GPU loading manual unless you choose otherwise.
8. [Configure MiniMax, device tools and optional web search](docs/minimax-deployment.md).
   Enter API keys through HA; never save them in a repo. Weather uses the included
   read-only script instead of a location-less web search.
9. [Add Music Assistant and Spotify](docs/spotify-on-dot.md). Distinguish the
   music service from the output speaker and use a unique default player name.
10. Test one full spoken turn, a real exposed-device action, a forecast and one
    music request. A passing HTTP request or generated confirmation alone does
    not prove that a physical device acted or the Dot played sound.

## Operation and maintenance

- Start Whisper/adapter/tunnels; load Orpheus explicitly when voice output is needed.
- Before training, stop GPU inference and hold the shared worker reservation.
  Do not restart inference until training exits and you explicitly start it.
- Keep training and inference environments separate; do not upgrade hardlinked
  CUDA payloads in place. Package snapshots are reproducibility inputs, not a
  promise that every wheel works on every platform.
- Back up controller and HA state privately. Never commit backups, token stores,
  corpus manifests, audio, checkpoints, browser data or configured environment files.
- Upstream upgrades can replace the small Home-LLM patches; inspect them afterward.

See [source provenance](SOURCE-PROVENANCE.md). Upstream licenses/model and dataset
terms remain applicable. No additional blanket license over third-party work is granted.
