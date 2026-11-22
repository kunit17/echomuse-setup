# Orpheus Voice desktop app

## Desktop on-demand activation

Default state is off. Click Orpheus Voice on the desktop to load the model and open the page. Closing the control window or clicking Stop unloads it. No automatic startup; sample commands cannot implicitly load it. Read DESKTOP.md.

# Orpheus voice

## Approved streaming default

Epoch3 now uses vLLM4-bit, CUDA graphs and GPU SNAC. Live playback: http://localhost:8812/. One-second audio buffer and sentence generation. `orpheus start|stop|status|sample` remain the public commands; read ACTIVATION.md for the protected production dependencies and retained16-bit fallback.

# Orpheus 3B, approved v1 corpus

Project: `/home/voiceuser/projects/orpheus-lora`.
Read the authoritative `gpt-sovits/TTS_WORKFLOW.md` before changing datasets.
This adapter consumes the immutable v1 manifests; it never segments recordings.

The project has a separate Python 3.11 interpreter. Existing pinned dependencies
are stored with hardlinks to save disk space; SNAC 1.2.1 is installed only here.
Do not edit shared dependency files or upgrade this environment in place.
Create a fresh environment for dependency changes. CSM's package set is unchanged.
Model and notebook revisions are pinned in setup.json; exact packages are in
requirements.lock.txt. Audio/token data, model weights and previews remain local.

```bash
orpheus status
orpheus prepare
orpheus train
orpheus sample --text-file /absolute/path/text.txt --out /absolute/path/sample.mp3
```

`train` performs full-length memory and checkpoint-resume preflights before the
three-epoch run. It tries 16-bit first and changes to 4-bit only after a confirmed
memory failure. It holds the shared reservation and passes its descriptor to
workers, which terminate children if the controller disappears. For future GPU work, stop the selected voice service first and restore it afterward. The owner selected Orpheus epoch 3 as main voice; see ACTIVATION.md. No GPU/RAM quotas are introduced.

SNAC resamples each clip to 24kHz and keeps every frame, including repeated quiet
frames. Encoding failures are fatal; no examples are filtered. The maximum
encoded sequence is 2,499 tokens; context is 4,096, with dynamic padding and no
truncation. Recording IDs are metadata, not speaker prefixes. Generation uses
the learned single voice and does not require CSM's reference audio.

The 1,083 training and 121 validation clips retain the original source split and
exclusions. All 1,204 file hashes were checked. Codec round-trip QA includes 11
long/edge examples. Two Whisper extra-suffix disagreements are retained in the
report and independently resolved by CTC. This does not claim human listening.

Full training: rank/alpha64, attention/MLP LoRA, lr2e-4, AdamW8bit, effective batch4,
three epochs, validation after every epoch, seed3407. Non-padding tokens use the
notebook's full causal loss. Compare eval_loss only within this run, not to CSM.
Three fixed-seed MP3 previews are generated at each epoch; the next training step
after preview is exercised in preflight. Earlier full checkpoints rotate only
after their previews are produced. Final epoch3 and its adapter/tokenizer remain.

Progress and previews: http://localhost:8769/ . Runtime metrics, preflight logs,
checkpoints and MP3s are under `runs/orpheus-v1-example01`. Full-length SNAC tokens
and round-trip evidence are in `data/v1-snac`. Original WAVs and manifests are
unchanged. The narrow inference-only tail cleanup is shared with CSM; raw
Orpheus previews and cleanup decisions are retained for comparison.

Tests: `.venv/bin/python scripts/test_features.py -v`.
The upstream notebook is saved locally with a SHA256 in setup.json.

## On-demand activation

Orpheus epoch 3 is the main model. Read ACTIVATION.md. `orpheus start|stop|status` controls the warm service; `orpheus sample` reuses it.

## Streaming benchmark

Read STREAMING.md for the isolated vLLM experiment and measured results. Listening: http://localhost:8814/. The selected production runtime is unchanged. For an explicit temporary live review, run `python3 scripts/stream_session.py`; Ctrl+C restores the previous production state.
