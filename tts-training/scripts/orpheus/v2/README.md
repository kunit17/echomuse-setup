# Orpheus V2 training

Owner authorized a fresh Orpheus run on the approved V2 corpus on November21,
2026, after asking whether V2 was available in the 4-bit streaming service.
The three-epoch training recipe matches the tested V1 run. This is Orpheus,
not a resume of the separate CSM V2 checkpoint.

- Corpus: `gpt-sovits/data/shecorpus/repaired/boundary-v2-20261116`.
- Fingerprint: `example02f5e3af078842b6be580e2c811fb57d4d884590d94eda53649d677005`.
- Train1856 clips/20432.04s; validation192 clips/2176.105s.
- Run: `/home/voiceuser/projects/orpheus-lora/runs/orpheus-v2-example02`.
- Immutable features: `orpheus-lora/data/v2-snac`.
- Controller: `python3 /home/voiceuser/projects/orpheus-lora/scripts/v2/pipeline.py`.
- Live progress and per-epoch previews: http://localhost:8770/ .

Use only the isolated scripts/v2 implementation for this run. common.py in that
directory deliberately selects V2 paths; serving and V1 scripts retain their
original paths. Training uses the existing pinned Python environment and base
revision, fresh16-bit LoRA rank/alpha64, all attention/MLP adapters, lr2e-4,
AdamW8bit, microbatch1/accumulation4, context4096, seed3407, exactly3epochs.
Expected1392 optimizer steps. Full validation and3comparison MP3s every epoch.
Only the last full checkpoint is retained; preview files and adapter hashes
retain evidence of earlier epochs. No user resource caps are introduced.

Stages: corpus audit; whole-clip SNAC features; round-trip ending checks with
cached Whisper and independent CTC for disagreements; longest-clip forward,
backward, generation and checkpoint-resume smoke tests; full training; preview
ASR review. A4-bit training fallback is permitted only after a confirmed16-bit
OOM. Ordinary code/data failures must be fixed, never bypassed or called OOM.

The corpus is already published and approved. No segmentation, transcription
replacement, waveform trimming, frame deduplication or row filtering is allowed.
Feature context overflow is an error rather than permission to truncate.
The audit checks every WAV hash and PCM edge, alignment margins, frozen source
partitions/exclusions, manifests, original source hashes and split leakage.
The300ms requirement is the source quiet-pause width; the permanent publisher
retains part of the pause on each side of the cut and checks120ms exported PCM
edges. Never reinterpret it as a new300ms-per-edge clipping/acceptance rule.

All GPU services are stopped for this run: Orpheus,16-bit fallback,CSM,Ollama,
Wyoming Whisper and Wyoming Orpheus. Gateway/search services are not changed.
The controller owns the global GPU reservation and passes it to asr_worker;
workers exit if their controller disappears. A systemd inference condition
checks that reservation before allowing Orpheus to load. Services remain off
after success/failure, respecting the owner's request and desktop activation
policy. Do not restore inference automatically at the end.

The selected V1 streaming deployment is preserved. A completed V2 adapter needs
listening review and a separate inference export before changing that selection.
Training precision and4-bit inference precision are independent. This run does
not activate or overwrite production weights, environment, model export or
the newer inference cache-memory configuration.

Keep source hashes, corpus-verification.json, dataset.json, plan.json, metrics,
preflight passes/logs, preview results, training-result.json and final adapter.
Check status.json for current progress; historical reports are not launch orders.
