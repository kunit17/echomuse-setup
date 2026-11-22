# Own-corpus Orpheus training and vLLM streaming

No private voice corpus, transcript, model, checkpoint or prior approval is
included. Use recordings you have the right to train and distribute. Reproducing
the workflow does not reproduce another person's trained voice without its data.

## Layout and environments

The reference implementation uses these sibling runtime directories:

- ~/projects/echomuse-provisioning — this public source checkout
- ~/projects/gpt-sovits — shared corpus scripts, ASR workers, QA and datasets
- ~/projects/orpheus-lora — Orpheus scripts, features, models and runs
- orpheus-lora/stream-benchmark/.venv — isolated vLLM serving environment
- orpheus-lora/stream-benchmark/merged-epoch3 — independently merged inference model

Configure example paths before use. Copy tts-training/scripts into the shared
runtime scripts directory and scripts/orpheus into orpheus-lora/scripts. Create
the .cache/reports and other working directories needed by the chosen supervisor.
Some legacy helpers refer to older CSM/GPT-SoVITS service layouts: they are
reference source, not automatically enabled dependencies of the Orpheus path.

Use separate Python environments for corpus/ASR, training, vLLM and the small
Wyoming adapter. Training and serving package snapshots are in
tts-training/deployment/metadata/. They are observed package versions, not
hash-locked wheels; check Python/CUDA/PyTorch compatibility and pip check.
Never alter an existing environment's hardlinked packages in place.

## Prepare and review your own immutable corpus

1. Keep original recordings unchanged and hash them. Create your own source
   recipe: sources with paths/SHA256, disjoint train/validation time partitions,
   exclusions and the exact builder policy. recipe.example.json is a schema
   example, deliberately not an approved or usable dataset.
2. Set CORPUS_REPAIR_RECIPE to your completed recipe and CORPUS_REPAIR_VERSION to
   a unique version. scripts/corpus_repair/build.py validates source hashes and
   frozen partitions before using cached work. Inspect its CLI --help.
3. Use only corpus_repair for cuts: acoustic pauses, VAD, independent alignment,
   full original-word boundaries and contextual listening. Do not revive older
   timestamp-cut pipelines. Run pilot scan/ASR/alignment and inspect its report
   before full ASR/alignment/publication. The run.py supervisor shows stage order.
4. Supply the ASR/alignment assets expected by asr_runtime and corpus preflight;
   run corpus_asr_preflight.py on the configured environment. Keep GPU work under
   asr_worker.py and resource_safe.py guards with one shared reservation.
5. Listen to the produced review clips and their original context. Check corpus.json,
   split manifests and final words. Create owner-review-approved.json using
   approve_dataset.py --dataset PATH --fingerprint ACTUAL_FINGERPRINT
   --note YOUR_REVIEW_NOTE --owner-approved only after YOUR review.
   Never distribute the resulting manifests/transcripts or copy another approval.

## Base models, SNAC features and training

Copy setup.example.json to orpheus-lora/setup.json. Its public model revisions
pin the base Orpheus model, quantized fallback and SNAC codec. Review access/model
terms, then use download.py to cache those revisions. Authentication stays in
your local environment; never write a token into scripts or published files.

Set common.py CORPUS to your approved dataset and keep ROOT/REPO/DATA/RUN
consistent. The public prepare.py derives split sizes rather than requiring the
private corpus's clip counts. It verifies approval/manifest/audio hashes, encodes
all SNAC frames, keeps seven-code frame groups and speech EOS, and fails if a
full example exceeds context 4096. It never silently truncates source speech.

Run prepare.py with the training environment and shared GPU reservation. Review
the generated dataset.json and QA audio. Run qa.py and resolve real QA failures
before training; a missing approval or model asset is a stop condition.

Review train.py and pipeline.py. The reference uses three epochs, rank/alpha 64,
learning rate 2e-4, microbatch 1, accumulation 4 and seed 3407. Run full-length and
checkpoint-resume smoke tests. Try 16-bit first; use 4-bit only for confirmed GPU
memory exhaustion. Preserve every frame and record actual metrics/previews.
The public pipeline leaves inference off at completion. Keep inference stopped
during training; do not launch an obsolete fallback service just to satisfy a
historical supervisor. Adapt any legacy service checks to your actual host.

## Select and merge your trained model

Listen to candidates on identical prompts, check endings and record the actual
adapter/tokenizer hashes. Create your own selected-model.json with adapter_path
and adapter_sha256; never treat an old epoch number as automatic approval.
stream_export.py merges that selection on CPU into a distinct inference export,
checks the adapter hash and writes per-file provenance. It preserves the LoRA
and refuses to rebuild an existing export with provenance. Use a new versioned
destination when changing models; do not delete an active export to force a run.

## Serve with vLLM and real PCM streaming

Create a NEW Python 3.11 environment at stream-benchmark/.venv and install the
saved requirements-vllm.lock.txt there; run pip check. The source's stream_setup.py
resolves packages dynamically and is a reference bootstrap, not a pinned rebuild.
Install the stream_* sources and worklet/browser assets via the reviewed
stream_install.py. Keep the shared audio_postprocess.py import available.

Example selected service command after configuring paths/model/codec caches:

    stream-benchmark/.venv/bin/python scripts/stream_server.py --precision 4bit --codec cuda --port 8812 --kv-cache-mib 1024

Keep loopback binding, context 4096, one sequence and explicit 1 GiB KV cache.
Review the 50% GPU startup-memory check separately from KV cache size. Do not
enable GPU-model autostart unless you intentionally choose that policy. Install
the systemd unit together with its on-demand/training-reservation drop-ins.

/health reports readiness; POST /stream returns signed little-endian 16-bit PCM,
24 kHz mono; /synthesize is the complete-WAV fallback. GPU SNAC decodes numeric
token IDs, preserves complete frame groups, handles right context and flushes
the final audio through speech EOS. Zero codec values are valid. Never trim
source or output just to improve a latency number. Sentence generation and the
browser AudioWorklet use a one-second buffer, or all available shorter audio.

Check a short and long response, first PCM arrival, continuity, final words,
cancel/reconnect behavior and complete trace state. A closed HTTP stream alone
is not proof that the final decoder audio was delivered. CPU framing tests are
in scripts/orpheus/test_stream.py; GPU/audio acceptance needs your machine.
Then follow orpheus-deployment.md to bridge this service into HA.
