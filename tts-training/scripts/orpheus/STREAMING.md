# Epoch-3 streaming benchmark

## Current production selection

Owner listening review is approved. The 4-bit vLLM runtime is now the default
on port8812 with a1-second buffer and sentence generation. Read ACTIVATION.md.
The stream-benchmark/.venv and merged-epoch3 folders are active serving
dependencies; do not delete them as temporary files. The following benchmark
sections retain their historical settings and do not override this selection.


This is an experiment, not a deployment. The selected warm service on port8812
is restored after the benchmark, including errors and cancellation. No corpus
or source recording is changed and no training is run.

Installed sources: `orpheus-lora/scripts/stream_*.py`. Results and independent
environment: `/home/voiceuser/projects/orpheus-lora/stream-benchmark`.
Run `python3 scripts/stream_bench.py` in a persistent WSL process. Never launch
two controllers. A completed run refuses to overwrite its report; changed
settings need a distinct result version. Setup: `stream_setup.py` installs
independent dependencies and records requirements.lock.txt. Do not upgrade the
hardlinked production/training packages.

Run `browser_watch.ps1` in a persistent hidden Windows PowerShell process before
the controller. It handles browser_probe status requests through the CPU results
server on8814; it does not require enabling WSL Windows executable interop.
Interrupted controllers reuse recorded completed requests and screening results
for unchanged settings; incomplete requests are regenerated.

The controller owns the shared reservation and passes it to asr_worker stage
supervisors. Child GPU process groups are terminated if the controller dies.
Graceful cancellation restores the main service. After a host crash or SIGKILL,
explicitly run `orpheus start` once orphan GPU workers are confirmed exited.

Benchmark order: warm production baseline (two warm-ups, six prompts × five
measurements), CPU merged export, then 16-bit and 4-bit vLLM. For each precision,
screen eager/CUDA-graphs × CPU/GPU SNAC and select the lowest worst-case RTF
across two probes for a full two-warm-up/30-request test. A Windows Edge
headless/muted AudioWorklet probe measures actual browser playback and checks
invalid-input rejection and disconnect cancellation. ASR runs after inference
workers exit. Temporary merged weights retain original base/adapter lineage.

No speaker prefix/reference is used. Keep current token framing, seed1700,
temperature.6, top-p.95, repetition penalty1.1 and explicit speech EOS128258.
Use numeric token IDs, not upstream's string-token decoder: zero codebook values
are valid, and every seven-code group must be preserved. Decode the complete
available prefix every four groups, keep eight groups as right context, and
flush all remaining samples at EOS. Save both streamed and full-decode WAVs
from identical token sequences with waveform-difference and sample-count checks.
Right context and the 300ms terminal holdback add latency intentionally; never
drop leading/trailing speech to make a faster number. The shared inference-only
tail policy is applied only at true completion. Missing EOS, invalid groups or
incomplete transport are failures, even if the user heard partial speech.

Experimental API: `POST http://127.0.0.1:8813/stream` with JSON text. Returns
chunked little-endian signed16 PCM, 24kHz mono, X-Request-ID. `/trace/ID` confirms
complete/failed/cancelled state and contains timing/coverage details. Playback
must distinguish the end of transport from confirmed successful generation.
`/health`, `/report`, `/`, `/worklet.js` support local testing. This API streams
audio for complete text; it does not provide streaming input text or microphone
conversation. Requests are serialized. Context4096, batch1 and codec CPU thread
selection are inference settings, not resource quotas.

Measurements: startup with cached disk weights, first token, first PCM, actual
browser playback, total generation/audio duration, RTF, chunk arrival trace,
GPU memory/power/temperature/utilization. Simulate .25/.5/1s buffers. Eligibility:
p95 first playback≤2s, p95 RTF≤.8, zero underflows at .5s, complete audio and no
unresolved ASR errors, followed by owner listening for identity and seams. Seed
agreement across different engines does not imply bit-identical speech. Report
ASR disagreement rather than treating it as proof of clipping or clarity.

CPU-only listening results remain on http://localhost:8814/ after restoration;
live generation on8813 exists only during the experiment. Keep local WAV/MP3,
JSONL/CSV, GPU telemetry, runtime logs, browser metrics and pinned environment.
Commit scripts, this guide and the compact factual report, not model/audio data.
The temporary ~7GB export and benchmark environment remain until owner review;
remove them only after verifying there is no selected runtime dependency.

## Current live policy: one-second buffer, sentence generation

At the owner's request, live streaming now accumulates one second of audio
before playback and generates text sentence by sentence. Every utterance uses
its own EOS and complete SNAC groups; only input text is partitioned. All audio
samples are retained. Titles/initials and decimal points are protected, and
very long sentences are divided on clause/word boundaries without dropping text.
Cleanup is applied at each utterance's genuine endpoint. This changes the live
experimental path; historical benchmark numbers above are not overwritten or
presented as measurements of the new settings.

Windows validation: Tony3sentences/7.25s audio started2.435s, zero underruns;
long passage5sentences/20.82s audio started1.716s, zero underruns. Both matched
every ASR word, passed full sample-count checks, invalid-input rejection and
disconnect cancellation. Two successful examples do not guarantee pause-free
playback for every future input. New clips are on the CPU review page and
`buffer-policy-check.json` records the exact results. The main service was
restored after validation; no streaming deployment selection was made.

For a fresh launch, start the CPU review server with
`python3 /home/voiceuser/projects/orpheus-lora/scripts/stream_review_server.py`
in a separate persistent process before starting the Windows helper. Keep it
running for the helper handshake and subsequent listening page. Install source
with stream_install.py; stream_setup.py reuses a ready pinned environment and
never upgrades it on a repeat invocation.

For a later interactive listening test, run
`python3 /home/voiceuser/projects/orpheus-lora/scripts/stream_session.py` in Ubuntu.
It temporarily runs the fastest measured candidate at http://localhost:8813/;
Ctrl+C restores the prior production service state. This enables hearing actual
streaming pauses rather than playing a prebuffered MP3. It never changes the
selected model, never starts on its own and does not mark listening approved.

## November 18 results

All eight configuration probes succeeded. Full measurement: 30 requests each
for baseline, vLLM16-bit graphs/GPU codec, and vLLM4-bit graphs/GPU codec.
Median RTF: 2.213 / 1.786 / 1.071 respectively. Corresponding p95 RTF:
3.328 / 1.968 / 1.235. Neither streaming candidate met the .8 headroom target
or zero-underrun requirement with a500ms playback buffer.

Windows Tony-prompt playback: 16-bit first audio2.524s, 17 underruns/4.336s
stalled; 4-bit first audio1.608s, 5 underruns/.224s stalled. Invalid requests
and disconnect cancellation passed on both. No completed stream lost samples.
All90 ASR checks matched the first and final words. Raw word-edit flags were
five baseline and four 4-bit long-passage results: ASR contracted "you have"
to "you've". Preserve those disagreements for listening rather than claiming
perfect transcription. Original selected epoch3 service is restored; no new
runtime was deployed. Temporary environment/export occupy approximately16GiB.
