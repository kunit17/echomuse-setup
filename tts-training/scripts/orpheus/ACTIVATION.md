# Approved Orpheus epoch-3 streaming service

## Current desktop activation policy

Owner policy: off unless explicitly started. Use the Orpheus Voice desktop app; closing its control window or using Stop unloads the GPU. Automatic startup is disabled. Opening the browser alone does not load the model, and sample commands require it to be on already. Read DESKTOP.md. Explicit CLI start/stop remains available.


The owner completed listening review and explicitly approved the default switch
on November18,2026. Main runtime: vLLM4-bit, CUDA graphs, GPU SNAC, sentence
generation, one-second client playback buffer. Original epoch3 LoRA weights and
full checkpoint are retained unchanged. No retraining was performed.

```bash
orpheus start
orpheus status
orpheus sample --text-file /absolute/path/text.txt --out /absolute/path/NEW.mp3
orpheus stop
```

Live playback: http://localhost:8812/ . It uses an AudioWorklet and one second
of queued audio, or the completed shorter utterance if it contains less audio.
The model stays loaded between requests. `stop` stops the main and fallback
services to release GPU memory. `start` reuses the correct selected runtime.

API on loopback8812: `/health`, streaming PCM `/stream`, and compatible
`/synthesize` returning a complete WAV. PCM is signed16 little endian,24kHz mono;
use X-Request-ID and `/trace/ID` to confirm complete generation. Existing WAV
consumers keep receiving complete files; actual streaming requires a streaming
client. Sentence generation preserves all tokens through every utterance's EOS.
The shared isolated-end-burst-v1 policy remains enabled. No gateway config changed.

Selection: `/home/voiceuser/projects/gpt-sovits/selected-tts-model.json` and
`/home/voiceuser/projects/orpheus-lora/selected-model.json`.

**Active production dependencies—do not delete as benchmark cache:**

- `/home/voiceuser/projects/orpheus-lora/stream-benchmark/.venv`
- `/home/voiceuser/projects/orpheus-lora/stream-benchmark/merged-epoch3`
- Pinned SNAC cache, epoch3 adapter/tokenizer and final checkpoint813.

The original16-bit service is retained, disabled by default. To use it manually:

```bash
systemctl --user stop orpheus-tts.service
systemctl --user start orpheus-tts-16bit.service
```

Both use port8812 and conflict so only one loads. `orpheus start` returns to the
selected4-bit streaming default. `orpheus stop` releases either runtime. The
fallback's raw health endpoint reports its16-bit state; the default CLI rejects
an unexpected runtime rather than silently accepting it as the approved4-bit.
Fallback service and original environment/weights remain intact.

Before GPU preparation/training stop the selected service. The corpus supervisor
uses selected-tts-model.json and restores the main service after workers exit.
Do not run the old promotion/cleanup scripts to reset runtime selection. The
historical benchmark used500ms whole-passage generation; its metrics are not
measurements of this owner-approved sentence-streaming configuration.


## GPU memory reservation (2026-11-20)

The approved 4-bit stream server defaults to --kv-cache-mib 1024, passed as
vLLM kv_cache_memory_bytes. It keeps max_model_len=4096 and max_num_seqs=1;
the allocated cache holds 9,344 tokens. gpu_memory_utilization=0.5 also lowers
the separate startup free-memory check; explicit KV bytes determine the cache.
This is not a hard cap on all CUDA working memory. Total card use measured
about 5.5 GiB during a complete 30.208-second speech test, versus 10.7 GiB
before. The selected adapter, GPU codec, CUDA graphs and sentence policy are
unchanged. Manual desktop activation remains in force. Health exposes the
cache size, context limit and request concurrency. The setup repository records
the patch, source hashes, test results and rollback instructions.
