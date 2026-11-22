# TTS source and dataset rules

Read TTS_WORKFLOW.md in this directory and ../docs/tts-training.md before corpus,
training or serving changes. This combined repository is the maintained code and
documentation location; deployed runtime paths still point to separate local
working directories. Do not deploy changes just because a repository was synced.

Use only scripts/corpus_repair for source preparation. Preserve original
recordings and complete words, frozen split/exclusion intervals, full SNAC
frames, and immutable dataset versions. No waveform/sequence truncation or
silent row filtering. Dataset/model approval records describe actual owner
decisions, not authorization to manufacture approvals or deploy a new model.

Keep the shared GPU training reservation and worker lifecycle checks. Do not
start inference during a training lease. The current Orpheus V2 run explicitly
leaves GPU services off at completion; that specific policy overrides older
generic restore-after-training text for that run. No training or deployment is
authorized by this repository integration itself.

Never edit hardlinked dependency payloads in place. Preserve the selected V1
epoch-3 adapter/checkpoint, stream-benchmark/.venv, merged-epoch3, manual desktop
activation, streaming EOS/sample coverage, and 1 GiB inference KV cache unless
the owner explicitly changes deployment. Historical benchmark/promotion scripts
must not override the current runbook or activation gates.
