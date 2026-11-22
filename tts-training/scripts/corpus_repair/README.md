# Source-first preparation

Read [TTS_WORKFLOW.md](../../TTS_WORKFLOW.md) for the
complete preparation, review, training, checkpoint selection and handoff process.
This directory is the sole supported corpus builder. DESIGN.md explains the
boundary checks; recipe.json stores permanent source hashes, exclusions and
partitions. Neither deleted old clips nor Windows review files are inputs.

Checks:

```bash
cd /home/voiceuser/projects/gpt-sovits
.venv/bin/python scripts/corpus_repair/build.py check
.venv/bin/python scripts/corpus_repair/test_boundaries.py
```

For a new recipe/version, follow section 3 of the runbook. The supervisor never
starts training. Completed versions are immutable. Use a new version and fresh
cache whenever sources/settings change. Review actual cut WAVs beside their
source context before recording owner approval for that exact fingerprint.

Current dataset: data/shecorpus/repaired/boundary-v1-20261115.
Current CSM training is complete. Owner selected epoch 5 for generation; epoch 2
had the lowest validation loss. Current state is recorded by the run and model
selection files listed in the runbook. Build-time reports retain past states.

Regression tests are automated evidence; they do not prove manual listening.
All earlier timestamp-cut builders and corpus-mode launchers are retired.
