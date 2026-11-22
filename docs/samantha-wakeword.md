# Correct custom wake-word training

Use synthetic wake-word audio plus background/room augmentation. No personal
voice recording or trained wake-word model is distributed here. The model learns
to recognize the word; it is independent from the voice used for spoken answers.

1. Use upstream EchoMuse oww_forge and its documented container/CLI. Record the
   actual image digest and asset/model revisions for your own run.
2. Create a persistent /data mount, then use the forge new and assets commands
   for Samantha. Review wakeword/samantha/config.yml: target samantha, 30,000
   positives, 2,000 validation samples, 50,000 steps, near-miss negative phrases,
   AudioSet/FMA backgrounds, room impulses and validation feature data.
3. Inspect train_canonical.py and build_canonical.py before training. The corrected
   path uses canonical clip lengths and validates voiced content. Do not create
   positives by blindly truncating generated speech or padding a clip without
   checking where the actual wake word occurs. A model can train successfully
   on unusable clips; training loss alone is not acceptance.
4. Put the corrected scripts in the mounted /data directory as expected by
   build_canonical.py. It redirects forge.TRAIN_PY to /data/train_canonical.py.
   Run the builder inside the pinned forge image with GPU access only after
   obtaining the shared training reservation and stopping GPU inference.
5. Evaluate using evaluate_samantha.py with held-out voices, positive phrases,
   silence/background and phonetic near-misses. Preserve complete words. Inspect
   positives acoustically/transcribe samples before accepting a model. Keep a
   final fresh suite separate from development evaluation.
6. Install only the model YOU trained and evaluated through your controller's
   supported wake-word model workflow. Select Samantha on the Dot and choose a
   detection threshold from your own false-positive/missed-word measurements.
7. Test in the actual room at normal distance and with background audio. Retain
   the previous known-good model for rollback; do not silently overwrite it.

pipeline.py/train_reserved.py are reference orchestration tied to a configured
shared worker, inference activation-request file and UID. Configure those
prerequisites first; do not launch them as a fresh-machine one-click installer.
The canonical builder/evaluator are the key reusable training implementation.

Upstream: https://github.com/wilbowes/EchoMuse/tree/main/oww_forge
