# Public distribution boundary

This is a fresh-history, sanitized source distribution. It includes reusable
source, package-version snapshots and example configuration, not a household
backup or an export of a running Home Assistant instance.

Excluded: private Git history, author metadata from that history, household
topology, owner/run logs, device/account IDs, training approval records, corpus
transcripts and manifests, recordings, trained model weights, generated samples,
browser profiles, credentials, tokens, databases, and uncommitted runtime edits.

Addresses in 192.0.2.0/24 and names voiceuser, piuser, DesktopUser and Example
City are examples. Replace them before use. No working login is supplied.
Generic ports, model names, public upstream URLs and dependency revisions remain
because they are needed to understand and reproduce the software workflow.

A scan reduces disclosure risk; it cannot mathematically prove that arbitrary
source text contains no sensitive information. Audit new contributions before
publication. Do not commit your configured copy, generated credentials or audio.

This distribution does not claim that another person's voice can be reproduced
without their recordings. Supply your own authorized corpus and train your own
weights. Historical helper programs are reference implementations; executable
setup steps are in README.md and docs/tts-training.md. Dataset approval must be
created for the new corpus, never copied from another training run.

Third-party projects and files retain their original licensing requirements.
This repository does not grant a new license over upstream software, models or
datasets. Review their licenses and hardware support before downloading them.
