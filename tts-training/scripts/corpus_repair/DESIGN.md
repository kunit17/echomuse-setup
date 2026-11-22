# Repaired corpus, version 1

Goal: rebuild training and validation WAV/text pairs from both immutable source
recordings. Operational commands and current state are in TTS_WORKFLOW.md.
Preparation itself never starts model training.

Sources: SheCorpus.flac (9,772 s, 48 kHz stereo) and
Myvoicetesting_32k_mono.wav (16,904.539125 s, 32 kHz mono).

## Boundary rules

1. Stream source audio in small windows; retain hashes, source frame ranges and
   reproducible mono/resampling rules. Never load a complete recording into RAM.
2. Measure 10 ms RMS frames and run Silero speech detection on overlapping
   windows. A cut is eligible only inside at least 300 ms of low-energy audio
   that is also outside conservatively padded speech-detection regions.
3. Keep approximately 150–250 ms of real source silence on either side. The
   silence must be present in the recording; no artificial padding or fades.
4. Prefer complete pause-delimited phrases, generally 8–25 seconds. Retain
   shorter complete phrases where necessary. Never force a cut to meet a target
   duration; regions with no safe boundary are quarantined.
5. Reject any exported clip whose first or last 120 ms fails the same measured
   silence test. Check the final samples, not just a word timestamp.

## Transcript/alignment checks

Fresh Whisper large-v3 CUDA float16 transcription is performed on the new cuts.
Original drafts remain archived; old cut-based verification is not reused.
An independent Wav2Vec2 CTC acoustic model aligns the text to each waveform.
Record per-word alignment confidence, first/last aligned character margins,
and an independently decoded transcript. Low confidence, unexplained missing
speech, disagreement at either edge, or uncertain boundaries go to quarantine.
An ASR word match alone is never sufficient. Model confidence is diagnostic,
not proof of complete phonemes or transcript correctness.

## Exclusions and partitioning

Preserve the owner's music exclusions and earlier reject-all-flags decisions
as source intervals; never reintroduce those intervals automatically. Preserve
the original source's frozen train/validation spans and the additional source's
five-minute validation blocks. Keep a five-second guard at new partition
boundaries. Cuts cannot cross a partition or exclusion boundary. Remove exact
audio duplicates; hold normalized transcript duplicates to prevent split
leakage. No proper-name filter.

Use all usable data from the approved source intervals, rather than filling an
arbitrary three-hour quota or discarding good speech to match a clip count.
Report accepted, quarantined, previously excluded, and unsegmented durations.

## Verification and publication

Exercise synthetic regression cases (false timestamp gaps, active speech at
an edge, insufficient silence, forbidden intervals, and split crossing).
Produce source-context comparisons for the reported clip 111 and examples
from both recordings. These are listening artifacts; automatic checks do not
constitute a human listening review.

Publish a new immutable version with train/validation manifests, WAVs, word
alignment evidence, rejection reasons, model/source/code hashes and QA summary.
Retire previous corpora only after authorized cleanup and dependency checks. Point the next CSM run at the new
version only after its automatic checks pass. Leave its training pause guard on.
