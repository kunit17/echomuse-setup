"""Report only observed continuous-corpus results, including failures."""
import json
from pathlib import Path
from corpus_ground_truth import ROOT,save

def load(name):
    path=ROOT/'reports'/name
    return json.loads(path.read_text()) if path.exists() else None

truth=load('corpus-ground-truth.json');corpus=load('corpus-verification.json');evaluation=load('continuous-evaluation.json')
curves=[]
path=ROOT/'reports/continuous-epochs.jsonl'
if path.exists():curves=[json.loads(line) for line in path.read_text().splitlines()]
text=['# Continuous-corpus rebuild report','',
    'Production decoder: faster-whisper large-v3, CUDA float16. Float32 expanded weights alone require 5.75 GiB and were not retried. The owner replaced the decoder comparison with ground-truth validation.',
    '', 'Hermes remains on she-xtts. The warm TTS service is left inactive; deployment belongs to the owner. No live Telegram message was sent.','']
if truth:
    text += [f"Ground-truth validation: status {truth['status']}; {truth['mapped_clips']} mapped clips; {truth['covered_seconds']:.3f} union seconds of the 9772-second corpus ({100*truth['coverage_fraction']:.2f}%). This is partial corpus coverage.",
             f"Controls: positive {truth['positive_control']:.6f}; negative {truth['negative_control']:.6f}. Both span endpoints were verified. Removed rows, repetition flags and high-disagreement audit rows were excluded.",'']
    if truth['status']=='complete':text += [f"True word-edit WER: micro {truth['micro_wer']:.5f}; median {truth['median_wer']:.5f}; mean {truth['mean_wer']:.5f}. Legacy SequenceMatcher disagreement median: {truth['median_legacy_sequence_disagreement']:.5f}. The historical 0.0088 figure uses the latter metric rather than edit-distance WER.",'']
if corpus:
    text += [f"Corpus status: {corpus['status']}.",'']
    if corpus.get('corpus_built'):
        text += [f"Built {corpus['clips']} clips, {corpus['hours']:.4f} hours; mean duration {corpus['mean_duration_seconds']:.3f} seconds. Splits: `{json.dumps(corpus['splits'])}`.",
            'Predictions were 1000–1400 clips and 2.3–2.6 hours. Strict privacy, pause-boundary and transcript-agreement exclusions can lower retention; see the verification JSON for actual exclusions.',
            'Clip audio was checked as 32000 Hz mono PCM s16le. Two-pass ASR agreement is repeatability evidence, not an independent human transcript accuracy estimate. Manual listening review remains unverified.','']
        if corpus.get('asr_pass_agreement'):
            text += ['ASR pass agreement: `'+json.dumps(corpus['asr_pass_agreement'])+'`. This is not reference-based WER.','']
        if corpus.get('filtering'):
            text += ['Filtering yield and review status: `'+json.dumps(corpus['filtering'])+'`. Privacy rules have not been relaxed.','']
for stage in ('s1','s2'):
    rows=sorted({r['epoch']:r for r in curves if r['stage']==stage}.values(),key=lambda r:r['epoch'])
    trained=[r for r in rows if r['epoch']>0]
    selection=load(f'continuous-{stage}-selection.json')
    if not trained:text += [f'{stage}: no completed new-corpus training epochs or optimizer steps verified.',''];continue
    total=sum(r['optimizer_steps'] for r in trained)
    text += [f"{stage}: {trained[-1]['epoch']} completed epochs; {total} recorded optimizer updates. Final global step: {trained[-1]['global_step']}.",'']
    if stage=='s2':text += [f"Actual optimizer calls: generator {sum(r['generator_optimizer_calls'] for r in trained)}, discriminator {sum(r['discriminator_optimizer_calls'] for r in trained)}.",'']
    if selection:text += [f"Selected epoch {selection['selected']['epoch']}: `{selection['selected']['checkpoint']}` by held-out {'token loss' if stage=='s1' else 'mel reconstruction, KL tie-breaker'}. Plateau criteria met: {selection['plateau']}.",'']
    text += ['| Epoch | Global step | Updates | Training metrics | Validation metrics |','|---:|---:|---:|---|---|']
    for row in rows:text.append(f"| {row['epoch']} | {row['global_step']} | {row['optimizer_steps']} | {json.dumps(row.get('train',{}),sort_keys=True)} | {json.dumps(row['validation'],sort_keys=True)} |")
    text += ['']
    if len(trained)>1:
        previous,final=trained[-2:]
        delta={k:final['train'][k]-previous['train'][k] for k in final['train'] if k in previous['train']}
        text += [f'Final consecutive training-metric deltas: `{json.dumps(delta,sort_keys=True)}`.','']
if evaluation and evaluation.get('status')=='complete':
    text += [f"Novel-text micro WER: {evaluation['novel_micro_wer']:.5f}; exact matches {evaluation['novel_exact_match_count']}/20. Dedicated termination probes have zero edits: {evaluation['termination_zero_edits']}.",
        f"Measured unsplit punctuation quiet runs, in seconds at both thresholds: `{json.dumps(evaluation['punctuation_quiet_seconds'])}`. Baseline comma/period/semicolon were each 0.00 seconds.",
        f"Punctuation thresholds met: {evaluation['punctuation_thresholds_met']}. Defects fixed: {evaluation['defects_fixed']}.",
        'The five-sentence unsplit sample and each aligned internal gap are in continuous-evaluation.json. Application-segmented output has five separately generated sentences and explicitly inserted gaps; those gaps do not demonstrate learned pauses.','']
else:text += ['New-model pause and termination measurements: not yet available. No improvement over the 0.00-second baseline is claimed.','']
# Do not write executable service activation into any script. The command is
# documentation for an explicit owner decision after checkpoint evaluation.
if all(load(f'continuous-{s}-selection.json') for s in ('s1','s2')):
    g=load('continuous-s1-selection.json')['selected']['checkpoint'];s=load('continuous-s2-selection.json')['selected']['checkpoint']
    text += ['Owner switch instructions (not executed): set `custom.t2s_weights_path` and `custom.vits_weights_path` in configs/tts-infer.yaml to the selected paths above; choose a reference from data/shecorpus/train rather than the old corpus. Then set the Hermes provider and activate the service at the owner’s discretion. These paths alone have not been live-provider verified.','',f'GPT: `{g}`',f'SoVITS: `{s}`','']
failures=load('continuous-failures.json') or []
for failure in failures:
    text += [f"Observed failure in {failure['stage']}: `{failure['error']}`. Resolved: {failure['resolved']}. {failure.get('resolution','')}",'']
for resource in sorted((ROOT/'reports').glob('she-continuous-*-resources.txt')):
    body=resource.read_text()
    if 'Result=success' not in body:text += [f'Failed stage: `{resource.name}`. Exact unit result:','```',body.strip(),'```','']
path=ROOT/'reports/retrain-report.md'
with path.open('w') as f:f.write('\n'.join(text).rstrip()+'\n');f.flush();__import__('os').fsync(f.fileno())
print('Updated observed-results report',flush=True)
