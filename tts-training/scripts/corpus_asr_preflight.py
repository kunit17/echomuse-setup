"""CUDA fp16 CUDA ASR timing preflight; never emit source transcripts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import re
import subprocess
import time
from asr_runtime import asr_context,heartbeat,load_model,verify_inputs

ROOT = Path(__file__).resolve().parents[1]
assert not os.environ.get('PYTHONPATH'), 'ASR requires isolated PYTHONPATH'
assert subprocess.run(['systemctl', '--user', 'is-active', '--quiet',
                       'she-gptsovits-tts.service']).returncode != 0

def save(value):
    path = ROOT / 'reports/corpus-asr-preflight.json'
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)

args = argparse.ArgumentParser(); args.add_argument('--compute-type', default='float16', choices=['float16']); args.add_argument('--device', default='cuda', choices=['cuda']); args = args.parse_args()
state = dict(status='running',run_id=os.environ.get('ASR_RUN_ID'),compute_type=args.compute_type,device=args.device,
             model_ready=False,resource_limits=None)
started = time.monotonic()
try:
    save(state)
    print('Loading cached large-v3:', args.device, args.compute_type, flush=True)
    context=asr_context(ROOT,Path('/home/voiceuser/projects/she-training/SheCorpus.flac'),benchmark_version=2)
    state['run_fingerprint']=context['fingerprint']
    model=load_model(ROOT)
    state.update(model_ready=True, load_seconds=time.monotonic()-started)
    state['actual_compute_type'] = model.model.compute_type
    assert state['actual_compute_type'] == args.compute_type
    save(state)
    print('Model loaded', state['load_seconds'], flush=True)
    import numpy as np
    work_start = time.monotonic()
    count = 0
    fingerprints = []
    # Twenty independent 30-second windows: ten minutes, without crossing spans.
    for offset in range(30, 630, 30):
        raw = subprocess.check_output([
            'ffmpeg', '-nostdin', '-v', 'error', '-ss', str(offset), '-i',
            '/home/voiceuser/projects/she-training/SheCorpus.flac', '-t', '30',
            '-af', 'pan=mono|c0=0.5*c0+0.5*c1', '-ar', '16000', '-f', 'f32le', '-'])
        audio = np.frombuffer(raw, dtype='<f4')
        segments, _ = model.transcribe(audio, language='en', task='transcribe', beam_size=5,
                                      temperature=0, condition_on_previous_text=False,
                                      vad_filter=False, word_timestamps=True)
        tokens = []
        words = 0
        for segment in segments:
            words += len(segment.words or [])
            tokens.extend(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", segment.text.lower()))
        fingerprints.append(dict(start_seconds=offset, duration_seconds=30,
                                 normalized_word_count=len(tokens),
                                 sha256=hashlib.sha256('\n'.join(tokens).encode()).hexdigest()))
        assert words > 0, 'No words in preflight window'
        count += words
        state.update(completed_audio_seconds=offset, words_detected=count)
        state['window_fingerprints'] = fingerprints
        save(state)
        heartbeat('preflight_window',completed_audio_seconds=offset,words_detected=count)
        print('Completed audio seconds', offset, flush=True)
    elapsed = time.monotonic()-work_start
    state.update(status='passed', transcription_completed=True, audio_seconds=600,
                 transcription_seconds=elapsed, audio_seconds_per_wall_second=600/elapsed,
                 first_pass_estimate_seconds=9772/(600/elapsed),
                 max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    verify_inputs(context)
except BaseException as exc:
    state.update(status='failed', failure_kind=str(exc) if isinstance(exc, TimeoutError) else type(exc).__name__,
                 elapsed_seconds=time.monotonic()-started)
    save(state)
    raise
save(state)
print(json.dumps(state), flush=True)
