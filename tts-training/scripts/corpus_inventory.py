"""Streaming, transcript-free SheCorpus pause inventory and frozen source split."""
import json
import os
from pathlib import Path
import random
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/voiceuser/projects/she-training/SheCorpus.flac')
RATE = 48000
FRAME = 480
WINDOW = RATE * 30
DURATION = 9772

def windows():
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-i', str(SOURCE),
               '-af', 'pan=mono|c0=0.5*c0+0.5*c1', '-ac', '1', '-ar', str(RATE), '-f', 'f32le', '-']
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    try:
        while True:
            raw = process.stdout.read(WINDOW * 4)
            if not raw:
                break
            yield np.frombuffer(raw, dtype='<f4')
        if process.wait() != 0:
            raise RuntimeError('Bounded streaming decode failed')
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.kill()
            process.wait()

def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def split_spans():
    selected = sorted(random.Random(20261112).sample(range(48), 5))
    validation = [(200 * i, 200 * (i + 1)) for i in selected]
    guards = []
    for start, end in validation:
        start, end = max(0, start - 15), min(DURATION, end + 15)
        if guards and start <= guards[-1][1]:
            guards[-1][1] = max(end, guards[-1][1])
        else:
            guards.append([start, end])
    training = []
    cursor = 0
    for start, end in guards:
        if cursor < start:
            training.append([cursor, start])
        cursor = end
    if cursor < DURATION:
        training.append([cursor, DURATION])
    return {'seed': 20261112, 'validation_blocks': selected,
            'train': training, 'validation': validation,
            'guard_seconds': DURATION - sum(b-a for a,b in training+validation),
            'source_rate': 48000,
            'source_frames': {key: [[int(a*48000), int(b*48000)] for a,b in spans]
                              for key, spans in [('train', training), ('validation', validation)]}}

def main():
    spans = split_spans()
    atomic_json(ROOT / 'reports/corpus-split.json', spans)
    peak, samples = 0., 0
    for audio in windows():
        peak = max(peak, float(np.abs(audio).max()))
        samples += len(audio)
    assert samples == RATE * DURATION, (samples, RATE * DURATION)
    thresholds = {'relative_2pct': peak * .02, 'minus_40_dbfs': .01}
    runs = {name: [] for name in thresholds}
    active = {name: None for name in thresholds}
    frame_index = 0
    for audio in windows():
        assert len(audio) % FRAME == 0
        rms = np.sqrt(np.mean(audio.reshape(-1, FRAME) ** 2, axis=1))
        for offset, value in enumerate(rms):
            index = frame_index + offset
            for name, threshold in thresholds.items():
                if value < threshold:
                    if active[name] is None:
                        active[name] = index
                elif active[name] is not None:
                    start = active[name]
                    if index - start >= 30:
                        runs[name].append([start / 100, index / 100])
                    active[name] = None
        frame_index += len(rms)
    for name in thresholds:
        if active[name] is not None and frame_index-active[name] >= 30:
            runs[name].append([active[name]/100, frame_index/100])
    result = {'source': 'SheCorpus recording', 'duration_seconds': samples/RATE,
              'analysis_rate': RATE, 'frame_seconds': .01, 'decode_window_seconds': 30,
              'mono_peak': peak, 'thresholds': thresholds, 'runs': runs,
              'summary': {name: {'count': len(items),
                                 'total_seconds': sum(b-a for a,b in items),
                                 'median_seconds': float(np.median([b-a for a,b in items])),
                                 'max_seconds': max(b-a for a,b in items)}
                          for name, items in runs.items()}}
    atomic_json(ROOT / 'reports/corpus-pause-inventory.json', result)
    print(json.dumps(result['summary']), flush=True)

if __name__ == '__main__':
    main()
