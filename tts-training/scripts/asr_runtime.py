"""Durable ASR provenance and cache utilities. Importing this never loads a model."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time
import uuid

SCHEMA = 2
DECODE_SETTINGS = dict(language='en', task='transcribe', beam_size=5, temperature=0,
                       condition_on_previous_text=False, vad_filter=False,
                       word_timestamps=True)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
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
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def atomic_text(path, text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('w') as stream:
            stream.write(text);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
        fd=os.open(path.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:temporary.unlink(missing_ok=True)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True).encode()).hexdigest()


def heartbeat(phase, **progress):
    path = os.environ.get('ASR_HEARTBEAT_PATH')
    if path:
        atomic_json(path, dict(run_id=os.environ.get('ASR_RUN_ID'), phase=phase,
                               monotonic=time.monotonic(), time=time.time(), **progress))


def snapshot(path):
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def file_digest(path):
    path = Path(path)
    before = snapshot(path)
    digest = hashlib.sha256()
    last = time.monotonic()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
            if time.monotonic() - last > 5:
                heartbeat('fingerprinting')
                last = time.monotonic()
    if snapshot(path) != before:
        raise RuntimeError('An input changed during fingerprinting')
    return dict(path=str(path.resolve()), bytes=before[2], sha256=digest.hexdigest(),
                snapshot=before)


def make_context(files, settings):
    descriptions = {name: file_digest(path) for name, path in sorted(files.items())}
    identity = dict(schema=SCHEMA, settings=settings,
                    inputs={key: {k: v for k, v in item.items() if k != 'snapshot'}
                            for key, item in descriptions.items()})
    return dict(fingerprint=fingerprint(identity), identity=identity,
                snapshots={item['path']: item['snapshot'] for item in descriptions.values()})


def verify_inputs(context):
    if any(snapshot(path) != expected for path, expected in context['snapshots'].items()):
        raise RuntimeError('An input changed during the ASR run; outputs must not be published')


def asr_context(root, source, **extra):
    root = Path(root)
    directory = root / '.cache/whisper-large-v3'
    files = {'source': source}
    for name in ['model.bin', 'config.json', 'tokenizer.json', 'vocabulary.json',
                 'preprocessor_config.json']:
        path = directory / name
        if name in ['model.bin', 'config.json'] or path.exists():
            files['model/' + name] = path
    versions = {}
    for package in ['faster-whisper', 'ctranslate2']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'not-installed'
    settings=dict(device='cuda',compute_type='float16',source_mix='0.5*c0+0.5*c1',
                  sample_rate=16000,decode=DECODE_SETTINGS,versions=versions)
    settings.update(extra)
    return make_context(files,settings)


def load_model(root):
    if not os.environ.get('ASR_RUN_ID'):
        raise RuntimeError('ASR must be launched through the worker controller')
    heartbeat('model_loading')
    from faster_whisper import WhisperModel
    model = WhisperModel(str(Path(root) / '.cache/whisper-large-v3'), device='cuda',
                         compute_type='float16', cpu_threads=0, num_workers=1,
                         local_files_only=True)
    if model.model.compute_type != 'float16':
        raise RuntimeError('The requested CUDA float16 profile was not honored')
    heartbeat('model_ready', actual_compute_type=model.model.compute_type)
    return model


class Cache:
    """Legacy, un-fingerprinted caches are preserved and deliberately never reused."""
    def __init__(self, directory, context):
        self.directory = Path(directory) / 'v2' / context['fingerprint']

    def path(self, key):
        return self.directory / (fingerprint(key) + '.json')

    def get(self, key):
        envelope = read_json(self.path(key))
        if not isinstance(envelope, dict) or envelope.get('key') != key:
            return None
        if envelope.get('value_hash') != fingerprint(envelope.get('value')):
            return None
        return envelope['value']

    def put(self, key, value):
        path = self.path(key)
        if path.exists() and self.get(key) is None:
            history = path.with_name(path.name + '.invalid-' + uuid.uuid4().hex)
            shutil.copy2(path, history)
        atomic_json(path, dict(key=key, value_hash=fingerprint(value), value=value))


def archive(path, directory):
    path = Path(path)
    if not path.exists():
        return
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / (path.stem + '-' + file_digest(path)['sha256'] + path.suffix)
    if not destination.exists():
        shutil.copy2(path, destination)
        with destination.open('rb') as stream:
            os.fsync(stream.fileno())


def agreement_report(rows):
    return dict(method='exact normalized-word agreement between two runs of the same ASR model',
                accepted_clips=len(rows), agreeing_clips=len(rows),
                agreement_rate=1.0 if rows else None, independent_accuracy_measurement=False)


def summarize_filters(exclusions, retained_seconds, source_seconds):
    from collections import Counter
    counts = Counter(item['reason'] for item in exclusions)
    if 'empty_or_identifying_reference' in counts:
        counts['legacy_empty_or_identifying_reference_unclassified'] = counts.pop('empty_or_identifying_reference')
    return dict(rejected_clips_by_reason=dict(counts), retained_seconds=retained_seconds,
                source_seconds=source_seconds, retained_fraction=retained_seconds/source_seconds if source_seconds>0 else None,
                privacy_policy_changed=False, rejected_sample_review_complete=False)
