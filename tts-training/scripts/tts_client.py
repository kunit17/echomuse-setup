"""Standard-library-only Hermes command client for the warm inference service."""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave

URL = 'http://127.0.0.1:8811'
# Never route loopback audio through an inherited HTTP proxy.
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def ready():
    try:
        with HTTP.open(URL + '/health', timeout=2) as response:
            health = json.load(response)
            return health.get('ready') is True and health.get('model_loaded') is True
    except (OSError, ValueError):
        return False


def synthesize(text, speed):
    request = urllib.request.Request(URL + '/synthesize',
        data=json.dumps({'text': text, 'speed': speed}).encode(),
        headers={'Content-Type': 'application/json'})
    with HTTP.open(request, timeout=180) as response:
        if response.headers.get_content_type() != 'audio/wav':
            raise ValueError('Server did not return audio/wav')
        audio = response.read()
    with wave.open(io.BytesIO(audio), 'rb') as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) != (32000, 1, 2, 'NONE') or wav.getnframes() == 0:
            raise ValueError('Invalid or empty PCM WAV')
        if len(wav.readframes(wav.getnframes())) != wav.getnframes() * 2:
            raise ValueError('Truncated WAV')
    return audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text-file', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--speed', type=float, default=1.0)
    args = parser.parse_args()
    text = args.text_file.read_text(encoding='utf-8')
    if not ready():
        raise RuntimeError('TTS service is unavailable; service activation is owner-controlled')
    audio = synthesize(text, args.speed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=args.out.parent, prefix='.tts-', delete=False) as output:
            temporary = output.name
            output.write(audio)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, args.out)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    print(f'Wrote {len(audio)} WAV bytes to {args.out}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'GPT-SoVITS TTS client failed: {exc}', file=sys.stderr)
        sys.exit(1)
