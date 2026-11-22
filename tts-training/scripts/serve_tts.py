"""Persistent loopback GPT-SoVITS inference worker; launch using scripts/env.sh."""
import io
import json
import logging
import math
import os
import sys
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CONFIG = ROOT / 'configs/tts-reference.json'
LOCK = threading.Lock()
LOG = logging.getLogger('she-gptsovits')


def load_reference(path=REFERENCE_CONFIG):
    import wave
    reference = json.loads(path.read_text())
    audio = Path(reference['audio_path']).resolve()
    allowed = (ROOT / 'data/shecorpus/train').resolve()
    if not audio.is_relative_to(allowed) or not audio.is_file():
        raise ValueError('Reference must be a retained training WAV')
    text = reference.get('text')
    if not isinstance(text, str) or not text.strip() or len(text) > 5000:
        raise ValueError('Invalid reference transcript')
    with wave.open(str(audio), 'rb') as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) != (32000, 1, 2, 'NONE') or wav.getnframes() == 0:
            raise ValueError('Invalid reference WAV')
    return {'audio_path': str(audio), 'text': text}


def synthesize(tts, text, speed, reference):
    import numpy as np
    import soundfile as sf
    inputs = {
        'text': text, 'text_lang': 'en',
        'ref_audio_path': reference['audio_path'],
        'prompt_text': reference['text'], 'prompt_lang': 'en',
        'top_k': 15, 'top_p': 1.0, 'temperature': 0.8, 'speed_factor': speed,
        'text_split_method': 'cut5', 'batch_size': 1, 'seed': 1700,
        'parallel_infer': False, 'use_cuda_graph': False,
        'use_flash_attention': False, 'repetition_penalty': 1.35,
    }
    # The generator must be consumed inside the lock: iteration mutates model state.
    with LOCK:
        chunks = list(tts.run(inputs))
    parts = []
    for sr, audio in chunks:
        if sr != 32000 or audio.ndim != 1 or not len(audio):
            raise RuntimeError('Unexpected inference audio format')
        if parts:
            parts.append(np.zeros(3200, dtype=np.float32))
        # Upstream yields int16; normalize before concatenating float silence.
        parts.append(audio.astype(np.float32) / 32768 if audio.dtype == np.int16 else audio.astype(np.float32))
    if not parts:
        raise RuntimeError('Inference returned no audio')
    output = io.BytesIO()
    sf.write(output, np.concatenate(parts), 32000, format='WAV', subtype='PCM_16')
    return output.getvalue(), len(chunks)


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    os.chdir(ROOT / 'upstream')
    reference = load_reference()
    from TTS_infer_pack.TTS import TTS, TTS_Config
    import torch
    tts = TTS(TTS_Config(str(ROOT / 'configs/tts-infer.yaml')))
    health = {'ready': True, 'model_loaded': True, 'device': torch.cuda.get_device_name(0),
              'gpt_checkpoint': str(tts.configs.t2s_weights_path),
              'sovits_checkpoint': str(tts.configs.vits_weights_path)}
    import yaml
    expected = yaml.safe_load((ROOT / 'configs/tts-infer.yaml').read_text())['custom']
    assert health['gpt_checkpoint'] == expected['t2s_weights_path']
    assert health['sovits_checkpoint'] == expected['vits_weights_path']

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(180)

        def log_message(self, fmt, *args):
            LOG.info(fmt, *args)

        def respond(self, status, body, content_type='application/json', chunks=None):
            try:
                self.send_response(status)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                if chunks is not None:
                    self.send_header('X-Audio-Chunks', str(chunks))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                LOG.info('Client disconnected')

        def do_GET(self):
            self.respond(200, json.dumps(health).encode()) if self.path == '/health' else self.respond(404, b'{}')

        def do_POST(self):
            if self.path != '/synthesize':
                self.respond(404, b'{}')
                return
            try:
                if self.headers.get_content_type() != 'application/json' or self.headers.get('Origin'):
                    raise ValueError('Expected local application/json request without Origin')
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 65536:
                    raise ValueError('Invalid body length')
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError('Expected JSON object')
                text = data.get('text')
                speed = data.get('speed', 1.0)
                if not isinstance(text, str) or not text.strip() or len(text) > 5000:
                    raise ValueError('text must contain 1–5000 characters')
                if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed) or not 0.25 <= speed <= 4:
                    raise ValueError('speed must be between 0.25 and 4')
            except (ValueError, UnicodeError, TimeoutError) as exc:
                self.close_connection = True
                self.respond(400, json.dumps({'error': str(exc)}).encode())
                return
            started = time.monotonic()
            try:
                audio, chunks = synthesize(tts, text, speed, reference)
                self.respond(200, audio, 'audio/wav', chunks)
                LOG.info('Synthesized chars=%d chunks=%d bytes=%d elapsed=%.3fs', len(text), chunks, len(audio), time.monotonic() - started)
            except Exception:
                LOG.exception('Synthesis failed')
                self.respond(500, b'{"error":"Synthesis failed; see service log"}')

    server = ThreadingHTTPServer(('127.0.0.1', 8811), Handler)
    server.daemon_threads = True
    LOG.info('Model loaded; listening on 127.0.0.1:8811: %s', health)
    server.serve_forever()


if __name__ == '__main__':
    main()
