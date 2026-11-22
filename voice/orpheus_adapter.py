"""Loopback Wyoming bridge for the approved Orpheus PCM streaming engine.

No model loading, dependency changes, or GPU work is performed by this adapter.
Disconnects cancel the upstream HTTP stream; completion requires its trace.
"""
import argparse
import asyncio
import contextlib
import json
import logging
import re
from functools import partial
from pathlib import Path

import aiohttp
from wyoming.audio import AudioStart, AudioChunk, AudioStop
from wyoming.error import Error
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize, SynthesizeStart, SynthesizeChunk, SynthesizeStop, SynthesizeStopped

LOG = logging.getLogger("orpheus-wyoming")
VOICE = "orpheus-epoch3"
RATE = 24000
MAX_TEXT = 5000


class VoiceError(Exception):
    def __init__(self, text, code="synthesis_failed"):
        super().__init__(text)
        self.code = code


def info():
    attribution = Attribution("Local trained Orpheus", "https://github.com/canopyai/Orpheus-TTS")
    return Info(tts=[TtsProgram(name="Orpheus", attribution=attribution, installed=True,
        description="Your trained Orpheus voice", version="1.0.0",
        voices=[TtsVoice(name=VOICE, attribution=attribution, installed=True,
            description="Trained voice (epoch 3)", version="epoch3-4bit", languages=["en"])],
        supports_synthesize_streaming=True)])


def completed_sentences(text):
    """Hold incomplete suffixes and ambiguous chunk-final punctuation.

    Mirrors the existing engine's abbreviation exclusions. The engine retains
    its own full sentence/EOS policy and safely splits long complete sentences.
    """
    ready = []
    start = 0
    for match in re.finditer(r'''[.!?]+["\u201d\u2019']*(?=\s)''', text):
        prefix = text[start:match.end()]
        if match.group().startswith('.') and re.search(r'\b(?:Mr|Mrs|Ms|Dr|Prof|St|[A-Z])\.$', prefix):
            continue
        sentence = prefix.strip()
        if sentence:
            ready.append(sentence)
        start = match.end()
    return ready, text[start:]


class Backend:
    def __init__(self, session, endpoint, adapter_hash):
        self.session = session
        self.endpoint = endpoint.rstrip('/')
        self.adapter_hash = adapter_hash
        self.owner = None

    async def check_ready(self):
        try:
            async with self.session.get(self.endpoint + '/health', timeout=aiohttp.ClientTimeout(total=5)) as response:
                response.raise_for_status()
                health = await response.json()
        except (aiohttp.ClientError, TimeoutError):
            raise VoiceError("Orpheus voice is off or unavailable. Start Orpheus explicitly.", "voice_off") from None
        if not health.get('ready'):
            raise VoiceError("Orpheus is still loading.", "voice_not_ready")
        required = dict(model='Orpheus', epoch=3, runtime='vllm-streaming', precision='4bit', codec='cuda')
        if any(health.get(k) != v for k, v in required.items()) or health.get('adapter_sha256') != self.adapter_hash:
            raise VoiceError("Loaded voice does not match the approved Orpheus model.", "wrong_model")

    async def audio(self, text):
        samples = 0
        async with self.session.post(self.endpoint + '/stream', json={'text': text}) as response:
            if response.status != 200:
                raise VoiceError("Orpheus rejected the speech request.")
            request_id = response.headers.get('X-Request-ID', '')
            if not re.fullmatch(r'[A-Za-z0-9-]{1,100}', request_id):
                raise VoiceError("Orpheus did not return a valid request ID.")
            if (response.headers.get('X-Sample-Rate'), response.headers.get('X-Channels'),
                    response.headers.get('X-Sample-Format')) != ('24000', '1', 's16le'):
                raise VoiceError("Orpheus returned an unexpected audio format.")
            remainder = b''
            async for chunk in response.content.iter_chunked(4096):
                chunk = remainder + chunk
                end = len(chunk) - len(chunk) % 2
                remainder = chunk[end:]
                if end:
                    samples += end // 2
                    yield chunk[:end]
            if remainder or not samples:
                raise VoiceError("Orpheus returned incomplete PCM audio.")
        # EOF alone is not success. Trace is committed in the engine's finally.
        trace = None
        for _ in range(30):
            async with self.session.get(self.endpoint + '/trace/' + request_id) as response:
                if response.status == 200:
                    trace = await response.json()
                    break
                if response.status != 404:
                    raise VoiceError("Could not verify Orpheus completion.")
            await asyncio.sleep(.1)
        if not trace or trace.get('status') != 'complete' or trace.get('sample_count') != samples:
            raise VoiceError("Orpheus generation was incomplete; audio is not a successful reply.")
        LOG.info("Completed request=%s samples=%d", request_id, samples)


class Handler(AsyncEventHandler):
    def __init__(self, backend, reader, writer, buffer_seconds=1.0):
        super().__init__(reader, writer)
        self.backend = backend
        self.buffer_bytes = int(buffer_seconds * RATE) * 2
        self.worker = None
        self.mode = None
        self.input_done = False
        self.queue = None
        self.pending = ''
        self.characters = 0
        self.write_lock = asyncio.Lock()

    async def write_event(self, event):
        async with self.write_lock:
            await super().write_event(event)

    def begin(self, streaming, voice=None, text_format=None):
        if self.worker and not self.worker.done():
            raise VoiceError("A speech request is already active.", "busy")
        if self.backend.owner is not None:
            raise VoiceError("Orpheus is speaking another reply. Try again.", "busy")
        if voice and (voice.name not in (None, VOICE) or voice.speaker is not None
                or (voice.language and voice.language.split('-')[0] != 'en')):
            raise VoiceError("Unsupported voice or language.", "invalid_request")
        if text_format not in (None, 'text'):
            raise VoiceError("Only plain text is supported.", "invalid_request")
        self.backend.owner = self
        self.mode = 'stream' if streaming else 'plain'
        self.input_done = False
        self.queue = asyncio.Queue()
        self.pending = ''
        self.characters = 0
        self.worker = asyncio.create_task(self.generate(), name='orpheus voice turn')

    def add_text(self, text, final=False):
        if not isinstance(text, str) or self.characters + len(text) > MAX_TEXT:
            raise VoiceError("Speech text must be at most 5000 characters.", "invalid_request")
        self.characters += len(text)
        self.pending += text
        sentences, self.pending = completed_sentences(self.pending)
        for sentence in sentences:
            self.queue.put_nowait(sentence)
        if final:
            if self.pending.strip():
                self.queue.put_nowait(self.pending.strip())
            self.pending = ''
            self.input_done = True
            self.queue.put_nowait(None)

    async def generate(self):
        started = False
        pending_audio = bytearray()
        try:
            async with asyncio.timeout(240):
                await self.backend.check_ready()
                while True:
                    text = await asyncio.wait_for(self.queue.get(), 30)
                    if text is None:
                        break
                    async for chunk in self.backend.audio(text):
                        if not started:
                            pending_audio.extend(chunk)
                            if len(pending_audio) < self.buffer_bytes:
                                continue
                            await self.write_event(AudioStart(rate=RATE, width=2, channels=1).event())
                            started = True
                            chunk = bytes(pending_audio)
                            pending_audio.clear()
                        await self.write_event(AudioChunk(rate=RATE, width=2, channels=1, audio=chunk).event())
                if pending_audio:
                    await self.write_event(AudioStart(rate=RATE, width=2, channels=1).event())
                    await self.write_event(AudioChunk(rate=RATE, width=2, channels=1, audio=bytes(pending_audio)).event())
                    started = True
                if not started:
                    raise VoiceError("There was no speech text.", "invalid_request")
                await self.write_event(AudioStop().event())
                if self.mode == 'stream':
                    await self.write_event(SynthesizeStopped().event())
        except asyncio.CancelledError:
            raise
        except Exception as error:
            code = error.code if isinstance(error, VoiceError) else 'synthesis_failed'
            message = str(error) if isinstance(error, VoiceError) else 'Orpheus synthesis failed or timed out.'
            LOG.warning("Voice turn failed: %s", code)
            with contextlib.suppress(ConnectionError, OSError):
                await self.write_event(Error(text=message, code=code).event())
        finally:
            if self.backend.owner is self:
                self.backend.owner = None
            if self.input_done:
                self.mode = None

    async def handle_event(self, event):
        try:
            if Describe.is_type(event.type):
                await self.write_event(info().event())
            elif SynthesizeStart.is_type(event.type):
                request = SynthesizeStart.from_event(event)
                self.begin(True, request.voice, request.text_format)
            elif Synthesize.is_type(event.type):
                if self.mode == 'stream':
                    return True  # HA's full-text compatibility copy, never speak twice.
                request = Synthesize.from_event(event)
                self.begin(False, request.voice, request.text_format)
                self.add_text(request.text, final=True)
            elif SynthesizeChunk.is_type(event.type):
                if self.mode != 'stream' or self.input_done:
                    raise VoiceError("Text chunk outside an active input stream.", "invalid_request")
                self.add_text(SynthesizeChunk.from_event(event).text)
            elif SynthesizeStop.is_type(event.type):
                if self.mode != 'stream' or self.input_done:
                    raise VoiceError("No active input stream to finish.", "invalid_request")
                self.add_text('', final=True)
            return True
        except (VoiceError, KeyError, TypeError, ValueError) as error:
            # Leave an existing valid turn alone if this connection sent a
            # second start; invalid input in its current turn cancels it.
            if not isinstance(error, VoiceError) or error.code != 'busy':
                await self.disconnect()
            await self.write_event(Error(text=str(error), code=getattr(error, 'code', 'invalid_request')).event())
            return True

    async def disconnect(self):
        if self.worker and not self.worker.done():
            self.worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.worker
        # Cancellation can happen before generate() reaches its try/finally.
        if self.backend.owner is self:
            self.backend.owner = None


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uri', default='tcp://127.0.0.1:10200')
    parser.add_argument('--endpoint', default='http://127.0.0.1:8812')
    parser.add_argument('--selection', type=Path, required=True)
    args = parser.parse_args()
    if args.endpoint != 'http://127.0.0.1:8812' or args.uri != 'tcp://127.0.0.1:10200':
        parser.error('Production bridge must remain on the approved loopback endpoints.')
    selection = json.loads(args.selection.read_text())
    if (selection.get('model'), selection.get('epoch'), selection.get('precision')) != ('Orpheus', 3, '4bit'):
        parser.error('Selection is not the approved epoch-3 4-bit model.')
    logging.basicConfig(level=logging.INFO)
    timeout = aiohttp.ClientTimeout(total=240, sock_connect=5, sock_read=90)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        backend = Backend(session, args.endpoint, selection['adapter_sha256'])
        LOG.info('Wyoming Orpheus adapter ready on loopback 10200; model activation is separate.')
        await AsyncServer.from_uri(args.uri).run(partial(Handler, backend))


if __name__ == '__main__':
    asyncio.run(main())
