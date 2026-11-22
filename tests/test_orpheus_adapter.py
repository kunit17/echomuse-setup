"""Real TCP Wyoming + fake HTTP engine tests; no model/GPU/network required."""
import asyncio
import contextlib
import sys
import unittest
import uuid
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, web
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info
from wyoming.tts import Synthesize, SynthesizeStart, SynthesizeChunk, SynthesizeStop, SynthesizeStopped, SynthesizeVoice

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'voice'))
from orpheus_adapter import Backend, Handler


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        self.traces = {}
        self.ready = True
        self.bad_trace = False
        self.odd = False
        self.slow = False
        self.upstream_disconnected = asyncio.Event()
        app = web.Application()
        app.router.add_get('/health', self.health)
        app.router.add_post('/stream', self.stream)
        app.router.add_get('/trace/{request}', self.trace)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.session = ClientSession(timeout=ClientTimeout(total=10), trust_env=False)
        self.backend = Backend(self.session, f'http://127.0.0.1:{port}', 'selected-hash')
        self.tasks = set()
        self.handlers = []
        self.server = await asyncio.start_server(self.accept, '127.0.0.1', 0)
        self.port = self.server.sockets[0].getsockname()[1]
        self.clients = []

    def accept(self, reader, writer):
        handler = Handler(self.backend, reader, writer)
        self.handlers.append(handler)
        task = asyncio.create_task(handler.run())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def asyncTearDown(self):
        for client in self.clients:
            await client.disconnect()
        for handler in self.handlers:
            await handler.disconnect()
        self.server.close()
        await self.server.wait_closed()
        for task in self.tasks:
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.session.close()
        await self.runner.cleanup()

    async def health(self, request):
        return web.json_response(dict(ready=self.ready, model='Orpheus', epoch=3,
            runtime='vllm-streaming', precision='4bit', codec='cuda', adapter_sha256='selected-hash'))

    async def stream(self, request):
        self.requests.append((await request.json())['text'])
        identifier = str(uuid.uuid4())
        response = web.StreamResponse(headers={'X-Request-ID': identifier, 'X-Sample-Rate': '24000',
            'X-Channels': '1', 'X-Sample-Format': 's16le'})
        await response.prepare(request)
        pcm = b'\x01\x02' * 36000 + (b'\x01' if self.odd else b'')
        try:
            for i in range(0, len(pcm), 1025):
                await response.write(pcm[i:i+1025])
                if self.slow:
                    await asyncio.sleep(.01)
            self.traces[identifier] = dict(status='failed' if self.bad_trace else 'complete',
                sample_count=len(pcm)//2)
            await response.write_eof()
        except (ConnectionError, asyncio.CancelledError):
            self.upstream_disconnected.set()
        return response

    async def trace(self, request):
        trace = self.traces.get(request.match_info['request'])
        return web.json_response(trace) if trace else web.Response(status=404)

    async def client(self):
        client = AsyncTcpClient('127.0.0.1', self.port)
        await client.connect()
        self.clients.append(client)
        return client

    async def read(self, client):
        event = await asyncio.wait_for(client.read_event(), 5)
        self.assertIsNotNone(event)
        return event

    async def collect(self, client, terminal='audio-stop'):
        result = []
        while True:
            event = await self.read(client)
            result.append(event)
            if event.type in (terminal, 'error'):
                return result

    async def test_discovery_does_not_load_model(self):
        self.ready = False
        client = await self.client()
        await client.write_event(Describe().event())
        result = Info.from_event(await self.read(client))
        self.assertTrue(result.tts[0].supports_synthesize_streaming)
        self.assertEqual(result.tts[0].voices[0].languages, ['en'])
        self.assertEqual(self.requests, [])

    async def test_plain_audio_covers_all_samples_and_allows_next_turn(self):
        client = await self.client()
        for _ in range(2):
            await client.write_event(Synthesize('Bedroom lights on.').event())
            events = await self.collect(client)
            self.assertEqual(events[0].type, 'audio-start')
            self.assertEqual(events[-1].type, 'audio-stop')
            self.assertEqual(sum(len(x.payload or b'') for x in events), 72000)
        self.assertEqual(len(self.requests), 2)

    async def test_stream_ignores_compatibility_copy_and_flushes_remainder(self):
        client = await self.client()
        await client.write_event(SynthesizeStart().event())
        await client.write_event(SynthesizeChunk('Bedroom lights on. ').event())
        await client.write_event(SynthesizeChunk('The front light is off').event())
        await client.write_event(Synthesize('Bedroom lights on. The front light is off').event())
        await client.write_event(SynthesizeStop().event())
        events = await self.collect(client, 'synthesize-stopped')
        self.assertEqual(self.requests, ['Bedroom lights on.', 'The front light is off'])
        self.assertEqual([x.type for x in events].count('audio-start'), 1)
        self.assertEqual([x.type for x in events].count('audio-stop'), 1)
        self.assertEqual([x.type for x in events].count('synthesize-stopped'), 1)
        self.assertEqual(sum(len(x.payload or b'') for x in events), 144000)

    async def test_audio_begins_before_backend_finishes(self):
        self.slow = True
        client = await self.client()
        await client.write_event(Synthesize('Bedroom lights on.').event())
        self.assertEqual((await self.read(client)).type, 'audio-start')
        self.assertEqual(self.traces, {})
        self.assertEqual((await self.collect(client))[-1].type, 'audio-stop')

    async def test_incomplete_trace_never_reports_audio_stop(self):
        self.bad_trace = True
        client = await self.client()
        await client.write_event(Synthesize('Bedroom lights on.').event())
        events = await self.collect(client)
        self.assertEqual(events[-1].type, 'error')
        self.assertNotIn('audio-stop', [x.type for x in events])

    async def test_partial_pcm_never_reports_success(self):
        self.odd = True
        client = await self.client()
        await client.write_event(Synthesize('Bedroom lights on.').event())
        events = await self.collect(client)
        self.assertEqual(events[-1].type, 'error')

    async def test_off_model_is_error_without_loading(self):
        self.ready = False
        client = await self.client()
        await client.write_event(Synthesize('Bedroom lights on.').event())
        event = await self.read(client)
        self.assertEqual(event.type, 'error')
        self.assertEqual(event.data['code'], 'voice_not_ready')
        self.assertEqual(self.requests, [])

    async def test_busy_and_disconnect_cancel_backend_and_release_owner(self):
        self.slow = True
        first = await self.client()
        await first.write_event(Synthesize('Bedroom lights on.').event())
        self.assertEqual((await self.read(first)).type, 'audio-start')
        other = await self.client()
        await other.write_event(Synthesize('Front light off.').event())
        self.assertEqual((await self.read(other)).data['code'], 'busy')
        await first.disconnect()
        await asyncio.wait_for(self.upstream_disconnected.wait(), 3)
        self.assertIsNone(self.backend.owner)
        self.slow = False
        await other.write_event(Synthesize('Front light off.').event())
        self.assertEqual((await self.collect(other))[-1].type, 'audio-stop')

    async def test_invalid_text_releases_owner_before_worker_runs(self):
        client = await self.client()
        await client.write_event(Synthesize('x' * 5001).event())
        self.assertEqual((await self.read(client)).type, 'error')
        self.assertIsNone(self.backend.owner)
        await client.write_event(Synthesize('Bedroom lights on.').event())
        self.assertEqual((await self.collect(client))[-1].type, 'audio-stop')

    async def test_unsupported_voice_is_rejected(self):
        client = await self.client()
        await client.write_event(Synthesize('Hi.', voice=SynthesizeVoice(name='other')).event())
        self.assertEqual((await self.read(client)).data['code'], 'invalid_request')
        self.assertEqual(self.requests, [])


if __name__ == '__main__':
    unittest.main()
