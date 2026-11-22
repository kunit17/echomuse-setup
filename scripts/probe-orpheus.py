#!/usr/bin/env python3
"""Discover the trained voice and optionally collect streamed PCM as WAV.

Speech generation never executes Home Assistant actions. Model must already be on.
"""
import argparse,asyncio,json,time,wave
from pathlib import Path
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe,Info
from wyoming.audio import AudioStart,AudioChunk
from wyoming.tts import Synthesize,SynthesizeStart,SynthesizeChunk,SynthesizeStop

async def main():
    p=argparse.ArgumentParser();p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=10200);p.add_argument('--text')
    p.add_argument('--stream',action='store_true');p.add_argument('--out',type=Path)
    args=p.parse_args()
    if args.out and args.out.exists():raise FileExistsError('Choose a new output file')
    async with AsyncTcpClient(args.host,args.port) as client:
        await client.write_event(Describe().event())
        event=await asyncio.wait_for(client.read_event(),15)
        if event is None or not Info.is_type(event.type):raise RuntimeError('No Wyoming info')
        info=Info.from_event(event)
        print(json.dumps({'ready':True,'tts':[x.name for x in info.tts],
                          'streaming':info.tts[0].supports_synthesize_streaming,
                          'voices':[x.name for x in info.tts[0].voices]}),flush=True)
    if not args.text:return
    chunks=[];first=None;format=None;audio_stops=0;started=time.monotonic()
    async with AsyncTcpClient(args.host,args.port) as client:
        if args.stream:
            await client.write_event(SynthesizeStart().event())
            for offset in range(0,len(args.text),11):
                await client.write_event(SynthesizeChunk(args.text[offset:offset+11]).event())
            await client.write_event(Synthesize(args.text).event())
            await client.write_event(SynthesizeStop().event())
        else:await client.write_event(Synthesize(args.text).event())
        while True:
            event=await asyncio.wait_for(client.read_event(),120)
            if event is None:raise RuntimeError('Disconnected before completion')
            if event.type=='error':raise RuntimeError(event.data)
            if event.type=='audio-start':
                if format is not None:raise RuntimeError('Duplicate audio-start')
                format=AudioStart.from_event(event)
            elif event.type=='audio-chunk':
                chunk=AudioChunk.from_event(event)
                if first is None:first=time.monotonic()-started
                chunks.append(chunk.audio)
            elif event.type=='audio-stop':
                audio_stops+=1
                if not args.stream:break
            elif event.type=='synthesize-stopped':break
    if audio_stops!=1 or format is None or not chunks:raise RuntimeError('Incomplete audio events')
    pcm=b''.join(chunks)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with wave.open(str(args.out),'wb') as wav:
            wav.setframerate(format.rate);wav.setsampwidth(format.width)
            wav.setnchannels(format.channels);wav.writeframes(pcm)
    print(json.dumps({'complete':True,'first_audio_seconds':first,
                      'total_seconds':time.monotonic()-started,'bytes':len(pcm),
                      'audio_seconds':len(pcm)/(format.rate*format.width*format.channels),
                      'chunks':len(chunks),'streaming_input':args.stream}),flush=True)
if __name__=='__main__':asyncio.run(main())
