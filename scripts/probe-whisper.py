#!/usr/bin/env python3
"""Wyoming readiness/optional transcription probe; never executes HA actions."""
import argparse,asyncio,json,time,wave
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe,Info
from wyoming.asr import Transcribe,Transcript
from wyoming.audio import AudioStart,AudioChunk,AudioStop
from wyoming.error import Error

async def main():
    p=argparse.ArgumentParser();p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',default=10300,type=int);p.add_argument('--wav')
    args=p.parse_args()
    async with AsyncTcpClient(args.host,args.port) as client:
        await client.write_event(Describe().event())
        event=await asyncio.wait_for(client.read_event(),15)
        if event is None or not Info.is_type(event.type):raise RuntimeError('No Wyoming info')
        info=Info.from_event(event)
        print(json.dumps({'ready':True,'asr':[x.name for x in info.asr]}),flush=True)
    if not args.wav:return
    with wave.open(args.wav,'rb') as wav:
        rate=wav.getframerate();width=wav.getsampwidth();channels=wav.getnchannels()
        audio=wav.readframes(wav.getnframes())
    async with AsyncTcpClient(args.host,args.port) as client:
        await client.write_event(Transcribe(language='en').event())
        await client.write_event(AudioStart(rate=rate,width=width,channels=channels).event())
        for i in range(0,len(audio),rate*width*channels//10):
            await client.write_event(AudioChunk(rate=rate,width=width,channels=channels,
                                               audio=audio[i:i+rate*width*channels//10]).event())
        start=time.monotonic();await client.write_event(AudioStop().event())
        while True:
            event=await asyncio.wait_for(client.read_event(),30)
            if event is None:raise RuntimeError('Connection closed without transcript')
            if Error.is_type(event.type):raise RuntimeError('Wyoming transcription error')
            if Transcript.is_type(event.type):
                print(json.dumps({'transcript':Transcript.from_event(event).text,
                                  'seconds_after_audio_stop':time.monotonic()-start}),flush=True)
                return
if __name__=='__main__':asyncio.run(main())
