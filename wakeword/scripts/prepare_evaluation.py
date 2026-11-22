#!/usr/bin/env python3
"""Pad independent Windows evaluation clips for openWakeWord inference."""
import argparse,json,wave
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
p.add_argument('--destination',type=Path,required=True);args=p.parse_args()
rows=json.loads((args.source/'manifest.json').read_text(encoding='utf-8-sig'))
args.destination.mkdir(parents=True,exist_ok=True)
for row in rows:
    name=row['file']
    if Path(name).name!=name or not name.endswith('.wav'):raise ValueError('Invalid fixture filename')
    target=args.destination/name
    if target.exists():raise FileExistsError('Choose a fresh evaluation destination')
    with wave.open(str(args.source/name),'rb') as source:
        if (source.getframerate(),source.getnchannels(),source.getsampwidth())!=(16000,1,2):
            raise ValueError('Expected 16 kHz mono PCM16')
        audio=source.readframes(source.getnframes())
    with wave.open(str(target),'wb') as output:
        output.setframerate(16000);output.setnchannels(1);output.setsampwidth(2)
        output.writeframes(bytes(32000)+audio+bytes(32000))
(args.destination/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
print('Prepared',len(rows),'evaluation clips with one second of silence on each side.')
