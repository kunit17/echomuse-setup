#!/usr/bin/env python3
"""Compare CPU INT8 Whisper on existing, identically encoded WAV fixtures.

Run with the isolated Wyoming environment. Synthetic fixtures test integration,
not far-field recognition accuracy. Never execute the recognized commands.
"""
import argparse, json, re, statistics, time, wave
from pathlib import Path
from faster_whisper import WhisperModel

def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def command_words(text):
    # Only normalize the number forms present in this explicit fixture set.
    text=text.lower().replace('%',' percent ')
    for phrase,value in [('twenty one','21'),('fifty','50'),('twenty','20')]:
        text=re.sub(r'\b'+phrase+r'\b',value,text)
    return words(text)

def distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        new=[i]
        for j,y in enumerate(b,1):
            new.append(min(new[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=new
    return row[-1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    args=p.parse_args()
    fixtures=json.loads((args.fixtures/'manifest.json').read_text(encoding='utf-8-sig'))
    result={'fixture_kind':'offline synthetic Windows Zira voice; not Dot recordings',
            'device':'cpu','compute_type':'int8','cpu_threads':8,'beam_size':5,'models':[]}
    for name in ['base.en','small.en']:
        started=time.monotonic()
        model=WhisperModel('Systran/faster-whisper-'+name,device='cpu',compute_type='int8',
                           cpu_threads=8,download_root=str(args.cache))
        load=time.monotonic()-started
        # Warm the same model before timing utterances.
        segments,_=model.transcribe(str(args.fixtures/fixtures[0]['file']),language='en',beam_size=5)
        list(segments)
        rows=[]
        for fixture in fixtures:
            path=args.fixtures/fixture['file']
            with wave.open(str(path),'rb') as wav: duration=wav.getnframes()/wav.getframerate()
            started=time.monotonic()
            segments,_=model.transcribe(str(path),language='en',beam_size=5)
            transcript=' '.join(s.text.strip() for s in segments)
            elapsed=time.monotonic()-started
            expected=words(fixture['text']);actual=words(transcript)
            rows.append(dict(file=fixture['file'],expected=fixture['text'],transcript=transcript,
                             seconds=elapsed,audio_seconds=duration,word_errors=distance(expected,actual),
                             expected_words=len(expected),exact=expected==actual,
                             normalized_exact=command_words(fixture['text'])==command_words(transcript)))
        row={'model':name,'load_and_download_seconds':load,'rows':rows,
             'median_seconds':statistics.median(x['seconds'] for x in rows),
             'max_seconds':max(x['seconds'] for x in rows),
             'exact_commands':sum(x['exact'] for x in rows),
             'normalized_exact_commands':sum(x['normalized_exact'] for x in rows),
             'word_error_rate':sum(x['word_errors'] for x in rows)/sum(x['expected_words'] for x in rows)}
        result['models'].append(row)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='rows'}),flush=True)
        del model
if __name__=='__main__':main()
