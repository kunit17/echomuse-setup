"""Pure streaming framing and playback accounting, independent of CUDA."""
import math,re
RATE=24000
GROUP_SAMPLES=2048
def split_sentences(text):
    """Split text only; each generated utterance must retain all audio through EOS."""
    text=text.strip();parts=[];start=0
    for match in re.finditer(r'[.!?]+["\u201d\u2019\']*(?=\s|$)',text):
        prefix=text[start:match.end()]
        if match.group().startswith('.') and re.search(r'\b(?:Mr|Mrs|Ms|Dr|Prof|St|[A-Z])\.$',prefix):continue
        part=prefix.strip()
        if part:parts.append(part)
        start=match.end()
    if text[start:].strip():parts.append(text[start:].strip())
    result=[]
    for part in parts:
        while len(part)>400:
            boundaries=list(re.finditer(r'[,;:]\s+',part[:401]))
            cut=boundaries[-1].end() if boundaries else part.rfind(' ',0,401)
            if cut<=0:raise ValueError('An unbroken word exceeds the text context')
            result.append(part[:cut].strip());part=part[cut:].strip()
        if part:result.append(part)
    return result
def safe_end(decoded_samples,groups,final=False):
    if final:return decoded_samples
    return max(0,min(decoded_samples-8*GROUP_SAMPLES,decoded_samples-7200))
def validate_codes(tokens):
    if len(tokens)%7:raise ValueError('Incomplete SNAC group')
    for i,token in enumerate(tokens):
        if not 0<=token-128266-(i%7)*4096<4096:raise ValueError('Invalid SNAC token')
def simulate(events,buffer_seconds):
    """events=(arrival seconds, newly arrived audio seconds). No assumed delivery."""
    if not events:return dict(first_playback=None,underruns=0,underrun_seconds=0)
    queued=0;started=None;end=None;count=0;stall=0
    for arrival,duration in events:
        if started is None:
            queued+=duration
            if queued>=buffer_seconds:
                started=arrival;end=arrival+queued
        else:
            if arrival>end+1e-6:count+=1;stall+=arrival-end;end=arrival
            end+=duration
    if started is None:started=events[-1][0]
    return dict(first_playback=started,underruns=count,underrun_seconds=stall)
def percentile(values,p=.95):
    return sorted(values)[max(0,math.ceil(len(values)*p)-1)] if values else None
