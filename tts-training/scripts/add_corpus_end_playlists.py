"""Append chronological start/end playlists without altering corpus or review IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import wave
import zipfile

def main(root,package,template):
    sys.path.insert(0,str(root/'scripts'))
    from asr_runtime import atomic_json,atomic_text
    original=json.loads((package/'queue.json').read_text(encoding='utf8'))
    rows=json.loads((package/'all-clips.json').read_text(encoding='utf8'))
    verification=json.loads((root/'reports/corpus-verification.json').read_text())
    assert original['corpus_fingerprint']==verification['run_fingerprint'] and len(rows)==verification['clips']==588
    ordered=sorted(rows,key=lambda r:(r['source_start_frame'],r['source_end_frame'],r['id']))
    first,last=ordered[:20],ordered[-10:]
    review=original.get('playlists',{}).get('review',[r['id'] for r in original['clips']])
    assert len(review)==50 and len(set(review))==50
    backup=Path(__file__).resolve().parent/'backups'/('before-end-playlists-'+time.strftime('%Y%m%dT%H%M%S'))
    backup.mkdir(parents=True)
    for name in ['index.html','queue.json','README.txt']:shutil.copy2(package/name,backup/name)
    protected=[root/f'data/shecorpus/{part}/{name}' for part in ['train','validation'] for name in ['manifest.csv','she.list']]
    before={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    existing={row['id']:row for row in original['clips']};merged=list(original['clips'])
    ffmpeg=shutil.which('ffmpeg') or '/home/voiceuser/.hermes/tools/ffmpeg-9.0.1-linux-x64/bin/ffmpeg'
    for source_row in first+last:
        source=root/source_row['filename']
        assert source.resolve().is_relative_to((root/'data/shecorpus/versions'/original['corpus_fingerprint']).resolve())
        assert hashlib.sha256(source.read_bytes()).hexdigest()==source_row['file_sha256']
        id=source_row['id'];audio=package/'audio'/(id+'.wav');context=package/'context'/(id+'.wav')
        shutil.copy2(source,audio)
        if not context.exists():
            start=max(0,source_row['source_start_frame']/48000-1)
            end=min(9772,source_row['source_end_frame']/48000+1)
            subprocess.run([ffmpeg,'-nostdin','-v','error','-y','-ss',str(start),'-i',
                '/home/voiceuser/projects/she-training/SheCorpus.flac','-t',str(end-start),
                '-af','pan=mono|c0=0.5*c0+0.5*c1','-ar','32000','-c:a','pcm_s16le',str(context)],check=True)
        with wave.open(str(audio)) as stream:assert (stream.getframerate(),stream.getnchannels(),stream.getsampwidth())==(32000,1,2)
        if id not in existing:
            row=dict(source_row,selection_reason='first 20 corpus clips' if source_row in first else 'last 10 corpus clips',
                     audio_url='audio/'+id+'.wav',context_url='context/'+id+'.wav',
                     context_clip_start_seconds=min(1,source_row['source_start_frame']/48000))
            merged.append(row);existing[id]=row
    original.update(clips=merged,playlists=dict(first20=[r['id'] for r in first],last10=[r['id'] for r in last],review=review),default_playlist='first20',
        playlist_order='chronological source_start_frame across both frozen corpus splits')
    atomic_json(package/'queue.json',original)
    payload=json.dumps(original,ensure_ascii=True).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    atomic_text(package/'index.html',template.read_text(encoding='utf8').replace('__REVIEW_PAYLOAD__',payload))
    atomic_text(package/'README.txt','Open index.html. Playlist: First 20 corpus clips / Last 10 corpus clips / Original review (50 clips).\n'
        'First and last use chronological source order across the full approved corpus.\n'
        'Check Play next clip automatically to listen sequentially. Playback does not mark clips reviewed.\n'
        'Previous/Next stay within the selected playlist. Source-context audio and transcripts are included.\n'
        'The same corpus IDs and browser storage key preserve existing decisions. Export JSON as a portable backup.\n'
        'The approved training audio, manifests and assignments are unchanged.\n')
    assert before=={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    report=dict(corpus_fingerprint=original['corpus_fingerprint'],order=original['playlist_order'],
        playlists={name:dict(clips=len(ids),ids=ids,seconds=sum(existing[id]['duration_seconds'] for id in ids)) for name,ids in original['playlists'].items()},
        available_unique_clips=len(merged),corpus_changed=False,review_storage_key_unchanged=True)
    atomic_json(package/'playlists.json',report)
    atomic_json(root/'reports/corpus-listening-playlists.json',report)
    shutil.copy2(root/'scripts/corpus_review.html',backup/'corpus_review.html')
    atomic_text(root/'scripts/corpus_review.html',template.read_text(encoding='utf8'))
    with zipfile.ZipFile(package.parent/'TTS-listening-review.zip','w',zipfile.ZIP_DEFLATED) as archive:
        names=['index.html','queue.json','summary.json','all-clips.json','audio-audit.json','README.txt','REVIEW_REPORT.md','playlists.json']
        names += [row[key] for row in merged for key in ['audio_url','context_url']]
        for name in dict.fromkeys(names):archive.write(package/name,name)
    print(json.dumps(dict(playlists={k:v['clips'] for k,v in report['playlists'].items()},
        available_unique_clips=len(merged),first_id=first[0]['id'],last_id=last[-1]['id'],corpus_changed=False),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--package',type=Path,required=True);parser.add_argument('--template',type=Path,required=True)
    args=parser.parse_args();main(args.root,args.package,args.template)
