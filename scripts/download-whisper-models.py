#!/usr/bin/env python3
"""Download only the recorded public model revisions into the voice cache."""
import argparse,json
from pathlib import Path
from huggingface_hub import snapshot_download
p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=Path.home()/'.cache/echomuse-whisper/models')
args=p.parse_args()
manifest=json.loads((Path(__file__).resolve().parents[1]/'voice/whisper-deployment.json').read_text())
for spec in manifest['models'].values():
    snapshot_download(repo_id=spec['repository'],revision=spec['revision'],cache_dir=str(args.cache))
    # The runtime uses a repository ID with local-files-only. Pin its cached
    # main reference to this recorded snapshot; do not fetch a moving main.
    ref=args.cache/('models--'+spec['repository'].replace('/','--'))/'refs/main'
    ref.parent.mkdir(parents=True,exist_ok=True);ref.write_text(spec['revision'])
    print(spec['repository']+' '+spec['revision'])
