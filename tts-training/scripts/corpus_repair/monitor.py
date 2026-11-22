"""Read-only local repair status and source-context pilot players."""
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
ROOT=Path('/home/voiceuser/projects/gpt-sovits')
PILOT=(ROOT/'data/shecorpus/repaired/boundary-v1-20261115/pilot').resolve()
DATASET=PILOT.parent
PAGE='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Corpus boundary repair</title><style>body{font:17px/1.6 system-ui;max-width:1000px;margin:auto;padding:30px;background:#f2f4ef;color:#19352d}section{background:white;border-radius:12px;padding:24px;margin:20px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px}a{color:#216b48}</style><h1>Corpus boundary repair</h1><p>Dataset preparation only. Model training stays paused.</p><section><h2 id="phase">Loading…</h2><p id="progress"></p><a href="/dataset/review/index.html">Listen to accepted training and validation clips</a><br><a href="/pilot/index.html">Boundary pilot and reported clip 111 comparison</a></section><section><h2>Measured status</h2><pre id="detail"></pre></section><script>async function refresh(){try{const d=await(await fetch('/api/status',{cache:'no-store'})).json();document.getElementById('phase').textContent=(d.status||'starting').replaceAll('_',' ');let p='';if(d.status==='acoustic_scan')p=d.source+': '+Math.round(d.through_seconds/60)+' of '+Math.round(d.source_seconds/60)+' minutes scanned';else if(d.status.includes('transcrib')||d.status.includes('align'))p=d.done+' / '+d.total+' clips';else if(d.hours)p=d.clips+' clips · '+d.hours.toFixed(2)+' hours';document.getElementById('progress').textContent=p;document.getElementById('detail').textContent=JSON.stringify(d,null,2);}catch(e){document.getElementById('phase').textContent='Waiting for status';}}refresh();setInterval(refresh,5000);</script></html>'''
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.headers.get('Host') not in ['localhost:8768','127.0.0.1:8768']:
            self.send_error(403);return
        if self.path=='/':body=PAGE.encode();mime='text/html; charset=utf-8'
        elif self.path=='/api/status':
            try:body=(ROOT/'reports/corpus-repair-status.json').read_bytes()
            except FileNotFoundError:body=b'{"status":"starting"}'
            mime='application/json'
        elif self.path.startswith('/pilot/') or self.path.startswith('/dataset/'):
            base=PILOT if self.path.startswith('/pilot/') else DATASET
            prefix='/pilot/' if base==PILOT else '/dataset/'
            p=(base/self.path[len(prefix):]).resolve()
            if not p.is_relative_to(base) or p.suffix not in ['.wav','.html'] or not p.is_file():
                self.send_error(404);return
            body=p.read_bytes();mime='audio/wav' if p.suffix=='.wav' else 'text/html; charset=utf-8'
        else:self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
    def log_message(self,*args):pass
if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',8768),Handler).serve_forever()
