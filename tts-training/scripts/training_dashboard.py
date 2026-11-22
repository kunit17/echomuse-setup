"""Loopback-only, read-only live view of approved TTS training. No model imports."""
import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import subprocess
import threading
import time
from urllib.parse import urlsplit

def read_json(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default

def epoch_rows(path,run_id):
    rows={};warnings=[]
    try:lines=path.read_text().splitlines()
    except OSError:return [],[]
    for number,line in enumerate(lines,1):
        if not line.strip():continue
        try:
            row=json.loads(line)
            if row.get('run_id')!=run_id:continue
            if row.get('stage') not in ['s1','s2'] or not isinstance(row.get('epoch'),int):continue
            if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in row.get('validation',{}).values()):continue
            rows[(row['stage'],row['epoch'])]=row
        except (ValueError,TypeError):
            if number<len(lines):warnings.append('Malformed historical metrics row '+str(number))
    return sorted(rows.values(),key=lambda r:(r['stage'],r['epoch'])),warnings

def tail(path):
    try:
        with path.open('rb') as stream:
            stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-32768))
            return stream.read().decode('utf8','replace').splitlines()[-14:]
    except OSError:return []

def patience(rows,policy):
    best=None;epoch=0
    for row in rows:
        if row['epoch']==0:continue
        value=row['validation'][policy['metric']]
        if best is None or value<best*(1-policy['min_relative_improvement']):best=value;epoch=row['epoch']
    return max(0,rows[-1]['epoch']-epoch) if rows else 0

class Feed:
    def __init__(self,root):
        self.root=root.resolve();self.lock=threading.Lock();self.gpu_time=0;self.gpu=None

    def gpu_sample(self):
        with self.lock:
            if time.monotonic()-self.gpu_time<4:return self.gpu
            try:
                result=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi',
                    '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu',
                    '--format=csv,noheader,nounits'],text=True,timeout=3).splitlines()[0]
                name,used,total,util,temp=[p.strip() for p in result.split(',')]
                self.gpu=dict(name=name,used_mib=float(used),total_mib=float(total),utilization=float(util),temperature=float(temp))
            except (OSError,ValueError,subprocess.SubprocessError):self.gpu=None
            self.gpu_time=time.monotonic();return self.gpu

    def snapshot(self):
        status=read_json(self.root/'reports/training-status.json',{})
        run_id=status.get('run_id','')
        if not re.fullmatch('[A-Za-z0-9_-]{1,100}',run_id):return dict(available=False,fetched=time.time())
        directory=self.root/'reports/training-runs'/run_id
        run=read_json(directory/'run.json',{})
        rows,warnings=epoch_rows(directory/'epochs.jsonl',run_id)
        active=status.get('active_stage');completed=status.get('completed_stages',[]);stages={}
        for stage,label in [('s1','gpt'),('s2','sovits')]:
            selected=read_json(directory/(stage+'-selection.json'),{})
            policy=run.get('policies',status.get('policies',{})).get(stage,{})
            history=[r for r in rows if r['stage']==stage];trained=[r for r in history if r['epoch']>0]
            last=trained[-1] if trained else None
            stage_state='complete' if label in completed else ('running' if label==active else 'queued')
            if status.get('status')=='failed' and active==label:stage_state='failed'
            remaining=None;epoch_wall=None
            if last and active==label and stage_state=='running':
                checkpoint=Path(last.get('full_checkpoint',last['checkpoint']))
                if checkpoint.is_relative_to(self.root/'experiments') or checkpoint.is_relative_to(self.root/'checkpoints'):
                    try:
                        saved=checkpoint.stat().st_mtime
                        epoch_wall=(saved-status['stage_started'])/last['epoch']
                        if epoch_wall>0:
                            remaining=max(0,(policy['max_epochs']-last['epoch'])*epoch_wall-(time.time()-saved))
                        else:epoch_wall=None
                    except (OSError,KeyError,ZeroDivisionError):pass
            stages[stage]=dict(state=stage_state,policy=policy,rows=history,last=last,
                selected=selected.get('selected'),patience_used=patience(history,policy) if history and policy else 0,
                ceiling_remaining_seconds=remaining,observed_wall_seconds_per_epoch=epoch_wall)
        controller_alive=False
        try:
            command=Path('/proc/'+str(int(status['controller_pid']))+'/cmdline').read_bytes()
            controller_alive=b'approved_pipeline.py' in command
        except (OSError,ValueError,KeyError):pass
        log=[]
        if active and re.fullmatch('[A-Za-z0-9_-]+',active):log=tail(self.root/'logs'/run_id/(active+'.log'))
        for stage in stages.values():
            # Paths are visible in checkpoint details, never used as HTTP file URLs.
            for row in stage['rows']:
                row.pop('full_checkpoints',None)
        return dict(available=True,fetched=time.time(),status=status,run=run,stages=stages,gpu=self.gpu_sample(),
                    controller_alive=controller_alive,log_tail=log,warnings=warnings,
                    log_updated=(self.root/'logs'/run_id/(active+'.log')).stat().st_mtime
                    if active and re.fullmatch('[A-Za-z0-9_-]+',active) and (self.root/'logs'/run_id/(active+'.log')).is_file() else None)

def make_handler(feed,assets,port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get('Host') not in [f'localhost:{port}',f'127.0.0.1:{port}']:
                self.send_error(403);return
            path=urlsplit(self.path).path
            files={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}
            try:
                if path=='/api/status':body=json.dumps(feed.snapshot(),allow_nan=False).encode();kind='application/json'
                elif path in files:
                    body=(assets/files[path]).read_bytes()
                    kind={'/':'text/html','/app.js':'text/javascript','/style.css':'text/css'}[path]
                elif path=='/favicon.ico':self.send_response(204);self.end_headers();return
                else:self.send_error(404);return
            except Exception:
                self.send_error(503,'Training snapshot temporarily unavailable');return
            self.send_response(200);self.send_header('Content-Type',kind+'; charset=utf-8')
            self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers();self.wfile.write(body)
        def do_POST(self):self.send_error(405)
        def log_message(self,*args):pass
    return Handler

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--assets',type=Path,required=True);parser.add_argument('--port',type=int,default=8766)
    args=parser.parse_args();server=ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(Feed(args.root),args.assets,args.port))
    print(f'Training dashboard: http://localhost:{args.port}',flush=True)
    server.serve_forever()
