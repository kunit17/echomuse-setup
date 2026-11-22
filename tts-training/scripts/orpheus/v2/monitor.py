from common import *
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import unquote,urlsplit
import html
PAGE='''<!doctype html><html><meta charset="utf-8"><title>Orpheus training</title><style>body{font:17px/1.6 system-ui;background:#f3f4ef;color:#18382f;max-width:1000px;margin:35px auto;padding:20px}section{padding:22px;background:white;border-radius:14px;margin:18px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}audio{width:100%}</style><h1>Orpheus 3B · v2 corpus</h1><section><h2 id="phase">Loading…</h2><p id="summary"></p><p id="loss"></p><pre id="error"></pre></section><section id="samples"></section><script>async function refresh(){const d=await(await fetch('/status',{cache:'no-store'})).json(),s=d.status;document.getElementById('phase').textContent=(s.status||'preparing')+' · '+(s.stage||'');document.getElementById('summary').textContent='Precision: '+(s.precision||'16-bit first')+' · epoch '+(s.epoch||0)+' / 3 · step '+(s.step||0)+' / '+(s.total_steps||'?');document.getElementById('loss').textContent=s.last_validation?'Validation loss: '+s.last_validation.eval_loss:'Encoding '+(s.prepared||0)+' / 2048 clips';document.getElementById('error').textContent=s.error||'';let root=document.getElementById('samples');root.replaceChildren();for(const item of d.samples){let h=document.createElement('h3'),p=document.createElement('p'),a=document.createElement('audio');h.textContent=item.label;p.textContent=item.text;a.controls=true;a.preload='none';a.src=item.url;root.append(h,p,a);}}refresh();setInterval(refresh,5000);</script></html>'''
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.headers.get('Host') not in ['localhost:8770','127.0.0.1:8770']:self.send_error(403);return
        route=unquote(urlsplit(self.path).path)
        if route=='/':body=PAGE.encode();kind='text/html'
        elif route=='/status':
            current=read_status()
            samples=[]
            for result in sorted((RUN/'samples').glob('*/results.json')):
                if result.parent.name.startswith('smoke'):continue
                for item in json.loads(result.read_text())['samples']:
                    label=item.get('model','Orpheus')+f' · epoch {item["epoch"]} · {item["name"]}'
                    samples.append(dict(label=label,text=item['text'],url='/samples/'+str(result.parent.relative_to(RUN/'samples'))+'/'+item['name']+'.mp3'))
            body=json.dumps(dict(status=current,samples=samples)).encode();kind='application/json'
        elif route=='/qa':
            cards=[]
            for item in json.loads((DATA/'qa.json').read_text()):
                ident=item['id'];cards.append('<h2>'+html.escape(ident)+'</h2><p>'+html.escape(item['text'])+'</p><p>Unchanged corpus clip</p><audio controls preload="none" src="/qa/original/'+ident+'.wav"></audio><p>SNAC encode/decode round trip</p><audio controls preload="none" src="/qa/roundtrip/'+ident+'.wav"></audio>')
            body=('<html><meta charset="utf-8"><title>SNAC round-trip checks</title><h1>SNAC round-trip comparisons</h1>'+''.join(cards)).encode();kind='text/html'
        elif route.startswith('/qa/'):
            parts=route.split('/')
            if len(parts)!=4 or parts[2] not in ['original','roundtrip']:self.send_error(404);return
            ident=parts[3].removesuffix('.wav')
            items={r['id']:r for r in json.loads((DATA/'qa.json').read_text())}
            if ident not in items:self.send_error(404);return
            body=Path(items[ident][parts[2]]).read_bytes();kind='audio/wav'
        elif route.startswith('/samples/'):
            base=(RUN/'samples').resolve();path=(base/route.removeprefix('/samples/')).resolve()
            if not path.is_relative_to(base) or path.suffix not in ['.mp3','.wav'] or not path.is_file():self.send_error(404);return
            body=path.read_bytes();kind='audio/mpeg' if path.suffix=='.mp3' else 'audio/wav'
        else:self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
    def log_message(self,*args):pass
if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',8770),Handler).serve_forever()
