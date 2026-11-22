"""CPU-only results page remains available after production is restored."""
from common import *
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
B=ROOT/'stream-benchmark'
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path=self.path.split('?',1)[0]
        if self.headers.get('Host') not in ('localhost:8814','127.0.0.1:8814'):self.send_error(400);return
        if path=='/':
            content=(B/'review.html').read_text().replace("$('play').onclick=playStream;","$('play').disabled=true; $('play').textContent='Live test available only while benchmark runtime is running';")
            data=content.encode();kind='text/html; charset=utf-8'
        elif path in ('/report','/status'):
            file=B/('report.json' if path=='/report' and (B/'report.json').exists() else 'status.json')
            data=file.read_bytes() if file.exists() else b'{"status":"preparing"}';kind='application/json'
        elif path.startswith('/audio/'):
            name=path[len('/audio/'):]
            if Path(name).name!=name or not name.endswith(('.wav','.mp3')):self.send_error(400);return
            file=B/'audio'/name
            if not file.is_file():self.send_error(404);return
            data=file.read_bytes();kind='audio/mpeg' if name.endswith('.mp3') else 'audio/wav'
        else:self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
ThreadingHTTPServer(('127.0.0.1',8814),Handler).serve_forever()
