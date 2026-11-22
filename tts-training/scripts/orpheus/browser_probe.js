const fs = require('fs');
const label=process.argv[2];
async function main(){
  let targets;
  for(let i=0;i<60;i++){
    try{targets=await(await fetch('http://127.0.0.1:9337/json/list')).json();if(targets.some(t=>t.type==='page'))break;}catch{}
    await new Promise(r=>setTimeout(r,500));
  }
  const target=targets.find(t=>t.type==='page');const ws=new WebSocket(target.webSocketDebuggerUrl);let id=0;const pending=new Map();
  await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
  ws.onmessage=event=>{const m=JSON.parse(event.data);if(m.id&&pending.has(m.id)){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}};
  const call=(method,params={})=>new Promise((resolve,reject)=>{pending.set(++id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
  const url=new URL('/?probe=1',process.argv[4]||'http://localhost:8813');
  if(process.argv[3]&&process.argv[3]!=='-')url.searchParams.set('text',fs.readFileSync(process.argv[3],'utf8').trim());
  await call('Page.navigate',{url:String(url)});
  let result;
  for(let i=0;i<240;i++){
    const r=await call('Runtime.evaluate',{expression:'window.benchmarkResult || null',returnByValue:true});
    result=r.result?.value;if(result)break;await new Promise(r=>setTimeout(r,500));
  }
  if(!result)throw Error('Browser stream playback timed out');
  const checks=await call('Runtime.evaluate',{awaitPromise:true,returnByValue:true,expression:`(async()=>{
    const invalid=await fetch('/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    const controller=new AbortController();
    const response=await fetch('/stream',{method:'POST',headers:{'Content-Type':'application/json'},signal:controller.signal,body:JSON.stringify({text:document.getElementById('text').value+' Tell me about your plans for the weekend.'})});
    const id=response.headers.get('X-Request-ID');await response.body.getReader().read();controller.abort();
    for(let i=0;i<100;i++){const r=await fetch('/trace/'+id);if(r.ok){const t=await r.json();return {invalidTextRejected:invalid.status===400,disconnectCancelled:t.status==='cancelled'};}await new Promise(r=>setTimeout(r,100));}
    return {invalidTextRejected:invalid.status===400,disconnectCancelled:false};
  })()`});
  result.apiChecks=checks.result?.value;
  fs.writeFileSync(require('path').join(__dirname,'browser-'+label+'.json'),JSON.stringify({...result,label,browser:'Windows Edge headless; muted AudioWorklet output'},null,2));
  try{await call('Browser.close');}catch{}ws.close();
  if(result.status!=='complete')throw Error(JSON.stringify(result));
}
main().catch(error=>{console.error(error);process.exitCode=1;});
