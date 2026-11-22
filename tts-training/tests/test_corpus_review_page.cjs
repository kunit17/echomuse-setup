// Pure local JavaScript tests with a synthetic DOM; no browser is controlled.
const fs=require('node:fs');
const vm=require('node:vm');
const assert=require('node:assert/strict');
const path=require('node:path');
const html=fs.readFileSync(path.join(__dirname,'../scripts/corpus_review.html'),'utf8');
const packageDir=path.resolve(process.argv[2]||path.join(__dirname,'../../listening-review'));
const queue=JSON.parse(fs.readFileSync(path.join(packageDir,'queue.json'),'utf8'));
queue.default_playlist='review';
const script=[...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)].at(-1)[1];
class Element{
  constructor(){this.value='';this.textContent='';this.children=[];this.handlers={};this.attrs={};this.paused=true;this.classList={toggle(){},add(){},remove(){}};}
  addEventListener(name,fn){this.handlers[name]=fn;}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;}
  setAttribute(k,v){this.attrs[k]=v;}
  pause(){this.paused=true;}
  play(){this.paused=false;return Promise.resolve();}
  focus(){}
  click(){}
  remove(){}
  matches(){return false;}
  async fire(name,event={}){return this.handlers[name]?.(event);}
}
const elements=new Map([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],new Element()]));
elements.get('review-data').textContent=JSON.stringify(queue);
elements.get('filter').value='all';
const saved=new Map();const downloads=[];
const document={getElementById:id=>{assert(elements.has(id),id);return elements.get(id);},createElement:()=>new Element(),body:new Element(),addEventListener(){}};
const context={document,localStorage:{getItem:k=>saved.get(k),setItem:(k,v)=>saved.set(k,v)},
  Blob,URL:{createObjectURL:b=>{downloads.push(b);return 'mock:download';},revokeObjectURL(){}},setTimeout:fn=>fn(),console};
vm.createContext(context);vm.runInContext(script,context);
const $=id=>elements.get(id);
(async()=>{
  assert.equal($('summary').textContent,'0 / 50 reviewed');
  assert.equal($('audio').src,queue.clips[0].audio_url);
  assert.equal($('transcript').textContent,queue.clips[0].text);
  await $('keep').fire('click');assert.equal($('summary').textContent,'1 / 50 reviewed');
  await $('next').fire('click');await $('reject').fire('click');
  await $('export').fire('click');assert.equal(downloads.length,0);assert.match($('message').textContent,/incomplete/);
  $('notes').value='Synthetic test reason';await $('notes').fire('input');
  assert.equal($('summary').textContent,'2 / 50 reviewed');
  await $('next').fire('click');await $('correct').fire('click');
  assert.equal($('summary').textContent,'2 / 50 reviewed');
  $('correction').value=queue.clips[2].text+' [TEST ONLY]';await $('correction').fire('input');
  assert.equal($('summary').textContent,'3 / 50 reviewed');
  await $('export').fire('click');assert.equal(downloads.length,1);
  const output=JSON.parse(await downloads[0].text());assert.equal(output.reviewed_count,3);
  assert.equal(output.corpus_fingerprint,queue.corpus_fingerprint);
  await $('unreviewed').fire('click');assert.equal($('summary').textContent,'2 / 50 reviewed');
  await $('import').fire('change',{target:{files:[{text:async()=>JSON.stringify(output)}],value:'mock'}});
  assert.equal($('summary').textContent,'3 / 50 reviewed');
  await $('import').fire('change',{target:{files:[{text:async()=>JSON.stringify({...output,corpus_fingerprint:'wrong'})}],value:'mock'}});
  assert.match($('message').textContent,/Import failed/);assert.equal($('summary').textContent,'3 / 50 reviewed');
  assert(saved.size>0);vm.runInNewContext(script,{...context});
  assert.equal($('summary').textContent,'3 / 50 reviewed');
  $('filter').value='validation';await $('filter').fire('change');assert.equal($('queue').children.length,10);
  // Audio/context assets and every queue entry reference the audited corpus.
  assert.equal(new Set(queue.playlists?.review||queue.clips.map(c=>c.id)).size,50);
  for(const c of queue.clips){assert(fs.existsSync(path.join(packageDir,c.audio_url)));assert(fs.existsSync(path.join(packageDir,c.context_url)));}
  if(queue.playlists){
    const savedBefore=JSON.parse([...saved.values()][0]);
    const allRows=JSON.parse(fs.readFileSync(path.join(packageDir,'all-clips.json'),'utf8')).sort((a,b)=>a.source_start_frame-b.source_start_frame);
    assert.deepEqual(queue.playlists.first20,allRows.slice(0,20).map(c=>c.id));
    assert.deepEqual(queue.playlists.last10,allRows.slice(-10).map(c=>c.id));
    $('playlist').value='first20';await $('playlist').fire('change');
    assert.equal($('queue').children.length,20);assert.equal($('clip-title').textContent,queue.playlists.first20[0]);
    for(let i=1;i<20;i++)await $('next').fire('click');
    assert.equal($('clip-title').textContent,queue.playlists.first20[19]);assert($('next').disabled);
    await $('next').fire('click');assert.equal($('clip-title').textContent,queue.playlists.first20[19]);
    $('playlist').value='last10';await $('playlist').fire('change');
    assert.equal($('queue').children.length,10);assert.equal($('clip-title').textContent,queue.playlists.last10[0]);
    $('autoplay').checked=true;await $('audio').fire('ended');
    assert.equal($('clip-title').textContent,queue.playlists.last10[1]);assert.equal($('audio').paused,false);
    for(let i=2;i<10;i++)await $('next').fire('click');
    await $('audio').fire('ended');assert.equal($('clip-title').textContent,queue.playlists.last10[9]);
    assert.equal($('message').textContent,'Playlist finished.');
    assert.deepEqual(JSON.parse([...saved.values()][0]),savedBefore);
  }
  console.log('Review page checks passed: first/last source ordering, playlist navigation, auto-next, decisions, export/import, storage, filters, and '+queue.clips.length*2+' audio assets.');
})().catch(error=>{console.error(error);process.exitCode=1;});
