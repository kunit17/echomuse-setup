'use strict';
const $=id=>document.getElementById(id);
let snapshot=null,paused=false,chosen='live',loading=false;
const labels={s1:'GPT',s2:'SoVITS'};
function number(value,digits=3){return Number.isFinite(value)?value.toFixed(digits):'—';}
function duration(seconds){if(!Number.isFinite(seconds))return '—';const m=Math.max(0,Math.ceil(seconds/60));return m>=60?Math.floor(m/60)+'h '+m%60+'m':m+'m';}
function batchProgress(lines){const matches=[...lines.join('\n').matchAll(/Train Epoch:\s*(\d+)\s*\[(\d+)%\]/g)];if(!matches.length)return null;const last=matches.at(-1);return {epoch:Number(last[1]),percent:Number(last[2])};}
function chart(id,points,best,label){
  const host=$(id);host.replaceChildren();const good=points.filter(p=>Number.isFinite(p.value));
  if(!good.length){host.textContent='Waiting for '+label+' measurements.';return;}
  const ns='http://www.w3.org/2000/svg',W=510,H=245,L=55,R=18,T=20,B=33;
  const svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('aria-label',label+' by epoch');
  function el(type,attrs,text){const node=document.createElementNS(ns,type);Object.entries(attrs).forEach(([k,v])=>node.setAttribute(k,v));if(text!==undefined)node.textContent=text;svg.append(node);return node;}
  let low=Math.min(...good.map(p=>p.value)),high=Math.max(...good.map(p=>p.value));let pad=(high-low)*.12||Math.max(Math.abs(high)*.05,.01);low=Math.max(0,low-pad);high+=pad;
  const max=Math.max(1,...good.map(p=>p.epoch)),x=e=>L+e/max*(W-L-R),y=v=>H-B-(v-low)/(high-low)*(H-T-B);
  for(let i=0;i<4;i++){const value=low+(high-low)*i/3,yy=y(value);el('line',{x1:L,y1:yy,x2:W-R,y2:yy,stroke:'#e4e9df'});el('text',{x:L-8,y:yy+4,'text-anchor':'end'},number(value,high>20?0:2));}
  const ticks=[...new Set([0,Math.round(max/2),max])];ticks.forEach(epoch=>el('text',{x:x(epoch),y:H-10,'text-anchor':'middle'},String(epoch)));
  el('path',{d:good.map((p,i)=>(i?'L':'M')+x(p.epoch)+','+y(p.value)).join(' '),class:'line'});
  for(const p of good){const circle=el('circle',{cx:x(p.epoch),cy:y(p.value),r:p.epoch===best?5.5:4,class:p.epoch===best?'best':'point'});const title=document.createElementNS(ns,'title');title.textContent=`Epoch ${p.epoch}: ${number(p.value,5)}${p.epoch===best?' · selected checkpoint':''}`;circle.append(title);}
  host.append(svg);
}
function trainingMetric(stage,row){return stage==='s1'?row.train?.total_loss_epoch:row.train?.['loss/g/mel'];}
function activeCurve(){return chosen==='live'?(snapshot?.status.active_stage==='sovits'?'s2':snapshot?.stages.s2.last?'s2':'s1'):chosen;}
function render(){
  if(!snapshot?.available){$('heading').textContent='No training run recorded';return;}
  const {status,run,gpu,stages}=snapshot,active=status.active_stage;
  const live=status.status==='running'||status.status==='starting';
  const title=status.status==='complete'?'Training and evaluation finished':status.status==='failed'?'Training needs attention':active==='gpt'?'GPT training is running':active==='sovits'?'SoVITS training is running':active==='synthesis'?'Generating test speech':active==='evaluation'?'Checking the generated speech':'Preparing training features';
  $('heading').textContent=title;
  $('subtitle').textContent='Fresh pretrained start · validation every epoch · best and latest checkpoints retained';
  const warning=status.error||(live&&!snapshot.controller_alive?'The training controller is no longer visible. The recorded status may be stale.':null);
  $('alert').classList.toggle('hidden',!warning);$('alert').textContent=warning||'';$('alert').className=warning?'notice warning':'notice hidden';
  $('elapsed').textContent=duration((status.finished||Date.now()/1000)-status.started);
  $('started').textContent='Started '+new Date(status.started*1000).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});
  $('gpu-util').textContent=gpu?number(gpu.utilization,0)+'%':'Unavailable';$('gpu-name').textContent=gpu?.name||'GPU telemetry unavailable';
  $('gpu-memory').textContent=gpu?number(gpu.used_mib/1024,1)+' / '+number(gpu.total_mib/1024,0)+' GiB':'—';
  $('gpu-temp').textContent=gpu?number(gpu.temperature,0)+' °C · total device memory in use':'No GPU memory cap applied';
  $('train-clips').textContent=run.train_clips+' training';$('val-clips').textContent=run.validation_clips+' validation · separate source spans';
  $('pipeline').replaceChildren();
  const completed=status.completed_stages||[];
  for(const [key,name,done] of [['features','Features',completed.filter(x=>x.endsWith('-verify')).length===2],['gpt','GPT',completed.includes('gpt')],['sovits','SoVITS',completed.includes('sovits')],['evaluation','Speech checks',completed.includes('evaluation')]]){
    const node=document.createElement('span');node.textContent=(done?'✓ ':'')+name;node.className=done?'done':(active===key||key==='features'&&active?.includes('-')||key==='evaluation'&&active==='synthesis')?'active':'';$('pipeline').append(node);
  }
  for(const key of ['s1','s2']){
    const stage=stages[key],epoch=stage.last?.epoch||0,max=stage.policy.max_epochs;
    $(key+'-state').textContent=stage.state;$(key+'-state').className='pill '+(stage.state==='running'?'live':stage.state);
    $(key+'-progress').replaceChildren(document.createTextNode(epoch+' '));const small=document.createElement('span');small.textContent='/ '+max+' epochs'+(stage.state==='complete'&&epoch<max?' · stopped early':stage.state==='running'||stage.state==='queued'?' · minimum '+stage.policy.min_epochs:'');$(key+'-progress').append(small);
    const batch=key==='s2'&&stage.state==='running'?batchProgress(snapshot.log_tail):null;
    $(key+'-bar').max=max;$(key+'-bar').value=stage.state==='complete'?max:epoch+(batch&&batch.epoch>epoch?batch.percent/100:0);
    $(key+'-best').textContent=stage.selected?'Epoch '+stage.selected.epoch:'—';
    $(key+'-loss').textContent=number(stage.last?.validation?.[stage.policy.metric]);
    const times=stage.rows.filter(r=>r.epoch>0&&Number.isFinite(r.seconds)).slice(-5).map(r=>r.seconds).sort((a,b)=>a-b);
    const average=times.length?times[Math.floor(times.length/2)]:null;
    const remaining=average===null?null:Math.max(0,(max-epoch-(batch&&batch.epoch>epoch?batch.percent/100:0))*average);
    $(key+'-eta').textContent=stage.state==='complete'?'Complete · selected by held-out validation':stage.state==='failed'?'Stopped with an error; see live log.':remaining!==null?'To the epoch ceiling: ~'+duration(remaining)+' training time, plus validation/saving. Early stopping may finish sooner.':stage.state==='queued'?'Queued · timing after its first epoch.':'Timing after the first saved checkpoint.';
    if(batch&&batch.epoch>epoch)$(key+'-eta').textContent='Epoch '+batch.epoch+' · '+batch.percent+'% of batches. '+$(key+'-eta').textContent;
  }
  const key=activeCurve(),stage=stages[key],metric=stage.policy.metric,rows=stage.rows,best=stage.selected;
  for(const tab of ['live','s1','s2'])$('tab-'+tab).setAttribute('aria-pressed',String(chosen===tab));
  $('validation-title').textContent=labels[key]+(key==='s1'?' held-out token loss':' held-out mel loss');
  $('training-title').textContent=labels[key]+(key==='s1'?' training loss · summed objective':' training mel loss · weighted objective');
  chart('validation-chart',rows.map(r=>({epoch:r.epoch,value:r.validation[metric]})),best?.epoch,'Validation loss');
  chart('training-chart',rows.filter(r=>r.epoch>0).map(r=>({epoch:r.epoch,value:trainingMetric(key,r)})),null,'Training loss');
  const baseline=rows.find(r=>r.epoch===0)?.validation[metric],last=stage.last?.validation[metric];
  let quality='Waiting for held-out measurements. Training updates never use the 66 validation clips.';
  let watch=false;
  if(best&&Number.isFinite(last)){
    const improvement=baseline?(baseline-last)/baseline*100:null;
    quality=`Best checkpoint: epoch ${best.epoch}. ${improvement!==null?'Latest validation loss is '+number(Math.abs(improvement),1)+'% '+(improvement>=0?'below':'above')+' the pretrained baseline. ':''}${stage.patience_used} / ${stage.policy.patience} epochs without a ≥${number(stage.policy.min_relative_improvement*100,1)}% validation improvement.`;
    const recent=rows.filter(r=>r.epoch>0).slice(-3);
    watch=recent.length===3&&recent.every(r=>r.validation[metric]>best.validation[metric]*1.005)&&trainingMetric(key,recent[2])<trainingMetric(key,recent[0]);
    if(watch)quality+=' Training loss is falling while validation remains worse than its best: watch for overfitting.';
  }
  quality+=' Early stopping is allowed from epoch '+stage.policy.min_epochs+'.';$('quality').textContent=quality;$('quality').className=watch?'notice warning':'notice';
  $('table-stage').textContent=labels[key];$('epoch-table').replaceChildren();
  for(const row of [...rows].reverse().slice(0,8)){
    const tr=document.createElement('tr');const values=[row.epoch===0?'0 · pretrained':String(row.epoch)+(row.epoch===best?.epoch?' · best':''),number(trainingMetric(key,row)),number(row.validation[metric]),row.optimizer_steps??'—',row.seconds?number(row.seconds,1)+' s':'—'];
    values.forEach((value,i)=>{const td=document.createElement('td');td.textContent=value;if(i===0&&row.epoch===best?.epoch)td.className='best-cell';tr.append(td);});$('epoch-table').append(tr);
  }
  if(!rows.length){const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=5;td.textContent='No completed epochs for this stage yet.';tr.append(td);$('epoch-table').append(tr);}
  $('checkpoints').textContent=['s1','s2'].map(key=>labels[key]+': '+(stages[key].selected?.checkpoint||'not selected yet')).join('\n');
  $('log').textContent=snapshot.log_tail.length?snapshot.log_tail.join('\n'):'No new log lines. Completed runs retain their logs on disk.';
  $('run-id').textContent='Run '+status.run_id;
  $('updated').textContent='Updated '+new Date(snapshot.fetched*1000).toLocaleTimeString()+' · read-only · every 5 seconds';
}
async function refresh(){if(loading)return;loading=true;try{const response=await fetch('/api/status',{cache:'no-store'});if(!response.ok)throw Error('HTTP '+response.status);snapshot=await response.json();render();$('connection').textContent=paused?'Updates paused':'Live';$('connection').className='pill '+(paused?'':'live');}catch(error){$('connection').textContent='Disconnected · retrying';$('connection').className='pill warning';$('alert').textContent='The dashboard cannot reach its local data feed. Training may still be running. '+error.message;$('alert').className='notice warning';}finally{loading=false;}}
$('refresh').addEventListener('click',refresh);$('pause').addEventListener('click',()=>{paused=!paused;$('pause').textContent=paused?'Resume updates':'Pause updates';$('connection').textContent=paused?'Updates paused':'Live';$('connection').className='pill '+(paused?'':'live');if(!paused)refresh();});
for(const tab of ['live','s1','s2'])$('tab-'+tab).addEventListener('click',()=>{chosen=tab;render();});
refresh();setInterval(()=>{if(!paused)refresh();},5000);
