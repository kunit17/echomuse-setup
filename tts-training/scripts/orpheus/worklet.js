class PCMPlayer extends AudioWorkletProcessor {
  constructor() {
    super();this.queue=[];this.offset=0;this.available=0;this.started=false;this.ended=false;
    this.played=0;this.starved=0;this.gaps=0;this.inGap=false;
    this.port.onmessage=({data})=>{
      if(data.end){this.ended=true;return;}
      this.queue.push(data.audio);this.available+=data.audio.length;
    };
  }
  process(inputs,outputs) {
    const output=outputs[0][0];output.fill(0);
    if(!this.started&&(this.available>=sampleRate*1.0||this.ended&&this.available)) {
      this.started=true;this.port.postMessage({firstPlayback:true});
    }
    if(!this.started)return true;
    for(let i=0;i<output.length;i++) {
      if(this.available) {
        output[i]=this.queue[0][this.offset++];this.available--;this.played++;this.inGap=false;
        if(this.offset===this.queue[0].length){this.queue.shift();this.offset=0;}
      } else if(!this.ended) {
        this.starved++;if(!this.inGap){this.gaps++;this.inGap=true;}
      }
    }
    if(this.ended&&!this.available){this.port.postMessage({done:true,samples:this.played,underruns:this.gaps,underrunSeconds:this.starved/sampleRate});return false;}
    return true;
  }
}
registerProcessor('pcm-player',PCMPlayer);
