from common import *
import argparse,sys
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--text-file',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path);args=parser.parse_args()
    text=args.text_file.read_text().strip();assert text and len(text)<=400,'Use a short utterance per call'
    result=json.loads((RUN/'training-result.json').read_text());precision=result['precision']
    checkpoint=RUN/'adapter-final';assert (checkpoint/'adapter_model.safetensors').exists()
    from unsloth import FastLanguageModel
    import numpy as np,soundfile as sf,torch
    from transformers import set_seed
    from peft import PeftModel
    from snac import SNAC
    from huggingface_hub import snapshot_download
    sys.path.insert(0,str(REPO/'scripts'))
    from audio_postprocess import clean_generated_tail
    started=time.time();name=MODEL if precision=='16bit' else QUANT
    base,tokenizer=FastLanguageModel.from_pretrained(name,revision=setup()['model_revisions'][name],
        max_seq_length=4096,dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        load_in_4bit=precision=='4bit',device_map='cpu' if precision=='16bit' else {'':'cuda:0'})
    model=PeftModel.from_pretrained(base,str(checkpoint),is_trainable=False).to('cuda')
    FastLanguageModel.for_inference(model)
    refresh_model_placement(model)
    codec=SNAC.from_pretrained(snapshot_download(CODEC,revision=setup()['model_revisions'][CODEC],local_files_only=True)).cpu().eval()
    load_seconds=time.time()-started
    ids=[TOKENS['start_human']]+tokenizer.encode(text,add_special_tokens=True)+[TOKENS['end_text'],TOKENS['end_human'],TOKENS['start_ai'],TOKENS['start_speech']]
    assert len(ids)+2000<=4096
    inputs=torch.tensor([ids],device='cuda');set_seed(1700);started=time.time()
    with torch.inference_mode():
        output=model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),max_new_tokens=2000,
            do_sample=True,temperature=.6,top_p=.95,repetition_penalty=1.1,
            eos_token_id=TOKENS['end_speech'],pad_token_id=TOKENS['pad'],use_cache=True)
    tokens=output[0,len(ids):].tolist()
    assert TOKENS['end_speech'] in tokens,'No end-of-speech; refuse truncated output'
    layers=unpack_codes(tokens[:tokens.index(TOKENS['end_speech'])])
    with torch.inference_mode():audio=codec.decode([torch.tensor([v]) for v in layers])[0,0].float().numpy()
    assert np.isfinite(audio).all() and len(audio)>0
    audio,decision=clean_generated_tail(audio,24000)
    destination=args.out.with_suffix('.wav');assert not destination.exists()
    destination.parent.mkdir(parents=True,exist_ok=True)
    sf.write(destination,audio,24000,subtype='PCM_16')
    import subprocess
    subprocess.run(['/home/voiceuser/.local/bin/ffmpeg','-hide_banner','-loglevel','error','-n','-i',str(destination),'-codec:a','libmp3lame','-b:a','128k',str(args.out.with_suffix('.mp3'))],check=True)
    atomic(args.out.with_suffix('.json'),dict(text=text,checkpoint=str(checkpoint),epoch=3,precision=precision,
        adapter_sha256=sha(checkpoint/'adapter_model.safetensors'),load_seconds=load_seconds,
        generation_seconds=time.time()-started,audio_seconds=len(audio)/24000,tail_cleanup=decision,manual_listening_complete=False))
    print(args.out.with_suffix('.mp3'),flush=True)
if __name__=='__main__':main()
