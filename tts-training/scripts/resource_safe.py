"""Local resource controls for the installed upstream trainers."""
import os
import gc
import ctypes
from pathlib import Path
import torch
from torch.utils.data import DataLoader

def loader(*args, **kwargs):
    kwargs.update(num_workers=0, persistent_workers=False, pin_memory=False)
    kwargs.pop('prefetch_factor', None)
    return DataLoader(*args, **kwargs)

def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; refusing CPU fallback')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one visible CUDA GPU')
    print('RESOURCE_PROFILE: single GPU; synchronous loading; no prefetch; batch 1', flush=True)
    print('CUDA_DEVICE:', torch.cuda.get_device_name(0), flush=True)

def bounded_hubert_loading():
    """Load frozen fp16 HuBERT directly on CUDA, avoiding full host copies."""
    from transformers import HubertModel
    original = HubertModel.from_pretrained
    def load(*args, **kwargs):
        kwargs.update(torch_dtype=torch.float16, low_cpu_mem_usage=True,
                      device_map={'': 'cuda:0'})
        return original(*args, **kwargs)
    HubertModel.from_pretrained = load

def atomic_save(original):
    def trim():
        gc.collect()
        try:ctypes.CDLL(None).malloc_trim(0)
        except AttributeError:pass
    class BoundedWriter:
        def __init__(self,stream):self.stream=stream;self.pending=0;self.position=0
        def write(self,data):
            view=memoryview(data)
            for start in range(0,len(view),1024*1024):
                piece=view[start:start+1024*1024]
                self.stream.write(piece);self.pending+=len(piece);self.position+=len(piece)
                if self.pending>=8*1024*1024:
                    self.flush();os.fdatasync(self.stream.fileno())
                    os.posix_fadvise(self.stream.fileno(),0,self.position,os.POSIX_FADV_DONTNEED)
                    self.pending=0
            return len(view)
        def flush(self):self.stream.flush()
    def save(obj, destination, *args, **kwargs):
        if not isinstance(destination, (str, os.PathLike)):
            return original(obj, destination, *args, **kwargs)
        path = Path(destination)
        tmp = path.with_name(path.name + '.saving')
        trim()
        with tmp.open('wb') as stream:
            original(obj, BoundedWriter(stream), *args, **kwargs)
            stream.flush()
            os.fsync(stream.fileno())
            os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
        os.replace(tmp, path)
        fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        trim()
    return save

def gpt_training_step(self, batch, batch_idx):
    # Upstream loss is summed. Eight microbatches of one example retain the
    # intended summed-gradient scale of four microbatches of two examples.
    optimizer = self.optimizers()
    scheduler = self.lr_schedulers()
    forward = self.model.forward if self.config['train'].get('if_dpo', False) else self.model.forward_old
    loss, accuracy = forward(batch['phoneme_ids'], batch['phoneme_ids_len'],
                            batch['semantic_ids'], batch['semantic_ids_len'], batch['bert_feature'])
    self.manual_backward(loss)
    if (batch_idx + 1) % 8 == 0 or batch_idx + 1 == self.trainer.num_training_batches:
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
    self.log('total_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=False)
    self.log('lr', scheduler.get_last_lr()[0], on_epoch=True, prog_bar=True, sync_dist=False)
    self.log('top_3_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True, sync_dist=False)
