"""Capture reproducibility metadata without initializing another GPU model."""
from common import *
import importlib.metadata,platform,subprocess,sys
cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown')
gpu=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
atomic(ROOT/'stream-benchmark/environment.json',dict(python=sys.version,platform=platform.platform(),cpu=cpu,gpu=gpu,
    packages={name:importlib.metadata.version(name) for name in ['torch','vllm','snac','bitsandbytes','vllm-bnb-plugin','transformers']},
    source_sha256={p.name:sha(p) for p in (ROOT/'scripts').glob('stream_*.py')},
    selection=json.loads((ROOT/'selected-model.json').read_text()),configuration='batch1 context4096; default vLLM GPU allocator; no custom resource caps'))
