import json,platform,sys,torch,torchaudio,torchvision
result={'python':platform.python_version(),'executable':sys.executable,'torch':torch.__version__,'torchaudio':torchaudio.__version__,'torchvision':torchvision.__version__,'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'cudnn':torch.backends.cudnn.version()}
assert result['cuda_available'] and 'RTX 3060' in result['device']
x=torch.randn(2048,2048,device='cuda',dtype=torch.float16);y=x@x;torch.cuda.synchronize()
result['gpu_operation_device']=str(y.device);result['allocated_bytes']=torch.cuda.memory_allocated()
print(json.dumps(result,indent=2))
from pathlib import Path
(Path(__file__).resolve().parents[1]/'reports/environment.json').write_text(json.dumps(result,indent=2)+'\n')
