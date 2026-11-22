"""Inspect cached model tensor headers without loading tensor payloads."""
import json
import os
from pathlib import Path
import struct
ROOT = Path(__file__).resolve().parents[1]
counts = {}
def read(fmt, stream):
    return struct.unpack('<'+fmt, stream.read(struct.calcsize('<'+fmt)))[0]
def string(stream):
    return stream.read(read('H', stream))[:-1].decode()
with (ROOT/'.cache/whisper-large-v3/model.bin').open('rb') as stream:
    version = read('I', stream)
    spec = string(stream)
    revision = read('I', stream)
    tensors = read('I', stream)
    for _ in range(tensors):
        name = string(stream)
        rank = read('B', stream)
        shape = [read('I', stream) for _ in range(rank)]
        dtype = read('B', stream)
        size = read('I', stream)
        entry = counts.setdefault(str(dtype), dict(tensors=0, payload_bytes=0, elements=0))
        entry['tensors'] += 1
        entry['payload_bytes'] += size
        elements = 1
        for dimension in shape: elements *= dimension
        entry['elements'] += elements
        stream.seek(size, 1)
value = dict(binary_version=version, spec=spec, revision=revision,
             tensors=tensors, by_dtype_id=counts,
             dtype_ids={'0':'float32','1':'int8','2':'int16','3':'int32','4':'float16','5':'bfloat16'},
             float32_tensor_payload_bytes=counts['4']['elements']*4,
             dtype_ids_source='installed ctranslate2/specs/model_spec.py')
path = ROOT/'reports/corpus-asr-model-inventory.json'
temporary = path.with_suffix('.json.tmp')
with temporary.open('w') as output:
    json.dump(value, output, indent=2); output.write('\n'); output.flush(); os.fsync(output.fileno())
os.replace(temporary, path)
fd = os.open(path.parent, os.O_DIRECTORY)
try: os.fsync(fd)
finally: os.close(fd)
print(json.dumps(value))
