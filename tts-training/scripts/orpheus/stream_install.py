"""Install benchmark sources only; do not start GPU work."""
from pathlib import Path
import shutil
ROOT=Path('/home/voiceuser/projects/orpheus-lora');B=ROOT/'stream-benchmark';B.mkdir(exist_ok=True)
SOURCE=Path(__file__).resolve().parent
names=[p.name for p in SOURCE.glob('stream_*.py')]+['test_stream.py']
for name in names:shutil.copy2(SOURCE/name,ROOT/'scripts'/name)
shutil.copy2(SOURCE/'stream_review.html',B/'review.html');shutil.copy2(SOURCE/'worklet.js',B/'worklet.js')
print('Installed experimental streaming benchmark; production selection unchanged.')
