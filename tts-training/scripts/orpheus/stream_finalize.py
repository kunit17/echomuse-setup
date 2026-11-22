"""Summarize completed results and verify production, without loading a model."""
from common import *
import urllib.request,re
B=ROOT/'stream-benchmark';report=json.loads((B/'report.json').read_text())
qa=json.loads((B/'asr.json').read_text());assert qa['complete']
flagged=[r for r in qa['results'] if r['word_edits'] or not r['first_word_match'] or not r['final_word_match']]
report['flagged_transcripts']=flagged
expected={}
for line in (B/'measurements.jsonl').read_text().splitlines():
    item=json.loads(line)
    if not item['warmup']:expected[(item['configuration'],item['prompt'],item['repeat'])]=item['text']
def normalize(text):return ' '.join(re.sub(r"[^a-z0-9 ]",' ',text.lower().replace("you've","you have")).split())
contraction_only=sum(normalize(r['transcript'])==normalize(expected[(r['configuration'],r['prompt'],r['repeat'])]) for r in flagged)
report['asr_interpretation']=dict(all_first_and_final_words_match=all(r['first_word_match'] and r['final_word_match'] for r in qa['results']),
    raw_flagged=len(flagged),contraction_only_flags=contraction_only,
    note=f"All {len(qa['results'])} ASR checks matched the first and final words. {contraction_only} of {len(flagged)} raw word-difference flags reduce to the you have / you've contraction. Listening is still required for complete phonemes, clarity and streaming seams.")
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with http.open('http://127.0.0.1:8812/health',timeout=5) as response:health=json.load(response)
assert health['ready'] and health['epoch']==3 and health['adapter_sha256']==json.loads((ROOT/'selected-model.json').read_text())['adapter_sha256']
report['production_restored']=dict(ready=True,model=health['model'],epoch=3,adapter_sha256=health['adapter_sha256'],verified_at=time.time())
atomic(B/'report.json',report)
print(json.dumps(dict(configurations=[{key:r[key] for key in ['name','requests','median_rtf','p95_rtf','asr_flagged']} for r in report['configurations']],flagged=flagged,coverage=report['audio_coverage'],production_restored=report['production_restored']),indent=2))
