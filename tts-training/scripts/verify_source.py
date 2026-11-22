import hashlib,json,pathlib
R=pathlib.Path(__file__).resolve().parents[1];S=pathlib.Path('/home/voiceuser/.hermes/cache/voices/she-xtts');d=json.loads((R/'reports/dataset.json').read_text())
assert len(list((S/'clips').glob('*.wav')))==1217
assert hashlib.sha256((S/'manifest.csv').read_bytes()).hexdigest()==d['manifest_sha256']
for c in d['source_clips']:assert hashlib.sha256((S/'clips'/c['filename']).read_bytes()).hexdigest()==c['source_sha256'],c['filename']
(R/'reports/source-integrity.json').write_text(json.dumps({'clips':1217,'all_wav_sha256_unchanged':True,'manifest_sha256_unchanged':True},indent=2)+'\n')
print('All 1217 source WAV SHA256 hashes and manifest unchanged.')
