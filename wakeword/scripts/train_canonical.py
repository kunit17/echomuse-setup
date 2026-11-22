"""Run pinned openWakeWord with canonical Piper BOS padding and intact audio."""
import sys,runpy
sys.path.insert(0,"/opt/piper-sample-generator")
import generate_samples
from piper_phonemize import phonemize_espeak,phoneme_ids_espeak

def canonical_ids(voice, config, text, verbose):
    return [value for sentence in phonemize_espeak(text,voice)
            for value in phoneme_ids_espeak(sentence,config["phoneme_id_map"])]

generate_samples.get_phonemes=canonical_ids
# Preserve unvoiced consonants and timing; do not concatenate only VAD frames.
generate_samples.remove_silence=lambda audio,*args,**kwargs:audio
runpy.run_path("/opt/openwakeword/openwakeword/train.py",run_name="__main__")
