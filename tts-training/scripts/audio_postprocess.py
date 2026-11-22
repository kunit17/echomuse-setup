"""Conservative cleanup of isolated CSM decoder end bursts in generated audio.

This is an inference-only operation. Never apply it to corpus segmentation.
"""
POLICY_ID='isolated-end-burst-v1'

def clean_generated_tail(audio,sample_rate):
    """Silence only a short, unvoiced low-band tail following near-silence.

    Keeps the duration and all samples before the final 100ms unchanged.
    Uncertain, voiced, loud, or high-frequency endings pass through untouched.
    Returns (audio, decision) so every adjustment can be logged.
    """
    import numpy as np
    x=np.asarray(audio)
    decision=dict(policy=POLICY_ID,applied=False,replaced_ms=0)
    if x.ndim!=1 or sample_rate<16000 or not np.isfinite(x).all():
        raise ValueError('Expected finite mono generated audio at >=16kHz')
    tail_length=round(.1*sample_rate);gap_length=round(.2*sample_rate)
    if len(x)<tail_length+gap_length:
        return x,dict(decision,reason='too_short')
    start=len(x)-tail_length
    gap=x[start-gap_length:start].astype('float64')
    tail=x[start:].astype('float64')
    gap_peak=float(np.max(np.abs(gap)))
    if gap_peak>=.001:
        return x,dict(decision,reason='no_verified_quiet_gap')
    gap_rms=float(np.sqrt(np.mean(gap**2)))
    tail_rms=float(np.sqrt(np.mean(tail**2)))
    tail_peak=float(np.max(np.abs(tail)))
    if tail_rms<max(.001,10*gap_rms):
        return x,dict(decision,reason='no_significant_burst')
    if tail_peak>.125:
        return x,dict(decision,reason='loud_ending_preserved')
    # Protect voiced short final words and high-frequency unvoiced consonants.
    y=tail-tail.mean()
    energy=float(y@y)
    if energy<=1e-15:
        return x,dict(decision,reason='uncertain_ending_preserved')
    correlation=np.correlate(y,y,mode='full')[len(y)-1:]
    voicing=float(np.max(correlation[round(sample_rate/500):round(sample_rate/70)])/energy)
    spectrum=np.abs(np.fft.rfft(y*np.hanning(len(y))))**2
    frequencies=np.fft.rfftfreq(len(y),1/sample_rate)
    high_band=float(spectrum[frequencies>4000].sum()/max(float(spectrum.sum()),1e-15))
    if voicing>.45 or high_band>.05:
        return x,dict(decision,reason='possible_speech_preserved',voicing=voicing,high_band_fraction=high_band)
    cleaned=x.copy();cleaned[start:]=0
    return cleaned,dict(decision,applied=True,reason='isolated_terminal_burst',replaced_ms=100,
        quiet_gap_ms=200,gap_peak=gap_peak,tail_rms=tail_rms,tail_peak=tail_peak,
        voicing=voicing,high_band_fraction=high_band)
