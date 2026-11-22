"""Ten-millisecond RMS quiet-run measurements, without audio modification."""
import numpy as np

def frame_rms(signal,sample_rate):
    width=sample_rate//100
    assert sample_rate%100==0 and signal.ndim==1
    frames=signal[:len(signal)//width*width].reshape(-1,width)
    return np.sqrt(np.mean(frames.astype(np.float64)**2,axis=1))

def longest_quiet(rms,threshold):
    count=best=0
    for value in rms:
        count=count+1 if value<threshold else 0
        best=max(best,count)
    return best*.01

def quiet_between(rms,left_seconds,right_seconds,threshold):
    left=max(0,int(np.ceil(left_seconds*100-1e-6)))
    right=max(left,int(np.floor(right_seconds*100+1e-6)))
    return longest_quiet(rms[left:right],threshold)

def quiet_at(rms,index,threshold):
    if not 0<=index<len(rms) or rms[index]>=threshold:return 0.
    left=right=index
    while left>0 and rms[left-1]<threshold:left-=1
    while right<len(rms) and rms[right]<threshold:right+=1
    return (right-left)*.01
