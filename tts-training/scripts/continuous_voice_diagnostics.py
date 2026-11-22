"""Describe performed-voice variation without inferring speaker identities."""
import csv
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from corpus_ground_truth import ROOT,save

EXP=ROOT/'experiments/she-continuous-v1'
features={};descriptors={}
for partition in ('train','validation'):
    rows=list(csv.DictReader((ROOT/f'data/shecorpus/{partition}/manifest.csv').open()))
    verified_embeddings=np.load(EXP/partition/'verified-speaker-embeddings.npy',mmap_mode='r')
    assert len(verified_embeddings)==len(rows)
    matrix=[];items=[]
    for index,row in enumerate(rows):
        name=Path(row['filename']).name
        embedding=np.array(verified_embeddings[index],dtype=np.float32,copy=True)
        embedding/=max(np.linalg.norm(embedding),1e-12);matrix.append(embedding)
        signal,sr=sf.read(ROOT/row['filename'],dtype='float32')
        signal=librosa.resample(signal,orig_sr=sr,target_sr=16000)
        f0=librosa.yin(signal,fmin=60,fmax=500,sr=16000,frame_length=1024,hop_length=160)
        rms=librosa.feature.rms(y=signal,frame_length=1024,hop_length=160)[0]
        voiced=f0[:len(rms)][rms[:len(f0)]>.02]
        frequency=np.fft.rfftfreq(1024,1/16000)
        spectrum=np.abs(librosa.stft(signal,n_fft=1024,hop_length=160))
        centroid=np.sum(spectrum*frequency[:,None],axis=0)/np.maximum(spectrum.sum(axis=0),1e-12)
        items.append(dict(filename=name,f0_median_hz=float(np.median(voiced)) if len(voiced) else None,
            f0_iqr_hz=float(np.quantile(voiced,.75)-np.quantile(voiced,.25)) if len(voiced) else None,
            spectral_centroid_median_hz=float(np.median(centroid))))
    features[partition]=np.stack(matrix);descriptors[partition]=items
train=features['train'];centroids=np.mean(train,axis=0,keepdims=True);chosen=1;best_score=0.;scores=[]
for k in range(2,min(6,len(train)-1)+1):
    model=KMeans(n_clusters=k,random_state=20261112,n_init=10).fit(train)
    score=float(silhouette_score(train,model.labels_,metric='cosine'));scores.append(dict(k=k,silhouette=score))
    if score-best_score>=.05:chosen=k;best_score=score;centroids=model.cluster_centers_
centroids=centroids/np.maximum(np.linalg.norm(centroids,axis=1,keepdims=True),1e-12)
distances=1-train@centroids.T
nearest=np.min(distances,axis=1);median=float(np.median(nearest));mad=float(np.median(abs(nearest-median)));distance_limit=median+3*mad
f0values=np.array([r['f0_median_hz'] or np.nan for r in descriptors['train']]);f0median=float(np.nanmedian(f0values));f0mad=float(np.nanmedian(abs(f0values-f0median)))
for partition,items in descriptors.items():
    distances=1-features[partition]@centroids.T
    for i,row in enumerate(items):
        row.update(cluster=int(np.argmin(distances[i])),cosine_distance=float(np.min(distances[i])),
            embedding_outlier=bool(np.min(distances[i])>distance_limit),
            f0_outlier=bool(row['f0_median_hz'] is not None and abs(row['f0_median_hz']-f0median)>3*f0mad))
report=dict(status='descriptive_measurements_complete_listening_pending',selected_clusters=chosen,silhouette_candidates=scores,
    training_fitted_thresholds=dict(cosine_distance=distance_limit,f0_median=f0median,f0_mad=f0mad),
    partitions={p:dict(clips=len(rows),flagged=sum(r['embedding_outlier'] or r['f0_outlier'] for r in rows),clips_data=rows) for p,rows in descriptors.items()},
    exclusions_for_performed_voice=0,listened_clips=0,identity_inference=False,
    f0_limitation='YIN on RMS-selected frames is descriptive; it does not reliably distinguish all voicing or identities.')
save(ROOT/'reports/continuous-voice-diagnostics.json',report)
verification=json.loads((ROOT/'reports/corpus-verification.json').read_text())
verification['speaker_diagnostics_complete']=True
verification['voice_detection']={
    'status':report['status'],'selected_clusters':chosen,
    'flagged':{p:sum(r['embedding_outlier'] or r['f0_outlier'] for r in rows) for p,rows in descriptors.items()},
    'exclusions_for_performed_voice':0,'listened_clips':0}
save(ROOT/'reports/corpus-verification.json',verification)
print('Voice diagnostics completed; descriptive clusters',chosen,flush=True)
