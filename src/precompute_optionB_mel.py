"""
Option B - convert the 8670 synthetic audio files into a mel spectrogram cache (old parameters, consistent with real data).
Saved to G:\bird_melcache_optionB, labels use relabel (Perch's highest score).
Old parameters: n_fft=1024, fmin=0 default, n_mels=128, hop=512, 5 seconds @ 32kHz, 6 segments.
"""
import os
import numpy as np
import pandas as pd
import librosa

SYNTH_DIR  = r"G:\bird\synth_data_optionB"
MANIFEST   = os.path.join(SYNTH_DIR, "synth_manifest_optionB.csv")
CACHE_DIR  = r"G:\bird_melcache_optionB"
INDEX_OUT  = os.path.join(CACHE_DIR, "optionB_cache_index.csv")

SAMPLE_RATE = 32000
SEG_SEC     = 5
SEG_SAMPLES = SAMPLE_RATE * SEG_SEC
N_MELS      = 128
N_FFT       = 1024        # old parameter
HOP         = 512
N_SEG       = 6

os.makedirs(CACHE_DIR, exist_ok=True)

def to_mel(wav):
    try:
        y,_ = librosa.load(wav, sr=SAMPLE_RATE, mono=True)
    except Exception:
        return None
    if len(y) < SEG_SAMPLES: y = np.pad(y,(0,SEG_SAMPLES-len(y)))
    n = max(1, len(y)//SEG_SAMPLES)
    idxs = np.linspace(0,n-1,N_SEG).astype(int) if n>=N_SEG else np.resize(np.arange(n),N_SEG)
    mels=[]
    for si in idxs:
        seg = y[si*SEG_SAMPLES:(si+1)*SEG_SAMPLES]
        if len(seg)<SEG_SAMPLES: seg=np.pad(seg,(0,SEG_SAMPLES-len(seg)))
        m = librosa.feature.melspectrogram(y=seg.astype(np.float32), sr=SAMPLE_RATE,
                                            n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
        m = librosa.power_to_db(m, ref=np.max)
        m = (m-m.min())/(m.max()-m.min()+1e-9)
        mels.append(m.astype(np.float32))
    return np.stack(mels)

def main():
    man = pd.read_csv(MANIFEST)
    print(f"To convert: {len(man)} items")
    rows=[]
    for i,r in man.iterrows():
        wav = os.path.join(SYNTH_DIR, r["file"])
        if not os.path.exists(wav): continue
        mels = to_mel(wav)
        if mels is None: continue
        out = "OB__" + r["file"].replace(".wav",".npy")
        np.save(os.path.join(CACHE_DIR, out), mels)
        rows.append({"cache":out, "relabel":r["relabel"], "target_sci":r["target_sci"]})
        if (i+1)%500==0: print(f"  {i+1}/{len(man)}")
    pd.DataFrame(rows).to_csv(INDEX_OUT, index=False)
    print(f"Done {len(rows)} items -> {CACHE_DIR}")

if __name__ == "__main__":
    main()
