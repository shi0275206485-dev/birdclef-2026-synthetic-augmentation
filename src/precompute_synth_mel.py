"""
Convert Group B's 50 synthetic audio clips into mel spectrogram cache (using old parameters, consistent with real data).
Saved into a separate directory G:\bird_melcache_synth, read together with the real cache during training.

Spectrogram parameters must exactly match the old cache for real data:
  n_fft=1024, fmin=0(default), fmax=16000, n_mels=128, hop=512, 5 seconds @32kHz
"""

import os
import numpy as np
import pandas as pd
import librosa

SYNTH_DIR  = r"D:\bird\synth_data"                 # location of the 50 synthetic wav files
MANIFEST   = os.path.join(SYNTH_DIR, "synth_manifest.csv")
CACHE_DIR  = r"G:\bird_melcache_synth"             # spectrogram cache for synthetic data

SAMPLE_RATE = 32000
SEG_SEC     = 5
SEG_SAMPLES = SAMPLE_RATE * SEG_SEC
N_MELS      = 128
N_FFT       = 1024        # old parameters (consistent with real data!)
HOP         = 512
N_SEG       = 6
# Note: the old parameters don't set fmin/fmax, using librosa defaults (fmin=0, fmax=sr/2=16000)

os.makedirs(CACHE_DIR, exist_ok=True)

def to_mel_segments(wav_path):
    try:
        y, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print("err:", wav_path, e); return None
    if len(y) < SEG_SAMPLES:
        y = np.pad(y, (0, SEG_SAMPLES - len(y)))
    n_avail = max(1, len(y) // SEG_SAMPLES)
    if n_avail >= N_SEG:
        idxs = np.linspace(0, n_avail - 1, N_SEG).astype(int)
    else:
        idxs = np.resize(np.arange(n_avail), N_SEG)
    mels = []
    for si in idxs:
        seg = y[si*SEG_SAMPLES:(si+1)*SEG_SAMPLES]
        if len(seg) < SEG_SAMPLES:
            seg = np.pad(seg, (0, SEG_SAMPLES - len(seg)))
        m = librosa.feature.melspectrogram(
            y=seg.astype(np.float32), sr=SAMPLE_RATE,
            n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)   # no fmin/fmax passed = old parameters
        m = librosa.power_to_db(m, ref=np.max)
        m = (m - m.min()) / (m.max() - m.min() + 1e-9)
        mels.append(m.astype(np.float32))
    return np.stack(mels)

def main():
    man = pd.read_csv(MANIFEST)
    # group synthetic data by file (each wav is one "recording")
    rows = []
    for fn in man["file"].unique():
        sci = man[man["file"] == fn]["target_sci"].iloc[0]
        wav = os.path.join(SYNTH_DIR, fn)
        if not os.path.exists(wav):
            continue
        mels = to_mel_segments(wav)
        if mels is None:
            continue
        out = os.path.join(CACHE_DIR, "SYNTH__" + fn.replace(".wav", ".npy"))
        np.save(out, mels)
        rows.append({"cache": os.path.basename(out), "scientific_name": sci, "file": fn})
        print(f"  {fn} -> {sci}")
    pd.DataFrame(rows).to_csv(os.path.join(CACHE_DIR, "synth_cache_index.csv"), index=False)
    print(f"\nDone: {len(rows)} synthetic audio clips converted to spectrograms -> {CACHE_DIR}")

if __name__ == "__main__":
    main()
