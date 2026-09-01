"""
Preprocessing: convert each recording into N_SEG mel-spectrogram segments,
cached on the G drive. Only needs to run once. Afterward, the training script
reads the cache directly and no longer recomputes spectrograms every epoch.

Each recording is stored as one .npy: shape (N_SEG, n_mels, time)
About 35000 recordings x ~1MB each ~= 33GB, well within the G drive's 800GB.

Environment: pip install librosa numpy pandas soundfile
Windows multiprocessing safe.
"""

import os
import numpy as np
import pandas as pd
import librosa
from concurrent.futures import ProcessPoolExecutor, as_completed

TRAIN_CSV   = r"D:\bird\train.csv"
AUDIO_ROOT  = r"D:\bird\train_audio"
CACHE_DIR   = r"G:\bird_melcache"     # cache on the G drive
N_WORKERS   = 6

SAMPLE_RATE = 32000
SEG_SEC     = 5
SEG_SAMPLES = SAMPLE_RATE * SEG_SEC
N_MELS      = 128
N_FFT       = 1024
HOP         = 512
N_SEG       = 6

os.makedirs(CACHE_DIR, exist_ok=True)

def cache_path(fname):
    # turn "1161364/iNat1216197.ogg" into a safe cache filename
    safe = fname.replace("/", "__").replace("\\", "__").replace(".ogg", ".npy")
    return os.path.join(CACHE_DIR, safe)

def process_one(fname):
    out = cache_path(fname)
    if os.path.exists(out):
        return "skip"
    path = os.path.join(AUDIO_ROOT, fname)
    try:
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        return f"err:{fname}:{e}"
    if len(y) < SEG_SAMPLES:
        y = np.pad(y, (0, SEG_SAMPLES - len(y)))
    n_avail = max(1, len(y) // SEG_SAMPLES)
    # reproducible for validation: evenly sample N_SEG segments (randomness for training augmentation is handled separately in the training script)
    if n_avail >= N_SEG:
        idxs = np.linspace(0, n_avail - 1, N_SEG).astype(int)
    else:
        idxs = np.resize(np.arange(n_avail), N_SEG)
    mels = []
    for si in idxs:
        seg = y[si * SEG_SAMPLES:(si + 1) * SEG_SAMPLES]
        if len(seg) < SEG_SAMPLES:
            seg = np.pad(seg, (0, SEG_SAMPLES - len(seg)))
        m = librosa.feature.melspectrogram(
            y=seg.astype(np.float32), sr=SAMPLE_RATE,
            n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
        m = librosa.power_to_db(m, ref=np.max)
        m = (m - m.min()) / (m.max() - m.min() + 1e-9)
        mels.append(m.astype(np.float32))
    np.save(out, np.stack(mels))
    return "ok"

def main():
    df = pd.read_csv(TRAIN_CSV)
    files = df["filename"].dropna().unique().tolist()
    print(f"Recordings to process: {len(files)}")

    ok = skip = err = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(process_one, f): f for f in files}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r == "ok": ok += 1
            elif r == "skip": skip += 1
            else:
                err += 1
                if err <= 20: print(r)
            if i % 1000 == 0:
                print(f"  Progress {i}/{len(files)}  ok={ok} skip={skip} err={err}")
    print(f"\nDone: ok={ok} skip={skip} err={err}")
    print(f"Cache directory: {CACHE_DIR}")

if __name__ == "__main__":
    main()
