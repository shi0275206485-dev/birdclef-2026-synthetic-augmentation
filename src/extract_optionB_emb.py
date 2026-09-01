"""
Option B - Extract BirdNET embeddings for 8670 synthetic audio clips (reusing the validated tflite method).
Uses relabel (the Perch top-scoring label) as the label, saved as npz for merging into training.

Reuses the core of extract_synth_emb_tflite.py: tflite + preserve_all_tensors + EMBED_IDX=545
8670 clips takes longer than 50 (about 3-5 hours), with progress printing.
"""

import os, glob
import numpy as np
import pandas as pd
import librosa

from tensorflow import lite as tflite

MODEL_PATH = r"D:\BirdNET-Analyzer\_internal\birdnet_analyzer\checkpoints\V2.4\BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"
SYNTH_DIR  = r"G:\bird\synth_data_optionB"
MANIFEST   = os.path.join(SYNTH_DIR, "synth_manifest_optionB.csv")
OUT_NPZ    = r"G:\bird\synth_embeddings_optionB.npz"

SAMPLE_RATE = 48000
SEG_SAMPLES = 144000          # 3 seconds @ 48kHz
EMBED_IDX   = 545             # GLOBAL_AVG_POOL/Mean (validated)

print("Loading BirdNET tflite ...")
interp = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4,
                            experimental_preserve_all_tensors=True)
interp.allocate_tensors()
INPUT_IDX = interp.get_input_details()[0]["index"]

def extract(path):
    try:
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    except Exception:
        return np.zeros((0,1024), np.float32)
    embs = []
    for s in range(0, len(y), SEG_SAMPLES):
        ch = y[s:s+SEG_SAMPLES]
        if len(ch) < SEG_SAMPLES:
            if len(ch) < SEG_SAMPLES*0.5: continue
            ch = np.pad(ch, (0, SEG_SAMPLES-len(ch)))
        sig = ch.astype(np.float32).reshape(1,-1)
        interp.resize_tensor_input(INPUT_IDX, sig.shape)
        interp.allocate_tensors()
        interp.set_tensor(INPUT_IDX, sig)
        interp.invoke()
        embs.append(interp.get_tensor(EMBED_IDX)[0].astype(np.float32))
    return np.vstack(embs) if embs else np.zeros((0,1024), np.float32)

def main():
    man = pd.read_csv(MANIFEST)
    print(f"To extract: {len(man)} entries")
    X, Y, F = [], [], []
    for i, row in man.iterrows():
        wav = os.path.join(SYNTH_DIR, row["file"])
        if not os.path.exists(wav): continue
        embs = extract(wav)
        for e in embs:
            X.append(e)
            Y.append(row["relabel"])      # key: use relabel (Perch top score) as the label
            F.append(row["file"])
        if (i+1) % 200 == 0:
            print(f"  {i+1}/{len(man)} extracted, cumulative segments {len(X)}")
    X = np.vstack(X) if X else np.zeros((0,1024), np.float32)
    np.savez(OUT_NPZ, X=X, y=np.array(Y), filename=np.array(F))
    print(f"\nDone: {len(X)} segments, {len(set(Y))} labels -> {OUT_NPZ}")

if __name__ == "__main__":
    main()
