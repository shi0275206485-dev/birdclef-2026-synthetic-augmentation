"""
Group B - extract BirdNET embeddings directly via tflite (bypassing a birdnetlib
version bug). Uses the same model file as the GUI, ensuring embeddings are
consistent with the original data.

BirdNET V2.4 spec:
  Input: 3 seconds @ 48kHz = 144000 samples
  Embedding taken from the second-to-last layer (output_index - 1), 1024-dim

Environment: pip install tensorflow librosa (or tflite_runtime)
"""

import os
import glob
import numpy as np
import pandas as pd
import librosa

# tflite: prefer the lightweight runtime, otherwise fall back to tensorflow.lite
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow import lite as tflite

# =====================================================
# CONFIG
# =====================================================

MODEL_PATH = r"D:\BirdNET-Analyzer\_internal\birdnet_analyzer\checkpoints\V2.4\BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"

SYNTH_DIR  = r"D:\bird\synth_data"
MANIFEST   = os.path.join(SYNTH_DIR, "synth_manifest.csv")
OUT_NPZ    = r"D:\bird\synth_embeddings.npz"

# for verification
ORIG_OGG_DIR = r"D:\bird\train_audio"
PARQUET_DIR  = r"D:\bird\merged_parquet"
DO_VERIFY    = True

SAMPLE_RATE = 48000
SEG_SAMPLES = 144000          # 3 seconds @ 48kHz

# =====================================================
# load tflite model
# =====================================================

print("Loading BirdNET tflite model ...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
interpreter.allocate_tensors()

print("Loading BirdNET tflite model ...")
interpreter = tflite.Interpreter(
    model_path=MODEL_PATH,
    num_threads=4,
    experimental_preserve_all_tensors=True,   # key: preserve all intermediate tensors, otherwise GLOBAL_AVG_POOL cannot be retrieved
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
INPUT_IDX = input_details["index"]
print(f"Input layer index={INPUT_IDX}, shape={input_details['shape']}")

# embedding layer = GLOBAL_AVG_POOL/Mean, index=545 (confirmed by probing)
EMBED_IDX = 545


def extract_embeddings(audio_path):
    """Extract BirdNET embeddings, returns (n_segments, 1024)"""
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    embs = []
    for start in range(0, len(y), SEG_SAMPLES):
        chunk = y[start:start + SEG_SAMPLES]
        if len(chunk) < SEG_SAMPLES:
            if len(chunk) < SEG_SAMPLES * 0.5:
                continue
            chunk = np.pad(chunk, (0, SEG_SAMPLES - len(chunk)))
        sig = chunk.astype(np.float32).reshape(1, -1)
        interpreter.resize_tensor_input(INPUT_IDX, sig.shape)
        interpreter.allocate_tensors()
        interpreter.set_tensor(INPUT_IDX, sig)
        interpreter.invoke()
        emb = interpreter.get_tensor(EMBED_IDX)[0].astype(np.float32)
        embs.append(emb)
    if not embs:
        return np.zeros((0, 1024), dtype=np.float32)
    return np.vstack(embs)

# =====================================================
# 1. consistency verification
# =====================================================

if DO_VERIFY:
    print("\n" + "=" * 60)
    print("Consistency check: tflite script extraction vs parquet (GUI extraction)")
    print("=" * 60)
    import pyarrow.dataset as ds

    dataset = ds.dataset(PARQUET_DIR, format="parquet")
    fcols = [c for c in dataset.schema.names if c.startswith("f_")]
    tbl = dataset.head(3000).to_pandas()

    sample_oggs = glob.glob(os.path.join(ORIG_OGG_DIR, "*", "*.ogg"))[:5]
    if not sample_oggs:
        print("Warning: no original ogg files found, skipping verification. Check ORIG_OGG_DIR.")
    else:
        for ogg in sample_oggs:
            stem = os.path.basename(ogg).replace(".ogg", "")
            match = tbl[tbl["filename"].astype(str).str.contains(stem, na=False)]
            if len(match) == 0:
                continue
            parquet_emb = match[fcols].values[0].astype(np.float32)
            script_emb = extract_embeddings(ogg)
            if len(script_emb) == 0:
                continue
            cos = np.dot(parquet_emb, script_emb[0]) / (
                np.linalg.norm(parquet_emb) * np.linalg.norm(script_emb[0]) + 1e-9)
            diff = np.abs(parquet_emb - script_emb[0]).mean()
            flag = "OK match" if cos > 0.95 else "WARNING large difference"
            print(f"  {stem[:30]:<32} cos={cos:.4f} diff={diff:.4f}  {flag}")

    print("\nIf cos>0.95 -> consistent with GUI, safe to proceed with extracting synthetic data.")
    input("Press Enter to continue extracting synthetic audio embeddings (Ctrl+C to abort and investigate)...")

# =====================================================
# 2. extract synthetic audio embeddings
# =====================================================

print("\n" + "=" * 60)
print("Extracting synthetic audio embeddings")
print("=" * 60)

man = pd.read_csv(MANIFEST)
print(f"Number of synthetic audio entries: {len(man)}")

all_emb, all_label, all_file = [], [], []
for _, row in man.iterrows():
    wav = os.path.join(SYNTH_DIR, row["file"])
    if not os.path.exists(wav):
        continue
    embs = extract_embeddings(wav)
    for seg in embs:
        all_emb.append(seg)
        all_label.append(row["target_sci"])
        all_file.append(row["file"])
    print(f"  {row['file']}: {len(embs)} segments -> {row['target_sci']}")

X_synth = np.vstack(all_emb) if all_emb else np.zeros((0, 1024), np.float32)
y_synth = np.array(all_label)
f_synth = np.array(all_file)

print(f"\nTotal synthetic embedding segments: {len(X_synth)}, number of classes: {len(np.unique(y_synth))}")

np.savez(OUT_NPZ, X=X_synth, y=y_synth, filename=f_synth)
print(f"Saved -> {OUT_NPZ}")
print("Next step: add the synthetic embeddings to the training set only, rerun the Group A model, and compare B vs A")