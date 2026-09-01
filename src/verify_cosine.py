"""
Verify the correctness of the tflite extraction method:
For the same batch of [raw audio], extract embeddings with tflite,
and compare cosine similarity against the embeddings already present
for these audio files in the parquet.
High similarity = the tflite extraction method matches the standard method.

After running, paste the output to the assistant and update the report accordingly.
"""
import os, glob
import numpy as np
import pandas as pd
import librosa
import pyarrow.dataset as ds

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow import lite as tflite

MODEL_PATH  = r"D:\BirdNET-Analyzer\_internal\birdnet_analyzer\checkpoints\V2.4\BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"
AUDIO_ROOT  = r"D:\bird\train_audio"
PARQUET_DIR = r"D:\bird\merged_parquet"   # change to your parquet directory
N_CHECK     = 10        # number of recordings to verify

SAMPLE_RATE = 48000
SEG_SAMPLES = 144000     # 3 seconds @ 48kHz
EMBED_IDX   = 545

print("Loading tflite ...")
interp = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4,
                            experimental_preserve_all_tensors=True)
interp.allocate_tensors()
INPUT_IDX = interp.get_input_details()[0]["index"]

def tflite_embed_first_seg(wav):
    y,_ = librosa.load(wav, sr=SAMPLE_RATE, mono=True)
    seg = y[:SEG_SAMPLES]
    if len(seg) < SEG_SAMPLES:
        seg = np.pad(seg,(0,SEG_SAMPLES-len(seg)))
    sig = seg.astype(np.float32).reshape(1,-1)
    interp.resize_tensor_input(INPUT_IDX, sig.shape)
    interp.allocate_tensors()
    interp.set_tensor(INPUT_IDX, sig)
    interp.invoke()
    return interp.get_tensor(EMBED_IDX)[0].astype(np.float32)

def cosine(a,b):
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))

# Read parquet: get filename + embedding, pick the first segment of the first N_CHECK distinct recordings
print("Reading parquet ...")
dataset = ds.dataset(PARQUET_DIR, format="parquet")
fcols = [c for c in dataset.schema.names if c.startswith("f_")]
tab = dataset.to_table(columns=fcols+["filename"]).to_pandas()
# Take the first row (first segment) of each recording
first = tab.groupby("filename", as_index=False).first()
sample = first.head(N_CHECK)

print(f"\nVerifying the first segment of {len(sample)} recordings:\n")
sims = []
for _, row in sample.iterrows():
    fn = row["filename"]
    wav = os.path.join(AUDIO_ROOT, fn.replace("/", os.sep))
    if not os.path.exists(wav):
        # try a direct search
        cand = glob.glob(os.path.join(AUDIO_ROOT, "**", os.path.basename(fn)), recursive=True)
        if cand: wav = cand[0]
        else: print(f"  skipped (audio not found): {fn}"); continue
    parquet_emb = row[fcols].to_numpy(dtype=np.float32)
    tflite_emb = tflite_embed_first_seg(wav)
    c = cosine(parquet_emb, tflite_emb)
    sims.append(c)
    print(f"  {fn}: cosine = {c:.4f}")

if sims:
    print(f"\n=== Results ===")
    print(f"Cosine similarity range: {min(sims):.4f} ~ {max(sims):.4f}")
    print(f"Mean: {np.mean(sims):.4f}")
    print(f"\nFor the report: tflite extraction vs standard extraction cosine similarity on {len(sims)} audio files: {min(sims):.2f}-{max(sims):.3f}")
