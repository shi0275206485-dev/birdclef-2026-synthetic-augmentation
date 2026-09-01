"""
Option B - Sub-experiment 1: 8670 synthetic embeddings (relabel labels) merged into training, LogReg B vs A.
Comparison baseline = A (real data only). Presented alongside the results from the "strict QC 50 samples" version, in response to the professor's announcement.

Dependencies: split_utils.py, eval_recording.py
"""
import time
import numpy as np
import pyarrow.dataset as ds
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from src.split_utils import compliant_split
from src.eval_recording import evaluate_recording_level, aggregate_to_recording

PARQUET_DIR = r"D:\bird\merged_parquet"
SYNTH_NPZ   = r"G:\bird\synth_embeddings_optionB.npz"

print("Loading raw embeddings ...")
dataset = ds.dataset(PARQUET_DIR, format="parquet")
fcols = [c for c in dataset.schema.names if c.startswith("f_")]
table = dataset.to_table(columns=fcols+["label","filename"])
X = np.empty((table.num_rows, len(fcols)), dtype=np.float32)
for j,c in enumerate(fcols):
    X[:,j] = table.column(c).to_numpy(zero_copy_only=False)
labels = np.asarray(table.column("label").to_numpy(zero_copy_only=False))
filenames = np.asarray(table.column("filename").to_numpy(zero_copy_only=False))
del table

le = LabelEncoder()
y = le.fit_transform(labels).astype(np.int64)
num_class = len(le.classes_)
print(f"Raw: {X.shape}, classes {num_class}")

train_idx, val_idx = compliant_split(y, filenames, test_size=0.2, random_state=42, min_train_count=2)
X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]
val_filenames = filenames[val_idx]
print(f"Real training {X_train.shape}, validation {X_val.shape}")

# Option B synthetic data (relabel labels)
synth = np.load(SYNTH_NPZ, allow_pickle=True)
Xs, ys_raw = synth["X"].astype(np.float32), synth["y"]
mask = np.isin(ys_raw, le.classes_)      # keep only labels within the 206 classes
Xs, ys_raw = Xs[mask], ys_raw[mask]
ys = le.transform(ys_raw).astype(np.int64)
print(f"Option B synthetic: {Xs.shape}, covering {len(np.unique(ys))} labels")
import collections
top = collections.Counter(ys_raw).most_common(5)
print("Top 5 most frequent relabeled labels:", [(le.classes_ if False else t[0], t[1]) for t in top])

X_train_B = np.vstack([X_train, Xs])
y_train_B = np.concatenate([y_train, ys])
print(f"Group B training set: {X_train_B.shape} (real {len(X_train)} + synthetic {len(Xs)})")

def run(Xtr, ytr, tag):
    t0=time.time()
    m = SGDClassifier(loss="log_loss", class_weight="balanced",
                      max_iter=50, tol=1e-3, n_jobs=4, random_state=42)
    m.fit(Xtr, ytr)
    prob = m.predict_proba(X_val)
    rec,_ = evaluate_recording_level(prob, val_filenames, y_val, num_class,
                                     class_names=le.classes_, verbose=False)
    print(f"[{tag}] rec Acc {rec['mean']['acc']:.4f} MacroF1 {rec['mean']['macro_f1']:.4f} ({time.time()-t0:.0f}s)")
    return rec['mean']['acc'], rec['mean']['macro_f1']

# Group A has already been run (fixed seed, fixed data, deterministic result); use the known baseline directly instead of re-running
aA, fA = 0.8690, 0.8002

print("\n===== B (real + optionB synth) =====")
aB, fB = run(X_train_B, y_train_B, "B")

print("\n"+"="*55)
print("Option B  LogReg  B vs A (recording level)")
print("="*55)
print(f"A (real only, known baseline)   Acc {aA:.4f}  MacroF1 {fA:.4f}")
print(f"B (real+optionB)          Acc {aB:.4f}  MacroF1 {fB:.4f}")
print(f"Δ                         Acc {aB-aA:+.4f}  MacroF1 {fB-fA:+.4f}")
