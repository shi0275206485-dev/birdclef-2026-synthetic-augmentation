"""
Option B - sub-experiment 1: 8670 synthetic embeddings (relabel labels) merged into training, LogReg B vs A.
Comparison baseline = A (real only). Presented alongside the "strict QC 50 items" version's results, in response to the instructor's announcement.

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

# option B synthetic data (relabel labels)
synth = np.load(SYNTH_NPZ, allow_pickle=True)
Xs, ys_raw = synth["X"].astype(np.float32), synth["y"]
mask = np.isin(ys_raw, le.classes_)      # keep only labels within the 206 classes
Xs, ys_raw = Xs[mask], ys_raw[mask]
ys = le.transform(ys_raw).astype(np.int64)
print(f"Option B synthetic: {Xs.shape}, covering {len(np.unique(ys))} labels")
import collections
top = collections.Counter(ys_raw).most_common(5)
print("Top 5 relabeled labels by count:", [(le.classes_ if False else t[0], t[1]) for t in top])

X_train_B = np.vstack([X_train, Xs])
y_train_B = np.concatenate([y_train, ys])
print(f"Group B training set: {X_train_B.shape} (real {len(X_train)} + synthetic {len(Xs)})")

def run(Xtr, ytr, tag, return_pred=False):
    t0=time.time()
    m = SGDClassifier(loss="log_loss", class_weight="balanced",
                      max_iter=50, tol=1e-3, n_jobs=4, random_state=42)
    m.fit(Xtr, ytr)
    prob = m.predict_proba(X_val)
    rec,_ = evaluate_recording_level(prob, val_filenames, y_val, num_class,
                                     class_names=le.classes_, verbose=False)
    print(f"[{tag}] rec Acc {rec['mean']['acc']:.4f} MacroF1 {rec['mean']['macro_f1']:.4f} ({time.time()-t0:.0f}s)")
    if return_pred:
        return rec['mean']['acc'], rec['mean']['macro_f1'], prob
    return rec['mean']['acc'], rec['mean']['macro_f1']

# A must also be run this time (need per-class predictions for comparison), but only run once
from sklearn.metrics import f1_score
def per_class_f1(prob):
    rec_ids, y_rec, prob_mean, _ = aggregate_to_recording(prob, val_filenames, y_val, num_class)
    pred = prob_mean.argmax(1)
    f1s = f1_score(y_rec, pred, average=None, labels=np.arange(num_class), zero_division=0)
    return f1s, y_rec

print("\n===== A (real only) =====")
aA, fA, probA = run(X_train, y_train, "A", return_pred=True)
print("===== B (real + optionB synth) =====")
aB, fB, probB = run(X_train_B, y_train_B, "B", return_pred=True)

f1A, y_rec = per_class_f1(probA)
f1B, _     = per_class_f1(probB)

print("\n"+"="*55)
print("Option B  LogReg  B vs A (recording level)")
print("="*55)
print(f"A (real only)         Acc {aA:.4f}  MacroF1 {fA:.4f}")
print(f"B (real+optionB)      Acc {aB:.4f}  MacroF1 {fB:.4f}")
print(f"Delta                 Acc {aB-aA:+.4f}  MacroF1 {fB-fA:+.4f}")

# ===== per-class F1 change analysis =====
print("\n" + "="*55)
print("Classes with the largest per-class F1 change (B - A)")
print("="*55)
delta = f1B - f1A
order = np.argsort(delta)
# number of recordings per class in the validation set
import collections
val_counts = collections.Counter(y_rec.tolist())

print("\n--- Top 10 classes with the largest F1 increase ---")
for idx in order[::-1][:10]:
    nm = le.classes_[idx]
    print(f"  {nm:<32} A={f1A[idx]:.3f} B={f1B[idx]:.3f} Delta={delta[idx]:+.3f} (val recordings {val_counts.get(idx,0)})")

print("\n--- Top 10 classes with the largest F1 decrease ---")
for idx in order[:10]:
    nm = le.classes_[idx]
    print(f"  {nm:<32} A={f1A[idx]:.3f} B={f1B[idx]:.3f} Delta={delta[idx]:+.3f} (val recordings {val_counts.get(idx,0)})")

# specifically look at the sparrow
if "Passer domesticus" in le.classes_:
    pi = list(le.classes_).index("Passer domesticus")
    print(f"\n>>> Passer domesticus (sparrow, flooded with 12789 segments): "
          f"A={f1A[pi]:.3f} B={f1B[pi]:.3f} Delta={delta[pi]:+.3f} (val recordings {val_counts.get(pi,0)})")
