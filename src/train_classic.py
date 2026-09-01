"""
Group A - Sub-experiment 1: BirdNET embedding + classic machine learning algorithms
Includes two models, Logistic Regression (SGD) and XGBoost,
sharing split_utils's compliant split and eval_recording's recording-level evaluation.

Depends on files in the same directory:
  - split_utils.py
  - eval_recording.py
"""

import time
import numpy as np
import pyarrow.dataset as ds

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from xgboost import XGBClassifier

from src.split_utils import compliant_split
from src.eval_recording import evaluate_recording_level

# =====================================================
# CONFIG
# =====================================================

PARQUET_DIR = r"D:\bird\merged_parquet"   # change this to your own path

# =====================================================
# LOAD  (low-memory arrow -> numpy direct conversion)
# =====================================================

dataset = ds.dataset(PARQUET_DIR, format="parquet")
schema_names = dataset.schema.names
feature_cols = [c for c in schema_names if c.startswith("f_")]
needed_cols = feature_cols + ["label", "filename"]

print("Num parquet:", len(dataset.files))
print("Loading", len(needed_cols), "columns ...")

table = dataset.to_table(columns=needed_cols)
print("Rows:", table.num_rows)

t0 = time.time()
X = np.empty((table.num_rows, len(feature_cols)), dtype=np.float32)
for j, c in enumerate(feature_cols):
    X[:, j] = table.column(c).to_numpy(zero_copy_only=False)
print(f"Built X {X.shape} in {time.time()-t0:.1f}s")

labels = np.asarray(table.column("label").to_numpy(zero_copy_only=False))
filenames = np.asarray(table.column("filename").to_numpy(zero_copy_only=False))
del table

le = LabelEncoder()
y = le.fit_transform(labels).astype(np.int64)
num_class = len(le.classes_)
print("Num classes:", num_class)

# =====================================================
# COMPLIANT SPLIT  (single-sample classes are forced into the val set + split 80/20 by filename)
# =====================================================

train_idx, val_idx = compliant_split(
    y, filenames, test_size=0.2, random_state=42, min_train_count=2,
)
X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]
val_filenames = filenames[val_idx]    # needed for recording-level aggregation, in the same order as X_val
print("Train:", X_train.shape, " Val:", X_val.shape)

# Class weights for the training set (to handle imbalance); single-sample classes aren't in the
# training set, so they're automatically ignored
sample_weight = compute_sample_weight("balanced", y=y_train)

# Collect results from each model, then print a comparison table at the end
summary = {}


def report_both_levels(name, prob_val):
    """Print segment-level + recording-level metrics, and store them in summary"""
    pred = prob_val.argmax(1)
    seg_acc = accuracy_score(y_val, pred)
    seg_macro = f1_score(y_val, pred, average="macro", zero_division=0)
    seg_weighted = f1_score(y_val, pred, average="weighted", zero_division=0)
    print(f"\n----- {name} segment-level -----")
    print(f"  Accuracy {seg_acc:.4f} | Macro F1 {seg_macro:.4f} | "
          f"Weighted F1 {seg_weighted:.4f}")

    rec_results, _ = evaluate_recording_level(
        prob_val, val_filenames, y_val, num_class, class_names=le.classes_
    )
    summary[name] = {
        "seg_acc": seg_acc, "seg_macro": seg_macro,
        "rec_mean_acc": rec_results["mean"]["acc"],
        "rec_mean_macro": rec_results["mean"]["macro_f1"],
        "rec_max_acc": rec_results["max"]["acc"],
        "rec_max_macro": rec_results["max"]["macro_f1"],
    }


# =====================================================
# MODEL 1: Logistic Regression (SGD-optimized, suited to large data)
# =====================================================

print("\n" + "#" * 55)
print("# Logistic Regression (SGD)")
print("#" * 55)

t0 = time.time()
logreg = SGDClassifier(
    loss="log_loss",            # = logistic regression
    class_weight="balanced",
    max_iter=50,
    tol=1e-3,
    n_jobs=4,                   # don't use -1, to avoid the earlier memory issue
    random_state=42,
)
logreg.fit(X_train, y_train)
print(f"trained in {time.time()-t0:.1f}s")

prob_lr = logreg.predict_proba(X_val)
report_both_levels("LogReg", prob_lr)
import gc; 
gc.collect()
# =====================================================
# MODEL 2: XGBoost
# =====================================================

print("\n" + "#" * 55)
print("# XGBoost")
print("#" * 55)

xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=num_class,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=5.0,
    reg_alpha=1.0,
    gamma=0.3,
    max_bin=128,
    tree_method="hist",
    device="cuda",              # if you don't have a GPU, change to "cpu"
    # device="cpu",
    # n_jobs=8,
    eval_metric="mlogloss",
    early_stopping_rounds=30,
    random_state=42,
)

t0 = time.time()
xgb.fit(
    X_train, y_train,
    sample_weight=sample_weight,
    eval_set=[(X_val, y_val)],
    verbose=True,
)
print(f"trained in {time.time()-t0:.1f}s, best_iter={xgb.best_iteration}")

prob_xgb = xgb.predict_proba(X_val)
report_both_levels("XGBoost", prob_xgb)

# =====================================================
# Comparison table
# =====================================================

print("\n" + "=" * 70)
print("Group A Sub-experiment 1 results comparison (segment-level vs recording-level aggregation)")
print("=" * 70)
header = f"{'Model':<10} | {'seg Acc':>8} {'seg MF1':>8} | " \
         f"{'rec(mean) Acc':>13} {'MF1':>7} | {'rec(max) Acc':>12} {'MF1':>7}"
print(header)
print("-" * len(header))
for name, r in summary.items():
    print(f"{name:<10} | {r['seg_acc']:>8.4f} {r['seg_macro']:>8.4f} | "
          f"{r['rec_mean_acc']:>13.4f} {r['rec_mean_macro']:>7.4f} | "
          f"{r['rec_max_acc']:>12.4f} {r['rec_max_macro']:>7.4f}")

print("\nDONE")