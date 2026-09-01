"""
Embedding data health check: check common issues in one pass
- NaN / Inf
- all-zero rows (extraction failure)
- duplicate rows (possible duplicate extraction or misalignment)
- sample count distribution per class
- whether labels and embeddings might be misaligned (rough check via nearest-neighbor consistency)
- whether the value distribution looks normal
"""

import numpy as np
import pyarrow.dataset as ds

PARQUET_DIR = r"D:\bird\merged_parquet"   # change this to your own path

dataset = ds.dataset(PARQUET_DIR, format="parquet")
schema_names = dataset.schema.names
feature_cols = [c for c in schema_names if c.startswith("f_")]
table = dataset.to_table(columns=feature_cols + ["label", "filename"])

X = np.empty((table.num_rows, len(feature_cols)), dtype=np.float32)
for j, c in enumerate(feature_cols):
    X[:, j] = table.column(c).to_numpy(zero_copy_only=False)
labels = np.asarray(table.column("label").to_numpy(zero_copy_only=False))
filenames = np.asarray(table.column("filename").to_numpy(zero_copy_only=False))
del table

print("="*55)
print("Shape:", X.shape, " Dim:", X.shape[1])
print("="*55)

# 1. NaN / Inf
n_nan = np.isnan(X).sum()
n_inf = np.isinf(X).sum()
print(f"\n[1] NaN count: {n_nan}   Inf count: {n_inf}")
if n_nan or n_inf:
    print("    Warning: dirty values present! Cleaning needed.")
    bad_rows = np.where(np.isnan(X).any(1) | np.isinf(X).any(1))[0]
    print(f"    Affected rows: {len(bad_rows)}")

# 2. All-zero rows
row_sum = np.abs(X).sum(1)
zero_rows = np.where(row_sum == 0)[0]
print(f"\n[2] All-zero row count: {len(zero_rows)} / {len(X)}  "
      f"({100*len(zero_rows)/len(X):.2f}%)")
if len(zero_rows):
    print("    Warning: these samples have all-zero embeddings, likely extraction failures:")
    for i in zero_rows[:10]:
        print(f"      row{i}: {filenames[i]}  label={labels[i]}")

# 3. Value distribution
print(f"\n[3] Values: min={X.min():.3f} max={X.max():.3f} "
      f"mean={X.mean():.3f} std={X.std():.3f}")
print(f"    Mean non-zero elements per row: {(X>0).sum(1).mean():.1f} / {X.shape[1]} "
      f"(ReLU embeddings are usually 30%-60% non-zero)")

# 4. Class distribution
classes, counts = np.unique(labels, return_counts=True)
print(f"\n[4] Number of classes: {len(classes)}")
order = np.argsort(counts)
print(f"    5 classes with fewest samples: ", [(classes[i], int(counts[i])) for i in order[:5]])
print(f"    5 classes with most samples: ", [(classes[i], int(counts[i])) for i in order[-5:]])
print(f"    Number of single-sample classes (count==1): {(counts==1).sum()}")

# 5. Duplicate row check (sampled, to avoid a slow full pairwise comparison)
print(f"\n[5] Sampled duplicate row check ...")
rng = np.random.default_rng(0)
sample = rng.choice(len(X), size=min(20000, len(X)), replace=False)
Xs = X[sample]
# Use hashing to roughly detect duplicates
hashes = [hash(row.tobytes()) for row in Xs]
n_dup = len(hashes) - len(set(hashes))
print(f"    Exact duplicate rows among {len(sample)} sampled rows: {n_dup}")

# 6. Rough label-misalignment check: use nearest neighbors to see whether
#    "same-class samples cluster together in feature space"
#    If embeddings and labels are misaligned, the same-class nearest-neighbor ratio
#    will be close to the random baseline
print(f"\n[6] Label-feature consistency check (nearest neighbors) ...")
try:
    from sklearn.neighbors import NearestNeighbors
    # Sample to speed things up
    s2 = rng.choice(len(X), size=min(5000, len(X)), replace=False)
    Xq = X[s2]
    yq = labels[s2]
    nn = NearestNeighbors(n_neighbors=2).fit(Xq)
    _, idx = nn.kneighbors(Xq)
    # idx[:,0] is itself, idx[:,1] is the nearest neighbor
    same = (yq[idx[:, 1]] == yq).mean()
    # Random baseline = weighted by 1/num_classes
    p = counts / counts.sum()
    random_baseline = (p**2).sum()
    print(f"    Same-class nearest-neighbor ratio: {same:.3f}")
    print(f"    Random baseline (should be close to this if fully misaligned): {random_baseline:.4f}")
    if same > random_baseline * 5:
        print("    OK: far above the random baseline -> labels and embeddings correspond correctly, no misalignment")
    else:
        print("    Warning: close to the random baseline -> possible label misalignment! Check the extraction pipeline")
except Exception as e:
    print("    Skipped (sklearn unavailable):", e)

print("\n" + "="*55)
print("Health check complete")
print("="*55)