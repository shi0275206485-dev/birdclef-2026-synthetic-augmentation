"""
BirdNET embedding (1024-d) -> MLP classifier
Reuses the original XGBoost script's data loading + GroupShuffleSplit to prevent leakage.
Designed for ReLU embeddings: standard 2-layer MLP + BatchNorm + Dropout + class-weighted loss.
"""

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    top_k_accuracy_score,
    classification_report,
)

# =====================================================
# CONFIG
# =====================================================

PARQUET_DIR = r"D:\bird\merged_parquet"

BATCH_SIZE   = 1024          # embeddings are lightweight; bump to 2048/4096 if you have VRAM to spare
EPOCHS       = 100           # upper bound, cut short by early stopping
LR           = 1e-3
WEIGHT_DECAY = 1e-4          # L2 regularization
DROPOUT      = 0.3
PATIENCE     = 10            # early stopping: stop if val macro-F1 hasn't improved for N consecutive epochs
HIDDEN       = [512, 256]    # hidden layer dimensions

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# =====================================================
# LOAD DATA  (same as the original script)
# =====================================================

dataset = ds.dataset(PARQUET_DIR, format="parquet")
schema_names = dataset.schema.names
feature_cols = [c for c in schema_names if c.startswith("f_")]
needed_cols = feature_cols + ["label", "filename"]

print("Num parquet:", len(dataset.files))
print("Loading", len(needed_cols), "columns ...")

table = dataset.to_table(columns=needed_cols)
print("Loaded table rows:", table.num_rows)

# Feature matrix: convert column by column arrow -> numpy directly, float32, avoiding the
# Python list bloat of to_pydict
import time
t0 = time.time()
X = np.empty((table.num_rows, len(feature_cols)), dtype=np.float32)
for j, c in enumerate(feature_cols):
    X[:, j] = table.column(c).to_numpy(zero_copy_only=False)
print(f"Built X {X.shape} in {time.time()-t0:.1f}s")

# Labels and grouping column: pulled out separately, without keeping the whole DataFrame
labels = table.column("label").to_numpy(zero_copy_only=False)
filenames = table.column("filename").to_numpy(zero_copy_only=False)
del table   # release the arrow table as early as possible

# Label encoding
le = LabelEncoder()
y = le.fit_transform(labels).astype(np.int64)
num_class = len(le.classes_)
print("Num classes:", num_class, " Feature dim:", X.shape[1])

# =====================================================
# COMPLIANT SPLIT  (single-sample classes are forced into the val set + the rest split 80/20 by filename)
# Assignment requirement: classes with only 1 sample always go into the validation set
# =====================================================

from src.split_utils import compliant_split

train_idx, val_idx = compliant_split(
    y, filenames,
    test_size=0.2, random_state=42, min_train_count=2,
)

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]
val_filenames = filenames[val_idx]   # recording each validation segment belongs to, in the same order as X_val
print("Train:", X_train.shape, " Val:", X_val.shape)

# =====================================================
# CLASS WEIGHTS  (helps macro F1 for small classes)
# Note: single-sample classes are only in the validation set, not the training set,
# so missing classes need a default weight filled in
# =====================================================

present = np.unique(y_train)
cw_present = compute_class_weight("balanced", classes=present, y=y_train)
# default weight 1.0; classes present in the training set use the computed weight instead
cw = np.ones(num_class, dtype=np.float32)
cw[present] = cw_present
class_weights = torch.tensor(cw, dtype=torch.float32, device=DEVICE)

# =====================================================
# DATALOADERS
# =====================================================

train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=(DEVICE == "cuda"))
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=(DEVICE == "cuda"))

# =====================================================
# MODEL
# =====================================================

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, n_class, dropout):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, n_class))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = MLP(X.shape[1], HIDDEN, num_class, DROPOUT).to(DEVICE)
print(model)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3
)

# =====================================================
# TRAIN LOOP  (with early stopping, based on val macro-F1)
# =====================================================

@torch.no_grad()
def evaluate(loader):
    model.eval()
    all_logits, all_y = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        all_logits.append(model(xb).cpu())
        all_y.append(yb)
    logits = torch.cat(all_logits)
    yt = torch.cat(all_y).numpy()
    prob = torch.softmax(logits, dim=1).numpy()
    pred = prob.argmax(1)
    return yt, pred, prob

history = {"train_loss": [], "val_macro_f1": [], "val_acc": []}
best_f1, best_state, wait = -1.0, None, 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)
    train_loss = running / len(train_ds)

    yt, pred, _ = evaluate(val_loader)
    val_acc = accuracy_score(yt, pred)
    val_f1 = f1_score(yt, pred, average="macro")
    scheduler.step(val_f1)

    history["train_loss"].append(train_loss)
    history["val_macro_f1"].append(val_f1)
    history["val_acc"].append(val_acc)

    lr_now = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch:3d} | train_loss {train_loss:.4f} | "
          f"val_acc {val_acc:.4f} | val_macroF1 {val_f1:.4f} | lr {lr_now:.2e}")

    # Early stopping: keep the weights with the best macro-F1
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (best macroF1={best_f1:.4f})")
            break

# Load the best weights
model.load_state_dict(best_state)

# =====================================================
# FINAL EVAL
# =====================================================

yt, pred, prob = evaluate(val_loader)
acc = accuracy_score(yt, pred)
macro_f1 = f1_score(yt, pred, average="macro")
weighted_f1 = f1_score(yt, pred, average="weighted")
k = min(5, num_class)
top5 = top_k_accuracy_score(yt, prob, k=k, labels=np.arange(num_class))

print("\n========== SEGMENT-LEVEL METRICS (segment-level, best model) ==========")
print("Accuracy:    ", round(acc, 4))
print("Macro F1:    ", round(macro_f1, 4))
print("Weighted F1: ", round(weighted_f1, 4))
print(f"Top-{k} Accuracy:", round(top5, 4))

# ---- Recording-level aggregated evaluation ----
from src.eval_recording import evaluate_recording_level
rec_results, _ = evaluate_recording_level(
    prob, val_filenames, yt, num_class, class_names=le.classes_
)

print("\n========== CLASSIFICATION REPORT ==========\n")
print(classification_report(
    yt, pred,
    labels=np.arange(num_class),
    target_names=le.classes_,
    zero_division=0,
))

# =====================================================
# CURVES
# =====================================================

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(history["train_loss"], color="tab:blue", label="train loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Train Loss", color="tab:blue")
ax2 = ax1.twinx()
ax2.plot(history["val_macro_f1"], color="tab:orange", label="val macro F1")
ax2.plot(history["val_acc"], color="tab:green", label="val acc")
ax2.set_ylabel("Val metric")
fig.legend(loc="upper right")
plt.title("MLP Training Curve")
plt.tight_layout()
plt.savefig("mlp_training_curve.png", dpi=120)
plt.show()

# =====================================================
# SAVE
# =====================================================

torch.save({
    "model_state": model.state_dict(),
    "classes": le.classes_,
    "config": {"hidden": HIDDEN, "in_dim": X.shape[1],
               "num_class": num_class, "dropout": DROPOUT},
}, "birdnet_mlp.pt")
print("\nSaved birdnet_mlp.pt — DONE")