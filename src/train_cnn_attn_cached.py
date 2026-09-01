"""
Group A - Sub-experiment 2: CNN + Attention Pooling, trained from mel spectrograms
pre-cached on the G drive. Run precompute_melspec.py first to generate the cache.

Reading from cache is much faster than computing spectrograms on the fly,
and avoids recomputation every epoch.
Windows + GPU safe.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

TRAIN_CSV  = r"D:\bird\train.csv"
CACHE_DIR  = r"G:\bird_melcache"

N_SEG       = 6
BATCH_SIZE  = 32        # lower to 8 on OOM
EPOCHS      = 100       # raised cap, let early stopping decide when to stop (previously cut off at 30)
LR          = 1e-2
PATIENCE    = 15        # more patience, to avoid early stopping triggered by LR-decay fluctuations
NUM_WORKERS = 6
DROP_OUT = 0.4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def cache_path(fname):
    safe = fname.replace("/", "__").replace("\\", "__").replace(".ogg", ".npy")
    return os.path.join(CACHE_DIR, safe)

class CachedDataset(Dataset):
    def __init__(self, df, train=True):
        self.train = train
        self.items = [(cache_path(r["filename"]), r["y"]) for _, r in df.iterrows()]
        # keep only entries whose cache file exists
        self.items = [(p, y) for p, y in self.items if os.path.exists(p)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        mels = np.load(path)               # (N_SEG, n_mels, time)
        if self.train and mels.shape[0] >= N_SEG:
            # training augmentation: shuffle segment order (attention is order-insensitive, adds diversity)
            perm = np.random.permutation(mels.shape[0])[:N_SEG]
            mels = mels[perm]
        return torch.from_numpy(mels.astype(np.float32)), y

class CNNAttn(nn.Module):
    def __init__(self, n_class, feat_dim=128):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2))
        self.cnn = nn.Sequential(
            block(1, 32), block(32, 64), block(64, 128), block(128, feat_dim),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.Tanh(), nn.Linear(64, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(DROP_OUT), nn.Linear(feat_dim, n_class))

    def forward(self, x):
        B, N, M, T = x.shape
        x = x.view(B * N, 1, M, T)
        feats = self.cnn(x).view(B, N, -1)
        weights = torch.softmax(self.attn(feats), dim=1)
        pooled = (feats * weights).sum(dim=1)
        return self.classifier(pooled), weights

def main():
    print("Device:", DEVICE)
    df = pd.read_csv(TRAIN_CSV)
    df = df[["filename", "scientific_name"]].dropna()
    le = LabelEncoder()
    df["y"] = le.fit_transform(df["scientific_name"])
    num_class = len(le.classes_)
    print("Number of classes:", num_class, " Number of recordings:", len(df))

    # Split: single-sample classes (recording-level count<2) are forced into the validation set
    # (assignment requirement); the rest are split 80/20 by filename
    y_arr = df["y"].values
    classes, cnts = np.unique(y_arr, return_counts=True)
    rare = set(classes[cnts < 2].tolist())     # classes with only 1 recording
    rare_mask = df["y"].isin(rare).values
    df_rare = df[rare_mask]                      # forced into val
    df_rest = df[~rare_mask].reset_index(drop=True)

    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    tr_i, va_i = next(gss.split(df_rest, df_rest["y"], groups=df_rest["filename"]))
    df_train = df_rest.iloc[tr_i].reset_index(drop=True)
    df_val = pd.concat([df_rest.iloc[va_i], df_rare], ignore_index=True)
    print(f"Single-sample classes (forced into validation set): {len(rare)} classes, {len(df_rare)} recordings")
    print(f"Classes covered by training set: {df_train['y'].nunique()}/{num_class}")

    train_ds = CachedDataset(df_train, train=True)
    val_ds   = CachedDataset(df_val, train=False)
    print(f"Train {len(train_ds)}, Val {len(val_ds)} (with cache available)")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = CNNAttn(num_class).to(DEVICE)
    counts = np.bincount(df_train["y"], minlength=num_class)
    w = 1.0 / np.sqrt(counts + 1); w = w / w.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(w, dtype=torch.float32, device=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2)

    def evaluate():
        model.eval()
        ys, ps = [], []
        loss_sum, n = 0.0, 0
        with torch.no_grad():
            for mels, y in val_loader:
                mels = mels.to(DEVICE); yd = y.to(DEVICE)
                logits, _ = model(mels)
                loss_sum += criterion(logits, yd).item() * mels.size(0)
                n += mels.size(0)
                ps.append(logits.argmax(1).cpu().numpy()); ys.append(y.numpy())
        yt = np.concatenate(ys); yp = np.concatenate(ps)
        val_loss = loss_sum / n
        acc = accuracy_score(yt, yp)
        macro = f1_score(yt, yp, average="macro", zero_division=0)
        return acc, macro, val_loss, yt, yp

    best, best_state, wait = -1, None, 0
    hist = {"train_loss": [], "val_loss": [], "val_acc": [], "val_macro": []}
    for ep in range(1, EPOCHS + 1):
        model.train(); running = 0.0
        for mels, y in train_loader:
            mels, y = mels.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(mels)
            loss = criterion(logits, y)
            loss.backward(); optimizer.step()
            running += loss.item() * mels.size(0)
        tl = running / len(train_ds)
        acc, macro, vl, _, _ = evaluate(); scheduler.step(macro)
        hist["train_loss"].append(tl); hist["val_loss"].append(vl)
        hist["val_acc"].append(acc); hist["val_macro"].append(macro)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {ep:2d} | lr {current_lr:.2e} | train_loss {tl:.4f} | val_loss {vl:.4f} | "
              f"rec Acc {acc:.4f} | rec MacroF1 {macro:.4f}")
        if macro > best:
            best, wait = macro, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stop at epoch {ep}"); break

    model.load_state_dict(best_state)
    acc, macro, vl, yt, yp = evaluate()
    print(f"\n==== CNN+Attention (from scratch) final recording-level ====")
    print(f"Accuracy: {acc:.4f}  Macro-F1: {macro:.4f}")
    np.savez("cnn_attn_history.npz", **hist)
    torch.save({"model_state": model.state_dict(), "classes": le.classes_},
               "birdnet_cnn_attn.pt")

    # ===== dual loss curves + validation metrics =====
    import matplotlib.pyplot as plt
    ep_range = range(1, len(hist["train_loss"]) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(ep_range, hist["train_loss"], label="train loss")
    plt.plot(ep_range, hist["val_loss"], label="val loss")
    plt.xlabel("Epoch"); plt.ylabel("Cross-Entropy Loss")
    plt.title("CNN+Attention Loss Curve"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("cnn_loss_curve.png", dpi=120); plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(ep_range, hist["val_acc"], label="val accuracy")
    plt.plot(ep_range, hist["val_macro"], label="val macro F1")
    plt.xlabel("Epoch"); plt.ylabel("Score")
    plt.title("CNN+Attention Validation Metrics"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("cnn_val_metrics.png", dpi=120); plt.close()

    # ===== confusion matrix (first 30 classes; otherwise 206 classes is too dense to read) =====
    from sklearn.metrics import confusion_matrix
    n_show = 30
    cm = confusion_matrix(yt, yp, labels=np.arange(num_class))
    plt.figure(figsize=(11, 9))
    plt.imshow(cm[:n_show, :n_show], cmap="Blues", aspect="auto")
    plt.colorbar(); plt.title(f"Confusion Matrix (first {n_show} classes)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig("cnn_confusion_matrix.png", dpi=120); plt.close()

    print("Saved: cnn_loss_curve.png, cnn_val_metrics.png, cnn_confusion_matrix.png")
    print("Saved birdnet_cnn_attn.pt and cnn_attn_history.npz")

if __name__ == "__main__":
    main()