"""
Group B sub-experiment 2: CNN+Attention from scratch, with 50 synthetic audio clips added to the training set.
Compare A (real only) vs B (real+synthetic), validation set is always pure real.

Uses the old spectrogram parameters (proven to better suit this architecture). Must first run:
  1. precompute_melspec.py, changed back to the old parameters, to generate G:\bird_melcache (can reuse the old cache if it already exists)
  2. precompute_synth_mel.py to generate G:\bird_melcache_synth
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

TRAIN_CSV    = r"D:\bird\train.csv"
CACHE_DIR    = r"G:\bird_melcache"          # real data, old-parameter cache
SYNTH_CACHE  = r"G:\bird_melcache_synth"    # synthetic data cache
SYNTH_INDEX  = os.path.join(SYNTH_CACHE, "synth_cache_index.csv")

N_SEG       = 6
BATCH_SIZE  = 32
EPOCHS      = 150
LR          = 1e-2
PATIENCE    = 25      # allow enough patience to ensure convergence (past experience: convergence needs ~88 epochs)
NUM_WORKERS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def cache_path(fname):
    return os.path.join(CACHE_DIR, fname.replace("/", "__").replace("\\", "__").replace(".ogg", ".npy"))

class CachedDataset(Dataset):
    """items: list of (npy_path, label). When train=True, shuffle segment order for augmentation."""
    def __init__(self, items, train=True):
        self.train = train
        self.items = [(p, y) for p, y in items if os.path.exists(p)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        mels = np.load(path)
        if self.train and mels.shape[0] >= N_SEG:
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
            block(1,32), block(32,64), block(64,128), block(128,feat_dim),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.attn = nn.Sequential(nn.Linear(feat_dim,64), nn.Tanh(), nn.Linear(64,1))
        self.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(feat_dim, n_class))

    def forward(self, x):
        B,N,M,T = x.shape
        x = x.view(B*N,1,M,T)
        feats = self.cnn(x).view(B,N,-1)
        wts = torch.softmax(self.attn(feats), dim=1)
        pooled = (feats*wts).sum(1)
        return self.classifier(pooled), wts

def build_items(df, le):
    return [(cache_path(r["filename"]), le.transform([r["scientific_name"]])[0])
            for _, r in df.iterrows()]

def train_eval(train_items, val_items, num_class, w, tag):
    train_ds = CachedDataset(train_items, train=True)
    val_ds   = CachedDataset(val_items, train=False)
    print(f"[{tag}] train {len(train_ds)}, val {len(val_ds)}")
    tl_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    va_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)
    model = CNNAttn(num_class).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

    def evaluate():
        model.eval(); ys, ps = [], []
        with torch.no_grad():
            for mels, y in va_loader:
                logits,_ = model(mels.to(DEVICE))
                ps.append(logits.argmax(1).cpu().numpy()); ys.append(y.numpy())
        yt, yp = np.concatenate(ys), np.concatenate(ps)
        return accuracy_score(yt,yp), f1_score(yt,yp,average="macro",zero_division=0), yt, yp

    best, best_state, wait = -1, None, 0
    for ep in range(1, EPOCHS+1):
        model.train(); run=0.0
        for mels,y in tl_loader:
            mels,y = mels.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); logits,_=model(mels)
            loss=crit(logits,y); loss.backward(); opt.step()
            run += loss.item()*mels.size(0)
        acc,macro,_,_ = evaluate(); sch.step(macro)
        if ep % 5 == 0 or ep == 1:
            print(f"  [{tag}] ep{ep:3d} loss {run/len(train_ds):.3f} Acc {acc:.4f} MacroF1 {macro:.4f}")
        if macro > best:
            best, wait = macro, 0
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  [{tag}] early stop ep{ep}"); break
    model.load_state_dict(best_state)
    acc, macro, yt, yp = evaluate()
    print(f"==== [{tag}] final: Acc {acc:.4f}  MacroF1 {macro:.4f} ====")
    return acc, macro, yt, yp

def main():
    print("Device:", DEVICE)
    df = pd.read_csv(TRAIN_CSV)[["filename","scientific_name"]].dropna()
    le = LabelEncoder(); le.fit(df["scientific_name"])
    num_class = len(le.classes_)

    # Split: singleton classes go to validation, the rest split 80/20 by filename
    y_arr = le.transform(df["scientific_name"])
    df = df.assign(y=y_arr)
    classes, cnts = np.unique(y_arr, return_counts=True)
    rare = set(classes[cnts < 2].tolist())
    df_rare = df[df["y"].isin(rare)]
    df_rest = df[~df["y"].isin(rare)].reset_index(drop=True)
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    tr_i, va_i = next(gss.split(df_rest, df_rest["y"], groups=df_rest["filename"]))
    df_train = df_rest.iloc[tr_i].reset_index(drop=True)
    df_val = pd.concat([df_rest.iloc[va_i], df_rare], ignore_index=True)
    print(f"Train recordings {len(df_train)}, val recordings {len(df_val)}, singleton classes {len(rare)}")

    train_items_real = build_items(df_train, le)
    val_items = build_items(df_val, le)  # validation set is pure real

    # Class weights (based on the real training set)
    counts = np.bincount(df_train["y"], minlength=num_class)
    w = 1.0/np.sqrt(counts+1); w = w/w.mean()

    # Synthetic data items (added to training only)
    synth_items = []
    if os.path.exists(SYNTH_INDEX):
        sidx = pd.read_csv(SYNTH_INDEX)
        for _, r in sidx.iterrows():
            sci = r["scientific_name"]
            if sci in le.classes_:
                p = os.path.join(SYNTH_CACHE, r["cache"])
                synth_items.append((p, le.transform([sci])[0]))
    print(f"Synthetic data available: {len(synth_items)} clips")

    # A: real only
    print("\n===== A (real only) =====")
    accA, macroA, _, _ = train_eval(train_items_real, val_items, num_class, w, "A-CNN")

    # B: real + synthetic
    print("\n===== B (real + synthetic) =====")
    accB, macroB, _, _ = train_eval(train_items_real + synth_items, val_items, num_class, w, "B-CNN")

    print("\n" + "="*50)
    print("CNN+Attention  B vs A (recording level)")
    print("="*50)
    print(f"A (real only)    Acc {accA:.4f}  MacroF1 {macroA:.4f}")
    print(f"B (real+synth)   Acc {accB:.4f}  MacroF1 {macroB:.4f}")
    print(f"Δ                Acc {accB-accA:+.4f}  MacroF1 {macroB-macroA:+.4f}")

if __name__ == "__main__":
    main()
