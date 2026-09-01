"""
Option B - sub-experiment 2: CNN+Attention, retrained with 8670 synthetic samples added (relabel labels).
A (real only) vs B (real + option B synthetic) compared in the same run (CNN has randomness, must be the same run).

Prerequisites:
  1. G:\bird_melcache real-data cache with old parameters (already exists)
  2. precompute_optionB_mel.py generates G:\bird_melcache_optionB
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
CACHE_DIR    = r"G:\bird_melcache"
OB_CACHE     = r"G:\bird_melcache_optionB"
OB_INDEX     = os.path.join(OB_CACHE, "optionB_cache_index.csv")

N_SEG       = 6
BATCH_SIZE  = 32
EPOCHS      = 150
LR          = 1e-2
PATIENCE    = 25      # allow enough patience to ensure convergence (prior experience: convergence needs ~88 epochs)
NUM_WORKERS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def cache_path(fn):
    return os.path.join(CACHE_DIR, fn.replace("/","__").replace("\\","__").replace(".ogg",".npy"))

class DS(Dataset):
    def __init__(self, items, train=True):
        self.train=train
        self.items=[(p,y) for p,y in items if os.path.exists(p)]
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        p,y=self.items[i]; mels=np.load(p)
        if self.train and mels.shape[0]>=N_SEG:
            mels=mels[np.random.permutation(mels.shape[0])[:N_SEG]]
        return torch.from_numpy(mels.astype(np.float32)), y

class CNNAttn(nn.Module):
    def __init__(self,n,feat=128):
        super().__init__()
        def blk(ci,co): return nn.Sequential(nn.Conv2d(ci,co,3,padding=1),
            nn.BatchNorm2d(co),nn.ReLU(),nn.MaxPool2d(2))
        self.cnn=nn.Sequential(blk(1,32),blk(32,64),blk(64,128),blk(128,feat),
            nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.attn=nn.Sequential(nn.Linear(feat,64),nn.Tanh(),nn.Linear(64,1))
        self.cls=nn.Sequential(nn.Dropout(0.4),nn.Linear(feat,n))
    def forward(self,x):
        B,N,M,T=x.shape; x=x.view(B*N,1,M,T)
        f=self.cnn(x).view(B,N,-1)
        w=torch.softmax(self.attn(f),1)
        return self.cls((f*w).sum(1)), w

def train_eval(train_items, val_items, num_class, w, tag):
    tl=DataLoader(DS(train_items,True),batch_size=BATCH_SIZE,shuffle=True,
                  num_workers=NUM_WORKERS,pin_memory=True,drop_last=True)
    va=DataLoader(DS(val_items,False),batch_size=BATCH_SIZE,shuffle=False,
                  num_workers=NUM_WORKERS,pin_memory=True)
    print(f"[{tag}] Train {len(DS(train_items,True))}, validation {len(DS(val_items,False))}")
    model=CNNAttn(num_class).to(DEVICE)
    crit=nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32,device=DEVICE))
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=0.5,patience=2)
    def ev():
        model.eval(); ys,ps=[],[]
        with torch.no_grad():
            for m,y in va:
                lg,_=model(m.to(DEVICE)); ps.append(lg.argmax(1).cpu().numpy()); ys.append(y.numpy())
        yt,yp=np.concatenate(ys),np.concatenate(ps)
        return accuracy_score(yt,yp), f1_score(yt,yp,average="macro",zero_division=0)
    best,bs,wait=-1,None,0
    for ep in range(1,EPOCHS+1):
        model.train(); 
        for m,y in tl:
            m,y=m.to(DEVICE),y.to(DEVICE); opt.zero_grad()
            lg,_=model(m); loss=crit(lg,y); loss.backward(); opt.step()
        acc,mac=ev(); sch.step(mac)
        if ep%10==0 or ep==1: print(f"  [{tag}] ep{ep} Acc {acc:.4f} MacroF1 {mac:.4f}")
        if mac>best: best,wait=mac,0; bs={k:v.cpu().clone() for k,v in model.state_dict().items()}
        else:
            wait+=1
            if wait>=PATIENCE: print(f"  [{tag}] early stop ep{ep}"); break
    model.load_state_dict(bs); acc,mac=ev()
    print(f"==== [{tag}] Final: Acc {acc:.4f} MacroF1 {mac:.4f} ====")
    return acc,mac

def main():
    print("Device:",DEVICE)
    df=pd.read_csv(TRAIN_CSV)[["filename","scientific_name"]].dropna()
    le=LabelEncoder(); le.fit(df["scientific_name"]); num_class=len(le.classes_)
    y_arr=le.transform(df["scientific_name"]); df=df.assign(y=y_arr)
    classes,cnts=np.unique(y_arr,return_counts=True)
    rare=set(classes[cnts<2].tolist())
    df_rare=df[df["y"].isin(rare)]; df_rest=df[~df["y"].isin(rare)].reset_index(drop=True)
    gss=GroupShuffleSplit(test_size=0.2,n_splits=1,random_state=42)
    tr,va=next(gss.split(df_rest,df_rest["y"],groups=df_rest["filename"]))
    df_tr=df_rest.iloc[tr].reset_index(drop=True)
    df_va=pd.concat([df_rest.iloc[va],df_rare],ignore_index=True)
    print(f"Training recordings {len(df_tr)}, validation {len(df_va)}")

    real_items=[(cache_path(r["filename"]), r["y"]) for _,r in df_tr.iterrows()]
    val_items =[(cache_path(r["filename"]), r["y"]) for _,r in df_va.iterrows()]

    counts=np.bincount(df_tr["y"],minlength=num_class); w=1.0/np.sqrt(counts+1); w=w/w.mean()

    # option B synthetic items (relabel labels, training only)
    ob_items=[]
    if os.path.exists(OB_INDEX):
        idx=pd.read_csv(OB_INDEX)
        for _,r in idx.iterrows():
            lab=r["relabel"]
            if lab in le.classes_:
                ob_items.append((os.path.join(OB_CACHE,r["cache"]), le.transform([lab])[0]))
    print(f"Option B synthetic available: {len(ob_items)} items")

    print("\n===== A (real only) =====")
    aA,fA=train_eval(real_items, val_items, num_class, w, "A-CNN")
    print("\n===== B (real + optionB) =====")
    aB,fB=train_eval(real_items+ob_items, val_items, num_class, w, "B-CNN")

    print("\n"+"="*50)
    print("Option B CNN+Attention  B vs A (recording level)")
    print("="*50)
    print(f"A (real only)     Acc {aA:.4f} MacroF1 {fA:.4f}")
    print(f"B (real+optionB)  Acc {aB:.4f} MacroF1 {fB:.4f}")
    print(f"Delta             Acc {aB-aA:+.4f} MacroF1 {fB-fA:+.4f}")

if __name__=="__main__":
    main()
