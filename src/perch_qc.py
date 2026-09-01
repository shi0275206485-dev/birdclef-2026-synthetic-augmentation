"""
Group B - Step 3 (upgraded version): closed-loop QC using Perch 2.0.

Advantages over BirdNET:
  - Supports non-bird classes (frogs/insects/mammals), solving the problem that
    non-bird classes can't be verified
  - About 15000 classes, broader coverage (may recognize species BirdNET doesn't)
  - Uses iNaturalist taxon_id to precisely match target species (your
    taxonomy.csv has this column)

Environment:
  pip install git+https://github.com/google-research/perch-hoplite.git
  pip install "tensorflow[and-cuda]~=2.20.0" librosa soundfile

Note: Perch expects 5 seconds/32kHz; AudioLDM2 output is 16kHz, needs resampling.
"""

import os
import glob
import numpy as np
import pandas as pd
import librosa
from perch_hoplite.zoo import model_configs

# =====================================================
# CONFIG
# =====================================================

PLAN_CSV   = r"D:\bird\embedding\class_synthesis_plan.csv"
TAXONOMY   = r"D:\bird\taxonomy.csv"          # contains inat_taxon_id
SYNTH_DIR  = r"D:\bird\synth_test"
PERCH_SR   = 32000                            # Perch requires 32kHz
SEG_SEC    = 5                                # Perch expects 5-second segments

# =====================================================
# Load Perch
# =====================================================

print("Loading Perch v2 (will download the model from Kaggle on first run) ...")
model = model_configs.load_model_by_name("perch_v2")
print("Perch loaded")

# Perch's class labels: usually can be obtained from the model as a class list.
# In perch_hoplite, logits is a dict whose key is the label set name (e.g. 'label').
# The label details are in the model's assets/labels.csv, following iNaturalist taxonomy.
# First do a dummy inference below to get the label dimension info of logits.
dummy = np.zeros(SEG_SEC * PERCH_SR, dtype=np.float32)
out = model.embed(dummy)
print("embedding shape:", np.asarray(out.embeddings).shape)
logit_keys = list(out.logits.keys())
print("logits label sets:", logit_keys)
LABELSET = logit_keys[0]
n_classes = np.asarray(out.logits[LABELSET]).shape[-1]
print(f"Perch output class count ({LABELSET}): {n_classes}")

# =====================================================
# Adaptively obtain the class-name list (only accept sources whose length == n_classes)
# =====================================================

def find_class_list(model, n_classes):
    import glob as _glob
    candidates = []

    # 1) Common attributes on the model object
    for attr in ["class_list", "labels", "classes", "class_names", "label_list"]:
        if hasattr(model, attr):
            try:
                v = getattr(model, attr)
                for sub in ["classes", "names", "labels"]:
                    if hasattr(v, sub):
                        v = getattr(v, sub)
                        break
                lst = list(v)
                candidates.append((f"model.{attr}", lst))
            except Exception:
                pass

    # 2) If a logits key maps to a ClassList
    for k in logit_keys:
        if hasattr(model, "class_lists"):
            try:
                cl = model.class_lists[k]
                lst = list(getattr(cl, "classes", cl))
                candidates.append((f"model.class_lists['{k}']", lst))
            except Exception:
                pass

    # 3) Look for labels.csv in the KaggleHub cache
    home = os.path.expanduser("~")
    patterns = [
        os.path.join(home, ".cache", "kagglehub", "**", "labels.csv"),
        os.path.join(home, ".cache", "kagglehub", "**", "assets", "*.csv"),
    ]
    for pat in patterns:
        for f in _glob.glob(pat, recursive=True):
            try:
                df = pd.read_csv(f)
                col = "label" if "label" in df.columns else df.columns[0]
                lst = df[col].astype(str).tolist()
                candidates.append((f"csv:{f}", lst))
            except Exception:
                pass

    for src, lst in candidates:
        if len(lst) == n_classes:
            print(f"[OK] class-name source: {src} (length {len(lst)} matches)")
            return lst
    print("[WARN] No length-matching class-name list found, candidate sources and lengths:")
    for src, lst in candidates:
        print(f"   {src}: length {len(lst)}")
    return None

perch_classes = find_class_list(model, n_classes)

# =====================================================
# Target species: including non-bird classes! (Perch's core advantage)
# =====================================================

plan = pd.read_csv(PLAN_CSV)
tax = pd.read_csv(TAXONOMY)

# Map plan's label to inat_taxon_id
# plan["label"] may be scientific_name; cross-reference with taxonomy
sci2inat = tax.set_index("scientific_name")["inat_taxon_id"].to_dict()
targets = plan[plan["need_synth"] > 0].copy()
targets["inat_taxon_id"] = targets["label"].map(sci2inat)

print("\n" + "=" * 60)
print("Target species (need more data) statistics by class:")
print(targets["class_name"].value_counts())
print(f"\nOf which {(targets['class_name']!='Aves').sum()} are non-bird species "
      f"-- these can't be verified with BirdNET, Perch can try")

# =====================================================
# QC: predict on synthesized wav files
# =====================================================

def load_for_perch(path):
    """Read audio, resample to 32kHz, split into 5-second segments"""
    y, sr = librosa.load(path, sr=PERCH_SR, mono=True)
    seg_len = SEG_SEC * PERCH_SR
    segs = []
    for start in range(0, len(y), seg_len):
        chunk = y[start:start + seg_len]
        if len(chunk) < seg_len:
            chunk = np.pad(chunk, (0, seg_len - len(chunk)))
        segs.append(chunk.astype(np.float32))
    return segs

print("\n" + "=" * 60)
print("Perch QC: predicting on synthesized audio")
print("=" * 60)

wavs = sorted(glob.glob(os.path.join(SYNTH_DIR, "*.wav")))
print(f"Number of wavs to check: {len(wavs)}\n")

for wav in wavs:
    segs = load_for_perch(wav)
    # Predict on each 5-second segment, then average the logits
    all_logits = []
    for seg in segs:
        o = model.embed(seg)
        all_logits.append(np.asarray(o.logits[LABELSET]).reshape(-1))
    mean_logit = np.mean(all_logits, axis=0)
    top5 = np.argsort(mean_logit)[::-1][:5]

    print(f"[{os.path.basename(wav)}]  Perch top5 class indices:")
    for idx in top5:
        if perch_classes is not None and idx < len(perch_classes):
            name = perch_classes[idx]
        else:
            name = f"class_{idx}"
        print(f"   idx={idx:<6} {name:<40} logit={mean_logit[idx]:.3f}")
    print()

print("=" * 60)
print("Interpretation:")
print("  - A higher logit means Perch is more confident. But logits for rare species aren't calibrated, so you need to set your own threshold.")
print("  - The key thing to check: is top1 your target species (or a close relative).")
print("  - Non-bird species (frogs/mammals) can now also appear in predictions -- this is Perch's breakthrough over BirdNET.")
print("  - Use the iNaturalist id from perch_classes[idx] to cross-check against your taxonomy.csv and verify the target species.")