"""
Group B - Step 1: diagnose which classes need synthetic data augmentation.

Outputs for each class:
  - recording-level sample count (deduplicated by filename, since generation is
    done per "recording")
  - segment-level sample count
  - whether it's a small class
  - suggested number of synthetic recordings to add (to reach the target balance line)

Also tries to read train.csv's class_name (Aves/Insecta/...), because AudioLDM2
generates the best quality for birds (Aves); non-bird classes need special attention.
"""

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

PARQUET_DIR = r"D:\bird\merged_parquet"   # change to your path
TRAIN_CSV   = r"D:\bird\train.csv"        # the lookup table (contains class_name)

# Target: bring each class's recording count up to this level (classes below it need topping up)
TARGET_RECORDINGS = 100      # adjustable: top up to at least ~200 recordings per class
SMALL_CLASS_TH    = 50      # recording count < this value is considered a "small class"

# =====================================================
# Read parquet's label + filename (features not needed, saves memory)
# =====================================================

dataset = ds.dataset(PARQUET_DIR, format="parquet")
table = dataset.to_table(columns=["label", "filename"])
labels = np.asarray(table.column("label").to_numpy(zero_copy_only=False))
filenames = np.asarray(table.column("filename").to_numpy(zero_copy_only=False))
del table

df = pd.DataFrame({"label": labels, "filename": filenames})

# =====================================================
# Recording-level & segment-level sample counts
# =====================================================

# Segment-level: how many rows each label has
seg_count = df.groupby("label").size().rename("seg_count")

# Recording-level: how many distinct filenames each label has
rec_count = df.groupby("label")["filename"].nunique().rename("rec_count")

stat = pd.concat([rec_count, seg_count], axis=1).reset_index()

# =====================================================
# Merge in class_name (bird/insect/amphibian) to help judge generation feasibility
# =====================================================

try:
    meta = pd.read_csv(TRAIN_CSV)
    # primary_label is the species code; label in the parquet may be scientific_name or a code
    # try both mappings
    if "scientific_name" in meta.columns:
        name_map = meta.drop_duplicates("scientific_name").set_index(
            "scientific_name")["class_name"].to_dict()
        stat["class_name"] = stat["label"].map(name_map)
    if stat["class_name"].isna().all() and "primary_label" in meta.columns:
        name_map2 = meta.drop_duplicates("primary_label").set_index(
            "primary_label").get("class_name", pd.Series()).to_dict()
        stat["class_name"] = stat["label"].astype(str).map(name_map2)
except Exception as e:
    print("(failed to read class_name, skipping, does not affect the main diagnostics):", e)
    stat["class_name"] = "unknown"

# =====================================================
# Compute the number needed to top up
# =====================================================

stat["need_synth"] = (TARGET_RECORDINGS - stat["rec_count"]).clip(lower=0)
stat["is_small"] = stat["rec_count"] < SMALL_CLASS_TH

stat = stat.sort_values("rec_count").reset_index(drop=True)

# =====================================================
# Report
# =====================================================

print("=" * 70)
print(f"Total number of classes: {len(stat)}")
print(f"Total recordings: {df['filename'].nunique()}   Total segments: {len(df)}")
print(f"Number of small classes (<{SMALL_CLASS_TH} recordings): {stat['is_small'].sum()}")
print(f"Number of classes needing synthetic data: {(stat['need_synth']>0).sum()}")
print(f"Total synthetic recordings needed (top up each class to {TARGET_RECORDINGS}): {int(stat['need_synth'].sum())}")
print("=" * 70)

print(f"\nThe 25 classes with the fewest samples (most in need of topping up):")
print(stat[["label", "class_name", "rec_count", "seg_count", "need_synth"]]
      .head(25).to_string(index=False))

if "class_name" in stat.columns:
    print(f"\nSmall-class distribution by class_name:")
    print(stat[stat["is_small"]].groupby("class_name").size())

# Save as csv, used in step 2's generation
stat.to_csv("class_synthesis_plan.csv", index=False)
print(f"\nSynthesis plan saved -> class_synthesis_plan.csv")
print("Next step: use AudioLDM2 to generate synthetic bird calls for classes with need_synth > 0")