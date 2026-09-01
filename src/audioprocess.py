"""
Reorganize .ogg audio files that are split into folders by "species ID" into the folder
structure required for BirdNET-Analyzer training:
    training_data/<scientific_name>_<common_name>/xxx.ogg

- Read train.csv to build an ID -> scientific_name_common_name mapping
- Copy (leaving the original files untouched) to the new directory
- Automatically strip illegal characters from BirdNET labels
- Skip classes with too few samples (optional)
"""

import os
import csv
import shutil
import re

# =====================================================
# Path configuration -- change these to your own
# =====================================================

CSV_PATH    = r"D:\bird\train.csv"            # the lookup csv
AUDIO_ROOT  = r"D:\bird\train_audio"                # audio root currently split into folders by ID
OUTPUT_ROOT = r"D:\bird\training_data"        # organized output directory (for BirdNET)

MIN_FILES_PER_CLASS = 0   # species with fewer audio files than this are skipped outright
                          # (BirdNET recommends >=10 per class, too few is pointless)
                          # set to 0 to keep everything

# =====================================================
# 1. Read the csv to build an ID -> label mapping
# =====================================================

def clean_label(sci, common):
    """Build the BirdNET-recommended <scientific_name>_<common_name>, and strip illegal folder-name characters"""
    label = f"{sci}_{common}"
    # Illegal Windows folder-name characters \ / : * ? " < > | plus leading/trailing whitespace
    label = re.sub(r'[\\/:*?"<>|]', "", label).strip()
    # Collapse multiple spaces into one
    label = re.sub(r"\s+", " ", label)
    return label

id2label = {}
with open(CSV_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pid = row["primary_label"].strip()
        id2label[pid] = clean_label(row["scientific_name"], row["common_name"])

print(f"Number of mapped species: {len(id2label)}")

# =====================================================
# 2. Walk the ID folders, copying into scientific_name_common_name folders
# =====================================================

os.makedirs(OUTPUT_ROOT, exist_ok=True)

total_copied = 0
skipped_classes = []
missing_in_csv = []

for entry in os.listdir(AUDIO_ROOT):
    src_dir = os.path.join(AUDIO_ROOT, entry)
    if not os.path.isdir(src_dir):
        continue

    pid = entry.strip()
    if pid not in id2label:
        missing_in_csv.append(pid)   # folder exists, but the ID can't be found in the csv
        continue

    # Collect all .ogg files under this ID
    oggs = [f for f in os.listdir(src_dir) if f.lower().endswith(".ogg")]

    if MIN_FILES_PER_CLASS and len(oggs) < MIN_FILES_PER_CLASS:
        skipped_classes.append((id2label[pid], len(oggs)))
        continue

    dst_dir = os.path.join(OUTPUT_ROOT, id2label[pid])
    os.makedirs(dst_dir, exist_ok=True)

    for fn in oggs:
        src = os.path.join(src_dir, fn)
        dst = os.path.join(dst_dir, fn)
        if not os.path.exists(dst):       # resumable run: skip already-copied files
            shutil.copy2(src, dst)
            total_copied += 1

    print(f"[{pid}] -> {id2label[pid]}  ({len(oggs)} files)")

# =====================================================
# 3. Summary report
# =====================================================

print("\n========== Done ==========")
print(f"Total audio files copied: {total_copied}")
print(f"Output directory: {OUTPUT_ROOT}")

if skipped_classes:
    print(f"\nClasses skipped for having fewer than {MIN_FILES_PER_CLASS} samples ({len(skipped_classes)}):")
    for name, n in skipped_classes:
        print(f"  {name}: {n} files")

if missing_in_csv:
    print(f"\nWarning: the following folder IDs could not be found in the csv ({len(missing_in_csv)}):")
    for pid in missing_in_csv:
        print(f"  {pid}")

print("\nNext step: open the BirdNET-Analyzer GUI -> Train tab -> select the output directory as training data")