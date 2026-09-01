"""
Group B - Step 3: BirdNET QC closed loop (predict & check).

Two functions:
  A) Dictionary check: confirm whether BirdNET recognizes your target bird species
     (BirdNET covers about 6000 global bird species, but obscure South American
     species may not be in the table)
  B) QC: predict on synthesized wavs with BirdNET, see whether they get
     recognized as the target bird species, and report the pass rate

Environment: pip install birdnetlib librosa
"""

import os
import glob
import pandas as pd
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer

# =====================================================
# CONFIG
# =====================================================

PLAN_CSV   = r"D:\bird\embedding\class_synthesis_plan.csv"   # output of step 1
SYNTH_DIR  = r"D:\bird\synth_test"                 # directory of trial-generated wavs
CONF_TH    = 0.3                                 # BirdNET confidence threshold (adjustable)

# =====================================================
# Load BirdNET analyzer (will download the model on first run)
# =====================================================

print("Loading BirdNET analyzer ...")
analyzer = Analyzer()

# BirdNET's species label table: analyzer.labels looks like "Scientific name_Common name"
birdnet_species = analyzer.labels
print(f"Number of species BirdNET recognizes: {len(birdnet_species)}")

# Extract the scientific-name part (before the underscore)
birdnet_sci = set(s.split("_")[0].strip().lower() for s in birdnet_species)

# =====================================================
# A) Dictionary check: is the target bird species in BirdNET
# =====================================================

plan = pd.read_csv(PLAN_CSV)
bird_small = plan[(plan["class_name"] == "Aves") & (plan["need_synth"] > 0)]

print("\n" + "=" * 60)
print("Dictionary check: are the target bird subclasses recognized by BirdNET")
print("=" * 60)

known, unknown = [], []
for _, row in bird_small.iterrows():
    sci = str(row["label"]).strip().lower()
    in_dict = sci in birdnet_sci
    # fuzzy match (a matching genus name also counts as a clue)
    genus = sci.split()[0] if " " in sci else sci
    genus_hit = any(g.split()[0] == genus for g in birdnet_sci if " " in g)
    status = "[OK] in dictionary" if in_dict else ("[WARN] genus-only match" if genus_hit else "[MISS] not in dictionary")
    print(f"  {row['label']:<35} rec={int(row['rec_count'])}  {status}")
    (known if in_dict else unknown).append(row["label"])

print(f"\nBirdNET recognizes: {len(known)} species, does not recognize: {len(unknown)} species")
if unknown:
    print("Unrecognized bird species -> cannot be verified with BirdNET, this limitation should be noted in the report")

# =====================================================
# B) QC: predict on synthesized wavs
# =====================================================

print("\n" + "=" * 60)
print("QC: BirdNET predictions on synthesized audio")
print("=" * 60)

wavs = sorted(glob.glob(os.path.join(SYNTH_DIR, "*.wav")))
print(f"Number of wavs to check: {len(wavs)}\n")

for wav in wavs:
    rec = Recording(analyzer, wav, min_conf=CONF_TH)
    rec.analyze()
    dets = rec.detections  # list of dict: common_name, scientific_name, confidence, ...
    print(f"[{os.path.basename(wav)}]")
    if not dets:
        print(f"  (no detections: BirdNET did not recognize any bird above confidence {CONF_TH})")
    else:
        # sort by confidence, show top3
        dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)
        for d in dets[:3]:
            print(f"  {d['scientific_name']:<30} "
                  f"{d['common_name']:<28} conf={d['confidence']:.3f}")
    print()

print("=" * 60)
print("Interpretation:")
print("  - If you want to keep a synthesized audio clip as a training sample for 'species X',")
print("    BirdNET should predict X as top1 on that clip with conf > threshold.")
print("  - If the pass rate is too low -> generate several times more clips in bulk to filter enough.")
print("  - If the target bird species isn't in the BirdNET dictionary -> that species can't be QC'd (state this in the report).")