"""
Group B - information-rich prompt version generation (rerun across all 43 classes).

Improvements over the old version: the prompt combines three kinds of real information
  1. Common name (e.g. "Chaco Chachalaca")
  2. Actual call type (song / call, from train.csv annotations, not a guess)
  3. Ecoregion (mapped from latitude/longitude, e.g. Pantanal / Gran Chaco / Cerrado)

Purpose: use the "maximally informative" prompt to test the upper bound of prompt
         engineering. If the pass rate is still very low even with this prompt,
         it confirms that "the bottleneck is model capability, not the prompt."

Output goes to a separate directory synth_data_v2, to make it easy to compare
pass rates against the old version (synth_data).

Environment: diffusers (GPU) + perch_hoplite (CPU)
"""

import os
import glob
import time
import numpy as np
import pandas as pd
import scipy.io.wavfile
import librosa
import torch

# =====================================================
# CONFIG
# =====================================================

TARGET_INFO = r"D:\bird\target_info.csv"     # target species info generated in the previous step
OUT_DIR     = r"D:\bird\synth_data_v2"        # new directory, does not overwrite the old version
MANIFEST    = os.path.join(OUT_DIR, "synth_manifest_v2.csv")
LOG_FILE    = os.path.join(OUT_DIR, "generation_log_v2.txt")

TARGET_KEEP = 100
MAX_GEN     = 100
BATCH       = 10
AUDIO_LEN   = 10.0
NUM_STEPS   = 200
GUIDANCE    = 3.5
PERCH_TOPK  = 10
PERCH_SR    = 32000
SEG_SEC     = 5

os.makedirs(OUT_DIR, exist_ok=True)

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# =====================================================
# latitude/longitude -> ecoregion
# =====================================================

def ecoregion(lat, lon):
    # rough ecoregion split for the central South America target area
    if lat > 0 or lat < -35:        # north of the equator / far south -> generalize
        return "tropical South America"
    if -62 <= lon <= -55 and -23 <= lat <= -16:
        return "the Pantanal wetlands of South America"
    if lon <= -57 and lat <= -20:
        return "the dry forests of the Gran Chaco"
    if -55 <= lon <= -45 and -20 <= lat <= -10:
        return "the Cerrado savanna of central Brazil"
    return "the tropical forests of central South America"

# =====================================================
# information-rich prompt construction
# =====================================================

def make_prompt(common, call_type, lat, lon, class_name):
    region = ecoregion(lat, lon)
    if class_name == "Amphibia":
        return (f"the {call_type} of the {common}, a frog, "
                f"recorded at night in {region}")
    # birds
    ct = "song" if "song" in str(call_type).lower() else "call"
    return (f"the {ct} of the {common}, a wild bird, "
            f"recorded in {region}, natural field recording")

# =====================================================
# load target info
# =====================================================

info = pd.read_csv(TARGET_INFO)
log(f"Number of target classes: {len(info)}")

# resume from checkpoint
done = set()
if os.path.exists(MANIFEST):
    prev = pd.read_csv(MANIFEST)
    c = prev.groupby("target_sci").size()
    done = set(c[c >= TARGET_KEEP].index)
    log(f"Resuming from checkpoint: {len(done)} classes already done")

# =====================================================
# models
# =====================================================

log("Loading AudioLDM2 (GPU) ...")
from diffusers import AudioLDM2Pipeline
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pipe = AudioLDM2Pipeline.from_pretrained(
    "cvssp/audioldm2",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
).to(DEVICE)
pipe.enable_attention_slicing()

log("Loading Perch (CPU) ...")
from perch_hoplite.zoo import model_configs
perch = model_configs.load_model_by_name("perch_v2")
dummy = np.zeros(SEG_SEC * PERCH_SR, dtype=np.float32)
po = perch.embed(dummy)
LABELSET = list(po.logits.keys())[0]
n_perch = np.asarray(po.logits[LABELSET]).shape[-1]
perch_classes = None
home = os.path.expanduser("~")
for f in glob.glob(os.path.join(home, ".cache", "kagglehub", "**", "labels.csv"), recursive=True):
    df = pd.read_csv(f)
    col = "label" if "label" in df.columns else df.columns[0]
    lst = df[col].astype(str).tolist()
    if len(lst) == n_perch:
        perch_classes = lst
        break
assert perch_classes is not None
perch_genus = [c.split()[0] if " " in c else c for c in perch_classes]

def perch_check_genus(audio_16k, target_genus):
    y = librosa.resample(audio_16k, orig_sr=16000, target_sr=PERCH_SR)
    seg_len = SEG_SEC * PERCH_SR
    logits = []
    for s in range(0, max(1, len(y)), seg_len):
        chunk = y[s:s+seg_len]
        if len(chunk) < seg_len:
            chunk = np.pad(chunk, (0, seg_len - len(chunk)))
        o = perch.embed(chunk.astype(np.float32))
        logits.append(np.asarray(o.logits[LABELSET]).reshape(-1))
    ml = np.mean(logits, axis=0)
    topk = np.argsort(ml)[::-1][:PERCH_TOPK]
    return target_genus in set(perch_genus[i] for i in topk), perch_classes[int(np.argmax(ml))]

# =====================================================
# main loop
# =====================================================

generator = torch.Generator(DEVICE).manual_seed(42)
rows = []
rate_summary = []   # record the pass rate per class, for old-vs-new comparison

for _, r in info.iterrows():
    sci = r["sci"]
    if sci in done:
        continue
    genus = sci.split()[0]
    cls = "Amphibia" if "Frog" in str(r["common"]) else "Aves"
    prompt = make_prompt(r["common"], r["type"], r["lat"], r["lon"], cls)
    safe = sci.replace(" ", "_")
    log(f"\n=== {sci} ({r['common']}) ===")
    log(f"    prompt: {prompt}")

    kept, gen = 0, 0
    t0 = time.time()
    while kept < TARGET_KEEP and gen < MAX_GEN:
        audios = pipe(
            prompt, num_inference_steps=NUM_STEPS, audio_length_in_s=AUDIO_LEN,
            num_waveforms_per_prompt=BATCH, guidance_scale=GUIDANCE,
            generator=generator,
            negative_prompt="low quality, noise, silence, static, music, human voice",
        ).audios
        for audio in audios:
            gen += 1
            audio = np.asarray(audio, dtype=np.float32)
            passed, top1 = perch_check_genus(audio, genus)
            if passed:
                fn = f"{safe}__{kept:03d}.wav"
                scipy.io.wavfile.write(os.path.join(OUT_DIR, fn), 16000, audio)
                rows.append({"file": fn, "target_sci": sci, "class_name": cls,
                             "genus": genus, "perch_top1": top1, "prompt": prompt})
                kept += 1
            if kept >= TARGET_KEEP:
                break
        if rows:
            mode = "a" if os.path.exists(MANIFEST) else "w"
            header = not os.path.exists(MANIFEST)
            pd.DataFrame(rows).to_csv(MANIFEST, mode=mode, header=header, index=False)
            rows = []
    pr = kept / gen if gen else 0
    rate_summary.append({"sci": sci, "kept": kept, "gen": gen, "rate": pr})
    log(f"    Done: passed {kept}/{gen} (pass rate {pr:.1%}), took {time.time()-t0:.0f}s")

pd.DataFrame(rate_summary).to_csv(
    os.path.join(OUT_DIR, "passrate_v2.csv"), index=False)
log("\nAll classes processed. See passrate_v2.csv for pass-rate details.")
log("Compare against the old synth_data pass rate to quantify the effect of the information-rich prompt.")
