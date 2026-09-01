"""
Group B core script: batch-generate synthetic bird/frog audio + real-time
genus-level quality filtering with Perch.
Designed to run overnight, with checkpoint resume, timeout protection, and
detailed logging.

Strategy:
  - For each class, generate using an acoustic prompt (built from the common
    name type)
  - AudioLDM2 generates on GPU; Perch does quality checking on CPU (doesn't
    compete for VRAM)
  - Genus-level pass: kept if a "same-genus" species appears in Perch's top-k
  - Stop once TARGET_KEEP passing clips are collected per class; cap at
    MAX_GEN generated clips as a stop-loss
  - Passing clips are saved as wav + recorded to the manifest; checkpoint
    resume skips classes already completed

Environment:
  GPU: diffusers (AudioLDM2)
  CPU: perch_hoplite (Perch v2 CPU version)
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

TAXONOMY    = r"D:\bird\taxonomy.csv"
PLAN_CSV    = r"D:\bird\embedding\class_synthesis_plan.csv"
OUT_DIR     = r"D:\bird\synth_data"
MANIFEST    = os.path.join(OUT_DIR, "synth_manifest.csv")
LOG_FILE    = os.path.join(OUT_DIR, "generation_log.txt")

TARGET_KEEP = 100        # stop once this many passing clips are collected per class
MAX_GEN     = 200        # max clips to generate per class (stop-loss, prevents getting stuck on low pass rate)
BATCH       = 8          # clips generated per pipe call (increase if VRAM allows, for speed)
AUDIO_LEN   = 10.0
NUM_STEPS   = 200
GUIDANCE    = 3.5
PERCH_TOPK  = 5        # passes if the same genus appears within Perch's top-k
PERCH_SR    = 32000
SEG_SEC     = 5

# frog controls (3 manually specified representatives)
FROG_TARGETS = ["Boana lundii", "Physalaemus nattereri", "Dermatonotus muelleri"]

os.makedirs(OUT_DIR, exist_ok=True)

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# =====================================================
# acoustic prompt construction: common-name-type keyword -> acoustic description
# =====================================================

# Call-feature templates based on bird type (leveraging the finding that
# "acoustic prompts can reach genus level")
ACOUSTIC_TEMPLATES = {
    "Nighthawk":  "a nasal buzzing aerial call of a nightjar bird at dusk",
    "Nightjar":   "a repetitive whistling churring call of a nightjar at night",
    "Owl":        "a deep hooting call of an owl at night",
    "Parakeet":   "loud harsh screeching squawks of a small parrot flock",
    "Macaw":      "loud raucous squawking calls of a large macaw parrot",
    "Chachalaca": "a loud raucous repetitive cackling chorus of a chachalaca bird at dawn",
    "Curassow":   "a deep low-frequency booming resonant hum of a large ground bird",
    "Tinamou":    "a melancholic pure whistling tremolo call of a tinamou ground bird",
    "Antshrike":  "an accelerating series of nasal barking notes of an antshrike",
    "Antbird":    "a sharp repeated chipping song of an antbird in dense forest",
    "Flycatcher": "a sharp chattering high-pitched call of a tropical flycatcher",
    "Hawk":       "a piercing descending scream of a bird of prey",
    "Cardinal":   "clear sweet whistled musical phrases of a songbird",
    "Cacholote":  "a loud raucous chattering duet of a furnariid bird",
    "Spinetail":  "a fast staccato trill of a small furnariid bird",
    "Woodpecker": "a loud rattling drumming and sharp call of a woodpecker",
    "Dove":       "a soft repetitive cooing of a dove",
    "Cuckoo":     "a repetitive hollow whistling call of a cuckoo",
    "Ibis":       "a loud nasal honking call of an ibis",
    "Heron":      "a harsh croaking squawk of a heron",
    "Tanager":    "high thin musical whistles of a colorful tanager",
    "Finch":      "a rapid bright trilling song of a small finch",
    "Wren":       "a loud bubbling cascading musical song of a wren",
    "Jay":        "loud harsh raucous calls of a jay",
    "Motmot":     "a soft low hooting double note of a motmot",
}
# frog templates
FROG_TEMPLATES = {
    "Boana":        "a melodic pulsed croaking call of a tree frog at night",
    "Physalaemus":  "a short nasal whining call of a small dwarf frog",
    "Dermatonotus": "a long low resonant moaning call of a burrowing frog",
    "_default":     "a repetitive croaking call of a frog at night in a wetland",
}

def make_prompt(common_name, scientific_name, class_name):
    if class_name == "Amphibia":
        genus = scientific_name.split()[0]
        return FROG_TEMPLATES.get(genus, FROG_TEMPLATES["_default"])
    # birds: match by common-name keyword
    for kw, tmpl in ACOUSTIC_TEMPLATES.items():
        if kw.lower() in common_name.lower():
            return tmpl
    # for unmatched birds, use a description that's generic but stronger than "generic"
    return f"a clear natural bird call of a {common_name} in tropical forest"

# =====================================================
# load target list
# =====================================================

tax = pd.read_csv(TAXONOMY)
plan = pd.read_csv(PLAN_CSV)

# bird minority classes (need_synth>0)
bird_targets = plan[(plan["class_name"] == "Aves") & (plan["need_synth"] > 0)]["label"].tolist()
all_targets = bird_targets + FROG_TARGETS

# fill in common_name / scientific_name / class / genus for each target
sci2row = tax.set_index("scientific_name")
targets_info = []
for sci in all_targets:
    if sci in sci2row.index:
        r = sci2row.loc[sci]
        targets_info.append({
            "scientific_name": sci,
            "common_name": r["common_name"],
            "class_name": r["class_name"],
            "genus": sci.split()[0],
        })
log(f"Number of target classes: {len(targets_info)} (birds {len(bird_targets)} + frogs {len(FROG_TARGETS)})")

# =====================================================
# checkpoint resume: classes already completed
# =====================================================

done_classes = set()
if os.path.exists(MANIFEST):
    prev = pd.read_csv(MANIFEST)
    counts = prev.groupby("target_sci").size()
    done_classes = set(counts[counts >= TARGET_KEEP].index)
    log(f"Resuming from checkpoint: {len(done_classes)} classes already done, skipping")

# =====================================================
# load models: AudioLDM2 (GPU) + Perch (CPU)
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
# force Perch to use CPU, to avoid competing with AudioLDM2 for VRAM
os.environ.setdefault("CUDA_VISIBLE_DEVICES_BACKUP", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
from perch_hoplite.zoo import model_configs
perch = model_configs.load_model_by_name("perch_v2")  # the one downloaded is the CPU version

# Perch class names
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
assert perch_classes is not None, "Perch class list not found"
# genus corresponding to each class (used for genus-level matching)
perch_genus = [c.split()[0] if " " in c else c for c in perch_classes]

# =====================================================
# Perch genus-level quality check
# =====================================================

def perch_check_genus(audio_16k, target_genus):
    """audio is 16kHz numpy; resample to 32k, run Perch prediction, check whether the topk contains the target genus"""
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
    topk_genus = set(perch_genus[i] for i in topk)
    passed = target_genus in topk_genus
    top1 = perch_classes[int(np.argmax(ml))]
    return passed, top1

# =====================================================
# main loop
# =====================================================

generator = torch.Generator(DEVICE).manual_seed(42)
manifest_rows = []

for ti in targets_info:
    sci = ti["scientific_name"]
    if sci in done_classes:
        continue

    genus = ti["genus"]
    prompt = make_prompt(ti["common_name"], sci, ti["class_name"])
    safe_name = sci.replace(" ", "_")
    log(f"\n=== {sci} ({ti['class_name']}) genus={genus} ===")
    log(f"    prompt: {prompt}")

    kept, generated = 0, 0
    t_start = time.time()

    while kept < TARGET_KEEP and generated < MAX_GEN:
        audios = pipe(
            prompt,
            num_inference_steps=NUM_STEPS,
            audio_length_in_s=AUDIO_LEN,
            num_waveforms_per_prompt=BATCH,
            guidance_scale=GUIDANCE,
            generator=generator,
            negative_prompt="low quality, noise, silence, static, music, human voice",
        ).audios

        for audio in audios:
            generated += 1
            audio = np.asarray(audio, dtype=np.float32)
            passed, top1 = perch_check_genus(audio, genus)
            if passed:
                fn = f"{safe_name}__{kept:03d}.wav"
                scipy.io.wavfile.write(os.path.join(OUT_DIR, fn), 16000, audio)
                manifest_rows.append({
                    "file": fn, "target_sci": sci, "class_name": ti["class_name"],
                    "genus": genus, "perch_top1": top1, "prompt": prompt,
                })
                kept += 1
            if kept >= TARGET_KEEP:
                break

        # save manifest in real time (prevents data loss on crash)
        if manifest_rows:
            mode = "a" if os.path.exists(MANIFEST) else "w"
            header = not os.path.exists(MANIFEST)
            pd.DataFrame(manifest_rows).to_csv(MANIFEST, mode=mode, header=header, index=False)
            manifest_rows = []

    rate = kept / generated if generated else 0
    log(f"    Done: passed {kept}/{generated} (pass rate {rate:.1%}), "
        f"took {time.time()-t_start:.0f}s")

log("\nAll classes processed.")
log(f"Output directory: {OUT_DIR}, manifest: {MANIFEST}")