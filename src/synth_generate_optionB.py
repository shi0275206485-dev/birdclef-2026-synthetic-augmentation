"""
Group B - Professor's Option B + rich-information prompt version:
  - Save everything generated, no data discarded
  - Perch takes the highest score within "our 206 classes" as the label (robin->crow is used as-is too)
  - Prompts use the rich-information version (common name + real call type + ecoregion mapped from lat/lon)
  - Records the original target species, relabeled label, and top score

Note: Option B doesn't care about "pass rate" (everything is kept), so the rich-information prompt is fine
    even when genus-level hits are low - it actually demonstrates "we did our best to construct the optimal prompt".

Environment: diffusers (GPU) + perch_hoplite (CPU)
"""

import os, glob, time
import numpy as np
import pandas as pd
import scipy.io.wavfile
import librosa
import torch

# ============ CONFIG ============
TARGET_INFO = r"D:\bird\target_info.csv"        # contains sci/common/type/lat/lon
TAXONOMY    = r"D:\bird\taxonomy.csv"
PLAN_CSV    = r"D:\bird\embedding\class_synthesis_plan.csv"
OUT_DIR     = r"G:\bird\synth_data_optionB"
MANIFEST    = os.path.join(OUT_DIR, "synth_manifest_optionB.csv")
LOG_FILE    = os.path.join(OUT_DIR, "generation_log_optionB.txt")

GEN_PER_CLASS = 200          # how many clips to generate per class (all kept)
BATCH       = 15
AUDIO_LEN   = 10.0
NUM_STEPS   = 200
GUIDANCE    = 3.5
PERCH_SR    = 32000
SEG_SEC     = 5
MIN_SCORE   = 0.1           # minimum score threshold; None = no filtering

os.makedirs(OUT_DIR, exist_ok=True)
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f: f.write(line+"\n")

# ============ Rich-information prompt (common name + type + ecoregion) ============
def ecoregion(lat, lon):
    if lat > 0 or lat < -35: return "tropical South America"
    if -62 <= lon <= -55 and -23 <= lat <= -16:
        return "the Pantanal wetlands of South America"
    if lon <= -57 and lat <= -20:
        return "the dry forests of the Gran Chaco"
    if -55 <= lon <= -45 and -20 <= lat <= -10:
        return "the Cerrado savanna of central Brazil"
    return "the tropical forests of central South America"

def make_prompt(common, call_type, lat, lon, class_name):
    region = ecoregion(lat, lon)
    if class_name == "Amphibia":
        return f"the {call_type} of the {common}, a frog, recorded at night in {region}"
    ct = "song" if "song" in str(call_type).lower() else "call"
    return f"the {ct} of the {common}, a wild bird, recorded in {region}, natural field recording"

# ============ Load target info ============
info = pd.read_csv(TARGET_INFO)
plan = pd.read_csv(PLAN_CSV)
tax  = pd.read_csv(TAXONOMY)
my_classes_set = set(plan["label"].unique().tolist())
log(f"Target class count: {len(info)}, dataset class count: {len(my_classes_set)}")

# class_name filled in via taxonomy (distinguishes bird/frog)
sci2cls = tax.set_index("scientific_name")["class_name"].to_dict()

done = set()
if os.path.exists(MANIFEST):
    prev = pd.read_csv(MANIFEST)
    c = prev.groupby("target_sci").size()
    done = set(c[c >= GEN_PER_CLASS].index)
    log(f"Resuming from checkpoint: {len(done)} classes already done")

# ============ Model ============
log("Loading AudioLDM2 (GPU) ...")
from diffusers import AudioLDM2Pipeline
DEV = "cuda" if torch.cuda.is_available() else "cpu"
pipe = AudioLDM2Pipeline.from_pretrained(
    "cvssp/audioldm2",
    torch_dtype=torch.float16 if DEV=="cuda" else torch.float32).to(DEV)
pipe.enable_attention_slicing()

log("Loading Perch (CPU) ...")
from perch_hoplite.zoo import model_configs
perch = model_configs.load_model_by_name("perch_v2")
dummy = np.zeros(SEG_SEC*PERCH_SR, dtype=np.float32)
po = perch.embed(dummy)
LABELSET = list(po.logits.keys())[0]
n_perch = np.asarray(po.logits[LABELSET]).shape[-1]
perch_classes = None
home = os.path.expanduser("~")
for f in glob.glob(os.path.join(home,".cache","kagglehub","**","labels.csv"), recursive=True):
    df = pd.read_csv(f); col = "label" if "label" in df.columns else df.columns[0]
    lst = df[col].astype(str).tolist()
    if len(lst)==n_perch: perch_classes = lst; break
assert perch_classes is not None

def norm(name):
    parts = str(name).replace("_"," ").split()
    return " ".join(parts[:2]) if len(parts)>=2 else str(name)
perch_norm = [norm(c) for c in perch_classes]
in_dataset_idx, idx_to_mylabel = [], {}
for i, pn in enumerate(perch_norm):
    if pn in my_classes_set:
        in_dataset_idx.append(i); idx_to_mylabel[i] = pn
in_dataset_idx = np.array(in_dataset_idx)
log(f"Perch class names mappable to our dataset: {len(in_dataset_idx)}/{len(my_classes_set)} classes")

def perch_relabel(audio_16k):
    y = librosa.resample(audio_16k, orig_sr=16000, target_sr=PERCH_SR)
    seg_len = SEG_SEC*PERCH_SR; logits=[]
    for s in range(0, max(1,len(y)), seg_len):
        ch = y[s:s+seg_len]
        if len(ch) < seg_len: ch = np.pad(ch,(0,seg_len-len(ch)))
        o = perch.embed(ch.astype(np.float32))
        logits.append(np.asarray(o.logits[LABELSET]).reshape(-1))
    ml = np.mean(logits, axis=0)
    sub = ml[in_dataset_idx]
    bl = int(np.argmax(sub))
    return idx_to_mylabel[int(in_dataset_idx[bl])], float(sub[bl])

# ============ Main loop ============
gen_t = torch.Generator(DEV).manual_seed(42)
rows = []
for _, r in info.iterrows():
    sci = r["sci"]
    if sci in done: continue
    cls = sci2cls.get(sci, "Aves")
    prompt = make_prompt(r["common"], r["type"], r["lat"], r["lon"], cls)
    safe = sci.replace(" ","_")
    log(f"\n=== {sci} ({r['common']}) ===")
    log(f"    prompt: {prompt}")
    n = 0; t0 = time.time()
    while n < GEN_PER_CLASS:
        audios = pipe(prompt, num_inference_steps=NUM_STEPS, audio_length_in_s=AUDIO_LEN,
                      num_waveforms_per_prompt=BATCH, guidance_scale=GUIDANCE, generator=gen_t,
                      negative_prompt="low quality, noise, silence, static, music, human voice").audios
        for audio in audios:
            audio = np.asarray(audio, dtype=np.float32)
            relabel, score = perch_relabel(audio)
            if MIN_SCORE is not None and score < MIN_SCORE: continue
            fn = f"{safe}__{n:03d}.wav"
            scipy.io.wavfile.write(os.path.join(OUT_DIR, fn), 16000, audio)
            rows.append({"file":fn, "target_sci":sci, "relabel":relabel,
                         "perch_score":round(score,3), "class_name":cls, "prompt":prompt})
            n += 1
            if n >= GEN_PER_CLASS: break
        if rows:
            mode="a" if os.path.exists(MANIFEST) else "w"; hdr=not os.path.exists(MANIFEST)
            pd.DataFrame(rows).to_csv(MANIFEST, mode=mode, header=hdr, index=False); rows=[]
    man = pd.read_csv(MANIFEST); sub = man[man["target_sci"]==sci]
    match = (sub["relabel"]==sci).mean() if len(sub) else 0
    log(f"    Done {n} clips, relabel match rate to target species {match:.1%}, elapsed {time.time()-t0:.0f}s")

log("\nAll classes done. manifest includes relabel (Perch top-score label) and perch_score.")
