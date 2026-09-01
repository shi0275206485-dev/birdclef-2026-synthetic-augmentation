"""
Group B - prompt engineering ablation experiment.

Goal: test whether the specificity of the prompt can make AudioLDM2 generate calls
      closer to a "specific target species".
Method: for each target species, generate several clips with each of three prompt
      tiers, then run Perch predictions and see whether the recognition results
      trend toward the target species (or its close relatives).

Three prompt tiers:
  generic  - extremely generic ("a bird call")
  acoustic - describes the acoustic characteristics of the species' call
  species  - uses the scientific name directly (likely ineffective, used to prove it's ineffective)

Environment: diffusers + perch_hoplite (recommended to run in two steps: generate
      first, then QC; if TF and torch environments conflict, split generation and
      QC into two environments/two runs)
"""

import os
import numpy as np
import scipy.io.wavfile
import torch
from diffusers import AudioLDM2Pipeline

# =====================================================
# CONFIG
# =====================================================

OUT_DIR = r"D:\bird\synth_prompt_test"
os.makedirs(OUT_DIR, exist_ok=True)

AUDIO_LEN_SEC = 10.0
N_PER_PROMPT  = 4          # generate 4 clips per tier
NUM_STEPS     = 200
GUIDANCE      = 3.5

# Two target species with distinctive calls that Perch recognizes, plus their
# acoustic descriptions.
# Acoustic descriptions are based on publicly available characteristics of the
# species' calls (cite the source when writing the report).
TARGETS = {
    "Crax_fasciolata": {
        "generic":  "a bird call",
        # Bare-faced Curassow: a deep, resonant "booming" sound, similar to a low hum
        "acoustic": "a deep low-frequency booming resonant hum call of a large ground bird",
        "species":  "Crax fasciolata Bare-faced Curassow call",
    },
    "Ortalis_canicollis": {
        "generic":  "a bird call",
        # Chachalaca: a loud, harsh, repetitive chorus-like call
        "acoustic": "a loud raucous repetitive cackling chorus call of a chachalaca bird at dawn",
        "species":  "Ortalis canicollis Chaco Chachalaca call",
    },
}

# =====================================================
# Load AudioLDM2
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

pipe = AudioLDM2Pipeline.from_pretrained(
    "cvssp/audioldm2",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
).to(DEVICE)
pipe.enable_attention_slicing()

generator = torch.Generator(DEVICE).manual_seed(42)

# =====================================================
# Generate
# =====================================================

manifest = []   # record which species and prompt tier each file corresponds to

for species, prompts in TARGETS.items():
    for level, prompt in prompts.items():
        print(f"\n[{species} | {level}] {prompt}")
        audios = pipe(
            prompt,
            num_inference_steps=NUM_STEPS,
            audio_length_in_s=AUDIO_LEN_SEC,
            num_waveforms_per_prompt=N_PER_PROMPT,
            guidance_scale=GUIDANCE,
            generator=generator,
            negative_prompt="low quality, noise, silence, static, music, human voice",
        ).audios

        for i, audio in enumerate(audios):
            fn = f"{species}__{level}__{i:02d}.wav"
            path = os.path.join(OUT_DIR, fn)
            scipy.io.wavfile.write(path, rate=16000, data=audio)
            manifest.append({"file": fn, "species": species,
                             "prompt_level": level, "prompt": prompt})
            print(f"  saved {fn}")

# Save manifest, used by the QC script
import pandas as pd
pd.DataFrame(manifest).to_csv(
    os.path.join(OUT_DIR, "manifest.csv"), index=False)

print(f"\nGeneration complete -> {OUT_DIR}")
print(f"{len(manifest)} clips total, saved to manifest.csv")
print("\nNext step: point SYNTH_DIR in perch_qc.py at this directory and run Perch QC,")
print("      to compare whether the recognition results of the three prompt tiers trend toward the target species.")