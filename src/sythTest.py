"""
Group B - Step 2 (trial generation): AudioLDM2 generates synthetic bird audio +
small-batch verification.

Goal: run at small scale first to confirm:
  1. 16GB of VRAM is sufficient (enable the VRAM-saving option)
  2. generation speed is acceptable
  3. the generated bird calls sound "convincing" (manually listen to a few first)
Once confirmed OK, use the batch script to generate in bulk.

Environment: pip install diffusers transformers accelerate scipy soundfile
"""

import os
import torch
import scipy.io.wavfile
from diffusers import AudioLDM2Pipeline

# =====================================================
# CONFIG
# =====================================================

OUT_DIR = r"D:\bird\synth_test"        # trial generation output
os.makedirs(OUT_DIR, exist_ok=True)

AUDIO_LEN_SEC = 10.0
N_PER_PROMPT  = 3                       # generate 3 clips per prompt first for a listen test
NUM_INFERENCE_STEPS = 200               # diffusion steps, quality vs speed, 200 is a good default
GUIDANCE_SCALE = 3.5                    # text guidance strength

# Try with 1-2 small bird classes first. Prompts use "sound descriptions", not
# species scientific names
# (AudioLDM2 doesn't recognize specific bird species' scientific names, needs
# sound characteristics described instead)
TEST_PROMPTS = {
    # file name prefix : text prompt
    "bird_generic":   "a bird chirping and singing in a forest, clear bird call",
    "bird_tropical":  "a tropical bird call, high-pitched whistling bird song",
}

# =====================================================
# Load pipeline (16GB VRAM optimization)
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

repo_id = "cvssp/audioldm2"             # audioldm2-large also usable (more VRAM-hungry)
pipe = AudioLDM2Pipeline.from_pretrained(
    repo_id,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)
pipe = pipe.to(DEVICE)

# ---- Key VRAM-saving options (recommended to enable both for 16GB) ----
pipe.enable_attention_slicing()         # attention slicing, significantly saves VRAM
# If still OOM, uncomment the line below (saves more VRAM but slower, offloads some modules to CPU)
# pipe.enable_model_cpu_offload()

# =====================================================
# Generate
# =====================================================

generator = torch.Generator(DEVICE).manual_seed(42)

for prefix, prompt in TEST_PROMPTS.items():
    print(f"\nGenerating '{prefix}': {prompt}")
    audios = pipe(
        prompt,
        num_inference_steps=NUM_INFERENCE_STEPS,
        audio_length_in_s=AUDIO_LEN_SEC,
        num_waveforms_per_prompt=N_PER_PROMPT,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
        negative_prompt="low quality, noise, silence, static",  # improve quality
    ).audios

    for i, audio in enumerate(audios):
        path = os.path.join(OUT_DIR, f"{prefix}_{i:02d}.wav")
        # AudioLDM2 output sample rate is 16kHz
        scipy.io.wavfile.write(path, rate=16000, data=audio)
        print(f"  saved {path}")

print(f"\nTrial generation complete, output in {OUT_DIR}")
print("Next steps:")
print("  1. Manually listen to a few clips, confirm whether they sound like bird calls")
print("  2. Record the per-clip generation time and VRAM usage (Task Manager/nvidia-smi)")
print("  3. Once confirmed OK, verify these wavs with BirdNET's predict & check")