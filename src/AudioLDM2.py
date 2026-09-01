import torch
from diffusers import AudioLDM2Pipeline

pipe = AudioLDM2Pipeline.from_pretrained(
    "cvssp/audioldm2",
    torch_dtype=torch.float16
)

pipe = pipe.to("cuda")

prompt = "Birds chirping in a forest"

audio = pipe(
    prompt,
    num_inference_steps=50,
    audio_length_in_s=10.0
).audios[0]
import scipy

scipy.io.wavfile.write("techno.wav", rate=16000, data=audio)
print("done")