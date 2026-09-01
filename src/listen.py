import librosa, librosa.display, matplotlib.pyplot as plt, numpy as np
p= r"\coffal1\XC609210.ogg"
f = r"D:\bird\train_audio" + p   # change to the path above
y, sr = librosa.load(f, sr=32000)
# check duration, plot spectrogram
S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)
plt.figure(figsize=(12,4)); librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='mel')
plt.title(p); plt.colorbar(); plt.show()
# to listen, use IPython: from IPython.display import Audio; Audio(y, rate=sr)