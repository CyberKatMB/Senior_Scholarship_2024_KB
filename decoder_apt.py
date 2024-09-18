import scipy.io.wavfile as wav
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# N.B.: this version produces a very dark picture (not helped by the fact all my passes were at night time!)

fs, data = wav.read("2024-08-18-2055-RAW.wav")
data_crop = data[20*fs:21*fs]
# ----- # 
resample = 4
data = data[::resample]
fs = fs//resample
# ----- #
def hilbert(data):
    analytical_signal = signal.hilbert(data)
    amplitude_envelope = np.abs(analytical_signal)
    return amplitude_envelope
data_am = hilbert(data)
# ----- #

plt.figure(figsize=(12,4))
plt.plot(data_crop)
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.title("Signal")
plt.show()


frame_width = int(0.5*fs)
w, h = frame_width, data_am.shape[0]//frame_width
image = Image.new('RGB', (w, h))

data_am = data_am.reshape(-1)
px, py = 0, 0
for p in range(data_am.shape[0]):
    print(data_am[p])
    lum = int(data_am[p]//32 - 32)
    if lum < 0: lum = 0
    if lum > 255: lum = 255
    image.putpixel((px, py), (0, lum, 0))
    px += 1
    if px >= w:
        if (py % 50) == 0:
            print(f"Line saved {py} of {h}")
        px = 0
        py += 1
        if py >= h:
            break

image = image.resize((w, 4*h))
plt.imshow(image)
plt.show()
