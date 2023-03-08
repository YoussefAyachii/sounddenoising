"""Wavelet transform method to denoise sound signals"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy as sp
from functions import addgaussian_noise, dwt_wavelet_denoising, best_threshold

# audio_name
audio_name = "phonebip"

# load audio file in the player
audio_path = "audio/"+audio_name+".wav"

# output paths
output_paths = "audio/wavelet_transform/" + audio_name

# 1. load audio file
signal, sampling_rate = librosa.load(audio_path)

# plot original signal
plt.figure(figsize=(18, 8))
librosa.display.waveshow(signal, sr=sampling_rate, alpha=0.5)
plt.xlabel("time (s)")
plt.ylabel("amplitude")
plt.title("*Bass loop* waveform (time domain)")
plt.savefig("figures/wavelet_transform/"+audio_name+"_timedomain.png")

# 2. zoom in to the waveform
samples = range(len(signal))
t = librosa.samples_to_time(samples, sr=sampling_rate)

plt.figure(figsize=(18, 8))
plt.plot(t[10000:10400], signal[10000:10400]) 
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Zoom on the *Bass loop* waveform (time domain)")
plt.savefig("figures/wavelet_transform/"+audio_name+"_timedomain_zoom.png")

# 3. denoising

lambdas = np.linspace(0, 1, 20)

# add gaussian noise
gsignal = addgaussian_noise(audio_path, sampling_rate,
                            output_path_wav=output_paths+"_gaussian_noise.wav",
                            output_path_plot="figures/wavelet_transform/"+audio_name+"_gaussian_noise.png")

# find best threshold
best_lamb, mse_results = best_threshold(signal, sampling_rate, gsignal, lambdas)

print("waveelet best lambda \n", best_lamb)
print("wavelet min mse \n", np.min(mse_results))

# clean signal 
csignal = dwt_wavelet_denoising(gsignal, threshold=best_lamb,
                                wavelet='db2', level=3, sampling_rate=sampling_rate,
                                output_path_plot="figures/wavelet_transform/"+audio_name+"_gaussian_denoise.png",
                                output_path_wav=output_paths+"_gaussian_denoise.wav")



wav.write(output_paths+"_gaussian_denoise.wav",
          sampling_rate, csignal.astype(signal.dtype))
