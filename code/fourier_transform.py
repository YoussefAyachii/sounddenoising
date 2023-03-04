"""Fourier Transform method to decompose
and denoise sound signals"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy as sp
from functions import (addgaussian_noise, evaluate_denoisation,
                       denoising_with_ft, best_threshold_ft)

# audio name
audio_name = "phonebip"

# load audio file in the player
audio_path = "audio/" + audio_name + ".wav"

# A. Time domain and Frequency domain

# 1. load audio file
signal, sampling_rate = librosa.load(audio_path)

# plot original signal
plt.figure(figsize=(18, 8))
librosa.display.waveshow(signal, sr=sampling_rate, alpha=0.5)
plt.xlabel("time (s)")
plt.ylabel("amplitude")
plt.title(audio_name+" waveform (time domain)")
plt.savefig("figures/fourier_transform/"+audio_name+"_timedomain.png")

# 2. zoom in to the waveform
samples = range(len(signal))
t = librosa.samples_to_time(samples, sr=sampling_rate)

plt.figure(figsize=(18, 8))
plt.plot(t[10000:10400], signal[10000:10400]) 
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Zoom on the *Phone bip* waveform (time domain)")
plt.savefig("figures/fourier_transform/"+audio_name+"_timedomain_zoom.png")

# 3. frequency domain
# fourier transformation (ft) of the signal:
# from time domain to frequency domain
# fft: fast FT
ft = sp.fft.fft(signal)

# frequency and magnitude: by deriving the FT
magnitude = np.absolute(ft)  # magnitude = absolute value of Fourier tronsform
frequency = np.linspace(0, sampling_rate, len(magnitude)) # list from=0, to=len(magnitude), step=1

# Magnitude Spectrum
plt.figure(figsize=(18, 8))
plt.plot(frequency[:5000], magnitude[:5000])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title("Magnitude spectrum (frequency domain)")
plt.savefig("figures/fourier_transform/"+audio_name+"_frequencydomain.png")

# B. Signal Decomposition

# compute the discrete fourier transform (DFT) (f_hat) using the fft algorithm.
f_hat = np.fft.fft(signal)

# compute the power spectral density (power of each frequency)
PSD = f_hat * np.conj(f_hat) / len(signal)

# find all frequencies with large power
# find best PSD threshold
threshold = 50
indices_freq_high_power = PSD > threshold

# clean power spectral density: zero-out componenets with power < threshold
PSDclean = PSD * indices_freq_high_power
f_hat_clean = f_hat * indices_freq_high_power

# from f_hat (fourier domain/freq domain) to f (time domain)
signal_clean = np.fft.ifft(f_hat_clean)

# plot denoised signal
plt.figure(figsize=(18, 8))
plt.plot(t, signal_clean)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title("Denoised signal using FFT (time domain)")
plt.savefig("figures/fourier_transform/"+audio_name+"_denoised_timedomain.png")


# plot results in same graph
fig, ax = plt.subplots(5, 1, constrained_layout = True)

# original signal: time domain
ax[0].plot(t, signal)
ax[0].set_title("original signal (time domain)")
ax[0].set_ylim([-1, 1])

# original signal: frequency domain (=magnitude spectrum)
magnitude = np.absolute(f_hat)
frequency = np.linspace(0, sampling_rate, len(magnitude))
ax[1].plot(frequency, magnitude)
ax[1].set_title("magnitude spectrum (frequency domain)")
ax[1].set_ylim([0, 1000 + np.max(magnitude)])

# power spectral density (PSD)
ax[2].plot(t, PSD)
ax[2].set_title("power spectral density (PSD)")
#ax[2].set_ylim([0, 2000])

# clean power spectral density: zero-out componenets with power < threshold 
ax[3].plot(t, PSDclean)
ax[3].set_title("clean power spectral density (PSD) (PSD>threshold)")
ax[3].set_ylim([0, 100 + np.max(PSDclean)])

# clean signal
ax[4].plot(t, signal_clean)
ax[4].set_ylim([-1, 1])
ax[4].set_title("signal denoised")

#save fig
plt.savefig("figures/fourier_transform/"+audio_name+"_decomp_summary.png")

# save sound after ft decomposition
wav.write("audio/fourier_transform/"+audio_name+"_decomp.wav", sampling_rate,
          signal_clean.astype(signal.dtype))


# B. Signal Denoising

# add gaussian noise to the original signal: mu=mean(signal and sigma=std(signal)
gsignal = addgaussian_noise(audio_path, sampling_rate,
                            output_path_wav="audio/fourier_transform/"
                            +audio_name+"_gaussian_noise.wav",
                            output_path_plot="figures/fourier_transform/"
                            +audio_name+"_gaussian_noise.png")


# find best PSD threshold
thresholds = np.linspace(1, 2000, 50)
best_threshold = best_threshold_ft(
    signal, gsignal, sampling_rate, sampling_rate,
    thresholds, mean="NA", sd="NA",
    output_path="figures/fourier_transform/"+audio_name+"_best_threshold.png")

# denosing signal using fourier transform decomposition
csignal = denoising_with_ft(gsignal, sampling_rate, psd_threshold=best_threshold,
                            output_path="audio/fourier_transform/"
                            +audio_name+"_gaussian_denoise.wav",
                            plots=True, audio_name=audio_name)

# evaluate signal denoisation
evaluate_denoisation(audio_path,
                     sampling_rate=sampling_rate,
                     pse_thresholds=thresholds,
                     output_path="figures/fourier_transform/"+audio_name+"_denoising_eval.png")