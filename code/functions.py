"""Functions"""

import scipy.io
import librosa.display
import librosa
import scipy as sp
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import pywt


# add gaussian noise to the original signal

def addgaussian_noise(audio_path, sampling_rate, mean="NA", sd="NA",
                      output_path_wav=False, output_path_plot=False):
    """Add guassian noise to signal

    Args:
        audio_path (np.array): original signal to add noise to
        mean (float): mean of the chosen gaussian
        sd (float): sd of the chosen gaussian
        output_path (str): path to save figure (noised audio)

    Returns:
        noisy_signal: original image with gaussian noise.
    """
    
    input_sig, sampling_rate = librosa.load(audio_path)
    if mean == "NA":
        mean = np.mean(input_sig)
    if sd == "NA":
        sd = np.std(input_sig)
    
    # check args
    assert len(input_sig.shape) == 1

    gaussian = np.random.normal(mean, sd, input_sig.shape[0])
    noisy_sig = np.zeros(input_sig.shape, np.float32)
    noisy_sig = input_sig + gaussian

    # save signal sound
    if output_path_wav != False:
        scipy.io.wavfile.write(output_path_wav, sampling_rate,
                               noisy_sig.astype(input_sig.dtype))
    
    # save signal waveform
    if output_path_plot != False:
        plt.figure(figsize=(18, 8))
        librosa.display.waveshow(noisy_sig, sr=sampling_rate, alpha=0.5)
        plt.xlabel("time (s)")
        plt.ylabel("amplitude")
        plt.title("original signal with gaussian noise"
                  + "\n ({})".format(output_path_plot))
        plt.savefig(output_path_plot)

    return noisy_sig


# denising using fourier transform

def denoising_with_ft(input_sig, sampling_rate, psd_threshold,
                      output_path=False, plots=False,
                      audio_name="phonebip"):
    """denoise input signal using fourier transform.

    Args:
        input_sig (np.array): signal to remove noise from
        psd_threshold (float): minimum value of frequency power to select
        output_path (str): path to save signal (noised audio)
    
    Returns:
        signal_clean: denoised signal.
    """

    # check args
    assert len(input_sig.shape) == 1
    
    # 1. FFT: from time domain to frequency domain
    # compute the fft (f_hat) of the noisy signal using the fft algorithm.
    ft = sp.fft.fft(input_sig)
    # frequency and magnitude: by deriving the FT
    magnitude = np.absolute(ft)
    frequency = np.linspace(0, sampling_rate, len(magnitude))
    
    if plots != False:
        # plot denoised in frequency domain 
        plt.figure(figsize=(18, 8))
        plt.plot(frequency, magnitude)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title("Denoised signal using FFT (frequency domain)")
        plt.savefig("figures/fourier_transform/"
                    +audio_name+"_noised_frequencydomain.png")

    # compute the power spectral density (power of each frequency)
    PSD = ft * np.conj(ft) / len(input_sig)
    t = np.arange(len(PSD))

    if plots != False:
        plt.figure(figsize=(18, 8))
        plt.plot(t, PSD)
        plt.title("power spectral density (PSD)")
        plt.savefig("figures/fourier_transform/"
                    +audio_name+"_noised_PSD.png")
    
    # find all frequencies with large power
    f_hat_clean = ft * (PSD > psd_threshold)

    if plots != False:
        plt.figure(figsize=(18, 8))
        plt.plot(frequency, f_hat_clean)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title("Denoised signal using FFT (frequency domain)")
        plt.savefig("figures/fourier_transform/"
                    +audio_name+"_denoised_frequencydomain.png")

    # from f_hat (fourier domain/freq domain) to f (time domain)
    signal_clean = np.fft.ifft(f_hat_clean)
    
    if plots != False:
        plt.figure(figsize=(18, 8))
        plt.plot(t, signal_clean)
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.title("Denoised signal using FFT (time domain)")
        plt.savefig("figures/fourier_transform/"+
                    audio_name+"_denoised_timedomain.png")

    # save signal
    if output_path != False:
        scipy.io.wavfile.write(output_path, sampling_rate,
                            signal_clean.astype(input_sig.dtype))
    return signal_clean


#c. evaluating denoisation

def mse_fftdenoising(sig_original, sig_denoised):
    """returns mse between two vectors of data"""
    return np.mean(np.square(sig_original - sig_denoised))


def evaluate_denoisation(original_audio_path, sampling_rate, pse_thresholds,
                         mean="NA", sd="NA", output_path=False):
    """finding the best pse threshold that returns the lowest mse error
    between original signal and denoised signal using fft
    after adding gaussian signal to the original signal"""
    
    sig_original, sampling_rate = librosa.load(original_audio_path)
    
    mse_results = np.zeros(len(pse_thresholds))

    # gaussian noise params
    if mean == "NA":
        mean = np.mean(sig_original)
    if sd == "NA":
        sd = np.std(sig_original)

    # noising
    noisy_sig = addgaussian_noise(original_audio_path, sampling_rate,
                                  mean=mean, sd=sd)

    for i, t in enumerate(pse_thresholds):
        # denoising using fft
        clean_sig = denoising_with_ft(noisy_sig, sampling_rate, psd_threshold=t)
        
        # mse original vs denoised
        mse_results[i] = mse_fftdenoising(sig_original, clean_sig)

    # print result
    min_mse = np.min(mse_results)
    best_pse_threshold = pse_thresholds[np.argmin(mse_results)]
    print(" best pse threshold between {} is {} which returns mse={}"
        .format((np.min(pse_thresholds), np.max(pse_thresholds)),
                best_pse_threshold, min_mse))
    
    # save fig
    if output_path != False:
        fig = plt.figure(figsize=(4, 6))
        plt.plot(pse_thresholds, mse_results)
        plt.xlabel("pse threshold")
        plt.ylabel("loss (mse)")
        plt.scatter(x=best_pse_threshold, y=min_mse, color="red",marker="o",
                    label="best pse threshold = {}".format(round(best_pse_threshold)), alpha=1)
        plt.title("mse between original signal and \n signal after fft reconstruction")
        plt.legend()
        plt.savefig(output_path)



def best_threshold_ft(original_audio, noised_audio,
                      original_sampling_rate, noised_sampling_rate,
                      psd_thresholds, mean="NA", sd="NA",
                      output_path=False):
    """finding the best PSD threshold that returns the lowest mse error
    between original signal and denoised signal using fft
    after adding gaussian signal to the original signal"""

    mse_results = np.zeros(len(psd_thresholds))

    # gaussian noise params
    if mean == "NA":
        mean = np.mean(original_audio)
    if sd == "NA":
        sd = np.std(original_audio)

    for i, t in enumerate(psd_thresholds):
        # denoising using fft
        clean_sig = denoising_with_ft(
            noised_audio, noised_sampling_rate, psd_threshold=t)

        # mse original vs denoised
        mse_results[i] = mse_fftdenoising(noised_audio, clean_sig)

    # print result
    min_mse = np.min(mse_results)
    best_psd_threshold = psd_thresholds[np.argmin(mse_results)]
    print(" best pse threshold between {} is {} which returns mse={}"
        .format((np.min(psd_thresholds), np.max(psd_thresholds)),
                best_psd_threshold, min_mse))

    # save fig
    if output_path != False:
        fig = plt.figure(figsize=(4, 6))
        plt.plot(psd_thresholds, mse_results)
        plt.xlabel("psd threshold")
        plt.ylabel("mse")
        plt.scatter(x=best_psd_threshold, y=min_mse, color="red",marker="o",
                    label="best psd threshold = {}".format(
                        round(best_psd_threshold)), alpha=1)
        plt.title("finding the best threshold:"
                  + "mse between signal and denoised signal"
                  + "\n after adding guassian noise")
        plt.legend()
        plt.savefig(output_path)

    return best_psd_threshold

# wavelet functions 

def dwt_wavelet_denoising(noisy_sig, sampling_rate, threshold=0.7, wavelet='db2', level=1,
                          output_path_wav=False, output_path_plot=False):
    # signal decomposition
    coeff = pywt.wavedec(noisy_sig, wavelet, mode="per")
    # fixing threshold
    coeff[1:] = (pywt.threshold(i, value=threshold, mode='hard') for i in coeff[1:])    
    # signal reconstruction
    clean_signal = pywt.waverec(coeff, wavelet, mode='per')

    # save signal sound
    if output_path_wav != False:
        scipy.io.wavfile.write(output_path_wav, sampling_rate,
                               clean_signal.astype(noisy_sig.dtype))

    # save signal waveform
    if output_path_plot != False:
        plt.figure(figsize=(18, 8))
        librosa.display.waveshow(clean_signal, sr=sampling_rate, alpha=0.5)
        plt.xlabel("time (s)")
        plt.ylabel("amplitude")
        plt.title("signal after wavelet denoising"
                  + "\n ({})".format(output_path_plot))
        plt.savefig(output_path_plot)

    return clean_signal[:-1]


def best_threshold(original_sig, sampling_rate, noised_sig, lambdas):

    mse_results = np.zeros(lambdas.shape)
    for i, lamb in enumerate(lambdas):
        clean_signal = dwt_wavelet_denoising(noised_sig, sampling_rate, threshold=lamb)
        mse_results[i] = np.mean(np.square(original_sig - clean_signal[-1]))
    best_lambda = lambdas[np.argmin(mse_results)]

    return best_lambda, mse_results
