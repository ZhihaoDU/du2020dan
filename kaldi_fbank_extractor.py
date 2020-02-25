# coding = utf-8
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt


def hz2mel(f):
    return 2595. * np.log10(1. + f / 700.)
def mel2hz(z):
    return 700. * (np.power(10., z / 2595.) - 1.)
def kaldi_mel2hz(z):
    return 700. * (np.exp(z / 1127.) - 1.)
def kaldi_hz2mel(f):
    return 1127. * np.log(1. + f / 700.)


def get_window(win_len, win_type):
    if win_type == 'hanning':
        win_len += 2
        window = np.hanning(win_len)
        window = window[1: -1]
    elif win_type == 'hamming':
        a = 2. * np.pi / (win_len - 1)
        window = 0.54 - 0.46 * np.cos(a * np.arange(win_len))
    elif win_type == 'triangle':
        window = 1. - (np.abs(win_len + 1. - 2.*np.arange(0., win_len+2., 1.)) / (win_len+1.))
        window = window[1: -1]
    elif win_type == 'povey':
        a = 2. * np.pi / (win_len-1)
        window = np.power(0.5 - 0.5 * np.cos(np.arange(win_len) * a), 0.85)
    elif win_type == 'blackman':
        blackman_coeff = 0.42
        a = 2. * np.pi / (win_len - 1)
        window = blackman_coeff - 0.5 * np.cos(a * np.arange(win_len)) + \
                 (0.5 - blackman_coeff) * np.cos(2 * a * np.arange(win_len))
    else:
        window = np.ones(win_len)
    return window

def get_fft_mel_mat(nfft, sr=8000, nfilts=None, width=1.0, minfrq=64, maxfrq=None):
    if nfilts is None:
        nfilts = nfft
    if maxfrq is None:
        maxfrq = sr // 2
    wts = np.zeros((nfilts, nfft//2+1))
    fftfrqs = np.arange(0, nfft//2+1) / (1. * nfft) * (sr)
    minmel = kaldi_hz2mel(minfrq)
    maxmel = kaldi_hz2mel(maxfrq)
    binfrqs = kaldi_mel2hz(minmel + np.arange(0, nfilts+2) / (nfilts+1.) * (maxmel - minmel))
    for i in range(nfilts):
        fs = binfrqs[[i+0, i+1, i+2]]
        fs = fs[1] + width * (fs - fs[1])
        loslope = (fftfrqs - fs[0]) / (fs[1] - fs[0])
        hislope = (fs[2] - fftfrqs) / (fs[2] - fs[1])
        wts[i, :] = np.maximum(0, np.minimum(loslope, hislope))
    return wts

def log_fbank(xx, is_log, is_kalid, need_mag, rs_method):
    sr = 16000
    win_len = 400
    shift_len = 160
    mel_channel = 40
    win_type = 'hamming'
    next_power = np.int32(np.floor(np.power(2., np.ceil(np.log2(win_len)))))
    my_melbank = get_fft_mel_mat(next_power, sr, mel_channel)
    samples = xx.shape[0]
    if is_kalid:
        pre_emphasis_weight = 0.97
        x = np.append(xx[0] - pre_emphasis_weight * xx[0], xx[1:] - pre_emphasis_weight*xx[:-1]).astype(np.float32)
        dither = np.random.standard_normal(samples)
        # x += dither
    else:
        x = xx
    if rs_method is not None:
        if rs_method.lower() == 'zhang':
            # Zhang's Rescale
            c = 32768.
        else:
            # Du's Rescale
            c = np.max(np.abs(x))
        x = x / c

    frames = 1 + (samples - win_len) // shift_len
    window = get_window(win_len, win_type)
    enframe = np.zeros([win_len, frames], np.float32)
    for i in range(frames):
        enframe[:, i] = x[i * shift_len: i * shift_len + win_len]
    if is_kalid:
        enframe = enframe - np.mean(enframe, 0, keepdims=True)
        # enframe = enframe - np.mean(enframe, 1, keepdims=True)
    enframe = np.multiply(enframe, window[:, np.newaxis])
    stft = np.fft.fft(enframe, 512, axis=0)
    mag_spectrum = np.abs(stft[:next_power//2 + 1, :])
    spectrum = np.power(mag_spectrum, 2.)
    c1 = np.matmul(my_melbank, spectrum)
    if is_log:
        c1 = np.log(c1)

    result = [c1.astype(np.float32)]
    if need_mag:
        result.append(mag_spectrum.astype(np.float32))
    if rs_method:
        result.append(c)
    return result


def log_fbank_for_wu(xx, is_log, is_kalid, need_mag, rs_method):
    sr = 16000
    win_len = 400
    shift_len = 160
    mel_channel = 40
    win_type = 'povey'
    next_power = np.int32(np.floor(np.power(2., np.ceil(np.log2(win_len)))))
    my_melbank = get_fft_mel_mat(next_power, sr, mel_channel, minfrq=20)
    samples = xx.shape[0]
    if is_kalid:
        pre_emphasis_weight = 0.97
        x = np.append(xx[0] - pre_emphasis_weight * xx[0], xx[1:] - pre_emphasis_weight*xx[:-1]).astype(np.float32)
        dither = 0.5 * np.random.standard_normal(samples)
        x += dither
    else:
        x = xx
    if rs_method is not None:
        if rs_method.lower() == 'zhang':
            # Zhang's Rescale
            c = 32768.
        else:
            # Du's Rescale
            c = np.max(np.abs(x))
        x = x / c

    frames = 1 + (samples - win_len) // shift_len
    window = get_window(win_len, win_type)
    enframe = np.zeros([win_len, frames], np.float32)
    for i in range(frames):
        enframe[:, i] = x[i * shift_len: i * shift_len + win_len]
    if is_kalid:
        enframe = enframe - np.mean(enframe, 0, keepdims=True)
        # enframe = enframe - np.mean(enframe, 1, keepdims=True)
    enframe = np.multiply(enframe, window[:, np.newaxis])
    stft = np.fft.fft(enframe, 512, axis=0)
    mag_spectrum = np.abs(stft[:next_power//2 + 1, :])
    spectrum = np.power(mag_spectrum, 2.)
    c1 = np.matmul(my_melbank, spectrum)
    if is_log:
        c1 = np.log(c1)

    result = [c1.astype(np.float32)]
    if need_mag:
        result.append(mag_spectrum.astype(np.float32))
    if rs_method:
        result.append(c)
    return result


# if __name__ == '__main__':
#     wav_path = '/home/duzhihao/data/CHiME3/data/audio/16kHz/isolated/dt05_ped_real/F01_050C0101_PED.CH5.wav'
#     sr, data = wavfile.read(wav_path)
#     fbank_feat = log_mel_filterbank_feature(data)
#     print("Done.")