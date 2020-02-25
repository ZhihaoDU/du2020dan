import torch
import numpy as np
from matplotlib import cm
import torchvision.utils as vutils


def wav2mel(wave, win_len, shift_len, window, mel_wts):
    nn = (wave.size(1) - win_len) // shift_len + 1
    frames_list = [wave[:, i*shift_len: i*shift_len+win_len].unsqueeze(1) for i in range(nn)]
    frames = torch.cat(frames_list, 1)
    spect = torch.rfft(frames * window, 1)
    power = (spect[:, :, :, 0].pow(2.) + spect[:, :, :, 1].pow(2.))
    mel_spectrogram = torch.matmul(power, mel_wts)

    return mel_spectrogram


def enframes(wave, win_len, shift_len, window=None):
    nn = (wave.shape[1] - win_len) // shift_len + 1
    frames_list = [wave[:, np.newaxis, i * shift_len: i * shift_len + win_len] for i in range(nn)]
    frames = np.concatenate(frames_list, 1)
    if window is not None:
        frames = frames * window
    return frames


def add_overlap(frames, shift_len):
    bb, tt, win_len = frames.size()
    wav_length = (tt - 1) * shift_len + win_len
    wavs = torch.zeros(bb, wav_length).to(frames)
    weight = torch.zeros(1, wav_length).to(frames)
    for i in range(tt):
        wavs[:, i * shift_len: i * shift_len + win_len] += frames[:, i, :]
        weight[:, i * shift_len: i * shift_len + win_len] += 1.
    return wavs / weight


def mel_post_process(mel_spectrogram):
    reference = 20.0
    min_db = -100

    mel_spectrogram[mel_spectrogram < 1e-4] = 1e-4
    mel_spectrogram = 20 * torch.log10(mel_spectrogram) - reference
    mel_spectrogram = (mel_spectrogram - min_db) / (-min_db)
    mel_spectrogram[mel_spectrogram < 0] = 0
    mel_spectrogram[mel_spectrogram > 1] = 1

    return mel_spectrogram


def wav2mag(wave, win_len, shift_len, window):
    nn = (wave.size(1) - win_len) // shift_len + 1
    frames_list = [wave[:, i*shift_len: i*shift_len+win_len].unsqueeze(1) for i in range(nn)]
    frames = torch.cat(frames_list, 1)
    spect = torch.rfft(frames * window, 1)
    mag_spect = (spect[:, :, :, 0].pow(2.) + spect[:, :, :, 1].pow(2.)).pow(0.5)
    return mag_spect


def wav2spect(wave, win_len, shift_len, window):
    nn = (wave.size(1) - win_len) // shift_len + 1
    frames_list = [wave[:, i*shift_len: i*shift_len+win_len].unsqueeze(1) for i in range(nn)]
    frames = torch.cat(frames_list, 1)
    spect = torch.rfft(frames * window, 1)
    return spect


def add_tf_image(writer, spect, name, num, nrow, step, colorful=True):
    # input state: bb x tt x dd
    x = spect.cpu().data.float()
    x = (x - x.min()) / (x.max() - x.min())
    num = min(x.size(0), num)
    if colorful:
        x = torch.Tensor(np.transpose(cm.get_cmap('jet')(x)[:num, :, :, :3], [0, 3, 2, 1]))
        # bb x 3 x dd x tt
    else:
        x = x.unsqueeze(1).permute(0, 1, 3, 2)
        # bb x 1 x dd x tt
    x = torch.flip(x, [2])
    plot_image = vutils.make_grid(x, nrow, padding=4, normalize=True, scale_each=True)
    writer.add_image(name, plot_image, step)
