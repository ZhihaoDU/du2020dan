# coding = utf-8
import numpy as np
from speech_utils import read_sphere_wav, random_mix_speech_noise, print_with_time, calc_rescale_c
from scipy.io import wavfile
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import scipy.io as sio
from utils.emph import pre_emphasis


class FrameDataset(Dataset):

    @staticmethod
    def read_path_list(list_file_path):
        f = open(list_file_path, 'r')
        file_list = f.readlines()
        for i in range(len(file_list)):
            file_list[i] = file_list[i].replace('\n', '')
            file_list[i] = file_list[i].split(' ')[-1]
        return np.array(file_list)

    @staticmethod
    def get_id(utt_path):
        ll = utt_path.split('/')
        ll = ll[-1].split('.')
        ll = ll[0].split('_')
        return ll[0]

    def __init__(self, mix_dbs, noise_scp, simu_noisy, target_type, is_train, rs_method, data_augment=None) -> None:
        super().__init__()
        self.sample_freq = 16000
        self.win_len = 25 * self.sample_freq // 1000
        self.win_shift = 10 * self.sample_freq // 1000

        self.mix_dbs = mix_dbs
        self.is_train = is_train
        self.target_type = target_type

        self.simu_path_list = self.read_path_list(simu_noisy)

        self.noise_path_list = self.read_path_list(noise_scp)[:1]
        self.noise_number = len(self.noise_path_list)
        self.noise_list = []
        for i in range(self.noise_number):
            # sr, noise = wavfile.read(self.noise_path_list[i])
            # noise, sr = librosa.load(self.noise_path_list[i])
            noise = self.read_wav(self.noise_path_list[i])
            if noise.shape[0] > self.sample_freq * 2 * 60:
                self.noise_list.append(noise)

        self.epoch = 0
        self.rs_method = rs_method
        self.data_augment = data_augment

    @staticmethod
    def read_wav(wav_path):
        if wav_path.endswith('wv1') or wav_path.endswith('sph'):
            data, header = read_sphere_wav(wav_path)
        else:
            sr, data = wavfile.read(wav_path)
            # data, sr = librosa.load(wav_path, 16000, mono=True)
            if len(data.shape) > 1:
                data = np.mean(data, 1)
        return data

    @staticmethod
    def split_speech_pair(_clean_speech, _noisy_speech):
        _noisy_speech = pre_emphasis(_noisy_speech, 0.97)
        _clean_speech = pre_emphasis(_clean_speech, 0.97)

        return _clean_speech, _noisy_speech

    def __getitem__(self, index):

        frame_list = []
        # origin data
        idx = index % len(self.simu_path_list)
        noisy_path = self.simu_path_list[idx]
        noisy_wav = self.read_wav(noisy_path)
        if "early" in self.target_type or "clean" in self.target_type:
            clean_path = noisy_path.replace(".wav", "_%s.wav" % self.target_type)
        else:
            clean_path = noisy_path.replace("isolated", "scaled")
        clean_wav = self.read_wav(clean_path)
        # use "power_norm" to normalize the waveform
        # c = np.sqrt(np.mean(np.square(noisy_wav)))
        c = calc_rescale_c(noisy_wav, self.rs_method)
        frame_list.append(self.split_speech_pair(clean_wav / c, noisy_wav / c))

        # data augmentation
        if self.data_augment != "None":
            # naive method
            if self.data_augment == "naive":
                snr = self.mix_dbs[np.random.randint(len(self.mix_dbs))]
                noise_wav = self.noise_list[np.random.randint(len(self.noise_list))]
                if self.is_train:
                    noise_start = np.random.randint(0, len(noise_wav) // 2 - len(clean_wav))
                else:
                    noise_start = np.random.randint(len(noise_wav) // 2, len(noise_wav) - len(clean_wav))
                scaled_wav = self.read_wav(noisy_path.replace("isolated", "scaled"))
                _noisy_speech, _clean_speech = random_mix_speech_noise(scaled_wav, noise_wav, snr, noise_start,
                                                                       noise_start + len(clean_wav), False)
                # c = np.sqrt(np.mean(np.square(_noisy_speech)))
                c = calc_rescale_c(_noisy_speech, self.rs_method)
                frame_list.append(self.split_speech_pair(clean_wav / c, _noisy_speech / c))

        return frame_list

    def __len__(self):
        return len(self.simu_path_list)


class FbankDataloader(DataLoader):

    def __init__(self, dataset, opts, batch_size, shuffle=True, sampler=None, batch_sampler=None, num_workers=8,
                 collate_fn=None, pin_memory=False, drop_last=True, timeout=0, worker_init_fn=None):
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, self.my_collate_func, pin_memory,
                         drop_last, timeout, worker_init_fn)
        self.opts = opts
        window = self.get_window(self.dataset.win_len, opts['win_type'])
        self.window = torch.Tensor(window[np.newaxis, np.newaxis, :]).to(self.opts['device'])
        self.next_power = next_power = np.int(np.power(2, np.ceil(np.log2(opts['win_len']))))
        my_melbank = self.get_fft_mel_mat(next_power, opts['sr'], opts['mel_channels'])
        self.melbank = torch.Tensor(my_melbank.T).to(opts['device'])

    @staticmethod
    def kaldi_mel2hz(z):
        return 700. * (np.exp(z / 1127.) - 1.)
    @staticmethod
    def kaldi_hz2mel(f):
        return 1127. * np.log(1. + f / 700.)

    def get_fft_mel_mat(self, nfft, sr=8000, nfilts=None, width=1.0, minfrq=64, maxfrq=None):
        if nfilts is None:
            nfilts = nfft
        if maxfrq is None:
            maxfrq = sr // 2
        wts = np.zeros((nfilts, nfft // 2 + 1))
        fftfrqs = np.arange(0, nfft // 2 + 1) / (1. * nfft) * (sr)
        minmel = self.kaldi_hz2mel(minfrq)
        maxmel = self.kaldi_hz2mel(maxfrq)
        binfrqs = self.kaldi_mel2hz(minmel + np.arange(0, nfilts + 2) / (nfilts + 1.) * (maxmel - minmel))
        for i in range(nfilts):
            fs = binfrqs[[i + 0, i + 1, i + 2]]
            fs = fs[1] + width * (fs - fs[1])
            loslope = (fftfrqs - fs[0]) / (fs[1] - fs[0])
            hislope = (fs[2] - fftfrqs) / (fs[2] - fs[1])
            wts[i, :] = np.maximum(0, np.minimum(loslope, hislope))
        return wts

    def extract_power(self, wavs):
        t_frames = self.enframes(wavs)
        t_frames -= t_frames.mean(2, True)
        t_frames = t_frames * self.window

        t_frames = F.pad(t_frames, (0, self.next_power - self.opts['win_len']))
        spect = torch.rfft(t_frames, 1)
        power_spect = torch.pow(spect[:, :, :, 0], 2.0) + torch.pow(spect[:, :, :, 1], 2.0)
        # c1 =torch.matmul(power_spect, self.melbank)
        return power_spect

    def enframes(self, wavs):
        bb, tt = wavs.size()
        frame_number = (tt - 400) // 160
        t_frames = torch.zeros([bb, frame_number, 400]).to(wavs)
        for i in range(frame_number):
            t_frames[:, i, :] = wavs[:, i * 160: i * 160 + 400]
        return t_frames

    def extract_fbank(self, wavs):
        t_frames = self.enframes(wavs)
        t_frames -= t_frames.mean(2, True)
        t_frames = t_frames * self.window

        t_frames = F.pad(t_frames, (0, self.next_power - self.opts['win_len']))
        spect = torch.rfft(t_frames, 1)
        power_spect = torch.pow(spect[:, :, :, 0], 2.0) + torch.pow(spect[:, :, :, 1], 2.0)
        c1 = torch.matmul(power_spect, self.melbank)
        return c1

    @staticmethod
    def get_window(win_len, win_type):
        if win_type == 'hanning':
            win_len += 2
            window = np.hanning(win_len)
            window = window[1: -1]
        elif win_type == 'hamming':
            a = 2. * np.pi / (win_len - 1)
            window = 0.54 - 0.46 * np.cos(a * np.arange(win_len))
        elif win_type == 'triangle':
            window = 1. - (np.abs(win_len + 1. - 2. * np.arange(0., win_len + 2., 1.)) / (win_len + 1.))
            window = window[1: -1]
        elif win_type == 'povey':
            a = 2. * np.pi / (win_len - 1)
            window = np.power(0.5 - 0.5 * np.cos(np.arange(win_len) * a), 0.85)
        elif win_type == 'blackman':
            blackman_coeff = 0.42
            a = 2. * np.pi / (win_len - 1)
            window = blackman_coeff - 0.5 * np.cos(a * np.arange(win_len)) + \
                     (0.5 - blackman_coeff) * np.cos(2 * a * np.arange(win_len))
        else:
            window = np.ones(win_len)
        return window

    @staticmethod
    def my_collate_func(speech_pair_list):
        seg_len = 16384
        n_item = len(speech_pair_list[0])
        batch_size = len(speech_pair_list)
        batch_clean = np.zeros([batch_size * n_item, 1, seg_len], np.float32)
        batch_noisy = np.zeros([batch_size * n_item, 1, seg_len], np.float32)
        for i, one_list in enumerate(speech_pair_list):
            for j, speech_pair in enumerate(one_list):
                clean_speech, noisy_speech = speech_pair
                n_samples = len(clean_speech)
                if n_samples > seg_len:
                    ss = np.random.randint(0, n_samples-seg_len)
                else:
                    ss = 0
                batch_clean[i*n_item+j, 0, :] = clean_speech[ss: ss + seg_len]
                batch_noisy[i*n_item+j, 0, :] = noisy_speech[ss: ss + seg_len]

        return batch_clean, batch_noisy

    def calc_feats_and_align(self, clean_frames, noisy_frames, left_frame=0, right_frame=0):
        with torch.no_grad():
            feats = self.extract_fbank(noisy_frames)
            tgts = self.extract_fbank(clean_frames)
            noises = self.extract_fbank(noisy_frames - clean_frames)
            n, t, d = feats.size()
            if left_frame > 0 or right_frame > 0:
                pad_feats = F.pad(feats.unsqueeze(1), (0, 0, left_frame, right_frame)).squeeze(1)
                ex_list = [pad_feats[:, i:i+t, :] for i in range(left_frame+1+right_frame)]
                ex_feats = torch.cat(ex_list, 2)
                return ex_feats.detach(), tgts.detach()
            return feats.detach(), tgts.detach(), noises.detach()

    def calc_feats_irms_and_align(self, clean_frames, noisy_frames, left_frame=0, right_frame=0):
        with torch.no_grad():
            feats = self.extract_fbank(noisy_frames)
            tgts = self.extract_fbank(clean_frames)
            noise = self.extract_fbank(noisy_frames - clean_frames)
            n, t, d = feats.size()
            irm = tgts / (tgts + noise)
            irm[torch.isnan(irm)] = 0.
            irm[torch.isinf(irm)] = 0.
            if left_frame > 0 or right_frame > 0:
                pad_feats = F.pad(feats.unsqueeze(1), (0, 0, left_frame, right_frame)).squeeze(1)
                ex_list = [pad_feats[:, i:i+t, :] for i in range(left_frame+1+right_frame)]
                ex_feats = torch.cat(ex_list, 2)
                return ex_feats.detach(), tgts.detach(), irm.detach()
            return feats.detach(), tgts.detach(), irm.detach()
