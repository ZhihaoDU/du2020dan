# coding = utf-8
import numpy as np
from speech_utils import read_sphere_wav, random_mix_speech_noise, print_with_time
from scipy.io import wavfile
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import scipy.io as sio


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

    def __init__(self, mix_dbs, noise_scp, simu_noisy, target_type, is_train, rs_method) -> None:
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
    def preprocess_speech(xx):
        pre_emphasis_weight = 0.97
        samples = xx.shape[0]
        x = np.append(xx[0] - pre_emphasis_weight * xx[0], xx[1:] - pre_emphasis_weight * xx[:-1]).astype(np.float32)
        dither = np.random.standard_normal(samples)
        # x += dither
        return x

    def enframe_speech_pair(self, _clean_speech, _noisy_speech):
        _noisy_speech = self.preprocess_speech(_noisy_speech)
        _clean_speech = self.preprocess_speech(_clean_speech)

        if self.rs_method is not None:
            if self.rs_method.lower() == 'zhang':
                c = 32768.
            else:
                c = np.max(np.abs(_noisy_speech))
            _noisy_speech /= c
            _clean_speech /= c

        frame_number = (len(_clean_speech) - self.win_len) // self.win_shift
        _clean_frames = np.zeros([frame_number, self.win_len], np.float32)
        _noisy_frames = np.zeros([frame_number, self.win_len], np.float32)
        for i in range(frame_number):
            _clean_frames[i, :] = _clean_speech[i * self.win_shift: i * self.win_shift + self.win_len]
            _noisy_frames[i, :] = _noisy_speech[i * self.win_shift: i * self.win_shift + self.win_len]
        return _clean_frames, _noisy_frames, frame_number

    def __getitem__(self, index):

        frame_list = {}
        # build chime2 simu data
        idx = index % len(self.simu_path_list)
        noisy_path = self.simu_path_list[idx]
        noisy_wav = self.read_wav(noisy_path)
        clean_path = noisy_path.replace(".wav", "_%s.wav" % self.target_type)
        clean_wav = self.read_wav(clean_path)
        c = np.sqrt(np.mean(np.square(noisy_wav)))
        noisy_wav = noisy_wav / c
        clean_wav = clean_wav / c
        frame_list['simu'] = self.enframe_speech_pair(clean_wav, noisy_wav)

        # build my simulate utterance
        # clean_wav = self.read_wav(self.speech_path_list[idx])
        snr = self.mix_dbs[np.random.randint(len(self.mix_dbs))]
        noise_wav = self.noise_list[np.random.randint(len(self.noise_list))]
        if self.is_train:
            noise_start = np.random.randint(0, len(noise_wav) // 2 - len(clean_wav))
        else:
            noise_start = np.random.randint(len(noise_wav) // 2, len(noise_wav) - len(clean_wav))
        _noisy_speech, _clean_speech = random_mix_speech_noise(clean_wav, noise_wav, snr, noise_start,
                                                               noise_start + len(clean_wav), False)
        c = np.sqrt(np.mean(np.square(_noisy_speech)))
        _noisy_speech = _noisy_speech / c
        _clean_speech = _clean_speech / c
        frame_list['my_simu'] = self.enframe_speech_pair(_clean_speech, _noisy_speech)

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

    def extract_power(self, frames):
        t_frames = torch.Tensor(frames).to(self.opts['device'])
        t_frames -= t_frames.mean(2, True)
        t_frames = t_frames * self.window

        t_frames = F.pad(t_frames, (0, self.next_power - self.opts['win_len']))
        spect = torch.rfft(t_frames, 1)
        power_spect = torch.pow(spect[:, :, :, 0], 2.0) + torch.pow(spect[:, :, :, 1], 2.0)
        # c1 =torch.matmul(power_spect, self.melbank)
        return power_spect

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
    def my_collate_func(frames_list):
        max_frame = 1024
        batch_size = len(frames_list)
        batch_clean = np.zeros([batch_size*2, max_frame, 400], np.float32)
        batch_noisy = np.zeros([batch_size*2, max_frame, 400], np.float32)
        batch_mask = np.zeros([batch_size*2, max_frame], np.float32)
        batch_frame_number = [0] * len(frames_list) * 2
        i = 0
        for one_dict in frames_list:
            tt = min(one_dict['my_simu'][2], max_frame)
            batch_clean[i, :tt, :] = one_dict['my_simu'][0][:tt, :]
            batch_noisy[i, :tt, :] = one_dict['my_simu'][1][:tt, :]
            batch_frame_number[i] = tt
            batch_mask[i, :tt] = 1.

            tt = min(one_dict['simu'][2], max_frame)
            batch_clean[i+batch_size, :tt, :] = one_dict['simu'][0][:tt, :]
            batch_noisy[i+batch_size, :tt, :] = one_dict['simu'][1][:tt, :]
            batch_frame_number[i+batch_size] = tt
            batch_mask[i + batch_size, :tt] = 1.

            i += 1
        return batch_clean, batch_noisy, batch_frame_number, batch_mask

    def calc_feats_and_align(self, clean_frames, noisy_frames, left_frame=0, right_frame=0):
        with torch.no_grad():
            feats = self.extract_power(noisy_frames)
            tgts = self.extract_power(clean_frames)
            noises = self.extract_power(noisy_frames - clean_frames)
            n, t, d = feats.size()
            if left_frame > 0 or right_frame > 0:
                pad_feats = F.pad(feats.unsqueeze(1), (0, 0, left_frame, right_frame)).squeeze(1)
                ex_list = [pad_feats[:, i:i+t, :] for i in range(left_frame+1+right_frame)]
                ex_feats = torch.cat(ex_list, 2)
                return ex_feats.detach(), tgts.detach()
            return feats.detach(), tgts.detach(), noises.detach()

    def calc_feats_irms_and_align(self, clean_frames, noisy_frames, left_frame=0, right_frame=0):
        with torch.no_grad():
            feats = self.extract_power(noisy_frames)
            tgts = self.extract_power(clean_frames)
            noise = self.extract_power(noisy_frames - clean_frames)
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
