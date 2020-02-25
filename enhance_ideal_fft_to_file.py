import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kaldi_helper import KaldiFeatHolder
from speech_utils import read_path_list, print_with_time
from scipy.io import wavfile
from kaldi_fbank_extractor import log_fbank, get_fft_mel_mat
import scipy.io as sio
from multiprocessing import Pool
import librosa
from models.Tan2018PowCRN import Generator


def post_process(src, target):
    print_with_time("Doing post process...")
    os.system("cp %s/spk2utt %s/" % (src, target))
    os.system("cp %s/text %s/" % (src, target))
    os.system("cp %s/utt2spk %s/" % (src, target))
    os.system("cp %s/wav.scp %s/" % (src, target))
    os.system("cp %s/spk2gender %s/" % (src, target))


def calc_func(noisy_dir_path):
    # nn.Module.dump_patches = True
    melbank = get_fft_mel_mat(512, 16000, 40)
    method = "Tan2018CRN_mag_early50"
    device = torch.device("cuda")
    print_with_time("Loading model...")
    model = Generator(64, 256).to(device)
    checkpoint = torch.load("Checkpoints/Tan2018CRN_mag_early50/checkpoint_000096336.pth")
    model.load_state_dict(checkpoint["generator"])
    model.eval()
    print_with_time("Start to enhance wav file in %s with method %s\n" % (noisy_dir_path, method))
    udir_path = "%s_%s" % (noisy_dir_path, method)
    if not os.path.exists(udir_path):
        os.mkdir(udir_path)
    wav_scp = read_path_list(os.path.join(noisy_dir_path, "wav.scp"))
    ark_file = open(os.path.join(udir_path, "feats.ark"), 'wb')
    scp_file = open(os.path.join(udir_path, "feats.scp"), 'w')
    key_len = wav_scp[0].find(' ')
    kaldi_holder = KaldiFeatHolder(key_len, 2000, 40)
    offset = key_len + 1
    enhanced_number = 0
    left_frame = 0
    right_frame = 0
    for one_wav in wav_scp:
        wav_id, wav_path = one_wav.split(' ')
        # wav_path = one_wav
        # if "dB" in wav_path:
        #     wav_path = wav_path.replace(".wav", "_early50.wav")
        sr, noisy_speech = wavfile.read(wav_path)
        # sr, early50_speech = wavfile.read(wav_path.replace(".wav", "_early50.wav"))
        # sr, direct_speech = wavfile.read(wav_path.replace(".wav", "_direct_sound.wav"))
        # process binaural waves.
        if len(noisy_speech.shape) > 1:
            noisy_speech = np.mean(noisy_speech, 1)
        # if len(early50_speech.shape) > 1:
        #     early50_speech = np.mean(early50_speech, 1)
        # if len(direct_speech.shape) > 1:
        #     direct_speech = np.mean(direct_speech, 1)
        # noisy_speech = noisy_speech.astype(np.int16)
        # librosa.output.write_wav("/home/duzhihao/440c0201_mono.wav", noisy_speech.astype(np.float32), 16000, True)

        c = np.sqrt(np.mean(np.square(noisy_speech)))
        noisy_speech = noisy_speech / c
        # early50_speech = early50_speech / c

        n_noisy_feat, n_noisy_mag = log_fbank(noisy_speech, False, True, True, None)
        # n_noise_feat, n_noise_mag = log_fbank(noisy_speech - early50_speech, False, True, True, None)
        # n_early50_feat, n_early50_mag = log_fbank(early50_speech, False, True, True, None)
        # n_irm = n_early50_feat[0] / (n_early50_feat[0] + n_noise_feat[0])
        log_noisy_power = np.log(n_noisy_mag ** 2 + 1e-12)
        # log_early50_power = np.log(n_early50_mag ** 2 + 1e-12)
        # log_mask = np.clip((log_early50_power+10) / (log_noisy_power+10), 0, 1)
        # log_enhanced_power = (log_noisy_power+10) * log_mask - 10
        # enhanced_power = np.exp(log_enhanced_power)
        # n_direct_feat = log_fbank(direct_speech, False, True, False, None)
        # n_noisy_feat = log_fbank_for_wu(noisy_speech, False, True, False, None)
        # log_noisy_feat = np.log(n_noisy_feat[0].T)
        # log_noisy_feat[np.isnan(log_noisy_feat)] = 0.
        # log_noisy_feat[np.isinf(log_noisy_feat)] = 0.
        # log_noisy_feat = log_noisy_feat[np.newaxis, :, :]
        # log_noisy_feat = torch.Tensor(log_noisy_feat).to(device)
        # n, t, d = log_noisy_feat.size()
        # if left_frame > 0 or right_frame > 0:
        #     pad_feats = F.pad(log_noisy_feat.unsqueeze(1), (0, 0, left_frame, right_frame)).squeeze(1)
        #     ex_list = [pad_feats[:, i:i+t, :] for i in range(left_frame+1+right_frame)]
        #     log_noisy_feat = torch.cat(ex_list, 2)

        # with torch.no_grad():
        #     feat_ex_list = [log_noisy_feat[:, :, i * 40:(i + 1) * 40].unsqueeze(1) for i in range(left_frame + 1 + right_frame)]
        #     nn_input = torch.cat(feat_ex_list, 1)
        #     mask = torch.sigmoid(model.forward(log_noisy_feat))
        #     mask = model.forward(log_noisy_feat)
        #     enhanced_feat = mask[0, :, :].cpu().numpy() * n_noisy_feat[0].T
        #     log_enhanced_feat = np.log(enhanced_feat)
        #     log_enhanced_feat = log_noisy_feat[:, :, 40*left_frame:40*(left_frame+1)] # * mask
        #     log_enhanced_feat = model.forward(log_noisy_feat)
        #     log_enhanced_feat = log_enhanced_feat * (log_pow_max - log_pow_min) + log_pow_min
        #     log_enhanced_feat = log_enhanced_feat.cpu().numpy()

        sio.savemat(method + "_chime2", {'noisy_mag': n_noisy_mag,
                                         'irm': log_mask,
                                         'early50_mag': n_early50_mag,
                                         'enhanced_mag': np.sqrt(enhanced_power),
                                         # 'direct_feat': n_direct_feat[0]
                                         })
        return

        kaldi_holder.set_key(wav_id)
        # kaldi_holder.set_value(log_enhanced_feat[0, :, :].cpu().numpy())
        # kaldi_holder.set_value(np.log((n_early50_feat[0]).T + 1e-8))
        kaldi_holder.set_value(np.log((n_noisy_feat[0] * n_irm).T + 1e-8))
        kaldi_holder.write_to(ark_file)
        scp_file.write("%s %s/feats.ark:%d\n" % (wav_id, udir_path, offset))
        offset += kaldi_holder.get_real_len()
        enhanced_number += 1
        if enhanced_number % 40 == 0:
            print_with_time(
                "Enhanced %5d(%6.2f%%) utterance" % (enhanced_number, 100. * enhanced_number / len(wav_scp)))
    print_with_time("Enhanced %d utterance" % enhanced_number)
    ark_file.close()
    scp_file.close()
    post_process(noisy_dir_path, udir_path)


if __name__ == '__main__':

    opts = {}
    opts['win_len'] = 400
    opts['sr'] = 16000
    opts['device'] = torch.device('cuda:0')

    noisy_dir_list = [
        # "/data/duzhihao/kaldi/egs/chime2/s5/data-fbank/dev_dt_05_clean",
        "/data/duzhihao/kaldi/egs/chime2/s5/data-fbank/dev_dt_05_noisy",
        # "/data/duzhihao/kaldi/egs/chime2/s5/data-fbank/dev_dt_05_reverb",
        # "/data/duzhihao/kaldi/egs/chime2/s5/data-fbank/test_eval92_5k_clean",
        # "/data/duzhihao/kaldi/egs/chime2/s5/data-fbank/test_eval92_5k_noisy",
    ]
    # noisy_dir_list = ["wav_scp"]
    # pool = Pool(4)
    pool = Pool(len(noisy_dir_list))
    pool.map(calc_func, noisy_dir_list)
    pool.close()



