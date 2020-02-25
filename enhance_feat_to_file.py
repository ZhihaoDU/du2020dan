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
from models.Tan2018MelCRN import Generator


def post_process(src, target):
    print_with_time("Doing post process...")
    os.system("cp %s/spk2utt %s/" % (src, target))
    os.system("cp %s/text %s/" % (src, target))
    os.system("cp %s/utt2spk %s/" % (src, target))
    os.system("cp %s/wav.scp %s/" % (src, target))
    os.system("cp %s/spk2gender %s/" % (src, target))


def calc_func(noisy_dir_path):
    debug_model = True
    # nn.Module.dump_patches = True
    # melbank = get_fft_mel_mat(512, 16000, 40)
    method = "mel_Tan2018MelCRN_mapping_mag_early50"
    if debug_model:
        os.system('mkdir -p debug/%s' % method)
    device = torch.device("cuda")
    print_with_time("Loading model...")
    model = Generator(64, 256).to(device)
    checkpoint = torch.load("Checkpoints/mel_Tan2018MelCRN_mapping_mag_early50/checkpoint_000048614.pth")
    model.load_state_dict(checkpoint["generator"])
    model.eval()
    print_with_time("Start to enhance wav file in %s with method %s\n" % (noisy_dir_path, method))
    udir_path = "%s_%s" % (noisy_dir_path, method)
    if not os.path.exists(udir_path):
        os.mkdir(udir_path)
    wav_scp = read_path_list(os.path.join(noisy_dir_path, "wav.scp"))
    if not debug_model:
        ark_file = open(os.path.join(udir_path, "feats.ark"), 'wb')
        scp_file = open(os.path.join(udir_path, "feats.scp"), 'w')
        key_len = wav_scp[0].find(' ')
        kaldi_holder = KaldiFeatHolder(key_len, 2000, 40)
        offset = key_len + 1
    enhanced_number = 0
    left_frame = 0
    right_frame = 0
    for it, (one_wav) in enumerate(wav_scp):
        wav_id, wav_path = one_wav.split(' ')
        sr, noisy_speech = wavfile.read(wav_path)
        # process binaural waves.
        if len(noisy_speech.shape) > 1:
            noisy_speech = np.mean(noisy_speech, 1)

        c = np.sqrt(np.mean(np.square(noisy_speech)))
        noisy_speech = noisy_speech / c

        n_noisy_feat, n_noisy_mag = log_fbank(noisy_speech, False, True, True, None)
        # n_log_noisy_power = np.log(n_noisy_mag ** 2 + 1e-12)
        # feat = torch.Tensor(n_log_noisy_power).to(device)
        feat = torch.Tensor(n_noisy_feat.T).unsqueeze(0).to(device)
        n, t, d = feat.size()
        if left_frame > 0 or right_frame > 0:
            pad_feats = F.pad(feat.unsqueeze(1), (0, 0, left_frame, right_frame)).squeeze(1)
            ex_list = [pad_feats[:, i:i+t, :] for i in range(left_frame+1+right_frame)]
            feat = torch.cat(ex_list, 2)

        with torch.no_grad():
            # mask = torch.sigmoid(model.forward(feat))
            # enhanced = mask * feat.pow(2.)
            enhanced_feat = F.softplus(model.forward(feat))
        log_enhanced_fbank = np.log(enhanced_feat * (c ** 2.) + 1e-12)

        if debug_model:
            early50_path = wav_path.replace('.wav', '_early50.wav')
            sr, early50 = wavfile.read(early50_path)
            if len(early50.shape) > 1:
                early50 = np.mean(early50, 1)
            early50 = early50 / c
            early50_feat, early50_mag = log_fbank(early50, False, True, True, None)
            sio.savemat("debug/%s/%s_%s" % (method, wav_id, wav_path.split('/')[-5]),
                        {'noisy_mag': n_noisy_mag, 'noisy_feat': n_noisy_feat,
                         'enhanced_feat': enhanced_feat[0, :, :].cpu().numpy().T * (c ** 2.),
                         'early50_mag': early50_mag, 'early50_feat': early50_feat,
                         'c': c
                         })
            if it >= 0:
                return
        else:
            kaldi_holder.set_key(wav_id)
            kaldi_holder.set_value(log_enhanced_fbank.T)
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
        "/data/duzhihao/kaldi/egs/chime2/s5/data-fbank/test_eval92_5k_noisy",
    ]
    # noisy_dir_list = ["wav_scp"]
    # pool = Pool(4)
    pool = Pool(len(noisy_dir_list))
    pool.map(calc_func, noisy_dir_list)
    pool.close()



