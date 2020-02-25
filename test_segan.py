import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kaldi_helper import KaldiFeatHolder
from speech_utils import read_path_list, print_with_time, calc_rescale_c
from scipy.io import wavfile
from kaldi_fbank_extractor import log_fbank, get_fft_mel_mat
import scipy.io as sio
from multiprocessing import Pool
import librosa
from models import get_model
from enhancement_targets import get_target
import json
import argparse
from utils.emph import pre_emphasis, de_emphasis


def post_process(src, target):
    print_with_time("Doing post process...")
    os.system("cp %s/spk2utt %s/" % (src, target))
    os.system("cp %s/text %s/" % (src, target))
    os.system("cp %s/utt2spk %s/" % (src, target))
    os.system("cp %s/wav.scp %s/" % (src, target))
    os.system("cp %s/spk2gender %s/" % (src, target))


def calc_func(noisy_dir_path):
    with torch.no_grad():
        debug_model = args.debug_model
        _method = method
        model_opts = json.load(open(os.path.join("configs/%s.json" % args.model_config), 'r'))
        gen_model = model_opts['gen_model_name']
        calc_target = get_target(args.target_type)

        device = torch.device("cuda")
        print_with_time("Loading model...")
        Generator, _ = get_model(gen_model, None)
        model = Generator(**model_opts['gen_model_opts']).to(device)

        checkpoint = torch.load("Checkpoints/%s/checkpoint_%09d.pth" % (_method, args.global_step))
        model.load_state_dict(checkpoint["generator"])
        # model.load_state_dict(checkpoint["enhancer"])
        model.eval()
        melbank = get_fft_mel_mat(512, 16000, 40)

        _method = "_".join([_method, str(args.global_step)])
        if debug_model:
            os.system('mkdir -p debug/%s' % _method)
        print_with_time("Start to enhance wav file in %s with method %s\n" % (noisy_dir_path, _method))
        udir_path = "%s_%s" % (noisy_dir_path, _method)
        if not os.path.exists(udir_path):
            os.mkdir(udir_path)
        wav_scp = read_path_list(os.path.join(noisy_dir_path, "wav.scp"))
        if not debug_model:
            ark_file = open(os.path.join(udir_path, "feats.ark"), 'wb')
            scp_file = open(os.path.join(udir_path, "feats.scp"), 'w')
            key_len = wav_scp[0].find(' ')
            kaldi_holder = KaldiFeatHolder(key_len, 3000, 40)
            offset = key_len + 1
        enhanced_number = 0
        for it, (one_wav) in enumerate(wav_scp):
            wav_id, wav_path = one_wav.split(' ')
            sr, noisy_speech = wavfile.read(wav_path)
            if len(noisy_speech.shape) > 1:
                noisy_speech = np.mean(noisy_speech, 1)

            early50_path = wav_path.replace('.wav', '_early50.wav')
            sr, early50 = wavfile.read(early50_path)
            if len(early50.shape) > 1:
                early50 = np.mean(early50, 1)
            # as the training dataset, use "power_norm" to normalize the waveform to match the input of model.
            # c = np.sqrt(np.mean(np.square(noisy_speech)))
            c = calc_rescale_c(noisy_speech, args.rescale_method)
            noisy_speech = noisy_speech / c
            noisy_speech = pre_emphasis(noisy_speech, 0.97)
            predict = model.forward(torch.Tensor(noisy_speech[np.newaxis, np.newaxis, :]).to(device))
            predict = predict.squeeze() * c
            predict = de_emphasis(predict.cpu().numpy())
            enhanced_fbank = log_fbank(predict, False, True, False, None)
            log_enhanced_fbank = np.log(enhanced_fbank + opts['eps'])

            noisy_fbank, noisy_mag = log_fbank(noisy_speech, False, True, True, None)
            early50_fbank, early50_mag = log_fbank(early50, False, True, True, None)

            if debug_model:
                sio.savemat("debug/%s/%s_%s" % (_method, wav_id, wav_path.split('/')[-5]),
                            {'noisy_mag': noisy_mag, 'noisy_fbank': noisy_fbank,
                             'enhanced_fbank': enhanced_fbank.T,
                             'early50_mag': early50_mag, 'early50_fbank': early50_fbank,
                             'predict': predict[0, 0, :].cpu().numpy(),
                             'log_enhanced_fbank': log_enhanced_fbank.T,
                             'log_early50_fbank': np.log(early50_fbank + opts['eps']),
                             'c': c
                             })
                if it >= 0:
                    return
            else:
                kaldi_holder.set_key(wav_id)
                kaldi_holder.set_value(log_enhanced_fbank)
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
        print_with_time("Done %s." % _method)


if __name__ == '__main__':

    opts = {}
    opts['win_len'] = 400
    opts['sr'] = 16000
    opts['device'] = torch.device('cuda:0')
    opts['mel_channels'] = 40
    opts['win_type'] = 'hamming'
    opts['eps'] = 1e-12
    opts['clip_low'] = 0.
    opts['clip_high'] = 1.
    opts['log_power_offset'] = 10.
    opts['compress_label'] = False
    opts['log_label_min'] = -27.63
    opts['log_label_max'] = 14.41

    parser = argparse.ArgumentParser()
    parser.add_argument('--script_note', type=str, default=None)
    parser.add_argument('--feature_domain', type=str, default="waveform")
    parser.add_argument('--adversarial_loss', type=str, default="lsgan")
    parser.add_argument('--model_config', type=str, default='SEGAN')
    parser.add_argument('--target_type', type=str, default="raw_waveform")
    parser.add_argument('--clean_type', type=str, default="early50")
    parser.add_argument('--name_note', type=str, default=None)
    parser.add_argument('--d_iter', type=int, default=1)
    parser.add_argument('--rescale_method', type=str, default="value_norm", choices=["None", "value_norm", "power_norm",
                                                                                     "st_power_norm", "max_norm"])
    parser.add_argument('--dist_alpha', type=float, default=1.)
    parser.add_argument('--data_augment', type=str, default="naive", choices=["None", "naive"])

    parser.add_argument('--global_step', type=int, default=0)
    parser.add_argument('--debug_model', type=bool, default=False)
    parser.add_argument('--l1_alpha', type=float, default=0.)
    parser.add_argument('--l2_alpha', type=float, default=0.)
    parser.add_argument('--glc_alpha', type=float, default=0., help="Lipschitz continuous penalty for generator")
    parser.add_argument('--feat_alpha', type=float, default=0.)

    args = parser.parse_args()
    if args.script_note is not None:
        model_name_list = [args.script_note, args.feature_domain]
    else:
        model_name_list = [args.feature_domain]
    # model_name_list.append("mse")
    if args.adversarial_loss is not None:
        model_name_list.append(args.adversarial_loss)
    model_name_list.extend([args.model_config, args.target_type, args.clean_type])
    if args.d_iter > 0:
        model_name_list.append("D%d" % args.d_iter)
    if args.name_note is not None:
        model_name_list.append(args.name_note)
    # if args.rescale_method != "None":
    #     model_name_list.append(args.rescale_method)
    if args.l1_alpha > 0:
        model_name_list.append("L1_%.6f" % args.l1_alpha)
    if args.l2_alpha > 0:
        model_name_list.append("L2_%.6f" % args.l2_alpha)
    if args.glc_alpha > 0:
        model_name_list.append("GLC_%.6f" % args.glc_alpha)
    if args.dist_alpha > 0:
        model_name_list.append("DIST_%.6f" % args.dist_alpha)
    if args.feat_alpha > 0:
        model_name_list.append("FEAT_%.6f" % args.feat_alpha)
    if args.data_augment != "None":
        model_name_list.append(args.data_augment)
    method = "_".join(model_name_list)
    print("|----------------------------------------------------------------------------|")
    print("|", method.center(74), "|")
    print("|----------------------------------------------------------------------------|")
    print(args)
    print(opts)
    input("Press any key to continue.")

    noisy_dir_list = [
        "/data/duzhihao/kaldi/egs/chime2/s5/data-fbank/train_si84_noisy",
        "/data/duzhihao/kaldi/egs/chime2/s5/data-fbank/dev_dt_05_noisy",
        "/data/duzhihao/kaldi/egs/chime2/s5/data-fbank/test_eval92_5k_noisy",
    ]
    if args.debug_model:
        noisy_dir_list = [
            "/data/duzhihao/kaldi/egs/chime2/s5/data-fbank/train_si84_noisy"
        ]
    pool = Pool(len(noisy_dir_list))
    pool.map(calc_func, noisy_dir_list)
    pool.close()
