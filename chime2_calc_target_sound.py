import librosa
import numpy as np
from tqdm import tqdm
import os
import argparse


def get_id(file_path):
    return file_path.split('/')[-1].split('.')[0]


def calc_ir(clean, reverb):
    C = np.fft.rfft(clean)
    R = np.fft.rfft(reverb)
    IR = R / C
    IR[np.isinf(IR)] = 0.
    IR[np.isnan(IR)] = 0.
    ir = np.fft.irfft(IR, n=len(clean))
    return ir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_type", type=str, default="early50")
    parser.add_argument("--clean_dir", type=str,
                        default="/data/duzhihao/corpus/CHiME2/chime2_wsj0/data/chime2-wsj0/isolated/si_et_05/clean")
    parser.add_argument("--target_dir", type=str, default=None)
    parser.add_argument("--reverb_scp", type=str,
                        default="/data/duzhihao/kaldi/egs/chime2/s5/data-fbank/test_eval92_5k_noisy/wav.scp")
    parser.add_argument("--early_ms", type=int, default=50, help="Work only in 'early' target_type.")
    args = parser.parse_args()
    target_type = args.target_type
    clean_dir = args.clean_dir
    early_ms = args.early_ms
    ir_len = 16 * early_ms
    postfix = "_%s" % args.target_type
    reverb_scp = args.reverb_scp
    reverb_path_list = open(reverb_scp, "r").readlines()
    # i = 0
    for one_line in tqdm(reverb_path_list, ascii=True, unit="wav"):
        one_line = one_line.replace("\n", "")
        wav_id, noisy_path = one_line.split(" ")
        reverb_path = noisy_path.replace("isolated", "scaled")
        utter_id = get_id(reverb_path)
        clean_path = os.path.join(clean_dir, "%s.wav" % utter_id)
        assert os.path.exists(clean_path)
        clean_wav, fs = librosa.load(clean_path, 16000, False)
        clean_wav *= 32768.
        reverb_wav, fs = librosa.load(reverb_path, 16000, False)
        reverb_wav *= 32768.
        reverb_wav = np.mean(reverb_wav, 0)
        assert len(clean_wav) == len(reverb_wav)
        ir = calc_ir(clean_wav, reverb_wav)
        if target_type == 'direct_sound':
            target_wav = clean_wav * np.max(np.abs(ir))
        else:
            target_wav = np.convolve(clean_wav, ir[:ir_len], 'full')
            target_wav = target_wav[:-ir_len+1]
        assert len(target_wav) == len(clean_wav)
        target_path = noisy_path.replace(".wav", "_%s.wav" % target_type)
        assert not os.path.exists(target_path)
        librosa.output.write_wav(target_path, target_wav, fs, False)
    print("Done.")
