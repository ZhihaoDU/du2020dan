import os
import sys
from tqdm import tqdm


def get_id(file_path):
    return file_path.split('/')[-1].split('.')[0]


if __name__ == '__main__':
    sph_list_file = open("/data/duzhihao/kaldi/egs/chime2/s5_mct/data-fbank/test_eval92_5k_clean/wav.scp", 'r')
    target_path = "/data/duzhihao/corpus/CHiME2/chime2_wsj0/data/chime2-wsj0/isolated/si_et_05/clean"
    scp_file = open(os.path.join(target_path, "wav.scp"), "w")
    if not os.path.exists(target_path):
        os.system("mkdir -p %s" % target_path)
    total_files = sph_list_file.readlines()
    print("Total number:", len(total_files))
    for one_line in tqdm(total_files, ascii=True, unit='wav'):
        parts = one_line.split(' ')
        cmd = " ".join(parts[1:-1])
        file_id = get_id(parts[-2])
        target_file_name = os.path.join(target_path, file_id+".wav")
        assert os.path.exists(target_file_name)
        cmd = cmd + " >> " + target_file_name
        # print(cmd)
        # os.system(cmd)
        scp_file.write("%s %s\n" % (parts[0], target_file_name))
    scp_file.close()
    print("Done.")

