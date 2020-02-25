import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
from kaldi_fbank_dataset_bak import FbankDataloader, FrameDataset
from speech_utils import print_with_time
import numpy as np
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import torch.nn.functional as F
import scipy.io as sio
import argparse

class BiLSTM(nn.Module):

    def __init__(self, input_size, layer_number, hidden_units, out_dim):
        super(BiLSTM, self).__init__()
        self.layer_number = layer_number
        self.hidden_units = hidden_units
        self.out_dim = out_dim
        self.lstm = nn.LSTM(input_size, hidden_units, layer_number, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_units*2, out_dim)
        self.device = torch.device('cuda')

    def forward(self, x):
        h0 = torch.zeros(self.layer_number*2, x.size(0), self.hidden_units).to(self.device)
        c0 = torch.zeros(self.layer_number*2, x.size(0), self.hidden_units).to(self.device)
        out, _ = self.lstm(x, (h0,c0))
        seq_len = out.shape[1]
        out = out.contiguous().view([-1, self.hidden_units*2])
        out = self.fc(out)
        out = out.contiguous().view([-1, seq_len, self.out_dim])
        return out

class DnnDiscriminator(nn.Module):

    def __init__(self):
        super(DnnDiscriminator, self).__init__()
        self.activation = nn.ReLU()
        self.fc_1 = nn.Linear(25*40, 1024)
        self.fc_2 = nn.Linear(1024, 1024)
        self.fc_3 = nn.Linear(1024, 1024)
        self.fc_out = nn.Linear(1024, 1)


    def forward(self, input):
        x = input.view(-1, 25*40)
        layer_1 = self.activation(self.fc_1(x))
        layer_2 = self.activation(self.fc_2(layer_1))
        layer_3 = self.activation(self.fc_3(layer_2))
        predict_y = self.fc_out(layer_3)
        return predict_y


class ConvDiscriminator(nn.Module):

    def __init__(self):
        super(ConvDiscriminator, self).__init__()
        self.conv_1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.conv_2 = nn.Conv2d(16, 16, 3, 1, 1)

        self.conv_3 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv_4 = nn.Conv2d(32, 32, 3, 1, 1)

        self.conv_5 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv_6 = nn.Conv2d(64, 64, 3, 1, 1)

        self.fc_1 = nn.Linear(5*5*64, 1024)
        self.fc_2 = nn.Linear(1024, 1024)
        self.fc_3 = nn.Linear(1024, 1)

    def forward(self, input):
        x = input.view(-1, 1, 1000, 40)
        layer_1 = F.relu(self.conv_1(x))
        layer_2 = F.relu((self.conv_2(layer_1)))
        pool_1 = F.max_pool2d(layer_2, 3, 2, 1)
        layer_3 = F.relu(self.conv_3(pool_1))
        layer_4 = F.relu(self.conv_4(layer_3))
        pool_2 = F.max_pool2d(layer_4, 3, 2, 1)
        layer_5 = F.relu(self.conv_5(pool_2))
        layer_6 = F.relu(self.conv_6(layer_5))
        pool_3 = F.max_pool2d(layer_6, 3, 2, 1)

        fc_in = pool_3.permute(0, 2, 1, 3).contiguous().view(-1, 5*5*64)
        fc_1 = F.relu(self.fc_1(fc_in))
        fc_2 = F.relu(self.fc_2(fc_1))
        fc_3 = self.fc_3(fc_2)
        return fc_3

class NoPoolingConvDiscriminator(nn.Module):

    def __init__(self):
        super(NoPoolingConvDiscriminator, self).__init__()
        self.conv_1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.conv_2 = nn.Conv2d(16, 16, 3, 2, 1)

        self.conv_3 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv_4 = nn.Conv2d(32, 32, 3, 2, 1)

        self.conv_5 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv_6 = nn.Conv2d(64, 64, 3, 2, 1)

        self.fc_1 = nn.Linear(5*5*64, 1024)
        self.fc_2 = nn.Linear(1024, 1024)
        self.fc_3 = nn.Linear(1024, 1)

    def forward(self, input):
        x = input.view(-1, 1, 1000, 40)
        layer_1 = F.relu(self.conv_1(x))
        layer_2 = F.relu((self.conv_2(layer_1)))
        layer_3 = F.relu(self.conv_3(layer_2))
        layer_4 = F.relu(self.conv_4(layer_3))
        layer_5 = F.relu(self.conv_5(layer_4))
        layer_6 = F.relu(self.conv_6(layer_5))

        fc_in = layer_6.permute(0, 2, 1, 3).contiguous().view(-1, 5*5*64)
        fc_1 = F.relu(self.fc_1(fc_in))
        fc_2 = F.relu(self.fc_2(fc_1))
        fc_3 = self.fc_3(fc_2)
        return fc_3


def get_d_loss(logits_real, logits_fake):
    true_labels = torch.ones_like(logits_real).to(device)
    real_loss = bce_loss(logits_real, true_labels)
    fake_loss = bce_loss(logits_fake, 1. - true_labels)
    #loss = real_loss + fake_loss
    return real_loss, fake_loss

def get_g_loss(logits_fake):
    true_labels = torch.ones_like(logits_fake)
    fake_image_loss = bce_loss(logits_fake, true_labels)
    return fake_image_loss


def add_image_summary(scope, feat, label, predict, mask, image_num, batch_size, iteration):
    for i in range(image_num):
        idx = (i+1)*batch_size - 1
        x = vutils.make_grid(feat[idx, :320, :].permute([1, 0]).contiguous().view(40, 320), normalize=True, scale_each=True)
        writer.add_image("%s/noisy_mfb_%d" % (scope, idx), x, iteration)
        x = vutils.make_grid(label[idx, :320, :].permute([1, 0]).contiguous().view(40, 320), normalize=True, scale_each=True)
        writer.add_image("%s/clean_mfb_%d" % (scope, idx), x, iteration)
        x = vutils.make_grid(predict[idx, :320, :].permute([1, 0]).contiguous().view(40, 320), normalize=True, scale_each=True)
        writer.add_image("%s/predict_%d" % (scope, idx), x, iteration)
        x = vutils.make_grid(mask[idx, :320, :].permute([1, 0]).contiguous().view(40, 320), normalize=True, scale_each=True)
        writer.add_image("%s/mask_%d" % (scope, idx), x, iteration)

def calc_l2_regularization(model):
    l2_regularization = None
    for param in model.parameters():
        if len(param.size()) == 4:
            if l2_regularization is None:
                l2_regularization = ((param.pow(2).sum([2,3]).pow(0.5) - 1) ** 2.).mean()
            else:
                l2_regularization = l2_regularization + ((param.pow(2).sum([2,3]).pow(0.5) - 1) ** 2.).mean()
        if len(param.size()) == 2:
            if l2_regularization is None:
                l2_regularization = ((param.pow(2).sum(1).pow(0.5) - 1) ** 2.).mean()
            else:
                l2_regularization = l2_regularization + ((param.pow(2).sum(1).pow(0.5) - 1) ** 2.).mean()
    return l2_regularization


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.0001)
    parser.add_argument('--use_weight_penalty', type=bool, default=False)
    parser.add_argument('--target_type', type=str, default="direct_sound")
    parser.add_argument('--batch_size', type=int, default=16)

    args = parser.parse_args()
    alpha = args.alpha
    use_weight_penalty = args.use_weight_penalty

    if use_weight_penalty:
        MODEL_NAME = "wp_chime2_dnngan_lm_bilstm_%s" % str(alpha).rstrip('0')
    else:
        MODEL_NAME = "chime2_dnngan_lm_bilstm_%s" % str(alpha).rstrip('0')
    BATCH_SIZE = args.batch_size
    TIME_STEPS = 1000
    FEAT_LENGTH = 320
    FRAME_LENGTH = 320 + (TIME_STEPS - 1) * 160
    FRAME_SHIFT = 16000 * 10

    print("|----------------------------------------------------------------------------|")
    print("|", ("Train %s: 4 layer, 512 units alpha=%s WP=%s" %
                (MODEL_NAME, str(alpha).rstrip('0'), str(use_weight_penalty))).center(74), "|")
    print("|----------------------------------------------------------------------------|")
    print_with_time("Start to load data...")

    NOISE_LIST = "wav_scp/chime2_background.scp"
    TR05_CLEA_LIST = "wav_scp/chime2_tri_direct_sound.scp"
    TR05_ORGN_LIST = "wav_scp/chime2_tri_direct_sound.scp"
    TR05_SIMU_LIST = "wav_scp/chime2_tri_noisy.scp"

    DT05_CLEA_LIST = "wav_scp/chime2_dev_direct_sound.scp"
    DT05_ORGN_LIST = "wav_scp/chime2_dev_direct_sound.scp"
    DT05_SIMU_LIST = "wav_scp/chime2_dev_noisy.scp"

    opts = {}
    opts['win_len'] = 400
    opts['sr'] = 16000
    opts['device'] = torch.device('cuda:0')
    opts['mel_channels'] = 40
    opts['win_type'] = 'hamming'

    train_dataset = FrameDataset([0, 3, 6], NOISE_LIST, TR05_CLEA_LIST, TR05_SIMU_LIST, args.target_type, True, None)
    train_dataloader = FbankDataloader(train_dataset, opts, BATCH_SIZE, True, num_workers=4)
    valid_dataset = FrameDataset([0, 3, 6], NOISE_LIST, DT05_CLEA_LIST, DT05_SIMU_LIST, args.target_type, False, None)
    valid_dataloader = FbankDataloader(valid_dataset, opts, BATCH_SIZE, True, num_workers=4)
    print_with_time("Done.")

    print_with_time("Start to construct model...")
    bce_loss = nn.BCEWithLogitsLoss()
    device = torch.device('cuda:0')
    # generator = torch.load("models/wp_cnngan_lm_mask_bilstm/epoch_49.pkl")
    generator = BiLSTM(40, 4, 512, 40).to(device)
    # discriminator = ConvDiscriminator().to(device)
    discriminator = DnnDiscriminator().to(device)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    g_optimizer = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    if not os.path.exists("models/%s" % MODEL_NAME):
        os.mkdir("models/%s" % MODEL_NAME)
    print_with_time("Backup current code to the model directory models/%s/%s" % (MODEL_NAME, __file__))
    os.system("cp %s models/%s/" % (__file__, MODEL_NAME))

    last_loss = 0.
    early_stop_loss = 0.
    early_stop_epoch = 0
    summary_count = 0
    print_interval = 10
    writer = SummaryWriter("Tensorboard/%s/" % MODEL_NAME)
    print_with_time("Model constructed, start to train the model.")
    epoch = 0
    tr_global_step = 0
    dt_global_step = 0
    # alpha = [0.001, 0.01, 0.1, 1, 1, 1]
    while True:
        trained_utter_number = 0
        for iteration, (clean_frame, noisy_frame, frame_number, batch_mask) in enumerate(train_dataloader):
            feat, label = train_dataloader.calc_feats_and_align(clean_frame, noisy_frame, frame_number)
            with torch.no_grad():
                trained_utter_number += feat.size(0)//2
                f_num = sum(frame_number)
                batch_mask = torch.Tensor(batch_mask[:, :, np.newaxis]).to(opts['device'])
                log_feat = torch.log(feat)
                log_feat[torch.isnan(log_feat)] = 0.
                log_feat[torch.isinf(log_feat)] = 0.

                log_label = torch.log(label)
                log_label[torch.isnan(log_label)] = 0.
                log_label[torch.isinf(log_label)] = 0.

                irm = log_label / log_feat
                irm[torch.isinf(irm)] = 0.
                irm[torch.isnan(irm)] = 0.
                irm[irm > 1] = 1.
                irm[irm < 0] = 0.

                # sio.savemat('aaa.mat', {'log_feat': log_feat.cpu().numpy(), 'log_label': log_label.cpu().numpy(),
                #                         'irm': irm.cpu().numpy()})
                # exit(0)

            mask = torch.sigmoid(generator.forward(log_feat))
            log_predict = log_feat * mask
            m_loss = (((irm - mask) ** 2.) * batch_mask).sum() / (batch_mask.sum() * 40)

            # calc the loss of discriminator
            logits_fake = discriminator.forward(log_feat * mask)
            logits_real = discriminator.forward(log_feat * irm)
            d_real_loss, d_fake_loss = get_d_loss(logits_real, logits_fake)
            d_loss = d_real_loss + d_fake_loss
            if use_weight_penalty:
                d_loss += 10. * calc_l2_regularization(discriminator)

            # backward for discriminator
            d_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)
            d_optimizer.step()

            # calc the loss of generator
            d_g_loss = get_g_loss(logits_fake)
            g_loss = alpha * d_g_loss + m_loss

            # backward for generator
            g_optimizer.zero_grad()
            g_loss.backward(retain_graph=False)
            g_optimizer.step()

            # log and summary
            if tr_global_step % print_interval == 0:
                print_with_time("Epoch {}, Step {}, Utterance {}, Loss: {:.4f}".format(epoch, tr_global_step // print_interval, trained_utter_number, m_loss.item()))
                writer.add_scalar('train/mse_loss', m_loss.item(), tr_global_step // print_interval)
                writer.add_scalar('train/d_loss', d_loss.item(), tr_global_step // print_interval)
                writer.add_scalar('train/d_real_loss', d_real_loss.item(), tr_global_step // print_interval)
                writer.add_scalar('train/d_fake_loss', d_fake_loss.item(), tr_global_step // print_interval)
                writer.add_scalar('train/g_loss', d_g_loss.item(), tr_global_step // print_interval)
                idx = 0
                for param in discriminator.parameters():
                    if len(param.size()) == 4:
                        writer.add_histogram("param_%d" % idx, param.pow(2).sum([2,3]).pow(0.5), tr_global_step // print_interval)
                        idx += 1
                    if len(param.size()) == 2:
                        writer.add_histogram("param_%d" % idx, param.pow(2).sum(1).pow(0.5), tr_global_step // print_interval)
                        idx += 1
                if tr_global_step % 100 == 0:
                    add_image_summary("train", log_feat, log_label, log_predict, mask, 2, BATCH_SIZE, tr_global_step // 100)
            tr_global_step += 1
        torch.save(generator, "models/%s/epoch_%d.gnt" % (MODEL_NAME, epoch))
        torch.save(discriminator, "models/%s/epoch_%d.dsc" % (MODEL_NAME, epoch))

        with torch.no_grad() :
            print_with_time("Complete train %d epochs, start to evaluate performance in valid dataset." % epoch)
            valid_loss = 0.
            for iteration, (clean_frame, noisy_frame, frame_number, batch_mask) in enumerate(valid_dataloader):
                with torch.no_grad():
                    feat, label = valid_dataloader.calc_feats_and_align(clean_frame, noisy_frame, frame_number)
                    f_num = sum(frame_number)
                    batch_mask = torch.Tensor(batch_mask[:, :, np.newaxis]).to(opts['device'])

                    log_feat = torch.log(feat + 1.)
                    log_feat[torch.isnan(log_feat)] = 0.
                    log_feat[torch.isinf(log_feat)] = 0.

                    log_label = torch.log(label + 1.)
                    log_label[torch.isnan(log_label)] = 0.
                    log_label[torch.isinf(log_label)] = 0.

                    irm = log_label / log_feat
                    irm[torch.isinf(irm)] = 0.
                    irm[torch.isnan(irm)] = 0.
                    irm[irm > 1.] = 1.
                    irm[irm < 0.] = 0.

                    mask = torch.sigmoid(generator.forward(log_feat))
                    log_predict = log_feat * mask
                    m_loss = (((irm - mask) ** 2.) * batch_mask).sum() / (batch_mask.sum() * 40)
                    valid_loss += m_loss.item()
                if dt_global_step % 100 == 0:
                    add_image_summary("valid", log_feat, log_label, log_predict, mask, 2, BATCH_SIZE, dt_global_step//100)
                dt_global_step += 1
        print_with_time("Loss on valid dataset is %.4f" % valid_loss)
        writer.add_scalar("valid/mse_loss", valid_loss, epoch)
        epoch += 1
    writer.close()