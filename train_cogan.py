import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
import argparse
from utils.my_logger import get_my_logger
from tqdm import tqdm
from utils.pytorch_feat_helper import add_tf_image
from loss_functions import get_disc_loss, get_recon_loss
from models import get_model
from enhancement_targets import get_target
import json
import scipy.io as sio
from utils.my_regularization import l1_regularization, l2_regularization
from utils.small_tools import print_dict


def set_seed(manual_seed):
    import random
    from torch.backends import cudnn
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    cudnn.benchmark = True
    cudnn.deterministic = True


def save_check_point(generator, discriminator, optim_g, optim_d, global_step, model_name):

    torch.save({
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optim_g": optim_g.state_dict(),
        "optim_d": optim_d.state_dict(),
        "global_step": global_step,
    }, "Checkpoints/GAN/%s/checkpoint_%09d.pth" % (model_name, global_step))



def load_check_point(generator, discriminator, optim_g, optim_d, global_step, model_name):
    check_point = torch.load("Checkpoints/GAN/%s/checkpoint_%09d.pth" % (model_name, global_step))
    generator.load_state_dict(check_point["generator"])
    optim_g.load_state_dict(check_point["optim_g"])
    if discriminator is not None:
        discriminator.load_state_dict(check_point["discriminator"])
        optim_d.load_state_dict(check_point["optim_d"])


def get_speech_slices(label, frame_number):
    frame_db = 20 * torch.log10(label.pow(2.).sum(2, keepdim=True).pow(0.5) + opts['eps'])
    vad_mask = frame_db - frame_db.max(1, keepdim=True)[0]
    vad_mask = torch.where(vad_mask > -40, torch.ones_like(vad_mask).to(label), torch.zeros_like(vad_mask).to(label))
    sum_kernel = torch.ones(1, 1, 5).to(label)
    vad_mask = (F.conv1d(vad_mask.permute(0, 2, 1), sum_kernel, padding=2).permute(0, 2, 1) >= 3).float()

    sum_kernel = torch.ones(1, 1, 40).to(label)
    slice_mask = F.conv1d(vad_mask.permute(0, 2, 1), sum_kernel).squeeze(1)

    bb, tt, dd = label.size()
    speech_slices = torch.zeros(bb * 4, 40, dd).to(label)
    slice_condition = torch.zeros(bb * 4, 40, 3).to(label)

    for i in range(bb):
        _idx = np.nonzero(slice_mask[i, :frame_number[i]-40].cpu().numpy() > 10)[0]
        np.random.shuffle(_idx)
        for j in range(4):
            ss = _idx[j]
            speech_slices[i * 4 + j, :, :] = label[i, ss: ss+40, :]
            slice_condition[i * 4 + j, :, 0] = (vad_mask[i, ss: ss+40, 0] == 0)
            # slice_condition[i * 4 + j, :, 1] = ((vad_mask[i, ss: ss+40, 0] == 1) *
            #                                     ((speech_slices[i * 4 + j, :, 20:].sum(1) /
            #                                       speech_slices[i * 4 + j, :, :].sum(1)) < 0.5))
            slice_condition[i * 4 + j, :, 1] = ((vad_mask[i, ss: ss + 40, 0] == 1) *
                                                (speech_slices[i * 4 + j, :, :20].sum(1) >
                                                 speech_slices[i * 4 + j, :, 30:].sum(1)))
            slice_condition[i * 4 + j, :, 2] = ((slice_condition[i * 4 + j, :, 0] == 0) *
                                                (slice_condition[i * 4 + j, :, 1] == 0))

    return speech_slices, slice_condition


if __name__ == '__main__':

    set_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_type', type=str, default="early50")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model_config', type=str, default='DCGAN64c128d')
    parser.add_argument('--global_step', type=int, default=0)
    parser.add_argument('--global_epoch', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--summary_interval', type=int, default=100)
    parser.add_argument('--feature_domain', type=str, default="mel")
    parser.add_argument('--total_epoch', type=int, default=-1)

    parser.add_argument('--adversarial_loss', type=str, default="wgangp")
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--d_iter', type=int, default=3)

    parser.add_argument('--l1_alpha', type=float, default=0.)
    parser.add_argument('--l2_alpha', type=float, default=0.)

    args = parser.parse_args()
    model_opts = json.load(open(os.path.join("configs/%s.json" % args.model_config), 'r'))
    model_name_list = [args.feature_domain, args.adversarial_loss, args.model_config]

    adversarial_loss = args.adversarial_loss
    gen_model = model_opts['gen_model_name']
    dis_model = model_opts['dis_model_name']
    model_name_list.append(args.clean_type)
    model_name_list.append("%ddpg" % args.d_iter)

    if args.l1_alpha > 0:
        model_name_list.append("L1_%.6f" % args.l1_alpha)
    if args.l2_alpha > 0:
        model_name_list.append("L2_%.6f" % args.l2_alpha)
    model_name = "_".join(model_name_list)

    logger = get_my_logger(model_name)

    BATCH_SIZE = args.batch_size

    print("|----------------------------------------------------------------------------|")
    print("|", model_name.center(74), "|")
    print("|----------------------------------------------------------------------------|")
    print(args)
    input("Make sure the args and press Enter key to continue.")
    logger.info("Start to load data...")

    NOISE_LIST = "wav_scp/chime2_background.scp"
    TR05_SIMU_LIST = "wav_scp/chime2_tri_noisy.scp"
    DT05_SIMU_LIST = "wav_scp/chime2_dev_noisy.scp"

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
    opts['compress_label'] = True
    opts['log_label_max'] = 0
    opts['log_label_min'] = 0

    dis_opts = {}
    if adversarial_loss == "wgangp":
        dis_opts["gp_alpha"] = 10.
    elif adversarial_loss == "hinge":
        dis_opts["hinge_margin"] = 1.
    dis_opts["l1_alpha"] = args.l1_alpha
    dis_opts["l2_alpha"] = args.l2_alpha

    if args.feature_domain.lower() == "mel":
        from kaldi_fbank_dataset import FbankDataloader, FrameDataset
    else:
        from kaldi_fft_dataset import FbankDataloader, FrameDataset

    train_dataset = FrameDataset([-6, -3, 0, 3, 6, 9], NOISE_LIST, TR05_SIMU_LIST, args.clean_type, True, None, None)
    train_dataloader = FbankDataloader(train_dataset, opts, BATCH_SIZE, True, num_workers=4, drop_last=True)
    valid_dataset = FrameDataset([-6, -3, 0, 3, 6, 9], NOISE_LIST, DT05_SIMU_LIST, args.clean_type, False, None, None)
    valid_dataloader = FbankDataloader(valid_dataset, opts, BATCH_SIZE, True, num_workers=4, drop_last=True)
    logger.info("Done.")

    logger.info("Start to construct model...")
    device = torch.device('cuda')

    Generator, Discriminator = get_model(gen_model, dis_model)
    disc_g_loss, disc_d_loss = get_disc_loss(adversarial_loss)

    generator = Generator(model_opts['gen_model_opts']).to(device)
    g_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    discriminator = Discriminator(model_opts['dis_model_opts']).to(device)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    global_step = args.global_step
    if args.global_step > 0:
        load_check_point(generator, discriminator, g_optimizer, d_optimizer, args.global_step, model_name)
        logger.info("Load generator and discriminator from checkpoint at %d." % args.global_step)

    if not os.path.exists("Checkpoints/GAN/%s" % model_name):
        os.system("mkdir -p Checkpoints/GAN/%s" % model_name)
    logger.info("Backup current code to the model directory Checkpoints/GAN/%s/%s" % (model_name, __file__))
    os.system("cp %s Checkpoints/GAN/%s/" % (__file__, model_name))

    writer = SummaryWriter("Tensorboard/GAN/%s/" % model_name)
    logger.info("Model constructed, start to train the model.")

    print(generator)
    input("Make sure the generator and press Enter key to continue.")
    print(discriminator)
    input("Make sure the discriminator and press Enter key to continue.")
    plot_z = torch.randn(64, args.z_dim, 1, 1).to(device)
    plot_c = None
    real_data_min = -27.64
    real_data_max = 13.77
    epoch = 0
    while epoch != args.total_epoch:
        generator.train()
        discriminator.train()
        pbar = tqdm(total=train_dataset.__len__(), ascii=True, unit="sp", desc="Train %d" % epoch)
        for iteration, (clean_frame, noisy_frame, frame_number, batch_mask) in enumerate(train_dataloader):
            feat, label, noise = train_dataloader.calc_feats_and_align(clean_frame, noisy_frame)
            with torch.no_grad():
                slices, conditions = get_speech_slices(label, frame_number)
                real_data = torch.log(slices + opts["eps"])
                real_data_min = min(real_data.min().item(), real_data_min)
                real_data_max = max(real_data.max().item(), real_data_max)
                real_data = (real_data - real_data_min) / (real_data_max - real_data_min)
                real_data = F.interpolate(real_data.unsqueeze(1), [64, 64])
                z = torch.randn(real_data.size(0), args.z_dim, 1, 1).to(real_data)
                c = conditions.permute(0, 2, 1).mean(2, keepdim=True).unsqueeze(3)
                if plot_c is None:
                    plot_c = conditions

            fake_data = generator.forward((z, c))
            # train discriminator
            for _ in range(args.d_iter):
                d_optimizer.zero_grad()
                disc_d_loss_dict = disc_d_loss(discriminator, (real_data.detach(), c.detach()),
                                               (fake_data.detach(), c.detach()), backward=True, opts=dis_opts)
                d_optimizer.step()

            # get discriminative loss for generator
            g_optimizer.zero_grad()
            disc_g_loss_dict = disc_g_loss(discriminator, (fake_data, c.detach()), backward=True, opts=dis_opts)
            g_optimizer.step()

            # log and summary
            if global_step % args.log_interval == 0:
                writer.add_scalar('train/d_fake_loss', disc_d_loss_dict["fake_loss"].item(), global_step // args.log_interval)
                writer.add_scalar('train/d_real_loss', disc_d_loss_dict["real_loss"].item(), global_step // args.log_interval)
                writer.add_scalar('train/w_dis', disc_d_loss_dict["w_dis"].item(), global_step // args.log_interval)
                writer.add_scalar('train/real_data_min', real_data_min, global_step // args.log_interval)
                writer.add_scalar('train/real_data_max', real_data_max, global_step // args.log_interval)
                if adversarial_loss == "wgangp":
                    writer.add_scalar("train/gradient_penalty", disc_d_loss_dict["gradient_penalty"].item(),
                                      global_step // args.log_interval)
                # add log of regularization
                if args.l1_alpha > 0:
                    writer.add_scalar("train/l1_penalty", args.l1_alpha * disc_d_loss_dict["l1_penalty"].item(),
                                      global_step // args.log_interval)
                if args.l2_alpha > 0:
                    writer.add_scalar("train/l2_penalty", args.l2_alpha * disc_d_loss_dict["l2_penalty"].item(),
                                      global_step // args.log_interval)

            if global_step % args.summary_interval == 0:
                add_tf_image(writer, real_data.squeeze(1), "train/real_data", 64, 8, global_step // args.summary_interval)
                add_tf_image(writer, fake_data.squeeze(1), "train/fake_data", 64, 8, global_step // args.summary_interval)
                c_plot = torch.argmax(conditions, 2, True).expand(-1, -1, 40)
                add_tf_image(writer, c_plot, "train/conditions", 64, 8,
                             global_step // args.summary_interval, colorful=True)
            global_step += 1
            pbar.update(BATCH_SIZE)
        pbar.close()

        # Save checkpoint at `global step`.
        logger.info("Save models at step %d" % global_step)
        save_check_point(generator, discriminator, g_optimizer, d_optimizer, global_step, model_name)
        logger.info("Done.")

        with torch.no_grad():
            logger.info("Complete train %d epochs, start to plot generated spectrogram." % epoch)
            generator.eval()
            discriminator.eval()
            plot_x = generator.forward((plot_z, plot_c.permute(0, 2, 1).mean(2, keepdim=True).unsqueeze(3)))
            add_tf_image(writer, plot_x.squeeze(1), "valid/plot_x", 64, 8, epoch + args.global_epoch)
            c_plot = torch.argmax(plot_c, 2, True).expand(-1, -1, 40)
            add_tf_image(writer, c_plot, "valid/conditions", 64, 8, epoch, colorful=True)

        epoch += 1
    writer.close()
