import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from speech_utils import print_with_time
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


def set_seed(manual_seed):
    import random
    from torch.backends import cudnn
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    cudnn.benchmark = True
    cudnn.deterministic = True


def save_check_point(generator, discriminator, optim_g, optim_d, global_step, model_name):
    if discriminator is not None and optim_d is not None:
        torch.save({
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optim_g": optim_g.state_dict(),
            "optim_d": optim_d.state_dict(),
            "global_step": global_step,
        }, "Checkpoints/%s/checkpoint_%09d.pth" % (model_name, global_step))
    else:
        torch.save({
            "generator": generator.state_dict(),
            "optim_g": optim_g.state_dict(),
            "global_step": global_step,
        }, "Checkpoints/%s/checkpoint_%09d.pth" % (model_name, global_step))


def load_check_point(generator, discriminator, optim_g, optim_d, global_step, model_name):
    check_point = torch.load("Checkpoints/%s/checkpoint_%09d.pth" % (model_name, global_step))
    generator.load_state_dict(check_point["generator"])
    optim_g.load_state_dict(check_point["optim_g"])
    if discriminator is not None:
        discriminator.load_state_dict(check_point["discriminator"])
        optim_d.load_state_dict(check_point["optim_d"])


def calc_gradient_penalty(model, batch, mask):
    from torch import autograd
    interpolates = autograd.Variable(batch, requires_grad=True)
    output = model(interpolates)
    mask = mask / mask.min()
    gradients = autograd.grad(outputs=output, inputs=interpolates,
                              grad_outputs=mask,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1).pow(2.)).mean()
    # gradient_penalty = F.relu(gradients.norm(2, dim=1) - 1).mean()
    return gradient_penalty


def calc_gradient_norm(model, batch, mask=None):
    from torch import autograd
    interpolates = autograd.Variable(batch, requires_grad=True)
    output = model((interpolates,))
    if mask is not None:
        mask = mask / mask.min()
    else:
        mask = torch.ones_like(output[0])
    gradients = autograd.grad(outputs=output[0], inputs=interpolates,
                              grad_outputs=mask,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_norm = gradients.norm(2, dim=1)
    # gradient_penalty = F.relu(gradients.norm(2, dim=1) - 1).mean()
    return gradient_norm


def get_speech_slices(label, results, frame_number):
    frame_db = 20 * torch.log10(label.pow(2.).sum(2, keepdim=True).pow(0.5) + opts['eps'])
    vad_mask = frame_db - frame_db.max(1, keepdim=True)[0]
    vad_mask = torch.where(vad_mask > -40, torch.ones_like(vad_mask).to(label), torch.zeros_like(vad_mask).to(label))
    sum_kernel = torch.ones(1, 1, 40).to(label)
    slice_mask = F.conv1d(vad_mask.permute(0, 2, 1), sum_kernel).squeeze(1)

    bb, tt, dd = label.size()
    slices = {}
    for key in results:
        if key == "real_data" or key == "fake_data":
            slices[key] = torch.zeros(bb * 4, 1, 40, dd).to(label)
    label_slices = torch.zeros(bb * 4, 1, 40, dd).to(label)
    for i in range(bb):
        _idx = np.nonzero(slice_mask[i, :frame_number[i]-40].cpu().numpy() > 10)[0]
        np.random.shuffle(_idx)
        for j in range(4):
            ss = _idx[j]
            label_slices[i*4+j, 0, :, :] = label[i, ss: ss+40, :]
            for key in slices:
                if key == "real_data" or key == "fake_data":
                    slices[key][i*4+j, 0, :, :] = results[key][i, ss: ss+40, :]
    for key in slices:
        if key == "real_data" or key == "fake_data":
            slices[key] = (slices[key] - real_data_min) / (real_data_max - real_data_min)
            slices[key] = F.interpolate(slices[key] * 2. - 1., [64, 64])
    return label_slices, slices['real_data'], slices['fake_data']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reconstruction_loss', type=str, default="mse")
    parser.add_argument('--clean_type', type=str, default="early50")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--model_config', type=str, default='BiLSTM')
    parser.add_argument('--name_note', type=str, default=None)
    parser.add_argument('--disc_name', type=str, default=None)
    parser.add_argument('--disc_step', type=int, default=0)
    parser.add_argument('--global_step', type=int, default=0)
    parser.add_argument('--global_epoch', type=int, default=0)
    parser.add_argument('--target_type', type=str, default="mask_irm")
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--summary_interval', type=int, default=100)
    parser.add_argument('--data_augment', type=str, default=None, choices=[None, "naive"])
    parser.add_argument('--feature_domain', type=str, default="fft")
    parser.add_argument('--total_epoch', type=int, default=200)
    parser.add_argument('--early_stopping_patience', type=int, default=0)
    parser.add_argument('--l1_alpha', type=float, default=0.)
    parser.add_argument('--l2_alpha', type=float, default=0.)
    parser.add_argument('--glc_alpha', type=float, default=0., help="Lipschitz continuous penalty for generator")
    parser.add_argument('--feat_alpha', type=float, default=0.)
    parser.add_argument('--dist_alpha', type=float, default=0.)
    parser.add_argument('--rescale_method', type=str, default="power_norm", choices=["None", "value_norm", "power_norm",
                                                                                     "max_norm"])
    parser.add_argument('--random_seed', type=int, default=0)
    args = parser.parse_args()

    set_seed(args.random_seed)

    model_opts = json.load(open(os.path.join("configs/%s.json" % args.model_config), 'r'))
    gen_model = model_opts['gen_model_name']
    dis_model = None
    target_type = args.target_type
    model_name_list = [args.feature_domain, args.model_config]
    reconstruction_loss = args.reconstruction_loss
    if args.dist_alpha >= 10000 or args.feat_alpha >= 10000:
        model_name_list.append("fMSE")
        if args.dist_alpha >= 10000:
            args.dist_alpha = 1.
        if args.feat_alpha >= 10000:
            args.feat_alpha = 1.

    if args.dist_alpha > 0 or args.feat_alpha > 0:
        dis_model = model_opts['dis_model_name']
    model_name_list.append(target_type)
    model_name_list.append(args.clean_type)
    if args.name_note is not None:
        model_name_list.append(args.name_note)
    if args.rescale_method != "None":
        model_name_list.append(args.rescale_method)
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
    if args.data_augment is not None:
        model_name_list.append(args.data_augment)
    model_name = "_".join(model_name_list)

    logger = get_my_logger(model_name)

    BATCH_SIZE = args.batch_size

    print("|----------------------------------------------------------------------------|")
    print("|", model_name.center(74), "|")
    print("|----------------------------------------------------------------------------|")
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
    opts['compress_label'] = False
    opts['log_label_max'] = 0
    opts['log_label_min'] = 0
    real_data_min = -19.35
    real_data_max = 13.76

    dis_opts = {}

    if args.feature_domain.lower() == "mel":
        from kaldi_fbank_dataset import FbankDataloader, FrameDataset
    else:
        from kaldi_fft_dataset import FbankDataloader, FrameDataset

    train_dataset = FrameDataset([-6, -3, 0, 3, 6, 9], NOISE_LIST, TR05_SIMU_LIST, args.clean_type, True, args.rescale_method, args.data_augment)
    train_dataloader = FbankDataloader(train_dataset, opts, BATCH_SIZE, True, num_workers=4, drop_last=True)
    valid_dataset = FrameDataset([-6, -3, 0, 3, 6, 9], NOISE_LIST, DT05_SIMU_LIST, args.clean_type, False, args.rescale_method, "None")
    valid_dataloader = FbankDataloader(valid_dataset, opts, BATCH_SIZE, True, num_workers=4, drop_last=True)
    logger.info("Done.")

    logger.info("Start to construct model...")
    device = torch.device('cuda')

    Generator, Discriminator = get_model(gen_model, dis_model)
    reconstruction_loss = get_recon_loss(reconstruction_loss)

    calc_target = get_target(args.target_type)

    if not os.path.exists("Checkpoints/%s" % model_name):
        os.system("mkdir -p Checkpoints/%s" % model_name)
    logger.info("Backup current code to the model directory Checkpoints/%s/%s" % (model_name, __file__))
    os.system("cp %s Checkpoints/%s/" % (__file__, model_name))

    generator = Generator(model_opts['gen_model_opts']).to(device)
    g_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    # discriminator = ConvDiscriminator().to(device)
    if args.dist_alpha > 0 or args.feat_alpha > 0:
        discriminator = Discriminator(model_opts['dis_model_opts']).to(device)
        if args.disc_step > 0:
            state_dict = torch.load("Checkpoints/GAN/%s/checkpoint_%09d.pth" % (args.disc_name, args.disc_step))
            discriminator.load_state_dict(state_dict["discriminator"])
            logger.info("Loaded discriminator from Checkpoints/GAN/%s at step %09d." % (args.disc_name, args.disc_step))

    global_step = args.global_step
    if args.global_step > 0:
        load_check_point(generator, None, g_optimizer, None, args.global_step, model_name)
        logger.info("Load generator from checkpoint at %d." % args.global_step)
    else:
        save_check_point(generator, None, g_optimizer, None, 1, model_name)
        logger.info("Save the init weight to step 1.")

    last_loss = 0.
    early_stop_loss = 0.
    early_stop_epoch = 0
    summary_count = 0
    print_interval = 10
    writer = SummaryWriter("Tensorboard/%s/" % model_name)
    logger.info("Model constructed, start to train the model.")
    print(args)
    input("Make sure the args and press Enter key to continue.")
    print(generator)
    input("Make sure the generator and press Enter key to continue.")
    if args.dist_alpha > 0 or args.feat_alpha > 0:
        print(discriminator)
        input("Make sure the discriminator and press Enter key to continue.")
    train_first = True
    # alpha = [0.001, 0.01, 0.1, 1, 1, 1]
    lowest_valid_loss = 1e8
    lowest_epoch = 0
    save_list = []
    for epoch in range(args.total_epoch):
        if epoch > 0 or train_first:
            generator.train()
            if args.dist_alpha > 0 or args.feat_alpha > 0:
                discriminator.eval()
            pbar = tqdm(total=train_dataset.__len__(), ascii=True, unit="sp", desc="Train %d" % epoch)
            for iteration, (clean_frame, noisy_frame, frame_number, batch_mask) in enumerate(train_dataloader):
                feat, label, noise = train_dataloader.calc_feats_and_align(clean_frame, noisy_frame)
                with torch.no_grad():
                    batch_mask = torch.Tensor(batch_mask[:, :, np.newaxis]).to(opts['device'])
                    batch_mask = batch_mask.expand_as(label)
                    batch_mask = batch_mask / batch_mask.sum(dim=(1, 2), keepdim=True)
                    log_feat = torch.log(feat + opts["eps"])
                    log_label = torch.log(label + opts["eps"])

                if args.target_type.lower() == "mapping_mag":
                    predict = generator.forward(feat.sqrt())
                else:
                    predict = generator.forward(log_feat)

                results = calc_target(feat, label, noise, predict, opts)
                enhanced = results["enhanced"]
                predict = results["predict"]
                target = results["target"]

                log_enhanced = torch.log(enhanced + opts["eps"])

                g_loss = 0
                # add l1/l2 regularization
                if args.l1_alpha > 0:
                    l1_penalty = l1_regularization(generator, "weight")
                    g_loss = g_loss + args.l1_alpha * l1_penalty
                if args.l2_alpha > 0:
                    l2_penalty = l2_regularization(generator, "weight")
                    g_loss = g_loss + args.l2_alpha * l2_penalty
                if args.dist_alpha > 0:
                    label_slices, real_data, fake_data = get_speech_slices(label, results, frame_number)
                    # get discriminative loss for generator
                    fake_logits, fake_feat = discriminator((fake_data,))
                    real_logits, real_feat = discriminator((real_data.detach(),))
                    rand_logits, rand_feat = discriminator((torch.rand_like(real_data) * 2. - 1.,))
                    dist_loss = (fake_logits - 1.).pow(2.).mean()
                    g_loss = g_loss + dist_loss * args.dist_alpha

                if args.feat_alpha > 0:
                    label_slices, real_data, fake_data = get_speech_slices(label, results, frame_number)
                    # get discriminative loss for generator
                    fake_logits, fake_feat = discriminator((fake_data,))
                    real_logits, real_feat = discriminator((real_data.detach(),))
                    feat_loss = (fake_feat - real_feat).pow(2.).mean()
                    g_loss = g_loss + feat_loss * args.feat_alpha

                recon_loss = reconstruction_loss(predict, target, batch_mask, backward=False, opts=None)["loss"] / BATCH_SIZE
                if "fMSE" not in model_name:
                    g_loss = g_loss + recon_loss

                g_optimizer.zero_grad()
                g_loss.backward()
                if args.glc_alpha > 0:
                    if args.target_type.lower() == "mapping_mag":
                        batch = feat.sqrt()
                    else:
                        batch = log_feat
                    gp = calc_gradient_penalty(generator, batch, batch_mask)
                    (gp * args.glc_alpha).backward()
                g_optimizer.step()

                # log and summary
                if global_step % args.log_interval == 0:
                    writer.add_scalar('train/recon_loss', recon_loss.item(), global_step // args.log_interval)
                    # add log of regularization
                    if args.l1_alpha > 0:
                        writer.add_scalar("train/l1_penalty", args.l1_alpha * l1_penalty.item(),
                                          global_step // args.log_interval)
                    if args.l2_alpha > 0:
                        writer.add_scalar("train/l2_penalty", args.l2_alpha * l2_penalty.item(),
                                          global_step // args.log_interval)
                    if args.glc_alpha > 0:
                        writer.add_scalar("train/glc_penalty", args.glc_alpha * gp.item(),
                                          global_step // args.log_interval)
                    if args.dist_alpha > 0:
                        writer.add_scalar("train/dist_penalty", args.dist_alpha * dist_loss.item(),
                                          global_step // args.log_interval)
                        writer.add_scalars("train", {"real_logits": real_logits.mean().item(),
                                                     "fake_logits": fake_logits.mean().item(),
                                                     "rand_logits": rand_logits.mean().item()},
                                           global_step // args.log_interval)
                    if args.feat_alpha > 0:
                        writer.add_scalar("train/feat_penalty", args.feat_alpha * feat_loss.item(),
                                          global_step // args.log_interval)
                        writer.add_scalars("train", {"real_logits": real_logits.mean().item(),
                                                     "fake_logits": fake_logits.mean().item()},
                                           global_step // args.log_interval)
                    # add min, max of clean feature
                    if args.target_type == "mapping_log_pow" and opts['compress_label']:
                        writer.add_scalars("train", {
                            "log_label_min": opts['log_label_min'],
                            "log_label_max": opts["log_label_max"],
                        }, global_step // args.log_interval)

                if global_step % args.summary_interval == 0:
                    add_tf_image(writer, log_feat[:, :480, :], "train/noisy_feat", 4, 1, global_step // args.summary_interval)
                    add_tf_image(writer, log_label[:, :480, :], "train/clean_feat", 4, 1, global_step // args.summary_interval)
                    add_tf_image(writer, log_enhanced[:, :480, :], "train/enhanced_feat", 4, 1, global_step // args.summary_interval)
                    if "mask" in target_type or "SigApp" in target_type:
                        add_tf_image(writer, predict[:, :480, :], "train/mask", 4, 1, global_step // args.summary_interval)
                        add_tf_image(writer, target[:, :480, :], "train/ideal_mask", 4, 1, global_step // args.summary_interval)
                        writer.add_histogram("train/hist_predict_mask", predict[0, :frame_number[0], :], global_step // args.summary_interval)
                        writer.add_histogram("train/hist_ideal_mask", target[0, :frame_number[0], :], global_step // args.summary_interval)
                    if args.dist_alpha > 0 or args.feat_alpha > 0:
                        add_tf_image(writer, real_data.squeeze(1), "train/real_data", 64, 8,
                                     global_step // args.summary_interval)
                        add_tf_image(writer, fake_data.squeeze(1), "train/fake_data", 64, 8,
                                     global_step // args.summary_interval)
                        writer.add_histogram("train/real_logits_hist", real_logits, global_step // args.summary_interval)
                        writer.add_histogram("train/fake_logits_hist", fake_logits, global_step // args.summary_interval)
                        writer.add_histogram("train/gradient_norm_hist",
                                             calc_gradient_norm(discriminator, fake_data, None),
                                             global_step // args.summary_interval
                                             )

                global_step += 1
                pbar.update(BATCH_SIZE)

        # Save checkpoint at `global step`.
        logger.info("Save models at step %d" % global_step)
        save_check_point(generator, None, g_optimizer, None, global_step, model_name)

        # Only save the last checkpoints
        save_list.append(global_step)
        if epoch > args.early_stopping_patience > 0:
            os.system("rm -rf Checkpoints/%s/checkpoint_%09d.pth" % (model_name, save_list[epoch - args.early_stopping_patience]))

        # Save min and max for 'mapping_low_pow' method in compress_label mode.
        if args.target_type == "mapping_log_pow" and opts['compress_label']:
            sio.savemat("Checkpoints/%s/checkpoint_%09d.mat" % (model_name, global_step),
                        {"log_label_min": opts['log_label_min'], "log_label_max": opts['log_label_max']})

        logger.info("Done.")
        with torch.no_grad():
            logger.info("Complete train %d epochs, start to evaluate performance in valid dataset." % epoch)
            # 0 for recon loss, 1 for disc loss, 2 for summation
            valid_loss = [0., 0., 0.]
            generator.eval()
            if args.dist_alpha > 0 or args.feat_alpha > 0:
                discriminator.eval()
            pbar = tqdm(total=valid_dataset.__len__(), ascii=True, unit="sp", desc="Valid %d" % epoch)
            for iteration, (clean_frame, noisy_frame, frame_number, batch_mask) in enumerate(valid_dataloader):
                feat, label, noise = train_dataloader.calc_feats_and_align(clean_frame, noisy_frame)

                batch_mask = torch.Tensor(batch_mask[:, :, np.newaxis]).to(opts['device'])
                batch_mask = batch_mask.expand_as(label)
                batch_mask = batch_mask / batch_mask.sum(dim=(1, 2), keepdim=True)
                log_feat = torch.log(feat + opts["eps"])
                log_label = torch.log(label + opts["eps"])

                if args.target_type.lower() == "mapping_mag":
                    predict = generator.forward(feat.sqrt())
                else:
                    predict = generator.forward(log_feat)

                results = calc_target(feat, label, noise, predict, opts)
                enhanced = results["enhanced"]
                predict = results["predict"]
                target = results["target"]

                log_enhanced = torch.log(enhanced + opts["eps"])
                if args.dist_alpha > 0:
                    label_slices, real_data, fake_data = get_speech_slices(label, results, frame_number)
                    # get discriminative loss for generator
                    fake_logits, fake_feat = discriminator((fake_data,))
                    real_logits, real_feat = discriminator((real_data,))
                    if "lsgan" in args.disc_name:
                        dist_loss = (fake_logits - 1.).pow(2.).mean()
                    elif "hinge" in args.disc_name:
                        dist_loss = F.relu(1. - fake_logits).mean()
                    else:
                        dist_loss = -fake_logits.mean()
                    valid_loss[1] += dist_loss.item() * args.dist_alpha

                if args.feat_alpha > 0:
                    label_slices, real_data, fake_data = get_speech_slices(label, results, frame_number)
                    # get discriminative loss for generator
                    fake_logits, fake_feat = discriminator((fake_data,))
                    real_logits, real_feat = discriminator((real_data,))
                    feat_loss = (fake_feat - real_feat).pow(2.).mean()
                    valid_loss[1] += feat_loss.item() * args.feat_alpha
                recon_loss = reconstruction_loss(predict, target, batch_mask, backward=False, opts=None)["loss"] / BATCH_SIZE
                valid_loss[0] += recon_loss.item()
                pbar.update(BATCH_SIZE)
            pbar.close()
        valid_loss[2] = valid_loss[0] + valid_loss[1]
        for i in range(len(valid_loss)):
            valid_loss[i] /= (iteration + 1.)

        add_tf_image(writer, log_feat[:, :480, :], "valid/noisy_feat", 4, 1, epoch + args.global_epoch)
        add_tf_image(writer, log_label[:, :480, :], "valid/clean_feat", 4, 1, epoch + args.global_epoch)
        add_tf_image(writer, log_enhanced[:, :480, :], "valid/enhanced_feat", 4, 1, epoch + args.global_epoch)
        if "mask" in target_type or "SigApp" in target_type:
            add_tf_image(writer, predict[:, :480, :], "valid/mask", 4, 1, epoch + args.global_epoch)
            add_tf_image(writer, target[:, :480, :], "valid/ideal_mask", 4, 1, epoch + args.global_epoch)
            writer.add_histogram("valid/hist_predict_mask", predict[0, :frame_number[0], :], epoch + args.global_epoch)
            writer.add_histogram("valid/hist_ideal_mask", target[0, :frame_number[0], :], global_step // args.summary_interval)

        if args.target_type == "mapping_log_pow" and opts['compress_label']:
            writer.add_scalars("valid", {
                "log_label_min": opts['log_label_min'],
                "log_label_max": opts["log_label_max"],
            }, epoch + args.global_epoch)

        logger.info("Loss on valid dataset is %.4f" % valid_loss[2])
        writer.add_scalar("valid/recon_loss", valid_loss[0], epoch + args.global_epoch)
        if args.dist_alpha > 0:
            writer.add_scalar('valid/dist_loss', valid_loss[1], epoch + args.global_epoch)
        if args.feat_alpha > 0:
            writer.add_scalar('valid/feat_loss', valid_loss[1], epoch + args.global_epoch)
        writer.add_scalar("valid/valid_loss", valid_loss[2], epoch + args.global_epoch)

        if "fMSE" in model_name:
            loss_crit = valid_loss[1]
        else:
            loss_crit = valid_loss[0]
        # Save the best model to step 0
        if loss_crit < lowest_valid_loss:
            logger.info("Save best models at step %d" % global_step)
            lowest_valid_loss = loss_crit
            save_check_point(generator, None, g_optimizer, None, 0, model_name)
            logger.info("Done.")
            lowest_epoch = epoch
        else:
            if epoch - lowest_epoch > args.early_stopping_patience > 0:
                break

    writer.close()
