import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
from kaldi_fft_dataset import FbankDataloader, FrameDataset
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


if __name__ == '__main__':

    set_seed(1314)

    parser = argparse.ArgumentParser()
    parser.add_argument('--reconstruction_loss', type=str, default="mse")
    parser.add_argument('--adversarial_loss', type=str, default=None)
    parser.add_argument('--gen_model', type=str, default="Tan2018PowCRN")
    parser.add_argument('--dis_model', type=str, default="Tan2018PowCRN")
    parser.add_argument('--alpha', type=float, default=0.0001)
    parser.add_argument('--clean_type', type=str, default="early50")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ngh', type=int, default=256)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--ndh', type=int, default=256)
    parser.add_argument('--global_step', type=int, default=0)
    parser.add_argument('--target_type', type=str, default="mask_irm")
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--summary_interval', type=int, default=100)
    parser.add_argument('--expand_training', type=bool, default=False)

    args = parser.parse_args()
    alpha = args.alpha
    gen_model = args.gen_model
    dis_model = None
    target_type = args.target_type
    model_name_list = [gen_model]
    adversarial_loss = args.adversarial_loss
    reconstruction_loss = args.reconstruction_loss
    if adversarial_loss:
        dis_model = args.dis_model
        model_name_list.append(dis_model)
        model_name_list.append(str(alpha))
    model_name_list.append(target_type)
    model_name_list.append(args.clean_type)
    model_name = "_".join(model_name_list)

    logger = get_my_logger(model_name)

    BATCH_SIZE = args.batch_size
    TIME_STEPS = 1000
    FEAT_LENGTH = 320
    FRAME_LENGTH = 320 + (TIME_STEPS - 1) * 160
    FRAME_SHIFT = 16000 * 10

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

    dis_opts = {}
    if adversarial_loss == "wgangp_loss":
        dis_opts["alpha"] = 10.
    elif adversarial_loss == "hinge_loss":
        dis_opts["margin"] = 1.

    train_dataset = FrameDataset([0, 3, 6], NOISE_LIST, TR05_SIMU_LIST, args.clean_type, True, None)
    train_dataloader = FbankDataloader(train_dataset, opts, BATCH_SIZE, True, num_workers=4, drop_last=True)
    valid_dataset = FrameDataset([0, 3, 6], NOISE_LIST, DT05_SIMU_LIST, args.clean_type, False, None)
    valid_dataloader = FbankDataloader(valid_dataset, opts, BATCH_SIZE, True, num_workers=4, drop_last=True)
    logger.info("Done.")

    logger.info("Start to construct model...")
    device = torch.device('cuda')

    Generator, Discriminator = get_model(gen_model, dis_model)
    if adversarial_loss:
        disc_d_loss, disc_g_loss = get_disc_loss(adversarial_loss)
    reconstruction_loss = get_recon_loss(reconstruction_loss)

    calc_target = get_target(args.target_type)

    generator = Generator(args.ngf, args.ngh).to(device)
    g_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    # discriminator = ConvDiscriminator().to(device)
    if adversarial_loss:
        discriminator = Discriminator(args.ndf, args.ndh).to(device)
        d_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    global_step = args.global_step
    if args.global_step > 0:
        if adversarial_loss:
            load_check_point(generator, discriminator, g_optimizer, d_optimizer, args.global_step, model_name)
        else:
            load_check_point(generator, None, g_optimizer, None, args.global_step, model_name)

    if not os.path.exists("Checkpoints/%s" % model_name):
        os.system("mkdir -p Checkpoints/%s" % model_name)
    logger.info("Backup current code to the model directory Checkpoints/%s/%s" % (model_name, __file__))
    os.system("cp %s Checkpoints/%s/" % (__file__, model_name))

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
    if adversarial_loss:
        print(discriminator)
        input("Make sure the discriminator and press Enter key to continue.")
    epoch = 0
    train_first = False
    # alpha = [0.001, 0.01, 0.1, 1, 1, 1]
    while True:
        if epoch > 0 or train_first:
            generator.train()
            if adversarial_loss:
                discriminator.train()
            pbar = tqdm(total=train_dataset.__len__(), ascii=True, unit="sp", desc="Train %d" % epoch)
            for iteration, (clean_frame, noisy_frame, frame_number, batch_mask) in enumerate(train_dataloader):
                if not args.expand_training:
                    clean_frame = clean_frame[BATCH_SIZE:, :, :]
                    noisy_frame = noisy_frame[BATCH_SIZE:, :, :]
                    batch_mask = batch_mask[BATCH_SIZE:, :]
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
                if adversarial_loss:
                    # train discriminator
                    d_optimizer.zero_grad()
                    disc_d_loss_dict = disc_d_loss(discriminator, log_label, log_enhanced, backward=True, opts=dis_opts)
                    d_optimizer.step()

                    # get discriminative loss for generator
                    disc_g_loss_dict = disc_g_loss(discriminator, log_enhanced, backward=False, opts=dis_opts)
                    g_loss = g_loss + disc_g_loss_dict["g_loss"] * alpha

                recon_loss = reconstruction_loss(predict, target, batch_mask, backward=False, opts=None)["loss"] / BATCH_SIZE
                g_loss = g_loss + recon_loss
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                # log and summary
                if global_step % args.log_interval == 0:
                    writer.add_scalar('train/recon_loss', recon_loss.item(), global_step // args.log_interval)
                    if adversarial_loss:
                        writer.add_scalar('train/d_real_loss', disc_d_loss_dict["real_loss"].item(),
                                          global_step // args.log_interval)
                        writer.add_scalar('train/d_fake_loss', disc_d_loss_dict["fake_loss"].item(),
                                          global_step // args.log_interval)
                        writer.add_scalar('train/g_loss', disc_g_loss_dict["g_loss"].item(),
                                          global_step // args.log_interval)
                if global_step % args.summary_interval == 0:
                    add_tf_image(writer, log_feat[:, :480, :], "train/noisy_feat", 4, 1, global_step // args.summary_interval)
                    add_tf_image(writer, log_label[:, :480, :], "train/clean_feat", 4, 1, global_step // args.summary_interval)
                    add_tf_image(writer, log_enhanced[:, :480, :], "train/enhanced_feat", 4, 1, global_step // args.summary_interval)
                    if "mask" in target_type:
                        add_tf_image(writer, predict[:, :480, :], "train/mask", 4, 1, global_step // args.summary_interval)
                        add_tf_image(writer, target[:, :480, :], "train/ideal_mask", 4, 1, global_step // args.summary_interval)
                        writer.add_histogram("train/hist_predict_mask", predict[0, :frame_number[0], :], global_step // args.summary_interval)
                        writer.add_histogram("train/hist_ideal_mask", target[0, :frame_number[0], :], global_step // args.summary_interval)

                global_step += 1
                pbar.update(BATCH_SIZE)

        logger.info("Save models at step %d" % global_step)
        if adversarial_loss:
            save_check_point(generator, discriminator, g_optimizer, d_optimizer, global_step, model_name)
        else:
            save_check_point(generator, None, g_optimizer, None, global_step, model_name)
        logger.info("Done.")
        with torch.no_grad():
            logger.info("Complete train %d epochs, start to evaluate performance in valid dataset." % epoch)
            valid_loss = 0.
            generator.eval()
            if adversarial_loss:
                discriminator.eval()
            pbar = tqdm(total=valid_dataset.__len__(), ascii=True, unit="sp", desc="Valid %d" % epoch)
            for iteration, (clean_frame, noisy_frame, frame_number, batch_mask) in enumerate(valid_dataloader):
                if not args.expand_training:
                    clean_frame = clean_frame[BATCH_SIZE:, :, :]
                    noisy_frame = noisy_frame[BATCH_SIZE:, :, :]
                    batch_mask = batch_mask[BATCH_SIZE:, :]
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
                recon_loss = reconstruction_loss(predict, target, batch_mask, backward=False, opts=None)["loss"] / BATCH_SIZE
                valid_loss += recon_loss.item()
                pbar.update(BATCH_SIZE)
            pbar.close()
        valid_loss = valid_loss / (iteration + 1.)

        add_tf_image(writer, log_feat[:, :480, :], "valid/noisy_feat", 4, 1, epoch)
        add_tf_image(writer, log_label[:, :480, :], "valid/clean_feat", 4, 1, epoch)
        add_tf_image(writer, log_enhanced[:, :480, :], "valid/enhanced_feat", 4, 1, epoch)
        if "mask" in target_type:
            add_tf_image(writer, predict[:, :480, :], "valid/mask", 4, 1, epoch)
            add_tf_image(writer, target[:, :480, :], "valid/ideal_mask", 4, 1, epoch)
            writer.add_histogram("valid/hist_predict_mask", predict[0, :frame_number[0], :], epoch)
            writer.add_histogram("valid/hist_ideal_mask", target[0, :frame_number[0], :], global_step // args.summary_interval)
        logger.info("Loss on valid dataset is %.4f" % valid_loss)
        writer.add_scalar("valid/recon_loss", valid_loss, epoch)
        epoch += 1
    writer.close()