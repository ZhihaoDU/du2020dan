import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_targets(feat, label, noise, predict, opts=None):
    """
    :param feat: Noisy spectrogram in linear format (no-logarithm)
    :param label: Clean spectrogram in linear format (no-logarithm)
    :param noise: Noise spectrogram in linear format (no-logarithm)
    :param predict: The output of model. For 'mask_log' method, the predict is the mask in log spectrogram domain.
    :param opts: Options for 'mask_log' methods
        clip_low: The low bound of log_mask (0 is experimentally good).
        clip_high: The high bound of log_mask (1 is experimentally good).
        log_power_offset: The offset of log spectrogram. Actually, it is a scale to zoom out the spectrogram to avoid
        the value is to small. Multiplies in linear is equal to additions in log domain.
        (10 is experimentally good for Root-Mean-Square normalized speech)
    :return: target: training target, predict: the predict log_mask, enhanced: the enhanced spectrogram in linear format
    """

    clip_low = opts["clip_low"]
    clip_high = opts["clip_high"]
    log_power_offset = opts["log_power_offset"]
    with torch.no_grad():
        eps = opts['eps']
        log_feat = torch.log(feat + eps)
        log_label = torch.log(label + eps)
        target = (log_label + log_power_offset) / (log_feat + log_power_offset)
        target[torch.isinf(target)] = 0.
        target[torch.isnan(target)] = 0.
        if clip_low is not None and clip_high is not None:
            target = F.hardtanh(target, clip_low, clip_high)
        log_ideal = (log_feat + log_power_offset) * target - log_power_offset
        ideal = torch.exp(log_ideal)
    if predict is None:
        predict = target
    else:
        predict = torch.sigmoid(predict) * (clip_high - clip_low) + clip_low
    log_enhanced = (log_feat + log_power_offset) * predict - log_power_offset
    enhanced = torch.exp(log_enhanced)
    return {"target": target, "predict": predict, "enhanced": enhanced.detach(),
            "ideal": ideal, "real_data": log_ideal, "fake_data": log_enhanced}
