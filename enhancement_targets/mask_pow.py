import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_targets(feat, label, noise, predict, opts=None):
    clip_low = opts["value_low"]
    clip_high = opts["value_high"]
    compress_function = opts["compress_function"]

    if clip_low and clip_high and predict:
        predict = F.hardtanh(predict, clip_low, clip_high)
    if compress_function and predict:
        predict = compress_function(predict)

    with torch.no_grad():
        target = label / feat
        target[torch.isinf(target)] = 0.
        target[torch.isnan(target)] = 0.
        if clip_low and clip_high:
            target = F.hardtanh(target, clip_low, clip_high)
        if compress_function:
            target = compress_function(target)
        ideal = feat * target
    if predict is None:
        predict = target
    enhanced = feat * predict
    return {"target": target, "predict": predict, "enhanced": enhanced,
            "ideal": ideal, "real_data": ideal, "fake_data": enhanced}
