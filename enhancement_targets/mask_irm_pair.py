import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_targets(feat, label, noise, predict, opts=None):
    with torch.no_grad():
        target = label / (label + noise)
        target[torch.isinf(target)] = 0.
        target[torch.isnan(target)] = 0.
        ideal = feat * target
    if predict is None:
        predict = target
    else:
        predict = torch.sigmoid(predict)
    enhanced = feat * predict
    return {"target": target, "predict": predict, "enhanced": enhanced,
            "ideal": ideal, "real_data": target, "fake_data": predict}
