import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_targets(feat, label, noise, predict, opts=None):
    predict = F.softplus(predict)
    with torch.no_grad():
        target = label.sqrt()
    if predict is None:
        predict = target
    enhanced = predict.pow(2.)
    return {"target": target, "predict": predict, "enhanced": enhanced,
            "ideal": label, "real_data": target, "fake_data": predict}
