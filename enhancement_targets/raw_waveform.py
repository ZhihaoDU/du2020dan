import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_targets(feat, label, noise, predict, opts=None):
    return {"target": label, "predict": predict, "enhanced": predict, "ideal": label,
            "real_data": torch.cat([label, feat], 1),
            "fake_data": torch.cat([predict, feat], 1)}
