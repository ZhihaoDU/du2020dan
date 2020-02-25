import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_targets(feat, label, noise, predict, opts=None):
    with torch.no_grad():
        eps = opts['eps']
        target = torch.log(label + eps)
    if opts['compress_label']:
        with torch.no_grad():
            opts['log_label_min'] = min(opts['log_label_min'], target.min().item())
            opts['log_label_max'] = max(opts['log_label_max'], target.max().item())
            target = (target - opts['log_label_min']) / (opts['log_label_max'] - opts['log_label_min'])
        if predict is None:
            predict = target
        else:
            predict = torch.sigmoid(predict)
        enhanced = torch.exp(predict * (opts['log_label_max'] - opts['log_label_min']) + opts['log_label_min'])
        return {"target": target, "predict": predict, "enhanced": enhanced, "ideal": label,
                "real_data": target * (opts['log_label_max'] - opts['log_label_min']) + opts['log_label_min'],
                "fake_data": predict * (opts['log_label_max'] - opts['log_label_min']) + opts['log_label_min']}
    else:
        if predict is None:
            predict = target
        enhanced = torch.exp(predict)
        return {"target": target, "predict": predict, "enhanced": enhanced,
                "ideal": label, "real_data": target, "fake_data": predict}
