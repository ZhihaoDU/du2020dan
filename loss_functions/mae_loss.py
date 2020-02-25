import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_function(predict, label, weights=None, backward=False, opts=None):
    if weights is None:
        weights = torch.ones_like(predict).to(predict)
    # weights = weights / weights.sum()
    loss = ((predict - label).abs() * weights).sum()
    if backward:
        loss.backward()
    return {"loss": loss}
