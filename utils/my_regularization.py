import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_regularization(model, mode="weight"):
    regularization_loss = 0.
    for name, para in model.named_parameters():
        if mode == "all" or (mode == "weight" and 'weight' in name):
            if para.requires_grad:
                regularization_loss = regularization_loss + para.pow(2.).sum()
    return regularization_loss


def l1_regularization(model, mode="weight"):
    regularization_loss = 0.
    for name, para in model.named_parameters():
        if mode == "all" or (mode == "weight" and 'weight' in name):
            if para.requires_grad:
                regularization_loss = regularization_loss + para.abs().sum()
    return regularization_loss
