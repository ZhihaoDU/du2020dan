import torch
import torch.nn as nn
import torch.nn.functional as F


def d_loss(discriminator, real_batch, fake_batch, backward=False, opts=None):
    margin = opts["hinge_margin"]
    # real batch train
    logits_real, _ = discriminator.forward(real_batch)
    real_mean = logits_real.mean()
    real_loss = F.relu(margin - logits_real).mean()
    if backward:
        real_loss.backward()

    # fake batch train
    logits_fake, _ = discriminator.forward(fake_batch)
    fake_mean = logits_fake.mean()
    fake_loss = F.relu(margin + logits_fake).mean()
    if backward:
        fake_loss.backward()
    w_dis = real_mean - fake_mean

    return {"real_loss": real_loss, "fake_loss": fake_loss, "w_dis": w_dis,
            "real_mean": real_mean, "fake_mean": fake_mean}


def g_loss(discriminator, fake_batch, backward=False, opts=None):
    logits, feat = discriminator.forward(fake_batch)
    logits_mean = logits.mean()
    g_loss = -logits_mean
    if backward:
        g_loss.backward()
    return {"g_loss": g_loss, "logits_mean": logits_mean, "feat": feat}
