import torch
import torch.nn as nn
import torch.nn.functional as F


def d_loss(discriminator, real_batch, fake_batch, backward=False, opts=None):
    # real batch train
    logits_real, _ = discriminator.forward(real_batch)
    real_mean = logits_real.mean()
    # (D(x) - 1)^2
    real_loss = 0.5 * (logits_real - 1).pow(2.).mean()
    if backward:
        real_loss.backward()

    # fake batch train
    logits_fake, _ = discriminator.forward(fake_batch)
    fake_mean = logits_fake.mean()
    # (D(G(z)) - 0)^2
    fake_loss = 0.5 * logits_fake.pow(2.).mean()
    if backward:
        fake_loss.backward()
    w_dis = real_mean - fake_mean

    return {"real_loss": real_loss, "fake_loss": fake_loss, "w_dis": w_dis,
            "real_mean": real_mean, "fake_mean": fake_mean}


def g_loss(g_out, discriminator, backward=False, opts=None):
    logits, feat = discriminator.forward(g_out)
    logits_mean = logits.mean()
    g_loss = 0.5 * (logits - 1.).pow(2.).mean()
    if backward:
        g_loss.backward()
    return {"g_loss": g_loss, "logits_mean": logits_mean, "feat": feat}
