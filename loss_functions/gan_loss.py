import torch
import torch.nn as nn
import torch.nn.functional as F


def d_loss(discriminator, real_batch, fake_batch, backward=False, opts=None):
    # real batch train
    logits_real, _ = discriminator.forward(real_batch.detach())
    real_mean = logits_real.mean()
    real_loss = F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real).to(logits_real))
    if backward:
        real_loss.backward()

    # fake batch train
    logits_fake, _ = discriminator.forward(fake_batch.detach())
    fake_mean = logits_fake.mean()
    fake_loss = F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake).to(logits_fake))
    if backward:
        fake_loss.backward()
    w_dis = real_mean - fake_mean

    return {"real_loss": real_loss, "fake_loss": fake_loss, "w_dis": w_dis,
            "real_mean": real_mean, "fake_mean": fake_mean}


def g_loss(g_out, discriminator, backward=False, opts=None):
    logits, _ = discriminator.forward(g_out)
    logits_mean = logits.mean()
    g_loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits).to(logits))
    if backward:
        g_loss.backward()
    return {"g_loss": g_loss, "logits_mean": logits_mean}
