import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_gradient_penalty(discriminator, real_batch, fake_batch):
    from torch import autograd
    real_data = real_batch[0]
    fake_data = fake_batch[0]
    if len(real_batch) > 1:
        condition = real_batch[1]
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(real_data)
    alpha = alpha.expand_as(real_data)

    interpolates = autograd.Variable(alpha * real_data + (1. - alpha) * fake_data, requires_grad=True)
    if len(real_batch) > 1:
        logits, _ = discriminator((interpolates, condition))
    else:
        logits, _ = discriminator((interpolates,))

    gradients = autograd.grad(outputs=logits, inputs=interpolates,
                              grad_outputs=torch.ones_like(logits).to(real_data),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    # gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    gradient_penalty = F.relu(gradients.norm(2, dim=1) - 2).mean()
    gradient_norm = gradients.norm(2, dim=1).mean()
    return gradient_penalty, gradient_norm


def d_loss(discriminator, real_batch, fake_batch, backward=False, opts=None):
    alpha = opts["gp_alpha"]
    # real batch train
    logits_real, _ = discriminator.forward(real_batch)
    real_mean = logits_real.mean()
    real_loss = 0.5 * (logits_real - 1.).pow(2.).mean()
    if backward:
        real_loss.backward()

    # fake batch train
    logits_fake, _ = discriminator.forward(fake_batch)
    fake_mean = logits_fake.mean()
    fake_loss = 0.5 * (logits_fake - 0.).pow(2.).mean()
    if backward:
        fake_loss.backward()
    gp, gn = calc_gradient_penalty(discriminator, real_batch, fake_batch)
    if backward:
        (gp * alpha).backward()
    w_dis = real_mean - fake_mean
    return {"real_loss": real_loss, "fake_loss": fake_loss, "w_dis": w_dis, "gradient_penalty": gp * alpha,
            "real_mean": real_mean, "fake_mean": fake_mean, "gradient_norm": gn}


def g_loss(discriminator, fake_batch, backward=False, opts=None):
    logits, feat = discriminator.forward(fake_batch)
    logits_mean = logits.mean()
    g_loss = 0.5 * (logits - 1.).pow(2.).mean()
    if backward:
        g_loss.backward()
    return {"g_loss": g_loss, "logits_mean": logits_mean, "feat": feat}
