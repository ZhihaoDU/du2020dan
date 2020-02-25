import torch.nn as nn
from utils.spectral_normalization import SpectralNorm
import torch.nn.functional as F
import math
import torch


def upsampling(x):
    # return F.upsample(x, scale_factor=2)
    return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)


def upsample_conv(x, conv):
    return conv(upsampling(x))


class GeneratorResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1, activation=F.relu, upsample=False):
        super(GeneratorResBlock, self).__init__()
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = (in_channels != out_channels) or upsample # False
        hidden_channels = out_channels if hidden_channels is None else hidden_channels

        self.c1 = nn.Conv2d(in_channels, hidden_channels, ksize, 1, pad)
        nn.init.xavier_uniform_(self.c1.weight, 1.)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, ksize, 1, pad)
        nn.init.xavier_uniform_(self.c2.weight, 1.)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(hidden_channels)

        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
            nn.init.xavier_uniform_(self.c_sc.weight, math.sqrt(2.))

    def residual(self, input):
        h = input
        h = self.bn1(h)
        h = self.activation(h)
        h = upsample_conv(h, self.c1)
        h = self.bn2(h)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def shortcut(self, input):
        if self.learnable_sc:
            if self.upsample:
                x = upsample_conv(input, self.c_sc)
            else:
                x = self.c_sc(input)
        else:
            if self.upsample:
                x = upsampling(input)
            else:
                x = input
        return x

    def forward(self, input):
        return self.residual(input) + self.shortcut(input)


class DiscriminatorResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1, activation=F.relu, downsample=False):
        super(DiscriminatorResBlock, self).__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample # True
        hidden_channels = in_channels if hidden_channels is None else hidden_channels

        self.c1 = nn.Conv2d(in_channels, hidden_channels, ksize, 1, pad)
        nn.init.xavier_uniform_(self.c1.weight, 1.)
        self.c1 = SpectralNorm(self.c1, power_iterations=5)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, ksize, 1, pad)
        nn.init.xavier_uniform_(self.c2.weight, 1.)
        self.c2 = SpectralNorm(self.c2, power_iterations=5)

        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
            nn.init.xavier_uniform_(self.c_sc.weight, math.sqrt(2.))
            self.c_sc = SpectralNorm(self.c_sc, power_iterations=5)

    def residual(self, input):
        h = input
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = downsample(h)
        return h

    def shortcut(self, input):
        if self.learnable_sc:
            h = self.c_sc(input)
            if self.downsample:
                h = downsample(h)
        else:
            h = input
        return h

    def forward(self, input):
        return self.residual(input) + self.shortcut(input)


def downsample(x):
    return F.avg_pool2d(x, 2)


class DiscriminatorOptimizedResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, ksize=3, pad=1, activation=F.relu):
        super(DiscriminatorOptimizedResBlock, self).__init__()
        self.activation = activation
        self.c1 = nn.Conv2d(in_channels, out_channels, ksize, 1, pad)
        nn.init.xavier_uniform_(self.c1.weight, 1.)
        self.c1 = SpectralNorm(self.c1, power_iterations=5)
        self.c2 = nn.Conv2d(out_channels, out_channels, ksize, 1, pad)
        nn.init.xavier_uniform_(self.c2.weight, 1.)
        self.c2 = SpectralNorm(self.c2, power_iterations=5)
        self.c_sc = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        nn.init.xavier_uniform_(self.c_sc.weight, math.sqrt(2))
        self.c_sc = SpectralNorm(self.c_sc, power_iterations=5)

    def residual(self, input):
        h = input
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = downsample(h)
        return h

    def shortcut(self, input):
        return self.c_sc(downsample(input))

    def forward(self, input):
        return self.residual(input) + self.shortcut(input)


class Generator(nn.Module):
    def __init__(self, opts):
        super(Generator, self).__init__()
        nz = opts["z_dim"]
        ngf = opts["nf"]
        nc = opts["nc"]
        self.conditional = opts["condition"]
        self.c_dim = opts["condition_dim"]
        self.c_type = opts["condition_type"]

        self.fc1 = nn.Linear(nz, 4 * 4 * ngf)
        nn.init.xavier_uniform_(self.fc1.weight, 1.)
        self.ngf = ngf
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # state size. ngf x 4 x 4
            GeneratorResBlock(ngf, ngf, upsample=True),
            # state size. ngf x 8 x 8
            GeneratorResBlock(ngf, ngf, upsample=True),
            # state size. ngf x 16 x 16
            GeneratorResBlock(ngf, ngf, upsample=True),
            # state size. ngf x 32 x 32
            GeneratorResBlock(ngf, ngf, upsample=True),
            # state size. ngf x 64 x 64
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
        )
        self.final_conv = nn.Conv2d(ngf, nc, 3, 1, 1)
        nn.init.xavier_uniform_(self.final_conv.weight, 1.)

    def forward(self, input):
        input = input[0].squeeze(3).squeeze(2)
        h1 = self.fc1(input)
        h2 = self.main(h1.view(-1, self.ngf, 4, 4))
        h3 = torch.tanh(self.final_conv(h2))
        return h3


class Discriminator(nn.Module):
    def __init__(self, opts):
        super(Discriminator, self).__init__()
        nc = opts["nc"]
        ndf = opts["nf"]
        self.conditional = opts["condition"]
        self.c_dim = opts["condition_dim"]
        self.c_type = opts["condition_type"]

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            DiscriminatorOptimizedResBlock(nc, ndf),
            # state size. (ndf) x 32 x 32
            DiscriminatorResBlock(ndf, ndf, downsample=True),
            # state size. (ndf) x 16 x 16
            DiscriminatorResBlock(ndf, ndf, downsample=True),
            # state size. (ndf) x 8 x 8
            DiscriminatorResBlock(ndf, ndf, downsample=False),
            # state size. (ndf) x 8 x 8
            DiscriminatorResBlock(ndf, ndf, downsample=False),
            # state size. (ndf) x 8 x 8
            nn.ReLU(),
        )
        self.dense = nn.Linear(ndf, 1, bias=False)
        nn.init.xavier_uniform_(self.dense.weight, 1.)
        self.dense = SpectralNorm(self.dense, power_iterations=5)

    def forward(self, input):
        feat = self.main(input[0])
        # state size. batch_size x (ndf) x 8 x 8
        output = feat.sum(dim=[2, 3])
        # state size. batch_size x (ndf)
        assert len(output.size()) == 2
        bb = output.size(0)
        logits = self.dense(output.view(bb, -1))
        assert logits.size(1) == 1
        # state size. batch_size x 1

        return logits.squeeze(1), feat
