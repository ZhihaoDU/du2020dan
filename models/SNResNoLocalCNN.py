import torch.nn as nn
from utils.spectral_normalization import SpectralNorm
import torch.nn.functional as F
import math
import torch
from modules.NoLocalLayer import NonLocalBlock2D


class GeneratorResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels=None, activation=F.relu, upsample=False):
        super(GeneratorResBlock, self).__init__()
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = (in_channels != out_channels) or upsample # False
        hidden_channels = out_channels if hidden_channels is None else hidden_channels

        self.c1 = nn.ConvTranspose2d(in_channels, hidden_channels, 3, 1, 1)
        # nn.init.xavier_uniform_(self.c1.weight, nn.init.calculate_gain('relu'))
        # self.c1 = SpectralNorm(self.c1, power_iterations=1)
        if self.upsample:
            self.c2 = nn.ConvTranspose2d(hidden_channels, out_channels, 4, 2, 1)
        else:
            self.c2 = nn.ConvTranspose2d(hidden_channels, out_channels, 3, 1, 1)
        # nn.init.xavier_uniform_(self.c2.weight, nn.init.calculate_gain('relu'))
        # self.c2 = SpectralNorm(self.c2, power_iterations=1)

        self.res = nn.Sequential(
            self.c1,
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            self.c2,
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        if self.learnable_sc:
            if self.upsample:
                self.sc_c = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
            else:
                self.sc_c = nn.ConvTranspose2d(in_channels, out_channels, 3, 1, 1)
            # nn.init.xavier_uniform_(self.sc_c.weight, nn.init.calculate_gain('relu'))
            # self.sc_c = SpectralNorm(self.sc_c, power_iterations=1)
            self.sc = nn.Sequential(
                self.sc_c,
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

    def residual(self, input):
        return self.res(input)

    def shortcut(self, input):
        x = input
        if self.learnable_sc:
            x = self.sc(x)
        return x

    def forward(self, input):
        return self.residual(input) + self.shortcut(input)


class DiscriminatorResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels=None, downsample=False):
        super(DiscriminatorResBlock, self).__init__()
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample # True
        hidden_channels = in_channels if hidden_channels is None else hidden_channels

        self.c1 = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)
        nn.init.xavier_uniform_(self.c1.weight, nn.init.calculate_gain("leaky_relu"))
        self.c1 = SpectralNorm(self.c1, power_iterations=1)
        if self.downsample:
            self.c2 = nn.Conv2d(hidden_channels, out_channels, 4, 2, 1)
        else:
            self.c2 = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)
        nn.init.xavier_uniform_(self.c2.weight, nn.init.calculate_gain("leaky_relu"))
        self.c2 = SpectralNorm(self.c2, power_iterations=1)
        self.res = nn.Sequential(
            self.c1,
            # nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(),
            self.c2,
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

        if self.learnable_sc:
            if self.downsample:
                self.c_sc = nn.Conv2d(in_channels, out_channels, 4, 2, 1)
            else:
                self.c_sc = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            nn.init.xavier_uniform_(self.c_sc.weight, nn.init.calculate_gain("leaky_relu"))
            self.c_sc = SpectralNorm(self.c_sc, power_iterations=1)
            self.sc = nn.Sequential(
                self.c_sc,
                # nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(),
            )

    def residual(self, input):
        return self.res(input)

    def shortcut(self, input):
        h = input
        if self.learnable_sc:
            h = self.sc(h)
        return h

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

        self.c1 = nn.ConvTranspose2d(nz, ngf*16, 4, 1, 0)
        # nn.init.xavier_uniform_(self.c1.weight, nn.init.calculate_gain('relu'))
        # self.c1 = SpectralNorm(self.c1, power_iterations=1)

        self.ngf = ngf
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # state size. nz
            self.c1,
            nn.BatchNorm2d(ngf*16),
            nn.ReLU(),
            # state size. ngf*16 x 4 x 4
            GeneratorResBlock(ngf * 16, ngf * 8, upsample=True),
            # state size. ngf*8 x 8 x 8
            NonLocalBlock2D(ngf * 8, ngf, "embedded_gaussian", False, False),
            # state size. ngf*8 x 8 x 8
            GeneratorResBlock(ngf * 8, ngf * 4, upsample=True),
            # state size. ngf*4 x 16 x 16
            GeneratorResBlock(ngf * 4, ngf * 2, upsample=True),
            # state size. ngf*2 x 32 x 32
            GeneratorResBlock(ngf * 2, ngf * 1, upsample=True),
            # state size. ngf*1 x 64 x 64
        )
        self.final_conv = nn.Conv2d(ngf, nc, 3, 1, 1)
        # nn.init.xavier_uniform_(self.final_conv.weight, nn.init.calculate_gain('tanh'))
        # self.final_conv = SpectralNorm(self.final_conv, power_iterations=1)

    def forward(self, input):
        h2 = self.main(input[0])
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
        self.c1 = nn.Conv2d(nc, ndf, 3, 1, 1)
        nn.init.xavier_uniform_(self.c1.weight, nn.init.calculate_gain("leaky_relu"))
        self.c1 = SpectralNorm(self.c1, power_iterations=1)
        self.first_conv = nn.Sequential(
            self.c1,
            nn.LeakyReLU(),
        )
        self.main = nn.Sequential(
            # input is (ndf) x 64 x 64
            DiscriminatorResBlock(ndf * 1, ndf * 2, downsample=True),
            # state size. (ndf*2) x 32 x 32
            DiscriminatorResBlock(ndf * 2, ndf * 4, downsample=True),
            # state size. (ndf*4) x 16 x 16
            DiscriminatorResBlock(ndf * 4, ndf * 8, downsample=True),
            # state size. (ndf*8) x 8 x 8
            NonLocalBlock2D(ndf * 8, ndf, "embedded_gaussian", False, False),
            # state size. (ndf*8) x 8 x 8
            DiscriminatorResBlock(ndf * 8, ndf * 16, downsample=True),
            # state size. (ndf*16) x 4 x 4
        )
        self.final_conv = nn.Conv2d(ndf*16, 1, 4, 1, 0)
        nn.init.xavier_uniform_(self.final_conv.weight, nn.init.calculate_gain('linear'))
        self.final_conv = SpectralNorm(self.final_conv, power_iterations=1)

    def forward(self, input):
        h1 = self.first_conv(input[0])
        feat = self.main(h1)
        # state size. batch_size x (ndf*16) x 4 x 4
        logits = self.final_conv(feat)
        # state size. batch_size x 1

        return logits.squeeze(), feat
