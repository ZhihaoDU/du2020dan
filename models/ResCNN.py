import torch.nn as nn
import torch.nn.functional as F


class GeneratorResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels=None):
        super(GeneratorResBlock, self).__init__()
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            # nn.UpsamplingNearest2d(scale_factor=2),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, 3, 1, 1),
        )
        self.shortcut = nn.Sequential(
            # nn.UpsamplingNearest2d(scale_factor=2),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        )

    def forward(self, input):
        return self.residual(input) + self.shortcut(input)


class DiscriminatorResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels=None):
        super(DiscriminatorResBlock, self).__init__()
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.residual = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, 3, 1, 1),
            nn.AvgPool2d(2),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.AvgPool2d(2),
        )

    def forward(self, input):
        return self.residual(input) + self.shortcut(input)


class DiscriminatorOptimizedResBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DiscriminatorOptimizedResBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.AvgPool2d(2),
        )
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
        )

    def forward(self, input):
        return self.residual(input) + self.shortcut(input)


class Generator(nn.Module):
    def __init__(self, opts):
        super(Generator, self).__init__()
        nz = opts["z_dim"]
        ngf = opts["nf"]
        nc = opts["nc"]
        self.fc1 = nn.Linear(nz, 4 * 4 * ngf * 16)
        self.ngf = ngf
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # state size. (ngf*16) x 4 x 4
            GeneratorResBlock(ngf * 16, ngf * 8),
            # state size. (ngf*8) x 8 x 8
            GeneratorResBlock(ngf * 8, ngf * 4),
            # state size. (ngf*4) x 16 x 16
            GeneratorResBlock(ngf * 4, ngf * 2),
            # state size. (ngf*2) x 32 x 32
            GeneratorResBlock(ngf * 2, ngf),
            # state size. (ngf) x 64 x 64
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            nn.Conv2d(ngf, nc, 3, 1, 1),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        input = input[0].squeeze(3).squeeze(2)
        h = self.fc1(input)
        output = self.main(h.view(-1, self.ngf * 16, 4, 4))
        return output


class Discriminator(nn.Module):
    def __init__(self, opts):
        super(Discriminator, self).__init__()
        nc = opts["nc"]
        ndf = opts["nf"]
        self.main = nn.Sequential(
            # input is (ndf) x 64 x 64
            DiscriminatorOptimizedResBlock(nc, ndf),
            # state size. (ndf) x 32 x 32
            DiscriminatorResBlock(ndf, ndf * 2),
            # state size. (ndf*2) x 16 x 16
            DiscriminatorResBlock(ndf * 2, ndf * 4),
            # state size. (ndf*4) x 8 x 8
            DiscriminatorResBlock(ndf * 4, ndf * 8),
            # state size. (ndf*8) x 4 x 4
            DiscriminatorResBlock(ndf * 8, ndf * 16),
            # state size. (ndf*16) x 2 x 2
            nn.ReLU(),
        )
        self.dense = nn.Linear(ndf * 16, 1)

    def forward(self, input):
        feat = self.main(input[0])
        # state size. batch_size x (ndf*16) x 2 x 2
        output = feat.sum(dim=[2, 3])
        # state size. batch_size x (ndf*16)
        bb = output.size(0)
        logits = self.dense(output.view(bb, -1))
        # state size. batch_size x 1

        return logits.squeeze(1), feat
