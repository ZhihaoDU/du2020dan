import torch.nn as nn
from utils.spectral_normalization import SpectralNorm


class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            SpectralNorm(nn.Conv2d(nc, ndf, 3, 1, 1, bias=False)),
            nn.LeakyReLU(0.1, inplace=True),
            SpectralNorm(nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf) x 32 x 32
            SpectralNorm(nn.Conv2d(ndf, ndf * 2, 3, 1, 1, bias=False)),
            nn.LeakyReLU(0.1, inplace=True),
            SpectralNorm(nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*2) x 16 x 16
            SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=False)),
            nn.LeakyReLU(0.1, inplace=True),
            SpectralNorm(nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*4) x 8 x 8
            SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=False)),
            nn.LeakyReLU(0.1, inplace=True),
            SpectralNorm(nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*8) x 4 x 4
            SpectralNorm(nn.Conv2d(ndf * 8, ndf * 8, 4, 1, 0, bias=False)),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*8) x 1 x 1
            # nn.Sigmoid()
        )
        self.dense = SpectralNorm(nn.Linear(ndf * 8, 1))

    def forward(self, input):
        output = self.main(input)
        bb = output.size(0)
        logits = self.dense(output.view(bb, -1))

        return logits.squeeze(1)
