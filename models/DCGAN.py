import torch.nn as nn
import torch
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, opts):
        super(Generator, self).__init__()
        nz = opts["z_dim"]
        ngf = opts["nf"]
        nc = opts["nc"]
        self.conditional = opts["condition"]
        self.c_dim = opts["condition_dim"]
        self.c_type = opts["condition_type"]
        self.l1 = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz + self.c_dim if self.conditional else nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
        )
        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8 + self.c_dim if self.conditional and self.c_type == "full" else ngf * 8,
                               ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
        )
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4 + self.c_dim if self.conditional and self.c_type == "full" else ngf * 4,
                               ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
        )
        self.l4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2 + self.c_dim if self.conditional and self.c_type == "full" else ngf * 2,
                               ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
        )
        self.l5 = nn.Sequential(
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if not self.conditional:
            x = input[0]
            h1 = self.l1(x)
            h2 = self.l2(h1)
            h3 = self.l3(h2)
            h4 = self.l4(h3)
            output = self.l5(h4)
        else:
            x = input[0]
            c = input[1]
            h1 = self.l1(torch.cat([x, c.expand(-1, -1, x.size(2), x.size(3))], 1))
            h2 = self.l2(torch.cat([h1, c.expand(-1, -1, h1.size(2), h1.size(3))], 1) if self.c_type == "full" else h1)
            h3 = self.l3(torch.cat([h2, c.expand(-1, -1, h2.size(2), h2.size(3))], 1) if self.c_type == "full" else h2)
            h4 = self.l4(torch.cat([h3, c.expand(-1, -1, h3.size(2), h3.size(3))], 1) if self.c_type == "full" else h3)
            output = self.l5(h4)
        return output


class InstanceNormalization2d(nn.Module):

    def __init__(self, num_features, affine=False):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        if self.affine:
            self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1, dtype=torch.float32))
            self.shift = nn.Parameter(torch.zeros(1, num_features, 1, 1, dtype=torch.float32))

    def forward(self, *input):
        x = input[0]
        # x = F.instance_norm(x, use_input_stats=True)
        bb, cc, hh, ww = x.size()
        x_mean = x.view(bb, cc, hh*ww).mean(dim=2, keepdim=True).view(bb, cc, 1, 1)
        x_std = x.view(bb, cc, hh*ww).std(dim=2, keepdim=True).view(bb, cc, 1, 1)
        x = (x - x_mean) / x_std
        if self.affine:
            x = (x - self.shift) * self.scale
        return x


class Discriminator(nn.Module):
    def __init__(self, opts):
        super(Discriminator, self).__init__()
        nc = opts["nc"]
        ndf = opts["nf"]
        self.conditional = opts["condition"]
        self.c_dim = opts["condition_dim"]
        self.c_type = opts["condition_type"]

        self.l1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc + self.c_dim if self.conditional else nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(ndf + self.c_dim if self.conditional and self.c_type == "full" else ndf,
                      ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            # nn.InstanceNorm2d(ndf * 2),
            # InstanceNormalization2d(ndf * 2, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(ndf * 2 + self.c_dim if self.conditional and self.c_type == "full" else ndf * 2,
                      ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            # nn.InstanceNorm2d(ndf * 4),
            # InstanceNormalization2d(ndf * 4, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
        )
        self.l4 = nn.Sequential(
            nn.Conv2d(ndf * 4 + self.c_dim if self.conditional and self.c_type == "full" else ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            # nn.InstanceNorm2d(ndf * 8),
            # InstanceNormalization2d(ndf * 8, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
        )
        self.l5 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, input):
        if not self.conditional:
            x = input[0]
            h1 = self.l1(x)
            h2 = self.l2(h1)
            h3 = self.l3(h2)
            h4 = self.l4(h3)
            output = self.l5(h4)
            return output, h4
        else:
            x = input[0]
            c = input[1]
            h1 = self.l1(torch.cat([x, c.expand(-1, -1, x.size(2), x.size(3))], 1))
            h2 = self.l2(torch.cat([h1, c.expand(-1, -1, h1.size(2), h1.size(3))], 1) if self.c_type == "full" else h1)
            h3 = self.l3(torch.cat([h2, c.expand(-1, -1, h2.size(2), h2.size(3))], 1) if self.c_type == "full" else h2)
            h4 = self.l4(torch.cat([h3, c.expand(-1, -1, h3.size(2), h3.size(3))], 1) if self.c_type == "full" else h3)
            output = self.l5(h4)
            return output, h4
