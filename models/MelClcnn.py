import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, opts):
        super(Generator, self).__init__()
        self.hidden_units = hidden_units = opts['hidden_units']
        self.layer_number = layer_number = opts['layer_number']
        self.ngf = ngf = opts['ngf']
        self.bidirectional = bidirectional = opts['bidirectional']

        self.conv1 = nn.Sequential(
            # 1 x t x 40
            nn.Conv2d(1, ngf, (3, 4), (1, 2), (1, 1)),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ngf, ngf, (3, 3), (1, 1), (1, 1), (1, 1)),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ngf, ngf, (3, 3), (1, 1), (2, 1), (2, 1)),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            # ngf x t x 20
            nn.Conv2d(ngf, ngf * 2, (3, 4), (1, 2), (1, 1)),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ngf * 2, ngf * 2, (3, 3), (1, 1), (4, 1), (4, 1)),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ngf * 2, ngf * 2, (3, 3), (1, 1), (8, 1), (8, 1)),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2),
        )
        self.conv3 = nn.Sequential(
            # ngf*2 x t x 10
            nn.Conv2d(ngf * 2, ngf * 4, (3, 4), (1, 2), (1, 1)),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2),
            # ngf*4 x t x 5
            nn.Conv2d(ngf * 4, ngf * 4, (3, 3), (1, 1), (16, 1), (16, 1)),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2),
        )
        self.lstm = nn.LSTM(ngf * 4 * 5, hidden_units, layer_number, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Sequential(
            nn.Linear(hidden_units * (bidirectional + 1), ngf * 4 * 5),
            nn.BatchNorm1d(ngf * 4 * 5),
            nn.LeakyReLU(0.2),
        )
        self.deconv1 = nn.Sequential(
            # ngf*4 x t x 5
            nn.ConvTranspose2d(ngf * 4 * 2, ngf * 4, (3, 3), (1, 1), (16, 1), dilation=(16, 1)),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, (3, 4), (1, 2), (1, 1)),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2),
            # ngf*2 x t x 10
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2 * 2, ngf *2, (3, 3), (1, 1), (8, 1), dilation=(8, 1)),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(ngf * 2, ngf * 2, (3, 3), (1, 1), (4, 1), dilation=(4, 1)),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(ngf * 2, ngf * 1, (3, 4), (1, 2), (1, 1)),
            nn.BatchNorm2d(ngf * 1),
            nn.LeakyReLU(0.2),
            # ngf*1 x t x 20
        )
        self.deconv3 = nn.Sequential(

            nn.ConvTranspose2d(ngf * 1 * 2, ngf * 1, (3, 3), (1, 1), (2, 1), dilation=(2, 1)),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(ngf * 1, ngf * 1, (3, 3), (1, 1), (1, 1), dilation=(1, 1)),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(ngf * 1, 1, (3, 4), (1, 2), (1, 1)),
            # 1 x t x 40
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        h1 = self.conv1(x)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        bb, cc, tt, dd = h3.size()
        lstm_in = h3.permute(0, 2, 1, 3).contiguous().view(bb, tt, cc * dd)

        h0 = torch.zeros(self.layer_number * (self.bidirectional + 1), x.size(0), self.hidden_units).to(x)
        c0 = torch.zeros(self.layer_number * (self.bidirectional + 1), x.size(0), self.hidden_units).to(x)
        lstm_out, _ = self.lstm(lstm_in, (h0, c0))
        lstm_out = self.fc(lstm_out.contiguous().view(bb * tt, self.hidden_units * (self.bidirectional + 1)))
        lstm_out = lstm_out.view([bb, tt, self.ngf * 4, 5]).contiguous().permute(0, 2, 1, 3)

        d1 = self.deconv1(torch.cat([h3, lstm_out], 1))
        d2 = self.deconv2(torch.cat([d1, h2], 1))
        d3 = self.deconv3(torch.cat([d2, h1], 1))
        return d3.squeeze(1)
