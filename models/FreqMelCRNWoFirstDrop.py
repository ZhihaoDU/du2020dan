import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, opts=None):
        super(Generator, self).__init__()
        self.bidirectional = bidirectional = opts['bidirectional']
        self.ngf = ngf = opts['ngf']
        self.hidden_units = hidden_units = opts["hidden_units"]
        self.layer_number = layer_number = opts["layer_number"]
        if opts["dropout"] is None:
            opts["dropout"] = 0
        self.dropout = dropout = opts["dropout"]
        # 1 x T x 40
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, ngf, (3, 4), (1, 2), (1, 1)),
            nn.BatchNorm2d(ngf),
            nn.ELU(),
        )
        # 16 x T x 20
        self.conv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf*2, (3, 4), (1, 2), (1, 1)),
            nn.BatchNorm2d(ngf*2),
            nn.ELU(),
            nn.Dropout2d(dropout)
        )
        # 32 x T x 10
        self.conv3 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*4, (3, 4), (1, 2), (1, 1)),
            nn.BatchNorm2d(ngf*4),
            nn.ELU(),
            nn.Dropout2d(dropout)
        )
        # 64 x T x 5
        self.conv4 = nn.Sequential(
            nn.Conv2d(ngf*4, ngf*8, (3, 4), (1, 2), (1, 1)),
            nn.BatchNorm2d(ngf*8),
            nn.ELU(),
            nn.Dropout2d(dropout)
        )
        # 128 x T x 2
        self.conv5 = nn.Sequential(
            nn.Conv2d(ngf*8, ngf*16, (1, 2), (1, 1), (0, 0)),
            nn.BatchNorm2d(ngf*16),
            nn.ELU(),
            nn.Dropout2d(dropout)
        )
        # 256 x T x 1
        # T x 256
        self.lstm = nn.LSTM(ngf*16, hidden_units, layer_number, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_units * (bidirectional + 1), ngf * 16),
            nn.BatchNorm1d(ngf * 16),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        # T x 256
        # 256 x T x 1
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(ngf*16*2, ngf*8, (1, 2), (1, 1), (0, 0)),
            nn.BatchNorm2d(ngf*8),
            nn.ELU(),
            nn.Dropout2d(dropout)
        )
        # 128 x T x 2
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(ngf*8*2, ngf*4, (3, 5), (1, 2), (1, 1)),
            nn.BatchNorm2d(ngf*4),
            nn.ELU(),
            nn.Dropout2d(dropout)
        )
        # 64 x T x 5
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(ngf*4*2, ngf*2, (3, 4), (1, 2), (1, 1)),
            nn.BatchNorm2d(ngf*2),
            nn.ELU(),
            nn.Dropout2d(dropout)
        )
        # 32 x T x 10
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(ngf*2*2, ngf, (3, 4), (1, 2), (1, 1)),
            nn.BatchNorm2d(ngf),
            nn.ELU(),
        )
        # 16 x T x 20
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, 1, (3, 4), (1, 2), (1, 1)),
        )
        # 1 x T x 40

    def forward(self, input):
        x = input.unsqueeze(1)
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        bb, cc, tt, dd = c5.size()
        lstm_in = c5.permute(0, 2, 1, 3).contiguous().view(bb, tt, cc * dd)
        h0 = torch.zeros(self.layer_number * (self.bidirectional + 1), x.size(0), self.hidden_units).to(x)
        c0 = torch.zeros(self.layer_number * (self.bidirectional + 1), x.size(0), self.hidden_units).to(x)
        lstm_out, _ = self.lstm(lstm_in, (h0, c0))

        fc_in = lstm_out.contiguous().view(bb * tt, self.hidden_units * (self.bidirectional + 1))
        fc_out = self.fc(fc_in)

        d5 = fc_out.view(bb, tt, cc, dd).contiguous().permute(0, 2, 1, 3)
        d4 = self.deconv5(torch.cat([d5, c5], 1))
        d3 = self.deconv4(torch.cat([d4, c4], 1))
        d2 = self.deconv3(torch.cat([d3, c3], 1))
        d1 = self.deconv2(torch.cat([d2, c2], 1))
        out = self.deconv1(torch.cat([d1, c1], 1))
        return out.squeeze(1)
