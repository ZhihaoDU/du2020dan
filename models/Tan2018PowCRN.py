import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, ngf=None, ngh=None, opts=None):
        super(Generator, self).__init__()
        # 1 x T x 257
        self.conv1 = nn.Sequential(
            nn.ReplicationPad2d((0, 0, 1, 0)),
            nn.Conv2d(1, 16, (2, 3), (1, 2)),
            nn.BatchNorm2d(16),
            nn.ELU(),
        )
        # 16 x T x 128
        self.conv2 = nn.Sequential(
            nn.ReplicationPad2d((0, 0, 1, 0)),
            nn.Conv2d(16, 32, (2, 3), (1, 2)),
            nn.BatchNorm2d(32),
            nn.ELU(),
        )
        # 32 x T x 63
        self.conv3 = nn.Sequential(
            nn.ReplicationPad2d((0, 0, 1, 0)),
            nn.Conv2d(32, 64, (2, 3), (1, 2)),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )
        # 64 x T x 31
        self.conv4 = nn.Sequential(
            nn.ReplicationPad2d((0, 0, 1, 0)),
            nn.Conv2d(64, 128, (2, 3), (1, 2)),
            nn.BatchNorm2d(128),
            nn.ELU(),
        )
        # 128 x T x 15
        self.conv5 = nn.Sequential(
            nn.ReplicationPad2d((0, 0, 1, 0)),
            nn.Conv2d(128, 256, (2, 3), (1, 2)),
            nn.BatchNorm2d(256),
            nn.ELU(),
        )
        # 256 x T x 7
        # T x 1024
        self.lstm1 = nn.LSTM(7 * 256, 1024, 1, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(1024, 7 * 256, 1, batch_first=True, bidirectional=False)
        # T x 1024
        # 256 x T x 4
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(256*2, 128, (2, 3), (1, 2)),
            nn.BatchNorm2d(128),
            nn.ELU(),
        )
        # 128 x T+1 x 9
        # 128 x T x 9
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128 * 2, 64, (2, 3), (1, 2)),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )
        # 64 x T+1 x 19
        # 64 x T x 19
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64 * 2, 32, (2, 3), (1, 2)),
            nn.BatchNorm2d(32),
            nn.ELU(),
        )
        # 32 x T+1 x 39
        # 32 x T x 39
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32 * 2, 16, (2, 3), (1, 2), output_padding=(0, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
        )
        # 16 x T+1 x 80
        # 16 x T x 80
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(16 * 2, 1, (2, 3), (1, 2)),
            # nn.BatchNorm2d(1),
            # nn.Softplus(),
        )
        # 16 x T+1 x 161
        # 16 x T x 161

    def forward(self, input):
        x = input.unsqueeze(1)
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        bb, cc, tt, dd = c5.size()
        lstm_in = c5.permute(0, 2, 1, 3).contiguous().view(bb, tt, cc * dd)
        h0 = torch.zeros(1, x.size(0), 1024).to(x)
        c0 = torch.zeros(1, x.size(0), 1024).to(x)
        lstm_out, _ = self.lstm1(lstm_in, (h0, c0))

        h0 = torch.zeros(1, x.size(0), 7 * 256).to(x)
        c0 = torch.zeros(1, x.size(0), 7 * 256).to(x)
        lstm_out, _ = self.lstm2(lstm_out, (h0, c0))

        d5 = lstm_out.view(bb, tt, cc, dd).contiguous().permute(0, 2, 1, 3)
        d4 = self.deconv5(torch.cat([d5, c5], 1))
        d4 = d4[:, :, 1:, :]
        d3 = self.deconv4(torch.cat([d4, c4], 1))
        d3 = d3[:, :, 1:, :]
        d2 = self.deconv3(torch.cat([d3, c3], 1))
        d2 = d2[:, :, 1:, :]
        d1 = self.deconv2(torch.cat([d2, c2], 1))
        d1 = d1[:, :, 1:, :]
        out = self.deconv1(torch.cat([d1, c1], 1))
        out = out[:, :, 1:, :]
        return out.squeeze(1)
