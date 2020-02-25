import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, ngf=None, ngh=None, opts=None):
        super(Generator, self).__init__()
        # 1 x T x 40
        self.conv1 = nn.Sequential(
            nn.ReplicationPad2d((0, 1, 1, 0)),
            nn.Conv2d(1, 16, (2, 3), (1, 2)),
            nn.BatchNorm2d(16),
            nn.ELU(),
        )
        # 16 x T x 20
        self.conv2 = nn.Sequential(
            nn.ReplicationPad2d((0, 1, 1, 0)),
            nn.Conv2d(16, 32, (2, 3), (1, 2)),
            nn.BatchNorm2d(32),
            nn.ELU(),
        )
        # 32 x T x 10
        self.conv3 = nn.Sequential(
            nn.ReplicationPad2d((0, 1, 1, 0)),
            nn.Conv2d(32, 64, (2, 3), (1, 2)),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )
        # 64 x T x 5
        self.conv4 = nn.Sequential(
            nn.ReplicationPad2d((0, 1, 1, 0)),
            nn.Conv2d(64, 128, (2, 3), (1, 2)),
            nn.BatchNorm2d(128),
            nn.ELU(),
        )
        # 128 x T x 2
        self.conv5 = nn.Sequential(
            nn.ReplicationPad2d((0, 0, 1, 0)),
            nn.Conv2d(128, 256, (2, 2), (1, 1)),
            nn.BatchNorm2d(256),
            nn.ELU(),
        )
        # 256 x T x 1
        # T x 256
        self.lstm1 = nn.LSTM(256, 1024, 1, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(1024, 256, 1, batch_first=True, bidirectional=False)
        # T x 256
        # 256 x T x 1
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(256*2, 128, (2, 2), (1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU(),
        )
        # 128 x T+1 x 2
        # 128 x T x 2
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128 * 2, 64, (2, 3), (1, 2)),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )
        # 64 x T+1 x 5
        # 64 x T x 5
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64 * 2, 32, (2, 3), (1, 2)),
            nn.BatchNorm2d(32),
            nn.ELU(),
        )
        # 32 x T+1 x 10
        # 32 x T x 10
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32 * 2, 16, (2, 3), (1, 2)),
            nn.BatchNorm2d(16),
            nn.ELU(),
        )
        # 16 x T+1 x 20
        # 16 x T x 20
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(16 * 2, 1, (2, 3), (1, 2)),
            # nn.BatchNorm2d(1),
            # nn.Softplus(),
        )
        # 16 x T+1 x 40
        # 16 x T x 40

    def forward(self, input):
        x = input.unsqueeze(1)
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        bb, cc, tt, dd = c5.size()
        lstm_in = c5.permute(0, 2, 1, 3).contiguous().view(bb, tt, cc * dd)
        hidden1 = torch.zeros(1, x.size(0), 1024).to(x)
        cell1 = torch.zeros(1, x.size(0), 1024).to(x)
        lstm_out, _ = self.lstm1(lstm_in, (hidden1, cell1))

        hidden2 = torch.zeros(1, x.size(0), 256).to(x)
        cell2 = torch.zeros(1, x.size(0), 256).to(x)
        lstm_out, _ = self.lstm2(lstm_out, (hidden2, cell2))

        d5 = lstm_out.view(bb, tt, cc, dd).contiguous().permute(0, 2, 1, 3)
        d4 = self.deconv5(torch.cat([d5, c5], 1))
        d4 = d4[:, :, 1:, :]
        d3 = self.deconv4(torch.cat([d4, c4], 1))
        d3 = d3[:, :, 1:, :]
        d2 = self.deconv3(torch.cat([d3, c3], 1))
        d2 = d2[:, :, 1:, :-1]
        d1 = self.deconv2(torch.cat([d2, c2], 1))
        d1 = d1[:, :, 1:, :-1]
        out = self.deconv1(torch.cat([d1, c1], 1))
        out = out[:, :, 1:, :-1]
        return out.squeeze(1)
