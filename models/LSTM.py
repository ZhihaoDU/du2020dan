import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, opts):
        super(Generator, self).__init__()
        self.layer_number = opts['layer_number']
        self.hidden_units = opts['hidden_units']
        self.input_size = opts['input_size']
        self.bidirectional = opts['bidirectional']
        if opts["dropout"] is None:
            opts["dropout"] = 0
        self.dropout = dropout = opts["dropout"]

        self.lstm = nn.LSTM(self.input_size, self.hidden_units, self.layer_number, batch_first=True,
                            bidirectional=self.bidirectional, dropout=dropout)
        self.fc = nn.Linear(self.hidden_units * (self.bidirectional + 1), self.input_size)

    def forward(self, input):
        h0 = torch.zeros(self.layer_number * (self.bidirectional+1), input.size(0), self.hidden_units).to(input)
        c0 = torch.zeros(self.layer_number * (self.bidirectional+1), input.size(0), self.hidden_units).to(input)
        x, _ = self.lstm(input, (h0, c0))
        bb, tt, dd = x.size()
        out = self.fc(x.contiguous().view(bb*tt, dd).contiguous())
        out = out.view(bb, tt, self.input_size)
        return out
