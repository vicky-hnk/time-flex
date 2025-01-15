import torch
import torch.nn as nn


class NLinear(nn.Module):
    """
    Normalization-Linear model.
    """

    def __init__(self, params):
        super().__init__()
        self.seq_len = params.seq_len
        self.pred_len = params.pred_len

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter(
        # (1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.variates = params.num_variates
        self.separate = params.separate
        if self.separate:
            self.Linear = nn.ModuleList()
            for i in range(self.variates):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.separate:
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)],
                                 dtype=x.dtype).to(x.device)
            for i in range(self.variates):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x  # [Batch, Output length, Channel]
