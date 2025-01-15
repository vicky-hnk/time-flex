import torch.nn as nn
import torch.nn.functional as F


class VCEncoderLayer(nn.Module):
    def __init__(self, attention, ktd, d_model, d_ff=None, dropout=0.1,
                 activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.ktd = ktd
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.conv1 = nn.Conv1d(in_channels=d_model,
                               out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attention_mask=None, tau=None, delta=None):
        # VCA module
        new_x, attn = self.attention(
            x, x, x,
            attention_mask=attention_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)

        # KTD Module
        x_ktd = y
        y = self.dropout(self.ktd(x_ktd))

        return self.norm2(x_ktd + y), attn
