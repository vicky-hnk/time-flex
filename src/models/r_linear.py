import torch
import torch.nn as nn
from src.models.layers.normalization import RevIN


class RLinear(nn.Module):
    """
    Remark: In contrast to the original implementation we calculate the loss
    outside of this model class to maintain consistency with other models.
    """
    def __init__(self, params):
        super().__init__()
        self.pred_len = params.pred_len

        # set device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.Linear = nn.ModuleList([
            nn.Linear(params.seq_len, params.pred_len) for _ in
            range(params.num_variates)
        ]) if params.separate else nn.Linear(params.seq_len,
                                             params.pred_len)

        self.dropout = nn.Dropout(params.dropout)
        self.revin = RevIN(params.num_variates) if params.use_revin else None
        self.separate = params.separate

    def forward(self, x):
        # x: [batch, pred_len, variates]
        x = self.revin(x, 'norm') if self.revin else x
        x = self.dropout(x)
        if self.separate:
            pred = torch.zeros([x.size(0), self.pred_len, x.size(2)],
                               dtype=x.dtype).to(self.device)
            for idx, proj in enumerate(self.Linear):
                pred[:, :, idx] = proj(x[:, :, idx])
        else:
            pred = self.Linear(x.transpose(1, 2)).transpose(1, 2)
        pred = self.revin(pred, 'denorm') if self.revin else pred

        return pred
