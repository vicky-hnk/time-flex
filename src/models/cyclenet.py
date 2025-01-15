import torch
import torch.nn as nn


class RecurrentCycle(torch.nn.Module):
    """
    A recurrent cycle buffer that stores and manipulates cyclic data using
    torch tensors.

    Attributes:
        cycle_len (int): Length of the cycle.
        channel_size (int): Number of features (channels).
        data (torch.nn.Parameter): A tensor of shape (cycle_len, channel_size)
        that stores the cyclic data and allows gradient updates.
    """

    def __init__(self, cycle_len, channel_size):
        super().__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size),
                                       requires_grad=True)

    def forward(self, index, length):
        """
        Forward pass for the RecurrentCycle module.
        Rolls the cyclic data according to the given index and extracts
        a portion of the data to match the specified length.

        Parameters:
            :params index: A tensor of indices used to roll the data along the
            cycle length. Each element represents how much to shift for each
            batch element.
            :params length: The length of the output sequence to be extracted
             the cyclic data.
        """
        rolled_data = torch.stack(
            [torch.roll(self.data, shifts=(-i.item(), 0), dims=(0, 1)) for i in
             index])

        outputs = []
        for i in range(rolled_data.size(0)):
            if length <= self.cycle_len:
                output = rolled_data[i, :length]
            else:
                # Repeat cyclic data if requested length exceeds cycle length.
                num_repeats = length // self.cycle_len
                remainder = length % self.cycle_len
                output = torch.cat([rolled_data[i]] * num_repeats + [
                    rolled_data[i, :remainder]])
            outputs.append(output)

        return torch.stack(outputs)


class RecurrentCycleNew(torch.nn.Module):
    # Thanks for the contribution of wayhoww.
    # The new implementation uses index arithmetic with modulo to directly
    # gather cyclic data in a single operation,
    # while the original implementation manually rolls and repeats the
    # data through looping.
    # It achieves a significant speed improvement (2x ~ 3x acceleration).
    # See https://github.com/ACAT-SCUT/CycleNet/pull/4 for more details.
    def __init__(self, cycle_len, channel_size):
        super().__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size),
                                       requires_grad=True)

    def forward(self, index, length):
        gather_index = (index.view(-1, 1) + torch.arange(
            length, device=index.device).view(
            1, -1)) % self.cycle_len
        return self.data[gather_index]


class CycleNet(nn.Module):
    """
    A forecasting model with optional recurrent cycle buffering and sequence
    normalization. The model can use either a linear or multi-layer perceptron
    (MLP) architecture.

    Attributes:
        seq_len (int): The input sequence length.
        pred_len (int): The output sequence length for prediction.
        enc_in (int): Number of input channels (features).
        cycle_len (int): The cycle length for the recurrent buffer.
        model_type (str): The type of model to use ('linear' or 'mlp').
        d_model (int): Hidden layer dimensionality (only used for 'mlp').
        use_revin (bool): Flag for using instance normalization.
        cycleQueue (RecurrentCycle): A recurrent cycle buffer for handling
        cyclic data.
        model (torch.nn.Module): The chosen model (linear or MLP).

    Methods:
        forward(x, cycle_index):
            Applies optional instance normalization, removes cyclic data from
            input, and performs forecasting using the model.
            Adds cyclic data back to the prediction before returning the final
            output.
    """

    def __init__(self, params):
        super().__init__()

        self.seq_len = params.seq_len
        self.pred_len = params.pred_len
        self.enc_in = params.enc_in
        self.cycle_len = params.cycle
        self.model_type = params.model_type
        self.d_model = params.model_dim
        self.use_revin = params.use_revin
        print("RevIN is set to ", self.use_revin)

        self.cycleQueue = RecurrentCycleNew(cycle_len=self.cycle_len,
                                            channel_size=self.enc_in)

        valid_model_types = ['linear', 'mlp']
        if self.model_type not in valid_model_types:
            raise ValueError(
                f"Invalid model_type '{self.model_type}'. "
                f"Expected one of: {valid_model_types}")

        if self.model_type == 'linear':
            self.model = nn.Linear(self.seq_len, self.pred_len)
        elif self.model_type == 'mlp':
            self.model = nn.Sequential(
                nn.Linear(self.seq_len, self.model_dim),
                nn.ReLU(),
                nn.Linear(self.model_dim, self.pred_len)
            )

    def forward(self, x, cycle_index):
        """
        Forward pass for the Model class. Performs forecasting on the input
        sequence with optional instance normalization.
        Cyclic data is subtracted from the input before passing it through the
        model, and added back to the model's output.

        Args:
            x (torch.Tensor): The input sequence tensor of shape
            (batch_size, seq_len, enc_in).
            cycle_index (torch.Tensor): A tensor of cycle indices for aligning
            the cyclic data.

        Returns:
            torch.Tensor: The output sequence tensor of shape
            (batch_size, pred_len, enc_in) after forecasting.
        """
        # Apply instance normalization if enabled
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # Remove the cycle from the input data
        x = x - self.cycleQueue(cycle_index, self.seq_len)

        # Forecasting using the model
        y = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Add the cycle back to the output data
        y = y + self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len,
                                self.pred_len)

        # Denormalize if instance normalization was applied
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y
