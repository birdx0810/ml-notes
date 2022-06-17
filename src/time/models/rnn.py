import torch

class RNN(torch.nn.Module):
    """A general RNN model for time-series analysis. This model could be used for 
    classification, regression, encoder, or decoder.

    .. figure:: _images/rnn-image.png
        :align: center

        RNN model architecture visualization. The RNN Layer could consist of one
        or more RNN layers.
    
    Args:
        module (str): The architecture for the model. Choose from: 
            ``["rnn", "lstm", "gru"]``.
        input_size (int): The input dimension of the model, this is the last 
            (feature) dimension of the dataset. Denoted as :math:`F`.
        hidden_size (int): The hidden dimension between the rnn layer and linear
            layer. Denoted as :math:`H`.
        output_size (int): The output dimension of the linear layer. Denoted as 
            :math:`O`.
        num_layers (int): The number of layers of the RNN layer.
        bidirectional (bool): Flag for bi-directional RNN.
        dropout (float): The dropout probability for the RNN layer.
        padding_value (float): The padding value of the dataset. 
        max_seq_len (int): The maximum sequence length of the dataset.
    """
    def __init__(
        self,
        module_name: str,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        bidirectional: bool,
        dropout: float,
        padding_value: float,
        max_seq_len: int,
        *args, 
        **kwargs
    ):
        super(RNN, self).__init__()
        
        # Sanity check for input
        available_modules = ["rnn", "lstm", "gru"]
        if module_name not in available_modules:
            raise ValueError(
                f"Module architecture should be either " +
                f"{available_modules}."
            )

        # Model hyperparameters
        self.module_name = module_name
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.padding_value = padding_value
        self.max_seq_len = max_seq_len

        # Model architecture
        self.rnn_module = self.__get_rnn_module()
        self.rnn_layer = self.rnn_module(
            input_size=self.input_size, 
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )

    def __get_rnn_module(self):
        r"""The method for initializing RNN layer. This is a private method and 
        should only be used during model initialization.
        """
        if self.module_name == "rnn":
            return torch.nn.RNN
        elif self.module_name == "lstm":
            return torch.nn.LSTM
        elif self.module_name == "gru":
            return torch.nn.GRU

    def forward(self, X, T) -> torch.FloatTensor:
        r"""The forward pass for the model.

        Args:
            X (torch.FloatTensor): The input data with shape :math:`(B, S, F)`.
            T: (torch.LongTensor): The sequence length for each data with shape 
                :math:`(B,)`. This should not be in cuda device.

        Returns:
            The output logits of the model with shape :math:`(B, S, O)`.
        """
        # Dynamic RNN input for ignoring paddings
        X_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=X, 
            lengths=T, 
            batch_first=True, 
            enforce_sorted=False
        )

        # (B, S, F) -> (B, S, H)
        H_o, H_t = self.rnn_layer(X_packed)
        
        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o, 
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len
        )

        # (B, S, H) -> (B, S, O)
        logits = self.linear_layer(H_o)

        return logits

if __name__ == "__main__":
    pass