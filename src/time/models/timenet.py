import torch

class TimeNetEncoder(torch.nn.Module):
    r"""The encoder model for TimeNet, this is also the model that will be used 
    after the pre-training process and will be used for downstream tasks.
    
    Args:
        feature_dim (int):
        hidden_dim (int):
        rnn_layers (int):
        bidirectional (bool)
        dropout (float)
        padding_value (float):
        max_seq_len (int):
    """
    def __init__(
        self, 
        feature_dim: int,
        hidden_dim: int,
        rnn_layers: int,
        bidirectional: bool,
        dropout: float,
        padding_value: float,
        max_seq_len: int,
    ):
        super(TimeNetEncoder, self).__init__()

        # Sanity check for input
        available_modules = ["rnn", "lstm", "gru"]
        if module_name not in available_modules:
            raise ValueError(
                f"Module architecture should be either " +
                f"{available_modules}."
            )

        # Model hyperparameters
        self.module_name = module_name
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = rnn_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.padding_value = padding_value
        self.max_seq_len = max_seq_len

        # Encoder Architecture
        self.rnn_module = self.__get_rnn_module()
        self.rnn_layer = self.rnn_module(
            input_size=self.feature_dim, 
            hidden_size=self.hidden_dim, 
            rnn_layers=self.num_layers, 
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

        # Init weights (Xavier Uniform and Ones/Zeros)
        with torch.no_grad():
            for name, param in self.rnn_layer.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)

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

    def forward(self, X, T):
        """Forward pass for encoding the time-series data to latent space.
        Args:
            - X: input time-series features (B, S, F)
            - T: input temporal information (B)
        Returns:
            - H_o: latent space embeddings (B, S, H)
            - H_t: (S, H)
        """
        # Dynamic RNN input for ignoring paddings
        X_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=X, 
            lengths=T, 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # (B, S, F) -> (D x B, S, H), if bidirectional D == 2, else 1.
        H_o, H_n = self.rnn_layer(X_packed)
        
        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o, 
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len
        )
        
        return H_o, H_n

class TimeNetDecoder(torch.nn.Module):
    """The decoder model for TimeNet, this is used for the SAE pre-training 
    objective. The decoder is thought of to be like a autoregressive generator, 
    and therefore should not be bidirectional.
    
    Args:
        hidden_dim (int): 
        feature_dim (int): 
        rnn_layers (int): 
        padding_value (float): 
        dropout (float): 
        max_seq_len (int): 
    """
    def __init__(
        self,
        hidden_dim: int,
        feature_dim: int,
        rnn_layers: int,
        padding_value: float,
        dropout: float,
        max_seq_len: int,
    ):
        super(TimeNetDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.num_layers = rnn_layers
        self.padding_value = padding_value
        self.dropout = dropout 
        self.max_seq_len = max_seq_len

        # Decoder Architecture
        self.rnn_module = self.__get_rnn_module()
        self.rnn_layer = self.rnn_module(
            input_size=self.hidden_dim, 
            hidden_size=self.hidden_dim, 
            rnn_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True
        )
        self.linear_layer = torch.nn.Linear(self.hidden_dim, self.feature_dim)

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

    def forward(self, H_o, H_n, T):
        """Forward pass for the recovering features from latent space to 
        original feature space.
        
        Args:
            - H_o (torch.FloatTensor): Output of encoder H_n with shape 
                (B, S, H).
            - H_n (torch.FloatTensor): Hidden state of encoder with shape 
                (D * L, B, H).
            - T (torch.LongTensor): input temporal information (B)
        Returns:
            - X_hat (torch.FloatTensor): recovered data (B, S, F)
        """
        # Get the last step of the hidden state as input
        H_o = H_o[:, -1, :]

        Y = []
        # When t = 0, the hidden state of the encoder is used. The hidden state 
        # of the previous time-step would be used otherwise
        for _ in self.max_seq_len:
            H_o, H_n = self.rnn_layer(H_o, H_n)
            Y.append(H_o)
        
        Y = torch.stack(Y)

        # (B, S, F)
        X_hat = self.linear_layer(Y)

        # Reverse sequences
        X_hat = X_hat.flip(dims=(0,1))
        
        # Pad sequences to target length
        for idx, t in enumerate(T):
            if t != self.max_seq_len:
                X_hat[:, t:, :] = -1

        return X_hat

class TimeNetClassifier(torch.nn.Module):
    r"""The classification module for downstream task (fine-tuning).
    """
    def __init__(
        self,
        hidden_dim,
        no_classes,
        cls_layers,
        dropout,
    ):
        # Model hyperparameters
        self.hidden_dim = hidden_dim
        self.output_dim = no_classes
        self.num_layers = cls_layers
        self.dropout = dropout
        if no_classes == 2:
            self.output_dim = 1

        # Model architecture
        self.hidden_layer = torch.nn.Sequential(
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
        )
        self.cls_layer = torch.nn.Sequential(
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_dim, self.output_dim)
        )

        if self.num_layers == 1:
            self.classifier = torch.nn.ModuleList([
                self.cls_layer
            ])
        else:
            self.classifier = torch.nn.ModuleList(
                [self.hidden_layer for _ in range(self.num_layers)-1] +
                self.cls_layer
            )

    def forward(self, X, T):
        logits = self.classifier(X)
        return logits

class TimeNet(torch.nn.Module):
    r"""A Sequence Auto-Encoder (SAE) for time-series as proposed by Malhotra et 
    al. (2017). 

    Reference: https://arxiv.org/abs/1706.08838
    """

    def __init__(self, *args, **kwargs):
        self.encoder = TimeNetEncoder(*args, **kwargs)
        self.decoder = TimeNetDecoder(*args, **kwargs)
        self.classifier = TimeNetClassifier(*args, **kwargs)

    def forward(self, X, T, obj) -> torch.FloatTensor:
        r"""The forward pass for TimeNet. 

        Args:
            X (torch.FloatTensor): The input time-series training data with 
                shape :math:`(B, S, F)`.
            T (torch.LongTensor): The sequence length of the input time-series 
                training data with shape :math:`(B,)`. 
            obj (str): The objective of the model. Objectives include `pretrain`
                and `finetune`.

        Returns:
            Output prediction of the models. The reconstructed time-series data 
            for the `pretrain` objective, and the output logits for the 
            `finetune` objective.
        """
        objectives = ["pretrain", "finetune"]
        
        # Module is in training mode
        self.train()

        if obj == "pretrain":
            pred = self.__pretraining(X, T)
        elif obj == "finetune":
            pred = self.__finetune(X, T)
        else:
            raise ValueError(
                f"The `obj` argument should be one of {objectives}."
            )

        return pred

    def __pretraining(self, X, T):
        H_o, H_n = self.encoder(X, T)
        X_hat = self.decoder(H_o, H_n, T)
        return X_hat

    def __finetune(self, X, T):
        H, _ = self.encoder(X, T)
        logits = self.classifier(H)
        return logits

    def inference(self, X, T):
        # Set model to evaluation mode
        self.eval()
        H, _ = self.encoder(X, T)
        logits = self.classifier(H)
        return logits
