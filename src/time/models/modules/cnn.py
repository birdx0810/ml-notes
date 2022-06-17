import torch

class CNNBlock(torch.nn.Module):
    r"""A basic CNN module with batch norm and activation.
    """
    def __init__(
        self, 
        input_dim, 
        output_dim, 
        kernel_size, 
        stride=1, 
        padding="same", 
        activation="relu",
    ):
        # Module architecture
        self.cnn_layer = torch.nn.Conv1d(
            in_channels=input_dim, 
            out_channels=output_dim, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
        )
        self.bn_layer = torch.nn.BatchNorm1d(output_dim),
        self.activation = self._get_activation_fn(activation)

    def _get_activation_fn(self, activation):
        if activation == "relu":
            return torch.nn.ReLU(inplace=True)
        elif activation == "sigmoid":
            return torch.nn.Sigmoid()
        elif activation == "tanh":
            return torch.nn.Tanh()
        elif activation == None:
            return None
        else:
            raise ValueError(f"Unknown activation function {activation}.")

    def forward(self, x):
        h = self.cnn_layer(x)
        y = self.bn_layer(h)
        if self.activation is not None:
            y = self.activation(y)
        return y

class HighwayCNNBlock(torch.nn.Module):
    r"""Basic highway block.
    """
    def __init__(
        self,
        input_dim,
    ):
        super(HighwayCNNBlock, self).__init__()
        self.transform_layer = CNNBlock(input_dim, input_dim, activation="sigmoid")
        self.hidden_layer = CNNBlock(input_dim, input_dim, activation="relu")

    def forward(self, x):
        t = self.transform_layer(x)
        h = self.linear_layer(x)
        c = 1 - t
        return (t * h) + (c * x)

class ResidualBlock(torch.nn.Module):
    r"""Basic residual block. The input and output dimensions are the same. 
    """
    def __init__(
        self,
        input_dim,
    ):
        super(ResidualBlock, self).__init__()
        # Module architecture
        self.cnn_layer = torch.nn.Sequential(
            CNNBlock(input_dim, input_dim, 8, activation=True),
            CNNBlock(input_dim, input_dim, 5, activation=True),
            CNNBlock(input_dim, input_dim, 3, activation=False),
        )

    def forward(self, x):
        copy = x
        h = self.cnn_layer(x)
        h = h + copy   # Shortcut
        y = self.relu(h)
        
        return y

class BottleneckBlock(torch.nn.Module):
    r"""Basic bottleneck block.
    """
    def __init__(
        self,
        input_dim,
        hidden_dim,
    ):
        super(BottleneckBlock, self).__init__()

        # Module architecture
        self.cnn_layer = torch.nn.Sequential(
            CNNBlock(input_dim, hidden_dim, 1, activation=True),    # Downsample
            CNNBlock(hidden_dim, hidden_dim, 3, activation=True), 
            CNNBlock(hidden_dim, input_dim, 1, activation=False),   # Upsample
        )
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        copy = x
        h = self.cnn_layer(x)
        h = h + copy   # Shortcut
        y = self.relu(h)

        return y

class SqueezeExciteBlock(torch.nn.Module):
    r"""TODO: Basic squeeze excite block.
    """
    def __init__(self):
        super(SqueezeExciteBlock).__init__()
        pass

    def forward(self):
        pass
