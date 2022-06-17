import torch
from models.general.moduls import CNNBlock

class FCN(torch.nn.Module):
    r"""A Fully Convolutional-NN model for time-series based on . 
    This model could be used for classification.

    """
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        dropout,
        *args,
        **kwargs,
    ):
        super(FCN, self).__init__()
        # Model hyperparameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        
        # Model architecture
        self.features = torch.nn.Sequential(
            CNNBlock(input_dim, hidden_dim, 8, activation=True),
            CNNBlock(hidden_dim, hidden_dim*2, 5, activation=True),
            CNNBlock(hidden_dim*2, hidden_dim, 3, activation=True),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, x):
        # (B, S, F) -> (B, S, H)
        x = self.features(x)
        # (B, S, H) -> (B, H)
        x = self.avgpool(x).squeeze()   # This could actually with `mean` over S-axis
        # (B, H) -> (B, C)
        logits = self.classifier(x)
        
        return logits
