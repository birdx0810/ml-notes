import torch

class ClassificationDataset(torch.utils.data.Dataset):
    r"""Time-Series classification dataset.

    Args:
        data (numpy.ndarray): The time-series dataset with shape 
            :math:`(B, S, F)`.
        time (numpy.ndarray, optional): The sequence lengths for each data with 
            shape :math:`(B,)`.
        labels (numpy.ndarray): The label for each sequence with shape 
            :math:`(B)`.

    Attributes:
        X (torch.FloatTensor): The time-series data with shape 
            :math:`(B, S, F)`.
        T (torch.LongTensor): The sequence lengths for each data with 
            shape :math:`(B,)`.
        labels (Union[torch.LongTensor, torch.FloatTensor]): The label for each 
            sequence with shape :math:`(B,)`. Use FloatTensor for binary 
            classification, and LongTensor otherwise.
    """

    def __init__(self, data, time, labels):
        self.X = torch.FloatTensor(data)     # Remove label column
        self.T = torch.LongTensor(time)
        self.Y = torch.LongTensor(labels)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx], self.Y[idx]

class NextStepPredictionDataset(torch.utils.data.Dataset):
    r"""Time-Series prediction dataset.
    
    Args:
        data (numpy.ndarray): The time-series dataset with shape (B, S, F).
        time (numpy.ndarray, optional): The sequence lengths for each data with 
            shape (B,).
    """

    def __init__(self):
        pass

class AnomalyDetectionDataset(torch.utils.data.Dataset):
    r"""
    """

    def __init__(self):
        pass

if __name__ == "__main__":
    pass
