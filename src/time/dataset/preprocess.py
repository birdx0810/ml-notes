import numpy as np
import pandas as pd
from sktime.utils.data_io import load_from_tsfile_to_dataframe

PATHS = {
    "EyesOpenShut": {
        "train": "./dataset/raw/EyesOpenShut/EyesOpenShut_TRAIN.ts",
        "test": "./dataset/raw/EyesOpenShut/EyesOpenShut_TEST.ts",
    },
    "StandWalkJump":{
        "train": "./dataset/raw/StandWalkJump/StandWalkJump_TRAIN.ts",
        "test": "./dataset/raw/StandWalkJump/StandWalkJump_TEST.ts",
    }
}

def load(dataset_name):
    r"""Function for loading raw data file and process them into 
    `numpy.ndarrays`.

    .. code-block:: python
        [
            {
                "data": np.ndarray,
                "label": int
            },
            ...
        ]
    
    Args:
        dataset_name (str): The name of the classification dataset.
    
    Returns:
        ``List`` of ``Tuples`` containing:
            - train_X, train_Y: Training data with shape (N, S, F) and labels
                with shape (N).
            - test_X, test_Y: Testing data with shape (N, S, F) and labels
                with shape (N).
    """
    output = []
    for split in ("train", "test"):    
        raw = load_from_tsfile_to_dataframe(PATHS[dataset_name][split])

        # X.shape: (N, F)
        df, labels = raw[0], raw[1]

        data = []
        for feats in df.to_numpy():
            instance = np.array([feat.to_numpy() for feat in feats]).transpose()
            data.append(instance)

        data = np.array(data)
        output.append((data, labels))

    return output

if __name__ == "__main__":
    (train_x, train_y), (test_x, test_y) = load("EyesOpenShut")
    print(f"train_x: {train_x.shape}")      # (56, 128, 14)
    print(f"train_y: {train_y.shape}")      # (56,)
    print(f"test_x: {test_x.shape}")        # (42, 128, 14)
    print(f"test_y: {test_y.shape}")        # (42,)

    (train_x, train_y), (test_x, test_y) = load("StandWalkJump")
    print(f"train_x: {train_x.shape}")      # (12, 2500, 4)
    print(f"train_y: {train_y.shape}")      # (12,)
    print(f"test_x: {test_x.shape}")        # (15, 2500, 4)
    print(f"test_y: {test_y.shape}")        # (15,)
