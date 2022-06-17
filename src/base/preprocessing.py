import numpy as np

def min_max_scaler(data):
    """Min-Max Normalizer.
    
    Args:
      - data: raw data
      
    Returns:
      - norm_data: normalized data
      - min_: minimum values (for renormalization)
      - max_: maximum values (for renormalization)
    """
    if len(data.shape) != 3:
        raise ValueError("Input data shape should be (N, S, F).")

    # Parameters
    no, seq, dim = data.shape

    data_2d = data.copy().reshape((no*seq, dim))
    min_ = np.nanmin(data_2d, axis = 0)
    data_2d = data_2d - min_
      
    max_ = np.nanmax(data_2d, axis = 0)
    norm_data = data_2d / (max_ + 1e-7)
    norm_data = norm_data.reshape((no, seq, dim))
    
    return norm_data, (min_, max_)

def standard_scaler(data):
    """Standard Scaler.
    
    Args:
      - data: raw data
      
    Returns:
      - norm_data: normalized data
      - mean_: mean (for renormalization)
      - std_: standard deviation (for renormalization)
    """
    if len(data.shape) != 3:
        raise ValueError("Input data shape should be (N, S, F).")

    # Parameters
    no, seq, dim = data.shape

    data_2d = data.copy().reshape((no*seq, dim))
    mean_ = np.nanmean(data_2d, axis = 0)
    var_ = np.nanvar(data_2d, axis = 0)

    scaled_data = (data_2d - mean_)/np.sqrt(var_)
    scaled_data = scaled_data.reshape((no, seq, dim))

    return scaled_data, (mean_, var_)
