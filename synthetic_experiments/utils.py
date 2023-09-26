import ot
import numpy as np
import torch

def wasserstein_distance(data1, data2):
    """
    Compute the Wasserstein distance between two datasets

    Parameters
    ----------
    data1 : Numpy array or Torch tensor
        First dataset
    data2 : Numpy array or Torch tensor
        Second dataset

    Returns
    -------
    wasserstein_distance : float
        Wasserstein distance between the two datasets
    """
    if isinstance(data1, torch.Tensor):
        data1 = data1.cpu().numpy()
    if isinstance(data2, torch.Tensor):
        data2 = data2.cpu().numpy()
    M = ot.dist(data1, data2)
    ab = np.ones(data1.shape[0]) / data1.shape[0]
    return ot.emd2(ab, ab, M)