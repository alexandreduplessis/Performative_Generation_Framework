import numpy as np
import torch
import ot
from sklearn.utils import check_random_state

def mix_data(data1, data2, random_state=0):
    """
    Mix two datasets

    Parameters
    ----------
    data1 : Numpy array or Torch tensor
        First dataset
    data2 : Numpy array or Torch tensor
        Second dataset

    Returns
    -------
    data : Numpy array or Torch tensor
        Mixed dataset (if both datasets are of the same type, this type is preserved, otherwise it is a Torch tensor)
    """
    rng = check_random_state(random_state)
    if isinstance(data1, np.ndarray) and isinstance(data2, np.ndarray):
        data = np.concatenate([data1, data2], axis=0)
        rng.shuffle(data)
    elif isinstance(data1, torch.Tensor) and isinstance(data2, torch.Tensor):
        data = torch.cat([data1, data2], axis=0)
        data = data[torch.randperm(data.size()[0])]
    elif isinstance(data1, np.ndarray) and isinstance(data2, torch.Tensor):
        data = torch.cat([torch.from_numpy(data1), data2], axis=0)
        data = data[torch.randperm(data.size()[0])]
    elif isinstance(data1, torch.Tensor) and isinstance(data2, np.ndarray):
        data = torch.cat([data1, torch.from_numpy(data2)], axis=0)
        data = data[torch.randperm(data.size()[0])]
    else:
        raise TypeError(f"Combinaison of data types {type(data1)} and {type(data2)} is not supported")
    return data

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
